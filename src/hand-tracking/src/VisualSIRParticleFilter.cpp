#include "VisualSIRParticleFilter.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <utility>

#include <Eigen/Dense>
#include <iCub/ctrl/math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <yarp/math/Math.h>
#include <yarp/eigen/Eigen.h>

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::eigen;
using namespace yarp::math;
using namespace yarp::os;
using yarp::sig::Vector;
using yarp::sig::ImageOf;
using yarp::sig::PixelRgb;


VisualSIRParticleFilter::VisualSIRParticleFilter(std::unique_ptr<Prediction> prediction, std::unique_ptr<VisualParticleFilterCorrection> correction,
                                                 std::unique_ptr<Resampling> resampling,
                                                 ConstString cam_sel, ConstString laterality, const int num_particles) :
    prediction_(std::move(prediction)), correction_(std::move(correction)),
    resampling_(std::move(resampling)),
    cam_sel_(cam_sel), laterality_(laterality), num_particles_(num_particles),
    icub_kin_arm_(iCubArm(laterality+"_v2")), icub_kin_finger_{iCubFinger(laterality+"_thumb"), iCubFinger(laterality+"_index"), iCubFinger(laterality+"_middle")}
{
    icub_kin_eye_ = iCubEye(cam_sel+"_v2");
    icub_kin_eye_.setAllConstraints(false);
    icub_kin_eye_.releaseLink(0);
    icub_kin_eye_.releaseLink(1);
    icub_kin_eye_.releaseLink(2);

    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    icub_kin_finger_[0].setAllConstraints(false);
    icub_kin_finger_[1].setAllConstraints(false);
    icub_kin_finger_[2].setAllConstraints(false);

    /* Left images:       /icub/camcalib/left/out
       Right images:      /icub/camcalib/right/out
       Head encoders:     /icub/head/state:o
       Arm encoders:      /icub/right_arm/state:o
       Torso encoders:    /icub/torso/state:o
       Right hand analog: /hand-tracking/right_hand/analog:i */
    port_image_in_left_.open    ("/hand-tracking/" + cam_sel_ + "_img:i");
    port_head_enc_.open         ("/hand-tracking/head:i");
    port_arm_enc_.open          ("/hand-tracking/" + laterality_ + "_arm:i");
    port_torso_enc_.open        ("/hand-tracking/torso:i");
    port_right_hand_analog_.open("/hand-tracking/right_hand/analog:i");

    right_hand_analogs_bounds_ = yarp::sig::Matrix(15, 2);
    
    right_hand_analogs_bounds_(0, 0)  = 218.0; right_hand_analogs_bounds_(0, 1)  =  61.0;
    right_hand_analogs_bounds_(1, 0)  = 210.0; right_hand_analogs_bounds_(1, 1)  =  20.0;
    right_hand_analogs_bounds_(2, 0)  = 234.0; right_hand_analogs_bounds_(2, 1)  =  16.0;

    right_hand_analogs_bounds_(3, 0)  = 224.0; right_hand_analogs_bounds_(3, 1)  =  15.0;
    right_hand_analogs_bounds_(4, 0)  = 206.0; right_hand_analogs_bounds_(4, 1)  =  18.0;
    right_hand_analogs_bounds_(5, 0)  = 237.0; right_hand_analogs_bounds_(5, 1)  =  23.0;

    right_hand_analogs_bounds_(6, 0)  = 250.0; right_hand_analogs_bounds_(6, 1)  =   8.0;
    right_hand_analogs_bounds_(7, 0)  = 195.0; right_hand_analogs_bounds_(7, 1)  =  21.0;
    right_hand_analogs_bounds_(8, 0)  = 218.0; right_hand_analogs_bounds_(8, 1)  =   0.0;

    right_hand_analogs_bounds_(9, 0)  = 220.0; right_hand_analogs_bounds_(9, 1)  =  39.0;
    right_hand_analogs_bounds_(10, 0) = 160.0; right_hand_analogs_bounds_(10, 1) =  10.0;
    right_hand_analogs_bounds_(11, 0) = 209.0; right_hand_analogs_bounds_(11, 1) = 101.0;
    right_hand_analogs_bounds_(12, 0) = 224.0; right_hand_analogs_bounds_(12, 1) =  63.0;
    right_hand_analogs_bounds_(13, 0) = 191.0; right_hand_analogs_bounds_(13, 1) =  36.0;
    right_hand_analogs_bounds_(14, 0) = 232.0; right_hand_analogs_bounds_(14, 1) =  98.0;

    port_estimates_out_.open  ("/hand-tracking/result/estimates:o");

    /* DEBUG ONLY */
    port_image_out_left_.open ("/hand-tracking/result/" + cam_sel_ + ":o");
    /* ********** */

    is_filter_init_ = false;
    is_running_     = false;

    setCommandPort();
}


VisualSIRParticleFilter::~VisualSIRParticleFilter() noexcept { }


void VisualSIRParticleFilter::runFilter()
{
    /* INITIALIZATION */
    // FIXME: page locked dovrebbe essere più veloce da utilizzate con CUDA, non sembra essere il caso.
//    Mat::setDefaultAllocator(cuda::HostMem::getAllocator(cuda::HostMem::PAGE_LOCKED));

    unsigned int  k = 0;

    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> init_hand_pose(ee_pose.data(), 6, 1);
    init_hand_pose.tail(3) *= ee_pose(6);

    VectorXd old_hand_pose = init_hand_pose;

    MatrixXf init_particle(6, num_particles_);
    for (int i = 0; i < num_particles_; ++i)
        init_particle.col(i) = init_hand_pose.cast<float>();

    VectorXf init_weight(num_particles_, 1);
    init_weight.setConstant(1.0/num_particles_);

    const int          block_size = 16;
    const int          img_width  = 320;
    const int          img_height = 240;
    const int          bin_number = 9;
    const unsigned int descriptor_length = (img_width/block_size*2-1) * (img_height/block_size*2-1) * bin_number * 4;
    Ptr<cuda::HOG> cuda_hog = cuda::HOG::create(Size(img_width, img_height), Size(block_size, block_size), Size(block_size/2, block_size/2), Size(block_size/2, block_size/2), bin_number);
    cuda_hog->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_COL_BY_COL);
    cuda_hog->setGammaCorrection(true);
    cuda_hog->setWinStride(Size(img_width, img_height));

    double left_cam_x[3];
    double left_cam_o[4];
    Vector left_eye_pose;
    Vector right_eye_pose;

    left_eye_pose = icub_kin_eye_.EndEffPose(CTRL_DEG2RAD * readRootToEye(cam_sel_));
    left_cam_x[0] = left_eye_pose(0); left_cam_x[1] = left_eye_pose(1); left_cam_x[2] = left_eye_pose(2);
    left_cam_o[0] = left_eye_pose(3); left_cam_o[1] = left_eye_pose(4); left_cam_o[2] = left_eye_pose(5); left_cam_o[3] = left_eye_pose(6);

    is_filter_init_ = true;


    /* FILTERING */
    ImageOf<PixelRgb>* imgin_left  = YARP_NULLPTR;
    while(is_running_)
    {
        Vector             q;
        Vector             analogs;
        std::vector<float> descriptors_cam_left (descriptor_length);
        cuda::GpuMat       cuda_img             (Size(img_width, img_height), CV_8UC3);
        cuda::GpuMat       cuda_img_alpha       (Size(img_width, img_height), CV_8UC4);
        cuda::GpuMat       descriptors_cam_cuda (Size(descriptor_length, 1),  CV_32F );

        imgin_left = port_image_in_left_.read(true);

        if (imgin_left != YARP_NULLPTR)
        {
            Vector& estimates_out = port_estimates_out_.prepare();

            ImageOf<PixelRgb>& imgout_left = port_image_out_left_.prepare();
            imgout_left = *imgin_left;

            MatrixXf temp_particle(6, num_particles_);
            VectorXf temp_weight(num_particles_, 1);
            VectorXf temp_parent(num_particles_, 1);

            Mat measurement;

            measurement = cvarrToMat(imgout_left.getIplImage());
            cuda_img.upload(measurement);
            cuda::cvtColor(cuda_img, cuda_img_alpha, COLOR_BGR2BGRA, 4);
            cuda_hog->compute(cuda_img_alpha, descriptors_cam_cuda);
            descriptors_cam_cuda.download(descriptors_cam_left);

            // FIXME: move the hand over time
            Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
            Map<VectorXd> new_arm_pose(ee_pose.data(), 6, 1);
            new_arm_pose.tail(3) *= ee_pose(6);

            VectorXd delta_hand_pose(6);
            double   delta_angle;
            delta_hand_pose.head(3) = new_arm_pose.head(3) - old_hand_pose.head(3);
            delta_angle             = new_arm_pose.tail(3).norm() - old_hand_pose.tail(3).norm();
            // FIXME: provare a tenerlo tra 0 e 2PI come fatto dopo
            if (delta_angle > 2.0 * M_PI) delta_angle -= 2.0 * M_PI;
//            if (delta_angle >  M_PI) delta_angle -= 2.0 * M_PI;
//            if (delta_angle < -M_PI) delta_angle += 2.0 * M_PI;
            delta_hand_pose.tail(3) = (new_arm_pose.tail(3) / new_arm_pose.tail(3).norm()) - (old_hand_pose.tail(3) / old_hand_pose.tail(3).norm());

            old_hand_pose = new_arm_pose;

            VectorXf sorted_pred = init_weight;
            std::sort(sorted_pred.data(), sorted_pred.data() + sorted_pred.size());

            float threshold = sorted_pred.tail(6)(0);
            for (int j = 0; j < num_particles_; ++j)
            {
                init_particle.col(j).head(3) += delta_hand_pose.head(3).cast<float>();

                float ang;
                ang = init_particle.col(j).tail(3).norm();
                init_particle.col(j).tail(3) /= ang;

                init_particle.col(j).tail(3) += delta_hand_pose.tail(3).cast<float>();
                init_particle.col(j).tail(3) /= init_particle.col(j).tail(3).norm();

                ang += static_cast<float>(delta_angle);
                if (ang > 2.0 * M_PI) ang -= 2.0 * M_PI;
//                if (ang >  M_PI) ang -= 2.0 * M_PI;
//                if (ang < -M_PI) ang += 2.0 * M_PI;
//                if (0      <= ang && ang < 0.002) ang =  0.002;
//                if (-0.002 <  ang && ang < 0)     ang = -0.002;
                init_particle.col(j).tail(3) *= ang;

                if(init_weight(j) <= threshold)
                    prediction_->predict(init_particle.col(j), init_particle.col(j));
            }

            /* Set parameters */
            // FIXME: da decidere come sistemare
            q = readRootToFingers();

            // FIXME: abduzione e apertura pollice fissata
            q(10) = 32.0;
            q(11) = 30.0;
//            q(12) = 0.0;
//            q(13) = 0.0;
//            q(14) = 0.0;
//            q(15) = 0.0;
//            q(16) = 0.0;
//            q(17) = 0.0;
            if (analogs_)
            {
                analogs = readRightHandAnalogs();
                itf_right_hand_analog_->read(analogs);
                correction_->setArmJoints(q, analogs, right_hand_analogs_bounds_);
            }
            else correction_->setArmJoints(q);

            VectorXf out_particle(6);

            correction_->setCamXO(left_cam_x, left_cam_o);
            correction_->correct(init_particle, descriptors_cam_left, init_weight);

            init_weight /= init_weight.sum();

            /* Extracting state estimate: weighted sum */
            out_particle = mean(init_particle, init_weight);
            /* Extracting state estimate: mode */
//            out_particle = mode(init_particle, init_weight);

            /* DEBUG ONLY */
            std::cout << "Step: " << k << std::endl;
            std::cout << "Neff: " << resampling_->neff(init_weight) << std::endl;
            /* ********** */

            if (resampling_->neff(init_weight) < std::round(num_particles_ / 5.f))
            {
                std::cout << "Resampling!" << std::endl;

                resampling_->resample(init_particle, init_weight,
                                      temp_particle, temp_weight,
                                      temp_parent);

                init_particle = temp_particle;
                init_weight   = temp_weight;
            }

            k++;

            /* STATE ESTIMATE OUTPUT */
            /* INDEX FINGERTIP */
//            icub_kin_arm_.setAng(q.subVector(0, 9) * (M_PI/180.0));
//            Vector chainjoints;
//            if (analogs_) icub_kin_finger_[1].getChainJoints(q.subVector(3, 18), analogs, chainjoints, right_hand_analogs_bounds_);
//            else          icub_kin_finger_[1].getChainJoints(q.subVector(3, 18), chainjoints);
//            icub_kin_finger_[1].setAng(chainjoints * (M_PI/180.0));
//
//            Vector l_ee_t(3);
//            toEigen(l_ee_t) = out_particle.col(0).head(3).cast<double>();
//            l_ee_t.push_back(1.0);
//
//            Vector l_ee_o(3);
//            toEigen(l_ee_o) = out_particle.col(0).tail(3).normalized().cast<double>();
//            l_ee_o.push_back(static_cast<double>(out_particle.col(0).tail(3).norm()));
//
//            yarp::sig::Matrix l_Ha = axis2dcm(l_ee_o);
//            l_Ha.setCol(3, l_ee_t);
//            Vector l_i_x = (l_Ha * (icub_kin_finger_[1].getH(3, true).getCol(3))).subVector(0, 2);
//            Vector l_i_o = dcm2axis(l_Ha * icub_kin_finger_[1].getH(3, true));
//            l_i_o.setSubvector(0, l_i_o.subVector(0, 2) * l_i_o[3]);
//
//
//            Vector r_ee_t(3);
//            toEigen(r_ee_t) = out_particle.col(1).head(3).cast<double>();
//            r_ee_t.push_back(1.0);
//
//            Vector r_ee_o(3);
//            toEigen(r_ee_o) = out_particle.col(1).tail(3).normalized().cast<double>();
//            r_ee_o.push_back(static_cast<double>(out_particle.col(1).tail(3).norm()));
//
//            yarp::sig::Matrix r_Ha = axis2dcm(r_ee_o);
//            r_Ha.setCol(3, r_ee_t);
//            Vector r_i_x = (r_Ha * (icub_kin_finger_[1].getH(3, true).getCol(3))).subVector(0, 2);
//            Vector r_i_o = dcm2axis(r_Ha * icub_kin_finger_[1].getH(3, true));
//            r_i_o.setSubvector(0, r_i_o.subVector(0, 2) * r_i_o[3]);
//
//
//            estimates_out.resize(12);
//            estimates_out.setSubvector(0, l_i_x);
//            estimates_out.setSubvector(3, l_i_o.subVector(0, 2));
//            estimates_out.setSubvector(6, r_i_x);
//            estimates_out.setSubvector(9, r_i_o.subVector(0, 2));
//            port_estimates_out_.write();

            /* PALM */
            estimates_out.resize(6);
            toEigen(estimates_out) = out_particle.cast<double>();
            port_estimates_out_.write();

            /* DEBUG ONLY */
            if (stream_images_)
            {
                SuperImpose::ObjPoseMap hand_pose;
                SuperImpose::ObjPose    pose;
                Vector ee_t(4);
                Vector ee_o(4);
                float ang;

                icub_kin_arm_.setAng(q.subVector(0, 9) * (M_PI/180.0));
                Vector chainjoints;
                for (size_t i = 0; i < 3; ++i)
                {
                    if (analogs_) icub_kin_finger_[i].getChainJoints(q.subVector(3, 18), analogs, chainjoints, right_hand_analogs_bounds_);
                    else          icub_kin_finger_[i].getChainJoints(q.subVector(3, 18), chainjoints);
                    icub_kin_finger_[i].setAng(chainjoints * (M_PI/180.0));
                }

                ee_t(0) = out_particle(0);
                ee_t(1) = out_particle(1);
                ee_t(2) = out_particle(2);
                ee_t(3) = 1.0;
                ang     = out_particle.tail(3).norm();
                ee_o(0) = out_particle(3) / ang;
                ee_o(1) = out_particle(4) / ang;
                ee_o(2) = out_particle(5) / ang;
                ee_o(3) = ang;

                pose.assign(ee_t.data(), ee_t.data()+3);
                pose.insert(pose.end(),  ee_o.data(), ee_o.data()+4);
                hand_pose.emplace("palm", pose);

                yarp::sig::Matrix Ha = axis2dcm(ee_o);
                Ha.setCol(3, ee_t);
                // FIXME: middle finger only!
                for (size_t fng = 0; fng < 3; ++fng)
                {
                    std::string finger_s;
                    pose.clear();
                    if (fng != 0)
                    {
                        Vector j_x = (Ha * (icub_kin_finger_[fng].getH0().getCol(3))).subVector(0, 2);
                        Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng].getH0());

                        if      (fng == 1) { finger_s = "index0"; }
                        else if (fng == 2) { finger_s = "medium0"; }

                        pose.assign(j_x.data(), j_x.data()+3);
                        pose.insert(pose.end(), j_o.data(), j_o.data()+4);
                        hand_pose.emplace(finger_s, pose);
                    }

                    for (size_t i = 0; i < icub_kin_finger_[fng].getN(); ++i)
                    {
                        Vector j_x = (Ha * (icub_kin_finger_[fng].getH(i, true).getCol(3))).subVector(0, 2);
                        Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng].getH(i, true));

                        if      (fng == 0) { finger_s = "thumb"+std::to_string(i+1); }
                        else if (fng == 1) { finger_s = "index"+std::to_string(i+1); }
                        else if (fng == 2) { finger_s = "medium"+std::to_string(i+1); }

                        pose.assign(j_x.data(), j_x.data()+3);
                        pose.insert(pose.end(), j_o.data(), j_o.data()+4);
                        hand_pose.emplace(finger_s, pose);
                    }
                }

//                yarp::sig::Matrix invH6 = Ha *
//                                          getInvertedH(-0.0625, -0.02598,       0,   -M_PI, -icub_kin_arm_.getAng(9)) *
//                                          getInvertedH(      0,        0, -M_PI_2, -M_PI_2, -icub_kin_arm_.getAng(8));
//                Vector j_x = invH6.getCol(3).subVector(0, 2);
//                Vector j_o = dcm2axis(invH6);
//                pose.clear();
//                pose.assign(j_x.data(), j_x.data()+3);
//                pose.insert(pose.end(), j_o.data(), j_o.data()+4);
//                hand_pose.emplace("forearm", pose);

                correction_->superimpose(hand_pose, measurement);

                glfwPostEmptyEvent();
                port_image_out_left_.write();
            }
            /* ********** */
        }
    }

    // FIXME: queste close devono andare da un'altra parte. Simil RFModule.
    port_image_in_left_.close();
    port_head_enc_.close();
    port_arm_enc_.close();
    port_torso_enc_.close();
    port_image_out_left_.close();
    port_estimates_out_.close();
    if (analogs_) drv_right_hand_analog_.close();
}


void VisualSIRParticleFilter::getResult() { }


std::future<void> VisualSIRParticleFilter::spawn()
{
    is_running_ = true;
    return std::async(std::launch::async, &VisualSIRParticleFilter::runFilter, this);
}


bool VisualSIRParticleFilter::isRunning()
{
    return is_running_;
}


bool VisualSIRParticleFilter::shouldStop()
{
    return correction_->oglWindowShouldClose();
}


void VisualSIRParticleFilter::stopThread()
{
    if (!is_filter_init_)
    {
        port_head_enc_.interrupt();
        port_arm_enc_.interrupt();
        port_torso_enc_.interrupt();
    }

    port_image_in_left_.interrupt();

    is_running_ = false;
}


bool VisualSIRParticleFilter::setCommandPort()
{
    std::cout << "Opening RPC command port." << std::endl;
    if (!port_rpc_command_.open("/hand-tracking/rpc"))
    {
        std::cerr << "Cannot open the RPC command port." << std::endl;
        return false;
    }
    if (!this->yarp().attachAsServer(port_rpc_command_))
    {
        std::cerr << "Cannot attach the RPC command port." << std::endl;
        return false;
    }
    std::cout << "RPC command port opened and attached. Ready to recieve commands!" << std::endl;

    return true;
}


bool VisualSIRParticleFilter::result_images(const bool status)
{
    stream_images_ = status;

    return true;
}


bool VisualSIRParticleFilter::lock_input(const bool status)
{
    lock_data_ = status;

    std::cerr << "lock_input not yet implemented!" << std::endl;
    return false;
}


bool VisualSIRParticleFilter::use_analogs(const bool status)
{
    if (status && !analogs_)
    {
        opt_right_hand_analog_.put("device", "analogsensorclient");
        opt_right_hand_analog_.put("remote", "/icub/right_hand/analog:o");
        opt_right_hand_analog_.put("local",  "/hand-tracking/right_hand/analog:i");
        if (!drv_right_hand_analog_.open(opt_right_hand_analog_))
        {
            std::cerr << "Cannot open right hand analog driver!" << std::endl;
            return false;
        }
        if (!drv_right_hand_analog_.view(itf_right_hand_analog_))
        {
            std::cerr << "Cannot get right hand analog interface!" << std::endl;
            drv_right_hand_analog_.close();
            return false;
        }

        analogs_ = true;

        return true;
    }
    else if (!status && analogs_)
    {
        drv_right_hand_analog_.close();

        analogs_ = false;

        return true;
    }
    else return false;
}


Eigen::MatrixXf VisualSIRParticleFilter::mean(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights) const
{
    return (particles.array().rowwise() * weights.array().transpose()).rowwise().sum();
}


Eigen::MatrixXf VisualSIRParticleFilter::mode(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights) const
{
    MatrixXf::Index maxRow;
    MatrixXf::Index maxCol;
    weights.maxCoeff(&maxRow, &maxCol);
    return particles.col(maxRow);
}


/* THIS CALL SHOULD BE IN OTHER CLASSES */

Vector VisualSIRParticleFilter::readTorso()
{
    Bottle* b = port_torso_enc_.read();
    Vector torso_enc(3);

    yAssert(b->size() == 3);

    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector VisualSIRParticleFilter::readRootToFingers()
{
    Bottle* b = port_arm_enc_.read();
    if (!b) return Vector(1, 0.0);

    yAssert(b->size() == 16);

    Vector root_fingers_enc(19);
    root_fingers_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 16; ++i)
    {
        root_fingers_enc(3+i) = b->get(i).asDouble();
    }

    return root_fingers_enc;
}


Vector VisualSIRParticleFilter::readRootToEye(const ConstString cam_sel)
{
    Bottle* b = port_head_enc_.read();
    if (!b) return Vector(1, 0.0);

    yAssert(b->size() == 6);

    Vector root_eye_enc(8);
    root_eye_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 4; ++i)
    {
        root_eye_enc(3+i) = b->get(i).asDouble();
    }
    if (cam_sel == "left")  root_eye_enc(7) = b->get(4).asDouble() + b->get(5).asDouble()/2.0;
    if (cam_sel == "right") root_eye_enc(7) = b->get(4).asDouble() - b->get(5).asDouble()/2.0;

    return root_eye_enc;
}


Vector VisualSIRParticleFilter::readRootToEE()
{
    Bottle* b = port_arm_enc_.read();
    if (!b) return Vector(1, 0.0);

    yAssert(b->size() == 16);

    Vector root_ee_enc(10);
    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
    {
        root_ee_enc(i+3) = b->get(i).asDouble();
    }

    return root_ee_enc;
}

Vector VisualSIRParticleFilter::readRightHandAnalogs()
{
    Bottle* b = port_right_hand_analog_.read();
    if (!b) return Vector(1, 0.0);

    yAssert(b->size() >= 15);

    Vector analogs(b->size());
    for (size_t i = 0; i < b->size(); ++i)
    {
        analogs(i) = b->get(i).asDouble();
    }

    return analogs;
}


yarp::sig::Matrix VisualSIRParticleFilter::getInvertedH(double a, double d, double alpha, double offset, double q)
{
    yarp::sig::Matrix H(4, 4);

    double theta = offset + q;
    double c_th  = cos(theta);
    double s_th  = sin(theta);
    double c_al  = cos(alpha);
    double s_al  = sin(alpha);

    H(0,0) =        c_th;
    H(0,1) =       -s_th;
    H(0,2) =           0;
    H(0,3) =           a;

    H(1,0) = s_th * c_al;
    H(1,1) = c_th * c_al;
    H(1,2) =       -s_al;
    H(1,3) =   -d * s_al;

    H(2,0) = s_th * s_al;
    H(2,1) = c_th * s_al;
    H(2,2) =        c_al;
    H(2,3) =    d * c_al;

    H(3,0) =           0;
    H(3,1) =           0;
    H(3,2) =           0;
    H(3,3) =           1;
    
    return H;
}
/* ************************************ */
