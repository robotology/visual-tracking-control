#include "VisualSIRParticleFilter.h"

#include <exception>
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


VisualSIRParticleFilter::VisualSIRParticleFilter(std::unique_ptr<ParticleFilterPrediction> prediction, std::unique_ptr<VisualCorrection> correction,
                                                 std::unique_ptr<Resampling> resampling,
                                                 ConstString cam_sel, ConstString laterality, const int num_particles) :
    prediction_(std::move(prediction)), correction_(std::move(correction)), resampling_(std::move(resampling)),
    cam_sel_(cam_sel), laterality_(laterality), num_particles_(num_particles),
    icub_kin_arm_(iCubArm(laterality+"_v2")), icub_kin_finger_{iCubFinger(laterality+"_thumb"), iCubFinger(laterality+"_index"), iCubFinger(laterality+"_middle")}
{
    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    icub_kin_finger_[0].setAllConstraints(false);
    icub_kin_finger_[1].setAllConstraints(false);
    icub_kin_finger_[2].setAllConstraints(false);

    /* Left images:       /icub/camcalib/left/out
       Right images:      /icub/camcalib/right/out
       Arm encoders:      /icub/right_arm/state:o
       Torso encoders:    /icub/torso/state:o      */
    port_image_in_left_.open ("/hand-tracking/" + cam_sel_ + "/img:i");
    port_arm_enc_.open       ("/hand-tracking/" + cam_sel_ + "/" + laterality_ + "_arm:i");
    port_torso_enc_.open     ("/hand-tracking/" + cam_sel_ + "/torso:i");
    port_estimates_out_.open ("/hand-tracking/" + cam_sel_ + "/result/estimates:o");

    is_filter_init_ = false;
    is_running_     = false;

    setCommandPort();
}


VisualSIRParticleFilter::~VisualSIRParticleFilter() noexcept
{
    port_image_in_left_.close();
    port_arm_enc_.close();
    port_torso_enc_.close();
    port_estimates_out_.close();
}


void VisualSIRParticleFilter::runFilter()
{
    is_running_ = true;

    /* INITIALIZATION */
    unsigned int  k = 0;

    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> init_hand_pose(ee_pose.data(), 6, 1);
    init_hand_pose.tail(3) *= ee_pose(6);

    MatrixXf init_particle(6, num_particles_);
    for (int i = 0; i < num_particles_; ++i)
        init_particle.col(i) = init_hand_pose.cast<float>();

    VectorXf init_weight(num_particles_, 1);
    init_weight.fill(1.0/num_particles_);

    const int          block_size = 16;
    const int          img_width  = 320;
    const int          img_height = 240;
    const int          bin_number = 9;
    const unsigned int descriptor_length = (img_width/block_size*2-1) * (img_height/block_size*2-1) * bin_number * 4;
    Ptr<cuda::HOG> cuda_hog = cuda::HOG::create(Size(img_width, img_height), Size(block_size, block_size), Size(block_size/2, block_size/2), Size(block_size/2, block_size/2), bin_number);
    cuda_hog->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_COL_BY_COL);
    cuda_hog->setGammaCorrection(true);
    cuda_hog->setWinStride(Size(img_width, img_height));

    // !!!: In generale si potrebbe prevedere, per ogni step dell'algoritmo, un passo INIT (senza prefissi), così da essere generico.
    prediction_->setStateModelProperty("ICFW_INIT");

    is_filter_init_ = true;


    /* FILTERING */
    ImageOf<PixelRgb>* imgin_left  = YARP_NULLPTR;
    while(is_running_)
    {
        std::vector<float> descriptors_cam_left (descriptor_length);
        cuda::GpuMat       cuda_img             (Size(img_width, img_height), CV_8UC3);
        cuda::GpuMat       cuda_img_alpha       (Size(img_width, img_height), CV_8UC4);
        cuda::GpuMat       descriptors_cam_cuda (Size(descriptor_length, 1),  CV_32F );

        imgin_left = port_image_in_left_.read(true);
        if (imgin_left != YARP_NULLPTR)
        {
            MatrixXf temp_particle(6, num_particles_);
            VectorXf temp_weight(num_particles_, 1);
            VectorXf temp_parent(num_particles_, 1);

            /* PROCESS CURRENT MEASUREMENT */
            Mat measurement;

            measurement = cvarrToMat(imgin_left->getIplImage());
            cuda_img.upload(measurement);
            cuda::cvtColor(cuda_img, cuda_img_alpha, COLOR_BGR2BGRA, 4);
            cuda_hog->compute(cuda_img_alpha, descriptors_cam_cuda);
            descriptors_cam_cuda.download(descriptors_cam_left);

            /* PREDICTION */
            VectorXf sorted_pred = init_weight;
            std::sort(sorted_pred.data(), sorted_pred.data() + sorted_pred.size());
            float threshold = sorted_pred.tail(6)(0);

            prediction_->setStateModelProperty("ICFW_DELTA");
            for (int j = 0; j < num_particles_; ++j)
            {
                if (init_weight(j) <= threshold)
                    prediction_->predict(init_particle.col(j), init_particle.col(j));
                else
                    prediction_->motion(init_particle.col(j), init_particle.col(j));
            }

            /* CORRECTION */
            correction_->setObservationModelProperty("VP_PARAMS");
            correction_->correct(init_particle, descriptors_cam_left, init_weight);
            init_weight /= init_weight.sum();

            /* ESTIMATE EXTRACTION */
            VectorXf out_particle(6);
            /* Extracting state estimate: weighted sum */
            out_particle = mean(init_particle, init_weight);
            /* Extracting state estimate: mode */
//            out_particle = mode(init_particle, init_weight);

            /* RESAMPLING */
            std::cout << "Step: " << k << "\nNeff: " << resampling_->neff(init_weight) << std::endl;
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

            /* DEBUG ONLY */
            if (stream_)
            {
                /* STATE ESTIMATE OUTPUT */
                /* INDEX FINGERTIP */
//                Vector q = readRootToEE();
//                icub_kin_arm_.setAng(q.subVector(0, 9) * (M_PI/180.0));
//                Vector chainjoints;
//                if (analogs_) icub_kin_finger_[1].getChainJoints(q.subVector(3, 18), analogs, chainjoints, right_hand_analogs_bounds_);
//                else          icub_kin_finger_[1].getChainJoints(q.subVector(3, 18), chainjoints);
//                icub_kin_finger_[1].setAng(chainjoints * (M_PI/180.0));
//
//                Vector l_ee_t(3);
//                toEigen(l_ee_t) = out_particle.col(0).head(3).cast<double>();
//                l_ee_t.push_back(1.0);
//
//                Vector l_ee_o(3);
//                toEigen(l_ee_o) = out_particle.col(0).tail(3).normalized().cast<double>();
//                l_ee_o.push_back(static_cast<double>(out_particle.col(0).tail(3).norm()));
//
//                yarp::sig::Matrix l_Ha = axis2dcm(l_ee_o);
//                l_Ha.setCol(3, l_ee_t);
//                Vector l_i_x = (l_Ha * (icub_kin_finger_[1].getH(3, true).getCol(3))).subVector(0, 2);
//                Vector l_i_o = dcm2axis(l_Ha * icub_kin_finger_[1].getH(3, true));
//                l_i_o.setSubvector(0, l_i_o.subVector(0, 2) * l_i_o[3]);
//
//
//                Vector r_ee_t(3);
//                toEigen(r_ee_t) = out_particle.col(1).head(3).cast<double>();
//                r_ee_t.push_back(1.0);
//
//                Vector r_ee_o(3);
//                toEigen(r_ee_o) = out_particle.col(1).tail(3).normalized().cast<double>();
//                r_ee_o.push_back(static_cast<double>(out_particle.col(1).tail(3).norm()));
//
//                yarp::sig::Matrix r_Ha = axis2dcm(r_ee_o);
//                r_Ha.setCol(3, r_ee_t);
//                Vector r_i_x = (r_Ha * (icub_kin_finger_[1].getH(3, true).getCol(3))).subVector(0, 2);
//                Vector r_i_o = dcm2axis(r_Ha * icub_kin_finger_[1].getH(3, true));
//                r_i_o.setSubvector(0, r_i_o.subVector(0, 2) * r_i_o[3]);
//
//
//                Vector& estimates_out = port_estimates_out_.prepare();
//                estimates_out.resize(12);
//                estimates_out.setSubvector(0, l_i_x);
//                estimates_out.setSubvector(3, l_i_o.subVector(0, 2));
//                estimates_out.setSubvector(6, r_i_x);
//                estimates_out.setSubvector(9, r_i_o.subVector(0, 2));
//                port_estimates_out_.write();

                /* PALM */
                Vector& estimates_out = port_estimates_out_.prepare();
                estimates_out.resize(6);
                toEigen(estimates_out) = out_particle.cast<double>();
                port_estimates_out_.write();

            }
            /* ********** */
        }
    }
}


void VisualSIRParticleFilter::getResult() { }


bool VisualSIRParticleFilter::setCommandPort()
{
    std::cout << "Opening RPC command port." << std::endl;
    if (!port_rpc_command_.open("/hand-tracking/" + cam_sel_ + "/cmd:i"))
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


bool VisualSIRParticleFilter::stream_result(const bool status)
{
    stream_ = status;

    return true;
}


bool VisualSIRParticleFilter::use_analogs(const bool status)
{
    if (status)
        return correction_->setObservationModelProperty("VP_ANALOGS_ON");
    else
        return correction_->setObservationModelProperty("VP_ANALOGS_OFF");
}


void VisualSIRParticleFilter::quit()
{
    if (!is_filter_init_)
    {
        port_arm_enc_.interrupt();
        port_torso_enc_.interrupt();
    }

    port_image_in_left_.interrupt();

    is_running_ = false;
}


VectorXf VisualSIRParticleFilter::mean(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights) const
{
    VectorXf out_particle = VectorXf::Zero(6);
    float    s_ang        = 0;
    float    c_ang        = 0;

    for (int i = 0; i < particles.cols(); ++i)
    {
        out_particle.head<3>() += weights(i) * particles.col(i).head<3>();
        out_particle.tail<3>() += weights(i) * particles.col(i).tail<3>().normalized();

        float ang = particles.col(i).tail<3>().norm();

        if      (ang >  2.0 * M_PI) ang -= 2.0 * M_PI;
        else if (ang <=        0.0) ang += 2.0 * M_PI;

        s_ang                  += weights(i) * std::sin(ang);
        c_ang                  += weights(i) * std::cos(ang);
    }

    float ang = std::atan2(s_ang, c_ang) + M_PI;

    if      (ang >  2.0 * M_PI) ang -= 2.0 * M_PI;
    else if (ang <=        0.0) ang += 2.0 * M_PI;

    out_particle.tail<3>() = out_particle.tail<3>().normalized() * std::atan2(s_ang, c_ang);

    return out_particle;
}


VectorXf VisualSIRParticleFilter::mode(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights) const
{
    MatrixXf::Index maxRow;
    MatrixXf::Index maxCol;
    weights.maxCoeff(&maxRow, &maxCol);
    return particles.col(maxRow);
}


/* THIS METHOD SHOULD BE IN INITIALIZATION CLASS */
Vector VisualSIRParticleFilter::readTorso()
{
    Bottle* b = port_torso_enc_.read();
    if (!b) return Vector(3, 0.0);

    yAssert(b->size() == 3);

    Vector torso_enc(3);
    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector VisualSIRParticleFilter::readRootToEE()
{
    Bottle* b = port_arm_enc_.read();
    if (!b) return Vector(10, 0.0);

    yAssert(b->size() == 16);

    Vector root_ee_enc(10);
    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
    {
        root_ee_enc(i+3) = b->get(i).asDouble();
    }

    return root_ee_enc;
}
/* ********************************************* */
