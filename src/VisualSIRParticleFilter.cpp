#include "VisualSIRParticleFilter.h"

#include <fstream>
#include <iostream>
#include <utility>

#include <Eigen/Dense>
#include <iCub/ctrl/math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <yarp/math/Math.h>

#include <SuperImpose/SICAD.h>

#include "VisualProprioception.h"

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;
typedef typename yarp::sig::Matrix YMatrix;

YMatrix getInvertedH(double a, double d, double alpha, double offset, double q);


VisualSIRParticleFilter::VisualSIRParticleFilter(std::shared_ptr<StateModel> state_model, std::shared_ptr<Prediction> prediction, std::shared_ptr<VisualObservationModel> observation_model, std::shared_ptr<VisualCorrection> correction, std::shared_ptr<Resampling> resampling) noexcept :
    state_model_(state_model), prediction_(prediction), observation_model_(observation_model), correction_(correction), resampling_(resampling),
icub_kin_eye_(iCubEye("left_v2")), icub_kin_arm_(iCubArm("right_v2")), icub_kin_finger_{iCubFinger("right_thumb"), iCubFinger("right_index"), iCubFinger("right_middle")}
{
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

    /* Images:         /icub/camcalib/left/out
       Head encoders:  /icub/head/state:o
       Arm encoders:   /icub/right_arm/state:o
       Torso encoders: /icub/torso/state:o     */
    port_image_in_.open ("/hand-tracking/left_img:i");
    port_head_enc_.open ("/hand-tracking/head:i");
    port_arm_enc_.open  ("/hand-tracking/right_arm:i");
    port_torso_enc_.open("/hand-tracking/torso:i");

    /* DEBUG ONLY */
    port_image_out_.open("/hand-tracking/result:o");
    /* ********** */

    is_running_ = false;
}


VisualSIRParticleFilter::~VisualSIRParticleFilter() noexcept { }


void VisualSIRParticleFilter::runFilter()
{
    /* INITIALIZATION */
    unsigned int  k = 0;
    int           num_particle = 50;

    double        cam_x[3];
    double        cam_o[4];

    Vector        ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> q_arm(ee_pose.data(), 6, 1);
    q_arm.tail(3) *= ee_pose(6);

    MatrixXf      init_particle(6, num_particle);
    for (int i = 0; i < num_particle; ++i)
        init_particle.col(i) = q_arm.cast<float>();

    VectorXf      init_weight(num_particle, 1);
    init_weight.setConstant(1.0/num_particle);

    const int     block_size = 16;
    const int     img_width  = 320;
    const int     img_height = 240;
    HOGDescriptor hog(Size(img_width, img_height), Size(block_size, block_size), Size(block_size/2, block_size/2), Size(block_size/2, block_size/2), 9, 1, -1, HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS, false);

    /* FILTERING */
    ImageOf<PixelRgb>* imgin = YARP_NULLPTR;
    while(is_running_)
    {
        Vector q;
        Vector eye_pose;
        std::vector<float> descriptors_cam;
        if (imgin == YARP_NULLPTR)
        {
            imgin = port_image_in_.read(true);
            q = readRootToFingers();
            eye_pose = icub_kin_eye_.EndEffPose(CTRL_DEG2RAD * readRootToEye("left"));
            cam_x[0] = eye_pose(0); cam_x[1] = eye_pose(1); cam_x[2] = eye_pose(2);
            cam_o[0] = eye_pose(3); cam_o[1] = eye_pose(4); cam_o[2] = eye_pose(5); cam_o[3] = eye_pose(6);
        }
//        imgin = port_image_in_.read(true);
        if (imgin != YARP_NULLPTR)
        {
            ImageOf<PixelRgb>& imgout = port_image_out_.prepare();
            imgout = *imgin;

            MatrixXf temp_particle(6, num_particle);
            VectorXf temp_weight(num_particle, 1);
            VectorXf temp_parent(num_particle, 1);

            Mat measurement = cvarrToMat(imgout.getIplImage());
            hog.compute(measurement, descriptors_cam);

//            Vector eye_pose = icub_kin_eye_.EndEffPose(CTRL_DEG2RAD * readRootToEye("left"));
//            cam_x[0] = eye_pose(0); cam_x[1] = eye_pose(1); cam_x[2] = eye_pose(2);
//            cam_o[0] = eye_pose(3); cam_o[1] = eye_pose(4); cam_o[2] = eye_pose(5); cam_o[3] = eye_pose(6);

            VectorXf sorted_pred = init_weight;
            std::sort(sorted_pred.data(), sorted_pred.data() + sorted_pred.size());
            float threshold = sorted_pred.tail(6)(0);
            for (int i = 0; i < num_particle; ++i)
                if(init_weight(i) <= threshold) prediction_->predict(init_particle.col(i), init_particle.col(i));

            /* Set parameters */
            // FIXME: da decidere come sistemare
//            Vector q = readRootToFingers();
            std::dynamic_pointer_cast<VisualProprioception>(observation_model_)->setArmJoints(q);
            std::dynamic_pointer_cast<VisualProprioception>(observation_model_)->setCamXO(cam_x, cam_o);
            /* ************** */

            correction_->correct(init_particle, descriptors_cam, init_weight);

            init_weight = init_weight / init_weight.sum();

            VectorXf out_particle(6, 1);
            /* Weighted sum */
            out_particle = (init_particle.array().rowwise() * init_weight.array().transpose()).rowwise().sum();

            /* Mode */
//            MatrixXf::Index maxRow;
//            MatrixXf::Index maxCol;
//            init_weight.maxCoeff(&maxRow, &maxCol);
//            out_particle = init_particle.col(maxRow);

//            VectorXf sorted = init_weight;
//            std::sort(sorted.data(), sorted.data() + sorted.size());
//            std::cout <<  sorted << std::endl;
            std::cout << "Step: " << ++k << std::endl;
//            std::cout << "Neff: " << ht_pf_f_->Neff(init_weight) << std::endl;
            std::cout << "Neff: " << resampling_->neff(init_weight) << std::endl;

            if (resampling_->neff(init_weight) < 15)
            {
                std::cout << "Resampling!" << std::endl;

                resampling_->resample(init_particle, init_weight, temp_particle, temp_weight, temp_parent);

                init_particle = temp_particle;
                init_weight   = temp_weight;
            }



            /* DEBUG ONLY */
            // FIXME: out_particle is treatead as a Vector, but it's a Matrix.
            SuperImpose::ObjPoseMap hand_pose;
            SuperImpose::ObjPose    pose;
            Vector ee_t(4);
            Vector ee_o(4);
            float ang;

            icub_kin_arm_.setAng(q.subVector(0, 9) * (M_PI/180.0));

            ee_t(0) = out_particle(0);
            ee_t(1) = out_particle(1);
            ee_t(2) = out_particle(2);
            ee_t(3) =             1.0;
            ang     = out_particle.tail(3).norm();
            ee_o(0) = out_particle(3) / ang;
            ee_o(1) = out_particle(4) / ang;
            ee_o(2) = out_particle(5) / ang;
            ee_o(3) = ang;

            pose.assign(ee_t.data(), ee_t.data()+3);
            pose.insert(pose.end(),  ee_o.data(), ee_o.data()+4);
            hand_pose.emplace("palm", pose);
            
            YMatrix Ha = axis2dcm(ee_o);
            Ha.setCol(3, ee_t);
            // FIXME: middle finger only!
            for (size_t fng = 2; fng < 3; ++fng)
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
//            YMatrix invH6 = Ha *
//                            getInvertedH(-0.0625, -0.02598,       0,   -M_PI, -icub_kin_arm_.getAng(9)) *
//                            getInvertedH(      0,        0, -M_PI_2, -M_PI_2, -icub_kin_arm_.getAng(8));
//            Vector j_x = invH6.getCol(3).subVector(0, 2);
//            Vector j_o = dcm2axis(invH6);
//            pose.clear();
//            pose.assign(j_x.data(), j_x.data()+3);
//            pose.insert(pose.end(), j_o.data(), j_o.data()+4);
//            hand_pose.emplace("forearm", pose);

            std::dynamic_pointer_cast<VisualProprioception>(observation_model_)->superimpose(hand_pose, measurement);
            glfwPostEmptyEvent();

            port_image_out_.write();
            /* ********** */
        }
    }
    // FIXME: queste close devono andare da un'altra parte. Simil RFModule.
    port_image_in_.close();
    port_head_enc_.close();
    port_arm_enc_.close();
    port_torso_enc_.close();
    port_image_out_.close();
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


void VisualSIRParticleFilter::stopThread()
{
    is_running_ = false;
}


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

    yAssert(b->size() == 16);

    Vector root_fingers_enc(19);
    root_fingers_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 16; ++i)
    {
        root_fingers_enc(3+i) = b->get(i).asDouble();
    }

    return root_fingers_enc;
}


Vector VisualSIRParticleFilter::readRootToEye(const ConstString eye)
{
    Bottle* b = port_head_enc_.read();

    yAssert(b->size() == 6);

    Vector root_eye_enc(8);
    root_eye_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 4; ++i)
    {
        root_eye_enc(3+i) = b->get(i).asDouble();
    }
    if (eye == "left")  root_eye_enc(7) = b->get(4).asDouble() + b->get(5).asDouble()/2.0;
    if (eye == "right") root_eye_enc(7) = b->get(4).asDouble() - b->get(5).asDouble()/2.0;

    return root_eye_enc;
}


Vector VisualSIRParticleFilter::readRootToEE()
{
    Bottle* b = port_arm_enc_.read();

    yAssert(b->size() == 16);

    Vector root_ee_enc(10);
    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
    {
        root_ee_enc(i+3) = b->get(i).asDouble();
    }

    return root_ee_enc;
}


YMatrix getInvertedH(double a, double d, double alpha, double offset, double q)
{
    YMatrix H(4, 4);

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
