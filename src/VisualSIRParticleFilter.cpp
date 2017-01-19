#include "VisualSIRParticleFilter.h"

#include <fstream>
#include <iostream>
#include <utility>

#include <Eigen/Dense>
#include <iCub/ctrl/math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <yarp/math/Math.h>

#include "Proprioception.h"
#include "SICAD.h"

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;
typedef typename yarp::sig::Matrix YMatrix;


VisualSIRParticleFilter::VisualSIRParticleFilter(std::shared_ptr<StateModel> state_model, std::shared_ptr<Prediction> prediction, std::shared_ptr<VisualObservationModel> observation_model, std::shared_ptr<VisualCorrection> correction, std::shared_ptr<Resampling> resampling) noexcept :
    state_model_(state_model), prediction_(prediction), observation_model_(observation_model), correction_(correction), resampling_(resampling)
{
    if (!yarp_.checkNetwork(3.0)) throw std::runtime_error("Runtime error: YARP seems unavailable!");

    icub_kin_eye_ = new iCubEye("left_v2");
    icub_kin_eye_->setAllConstraints(false);
    icub_kin_eye_->releaseLink(0);
    icub_kin_eye_->releaseLink(1);
    icub_kin_eye_->releaseLink(2);

    icub_kin_arm_ = new iCubArm("right_v2");
    icub_kin_arm_->setAllConstraints(false);
    icub_kin_arm_->releaseLink(0);
    icub_kin_arm_->releaseLink(1);
    icub_kin_arm_->releaseLink(2);

    icub_kin_finger_[0] = new iCubFinger("right_thumb");
    icub_kin_finger_[1] = new iCubFinger("right_index");
    icub_kin_finger_[2] = new iCubFinger("right_middle");
    icub_kin_finger_[0]->setAllConstraints(false);
    icub_kin_finger_[1]->setAllConstraints(false);
    icub_kin_finger_[2]->setAllConstraints(false);

    /* Images:         /icub/camcalib/left/out
       Head encoders:  /icub/head/state:o
       Arm encoders:   /icub/right_arm/state:o
       Torso encoders: /icub/torso/state:o     */
    port_image_in_.open ("/left_img:i");
    port_head_enc_.open ("/head");
    port_arm_enc_.open  ("/right_arm");
    port_torso_enc_.open("/torso");

    if (!yarp_.connect("/icub/camcalib/left/out", "/left_img:i")) throw std::runtime_error("Runtime error: /icub/camcalib/left/out seems unavailable!");
    if (!yarp_.connect("/icub/head/state:o",      "/head"))       throw std::runtime_error("Runtime error: /icub/head/state:o seems unavailable!");
    if (!yarp_.connect("/icub/right_arm/state:o", "/right_arm"))  throw std::runtime_error("Runtime error: /icub/right_arm/state:o seems unavailable!");
    if (!yarp_.connect("/icub/torso/state:o",     "/torso"))      throw std::runtime_error("Runtime error: /icub/torso/state:o seems unavailable!");

    /* DEBUG ONLY */
    port_image_out_.open("/left_img:o");
    /* ********** */

    is_running_ = false;
}


VisualSIRParticleFilter::~VisualSIRParticleFilter() noexcept
{
    delete icub_kin_eye_;
    delete icub_kin_arm_;
}


void VisualSIRParticleFilter::runFilter()
{
    /* INITIALIZATION */
    unsigned int k = 0;
    MatrixXf init_particle;
    VectorXf init_weight;
    double cam_x[3];
    double cam_o[4];
    int num_particle = 50;

    init_weight.resize(num_particle, 1);
    init_weight.setConstant(1.0/num_particle);

    init_particle.resize(6, num_particle);

    Vector q = readRootToEE();
    Vector ee_pose = icub_kin_arm_->EndEffPose(CTRL_DEG2RAD * q);

    Map<VectorXd> q_arm(ee_pose.data(), 6, 1);
    q_arm.tail(3) *= ee_pose(6);
    for (int i = 0; i < num_particle; ++i)
    {
        init_particle.col(i) = q_arm.cast<float>();
    }

    /* FILTERING */
    ImageOf<PixelRgb> * imgin = YARP_NULLPTR;
    while(is_running_)
    {
        if (imgin == YARP_NULLPTR) imgin = port_image_in_.read(true);
//        imgin = port_image_in_.read(true);
        if (imgin != YARP_NULLPTR)
        {
            ImageOf<PixelRgb>& imgout = port_image_out_.prepare();
            imgout = *imgin;

            MatrixXf temp_particle(6, num_particle);
            VectorXf temp_weight(num_particle, 1);
            VectorXf temp_parent(num_particle, 1);

            Mat measurement = cvarrToMat(imgout.getIplImage());

            Vector eye_pose = icub_kin_eye_->EndEffPose(CTRL_DEG2RAD * readRootToEye("left"));
            cam_x[0] = eye_pose(0); cam_x[1] = eye_pose(1); cam_x[2] = eye_pose(2);
            cam_o[0] = eye_pose(3); cam_o[1] = eye_pose(4); cam_o[2] = eye_pose(5); cam_o[3] = eye_pose(6);

            VectorXf sorted_pred = init_weight;
            std::sort(sorted_pred.data(), sorted_pred.data() + sorted_pred.size());
            float threshold = sorted_pred.tail(6)(0);
            for (int i = 0; i < num_particle; ++i)
                if(init_weight(i) <= threshold) prediction_->predict(init_particle.col(i), init_particle.col(i));

            /* Set parameters */
            // FIXME: da decidere come sistemare
            Vector q = readArm();
            std::dynamic_pointer_cast<Proprioception>(observation_model_)->setArmJoints(q);
            std::dynamic_pointer_cast<Proprioception>(observation_model_)->setCamXO(cam_x, cam_o);
            std::dynamic_pointer_cast<Proprioception>(observation_model_)->setImgBackEdge(measurement);
            /* ************** */

            for (int i = 0; i < num_particle; ++i)
                correction_->correct(init_particle.col(i), measurement, init_weight.row(i));

            init_weight = init_weight / init_weight.sum();

            MatrixXf out_particle(6, 1);
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
            Vector ee_o(4);
            float ang;

            ang     = out_particle.col(0).tail(3).norm();
            ee_o(0) = out_particle(3) / ang;
            ee_o(1) = out_particle(4) / ang;
            ee_o(2) = out_particle(5) / ang;
            ee_o(3) = ang;

            pose.assign(out_particle.data(), out_particle.data()+3);
            pose.insert(pose.end(), ee_o.data(), ee_o.data()+4);
            hand_pose.emplace("palm", pose);

            Vector ee_t(3, pose.data());
            ee_t.push_back(1.0);
            YMatrix Ha = axis2dcm(ee_o);
            Ha.setCol(3, ee_t);
            // FIXME: middle finger only!
            for (size_t fng = 2; fng < 3; ++fng)
            {
                std::string finger_s;
                pose.clear();
                if (fng != 0)
                {
                    Vector j_x = (Ha * (icub_kin_finger_[fng]->getH0().getCol(3))).subVector(0, 2);
                    Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng]->getH0());

                    if      (fng == 1) { finger_s = "index0"; }
                    else if (fng == 2) { finger_s = "medium0"; }

                    pose.assign(j_x.data(), j_x.data()+3);
                    pose.insert(pose.end(), j_o.data(), j_o.data()+4);
                    hand_pose.emplace(finger_s, pose);
                }

                for (size_t i = 0; i < icub_kin_finger_[fng]->getN(); ++i)
                {
                    Vector j_x = (Ha * (icub_kin_finger_[fng]->getH(i, true).getCol(3))).subVector(0, 2);
                    Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng]->getH(i, true));

                    if      (fng == 0) { finger_s = "thumb"+std::to_string(i+1); }
                    else if (fng == 1) { finger_s = "index"+std::to_string(i+1); }
                    else if (fng == 2) { finger_s = "medium"+std::to_string(i+1); }

                    pose.assign(j_x.data(), j_x.data()+3);
                    pose.insert(pose.end(), j_o.data(), j_o.data()+4);
                    hand_pose.emplace(finger_s, pose);
                }
            }

            std::dynamic_pointer_cast<Proprioception>(observation_model_)->superimpose(hand_pose, measurement);

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
    Bottle * b = port_torso_enc_.read();
    Vector torso_enc(3);

    yAssert(b->size() == 3);

    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector VisualSIRParticleFilter::readArm()
{
    Bottle * b = port_arm_enc_.read();
    Vector arm_enc(16);

    yAssert(b->size() == 16);

    for (size_t i = 0; i < 16; ++i)
    {
        arm_enc(i) = b->get(i).asDouble();
    }

    return arm_enc;
}


Vector VisualSIRParticleFilter::readRootToEye(const ConstString eye)
{
    Bottle * b = port_head_enc_.read();
    Vector root_eye_enc(8);

    root_eye_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 4; ++i)
    {
        root_eye_enc(i+3) = b->get(i).asDouble();
    }
    if (eye == "left")  root_eye_enc(7) = b->get(4).asDouble() + b->get(5).asDouble()/2.0;
    if (eye == "right") root_eye_enc(7) = b->get(4).asDouble() - b->get(5).asDouble()/2.0;

    return root_eye_enc;
}


Vector VisualSIRParticleFilter::readRootToEE()
{
    Bottle * b = port_arm_enc_.read();
    Vector root_ee_enc(10);

    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
    {
        root_ee_enc(i+3) = b->get(i).asDouble();
    }

    return root_ee_enc;
}
