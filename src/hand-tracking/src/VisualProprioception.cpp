#include "VisualProprioception.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <utility>

#include <iCub/ctrl/math.h>
#include <yarp/math/Math.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Property.h>
#include <yarp/os/ResourceFinder.h>

using namespace cv;
using namespace Eigen;
using namespace iCub::iKin;
using namespace iCub::ctrl;
using namespace yarp::dev;
using namespace yarp::os;
using namespace yarp::math;
using yarp::sig::Vector;


VisualProprioception::VisualProprioception(const int num_images, const ConstString& cam_sel, const ConstString& laterality, const ConstString& context) :
    laterality_(laterality), icub_arm_(iCubArm(laterality+"_v2")), icub_kin_finger_{iCubFinger(laterality+"_thumb"), iCubFinger(laterality+"_index"), iCubFinger(laterality+"_middle")},
    cam_sel_(cam_sel)
{
    ResourceFinder rf;

    icub_kin_eye_ = iCubEye(cam_sel_+"_v2");
    icub_kin_eye_.setAllConstraints(false);
    icub_kin_eye_.releaseLink(0);
    icub_kin_eye_.releaseLink(1);
    icub_kin_eye_.releaseLink(2);

    if (openGazeController())
    {
        Bottle cam_info;
        itf_gaze_->getInfo(cam_info);
        yInfo() << log_ID_ << "[CAM PARAMS]" << cam_info.toString();
        Bottle* cam_sel_intrinsic  = cam_info.findGroup("camera_intrinsics_" + cam_sel_).get(1).asList();
        cam_width_  = cam_info.findGroup("camera_width_" + cam_sel_).get(1).asInt();
        cam_height_ = cam_info.findGroup("camera_height_" + cam_sel_).get(1).asInt();
        cam_fx_     = static_cast<float>(cam_sel_intrinsic->get(0).asDouble());
        cam_cx_     = static_cast<float>(cam_sel_intrinsic->get(2).asDouble());
        cam_fy_     = static_cast<float>(cam_sel_intrinsic->get(5).asDouble());
        cam_cy_     = static_cast<float>(cam_sel_intrinsic->get(6).asDouble());
    }
    else
    {
        yWarning() << log_ID_ << "[CAM PARAMS]" << "No intrinisc camera information could be found by the ctor. Looking for fallback values in parameters.ini.";

        rf.setVerbose();
        rf.setDefaultContext(context);
        rf.setDefaultConfigFile("parameters.ini");
        rf.configure(0, YARP_NULLPTR);

        Bottle* fallback_intrinsic = rf.findGroup("FALLBACK").find("intrinsic_" + cam_sel_).asList();
        if (fallback_intrinsic)
        {
            yInfo() << log_ID_ << "[FALLBACK][CAM PARAMS]" << fallback_intrinsic->toString();

            cam_width_  = static_cast<unsigned int>(fallback_intrinsic->get(0).asInt());
            cam_height_ = static_cast<unsigned int>(fallback_intrinsic->get(1).asInt());
            cam_fx_     = static_cast<float>(fallback_intrinsic->get(2).asDouble());
            cam_cx_     = static_cast<float>(fallback_intrinsic->get(3).asDouble());
            cam_fy_     = static_cast<float>(fallback_intrinsic->get(4).asDouble());
            cam_cy_     = static_cast<float>(fallback_intrinsic->get(5).asDouble());
        }
        else
        {
            yWarning() << log_ID_ << "[CAM PARAMS]" << "No fallback values could be found in parameters.ini by the ctor for the intrinisc camera parameters. Falling (even more) back to iCub_SIM values.";
            cam_width_  = 320;
            cam_height_ = 240;
            cam_fx_     = 257.34;
            cam_cx_     = 160.0;
            cam_fy_     = 257.34;
            cam_cy_     = 120.0;
        }
    }
    yInfo() << log_ID_ << "[CAM]" << "Running with:";
    yInfo() << log_ID_ << "[CAM]" << " - width:"  << cam_width_;
    yInfo() << log_ID_ << "[CAM]" << " - height:" << cam_height_;
    yInfo() << log_ID_ << "[CAM]" << " - fx:"     << cam_fx_;
    yInfo() << log_ID_ << "[CAM]" << " - fy:"     << cam_fy_;
    yInfo() << log_ID_ << "[CAM]" << " - cx:"     << cam_cx_;
    yInfo() << log_ID_ << "[CAM]" << " - cy:"     << cam_cy_;

    cam_x_[0] = 0;
    cam_x_[1] = 0;
    cam_x_[2] = 0;

    cam_o_[0] = 0;
    cam_o_[1] = 0;
    cam_o_[2] = 0;
    cam_o_[3] = 0;

    /* Comment/Uncomment to add/remove limbs */
    rf.setDefaultContext(context + "/mesh");

    cad_obj_["palm"] = rf.findFileByName("r_palm.obj");
    if (!file_found(cad_obj_["palm"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_palm.obj not found!");

//    cad_obj_["thumb1"] = rf.findFileByName("r_tl0.obj");
//    if (!file_found(cad_obj_["thumb1"]))
//        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl0.obj not found!");
//    cad_obj_["thumb2"] = rf.findFileByName("r_tl1.obj");
//    if (!file_found(cad_obj_["thumb2"]))
//        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl1.obj not found!");
//    cad_obj_["thumb3"] = rf.findFileByName("r_tl2.obj");
//    if (!file_found(cad_obj_["thumb3"]))
//        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl2.obj not found!");
//    cad_obj_["thumb4"] = rf.findFileByName("r_tl3.obj");
//    if (!file_found(cad_obj_["thumb4"]))
//        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl3.obj not found!");
//    cad_obj_["thumb5"] = rf.findFileByName("r_tl4.obj");
//    if (!file_found(cad_obj_["thumb5"]))
//        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl4.obj not found!");

    cad_obj_["index0"] = rf.findFileByName("r_indexbase.obj");
    if (!file_found(cad_obj_["index0"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_indexbase.obj not found!");
    cad_obj_["index1"] = rf.findFileByName("r_ail0.obj");
    if (!file_found(cad_obj_["index1"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ail0.obj not found!");
    cad_obj_["index2"] = rf.findFileByName("r_ail1.obj");
    if (!file_found(cad_obj_["index2"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ail1.obj not found!");
    cad_obj_["index3"] = rf.findFileByName("r_ail2.obj");
    if (!file_found(cad_obj_["index3"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ail2.obj not found!");
    cad_obj_["index4"] = rf.findFileByName("r_ail3.obj");
    if (!file_found(cad_obj_["index4"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ail3.obj not found!");

    cad_obj_["medium0"] = rf.findFileByName("r_ml0.obj");
    if (!file_found(cad_obj_["medium0"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ml0.obj not found!");
    cad_obj_["medium1"] = rf.findFileByName("r_ml1.obj");
    if (!file_found(cad_obj_["medium1"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ml1.obj not found!");
    cad_obj_["medium2"] = rf.findFileByName("r_ml2.obj");
    if (!file_found(cad_obj_["medium2"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ml2.obj not found!");
    cad_obj_["medium3"] = rf.findFileByName("r_ml3.obj");
    if (!file_found(cad_obj_["medium3"]))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ml3.obj not found!");

//    cad_obj_["forearm"] = rf.findFileByName("r_forearm.obj");
//    if (!file_found(cad_obj_["forearm"]))
//        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_forearm.obj not found!");

    rf.setDefaultContext(context + "/shader");
    ConstString shader_path = rf.findFileByName("shader_model.vert");
    if (!file_found(shader_path))
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR::DIR\nERROR: shader directory not found!");
    shader_path = shader_path.substr(0, shader_path.rfind("/"));

    try
    {
        si_cad_ = new SICAD(cad_obj_,
                            cam_width_, cam_height_,cam_fx_, cam_fy_, cam_cx_, cam_cy_,
                            num_images,
                            {1.0, 0.0, 0.0, M_PI},
                            shader_path);
    }
    catch (const std::runtime_error& e)
    {
        throw std::runtime_error(e.what());
    }

    icub_kin_finger_[0].setAllConstraints(false);
    icub_kin_finger_[1].setAllConstraints(false);
    icub_kin_finger_[2].setAllConstraints(false);

    icub_arm_.setAllConstraints(false);
    icub_arm_.releaseLink(0);
    icub_arm_.releaseLink(1);
    icub_arm_.releaseLink(2);

    /* Set analogs (optional by default) */
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

    /* Head encoders:     /icub/head/state:o
       Arm encoders:      /icub/right_arm/state:o
       Torso encoders:    /icub/torso/state:o     */
    port_head_enc_.open ("/hand-tracking/VisualProprioception/" + cam_sel_ + "/head:i");
    port_arm_enc_.open  ("/hand-tracking/VisualProprioception/" + cam_sel_ + "/" + laterality_ + "_arm:i");
    port_torso_enc_.open("/hand-tracking/VisualProprioception/" + cam_sel_ + "/torso:i");
}


VisualProprioception::~VisualProprioception() noexcept
{
    drv_gaze_.close();
    drv_right_hand_analog_.close();

    port_head_enc_.close();
    port_arm_enc_.close();
    port_torso_enc_.close();

    delete si_cad_;
}


VisualProprioception::VisualProprioception(const VisualProprioception& proprio) :
    cam_width_(proprio.cam_width_), cam_height_(proprio.cam_height_), cam_fx_(proprio.cam_fx_), cam_cx_(proprio.cam_cx_), cam_fy_(proprio.cam_fy_), cam_cy_(proprio.cam_cy_),
    cad_obj_(proprio.cad_obj_), si_cad_(proprio.si_cad_)
{
    cam_x_[0] = proprio.cam_x_[0];
    cam_x_[1] = proprio.cam_x_[1];
    cam_x_[2] = proprio.cam_x_[2];

    cam_o_[0] = proprio.cam_o_[0];
    cam_o_[1] = proprio.cam_o_[1];
    cam_o_[2] = proprio.cam_o_[2];
    cam_o_[3] = proprio.cam_o_[3];

    icub_kin_finger_[0] = proprio.icub_kin_finger_[0];
    icub_kin_finger_[1] = proprio.icub_kin_finger_[1];
    icub_kin_finger_[2] = proprio.icub_kin_finger_[2];

    icub_arm_ = proprio.icub_arm_;
}


VisualProprioception::VisualProprioception(VisualProprioception&& proprio) noexcept :
    icub_arm_(std::move(proprio.icub_arm_)),
    cam_width_(std::move(proprio.cam_width_)), cam_height_(std::move(proprio.cam_height_)), cam_fx_(std::move(proprio.cam_fx_)), cam_cx_(std::move(proprio.cam_cx_)), cam_fy_(std::move(proprio.cam_fy_)), cam_cy_(std::move(proprio.cam_cy_)),
    cad_obj_(std::move(proprio.cad_obj_)), si_cad_(std::move(proprio.si_cad_))
{
    cam_x_[0] = proprio.cam_x_[0];
    cam_x_[1] = proprio.cam_x_[1];
    cam_x_[2] = proprio.cam_x_[2];

    cam_o_[0] = proprio.cam_o_[0];
    cam_o_[1] = proprio.cam_o_[1];
    cam_o_[2] = proprio.cam_o_[2];
    cam_o_[3] = proprio.cam_o_[3];

    icub_kin_finger_[0] = std::move(proprio.icub_kin_finger_[0]);
    icub_kin_finger_[1] = std::move(proprio.icub_kin_finger_[1]);
    icub_kin_finger_[2] = std::move(proprio.icub_kin_finger_[2]);

    proprio.cam_x_[0] = 0.0;
    proprio.cam_x_[1] = 0.0;
    proprio.cam_x_[2] = 0.0;

    proprio.cam_o_[0] = 0.0;
    proprio.cam_o_[1] = 0.0;
    proprio.cam_o_[2] = 0.0;
    proprio.cam_o_[3] = 0.0;
}


VisualProprioception& VisualProprioception::operator=(const VisualProprioception& proprio)
{
    VisualProprioception tmp(proprio);
    *this = std::move(tmp);

    return *this;
}


VisualProprioception& VisualProprioception::operator=(VisualProprioception&& proprio) noexcept
{
    icub_arm_ = std::move(proprio.icub_arm_);

    icub_kin_finger_[0] = std::move(proprio.icub_kin_finger_[0]);
    icub_kin_finger_[1] = std::move(proprio.icub_kin_finger_[1]);
    icub_kin_finger_[2] = std::move(proprio.icub_kin_finger_[2]);

    cam_x_[0] = proprio.cam_x_[0];
    cam_x_[1] = proprio.cam_x_[1];
    cam_x_[2] = proprio.cam_x_[2];

    cam_o_[0] = proprio.cam_o_[0];
    cam_o_[1] = proprio.cam_o_[1];
    cam_o_[2] = proprio.cam_o_[2];
    cam_o_[3] = proprio.cam_o_[3];

    cam_width_  = std::move(proprio.cam_width_);
    cam_height_ = std::move(proprio.cam_height_);
    cam_fx_     = std::move(proprio.cam_fx_);
    cam_cx_     = std::move(proprio.cam_cx_);
    cam_fy_     = std::move(proprio.cam_fy_);
    cam_cy_     = std::move(proprio.cam_cy_);

    cad_obj_ = std::move(proprio.cad_obj_);
    si_cad_  = std::move(proprio.si_cad_);

    proprio.cam_x_[0] = 0.0;
    proprio.cam_x_[1] = 0.0;
    proprio.cam_x_[2] = 0.0;

    proprio.cam_o_[0] = 0.0;
    proprio.cam_o_[1] = 0.0;
    proprio.cam_o_[2] = 0.0;
    proprio.cam_o_[3] = 0.0;

    return *this;
}


void VisualProprioception::getModelPose(const Ref<const MatrixXf>& cur_state, std::vector<Superimpose::ModelPoseContainer>& hand_poses)
{
    for (int j = 0; j < cur_state.cols(); ++j)
    {
        Superimpose::ModelPoseContainer hand_pose;
        Superimpose::ModelPose          pose;
        Vector                          ee_t(4);
        Vector                          ee_o(4);


        ee_t(0) = cur_state(0, j);
        ee_t(1) = cur_state(1, j);
        ee_t(2) = cur_state(2, j);
        ee_t(3) =             1.0;

        ee_o(0) = cur_state(3, j);
        ee_o(1) = cur_state(4, j);
        ee_o(2) = cur_state(5, j);
        ee_o(3) = cur_state(6, j);

        pose.assign(ee_t.data(), ee_t.data()+3);
        pose.insert(pose.end(),  ee_o.data(), ee_o.data()+4);
        hand_pose.emplace("palm", pose);

        /* Change index to add/remove limbs */
        yarp::sig::Matrix Ha = axis2dcm(ee_o);
        Ha.setCol(3, ee_t);
        for (size_t fng = 1; fng < 3; ++fng)
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
        /* Comment/Uncomment to add/remove limbs */
//        yarp::sig::Matrix invH6 = Ha *
//                                  getInvertedH(-0.0625, -0.02598,       0,   -M_PI, -icub_arm_.getAng(9)) *
//                                  getInvertedH(      0,        0, -M_PI_2, -M_PI_2, -icub_arm_.getAng(8));
//        Vector j_x = invH6.getCol(3).subVector(0, 2);
//        Vector j_o = dcm2axis(invH6);
//        pose.clear();
//        pose.assign(j_x.data(), j_x.data()+3);
//        pose.insert(pose.end(), j_o.data(), j_o.data()+4);
//        hand_pose.emplace("forearm", pose);

        hand_poses.push_back(hand_pose);
    }
}


void VisualProprioception::observe(const Ref<const MatrixXf>& cur_state, OutputArray observation)
{
    std::vector<Superimpose::ModelPoseContainer> hand_poses;
    getModelPose(cur_state, hand_poses);

    observation.create(cam_height_ * si_cad_->getTilesRows(), cam_width_ * si_cad_->getTilesCols(), CV_8UC3);
    Mat hand_ogl = observation.getMat();

    si_cad_->superimpose(hand_poses, cam_x_, cam_o_, hand_ogl);
}


bool VisualProprioception::setProperty(const std::string property)
{
    if (property == "VP_PARAMS")
        return setiCubParams();

    if (property == "VP_ANALOGS_ON")
        return openAnalogs();
    if (property == "VP_ANALOGS_OFF")
        return closeAnalogs();

    return false;
}


bool VisualProprioception::setiCubParams()
{
    Vector left_eye_pose = icub_kin_eye_.EndEffPose(CTRL_DEG2RAD * readRootToEye(cam_sel_));

    cam_x_[0] = left_eye_pose(0);
    cam_x_[1] = left_eye_pose(1);
    cam_x_[2] = left_eye_pose(2);

    cam_o_[0] = left_eye_pose(3);
    cam_o_[1] = left_eye_pose(4);
    cam_o_[2] = left_eye_pose(5);
    cam_o_[3] = left_eye_pose(6);


    Vector q = readRootToFingers();
//    q(10) = 32.0;
//    q(11) = 30.0;
//    q(12) = 0.0;
//    q(13) = 0.0;
//    q(14) = 0.0;
//    q(15) = 0.0;
//    q(16) = 0.0;
//    q(17) = 0.0;


    if (analogs_)
    {
        Vector analogs;
        itf_right_hand_analog_->read(analogs);
        setArmJoints(q, analogs, right_hand_analogs_bounds_);
    }
    else
        setArmJoints(q);

    return true;
}


void VisualProprioception::setArmJoints(const Vector& q)
{
    icub_arm_.setAng(q.subVector(0, 9) * (M_PI/180.0));

    Vector chainjoints;
    for (size_t i = 0; i < 3; ++i)
    {
        icub_kin_finger_[i].getChainJoints(q.subVector(3, 18), chainjoints);
        icub_kin_finger_[i].setAng(chainjoints * (M_PI/180.0));
    }
}


void VisualProprioception::setArmJoints(const Vector& q, const Vector& analogs, const yarp::sig::Matrix& analog_bounds)
{
    icub_arm_.setAng(q.subVector(0, 9) * (M_PI/180.0));

    Vector chainjoints;
    for (size_t i = 0; i < 3; ++i)
    {
        icub_kin_finger_[i].getChainJoints(q.subVector(3, 18), analogs, chainjoints, analog_bounds);
        icub_kin_finger_[i].setAng(chainjoints * (M_PI/180.0));
    }
}


bool VisualProprioception::file_found(const ConstString& file)
{
    if (!file.empty())
    {
        yInfo() << log_ID_ << "File " + file.substr(file.rfind("/") + 1) + " found.";
        return true;
    }

    return false;
}


int VisualProprioception::getOGLTilesNumber()
{
    return si_cad_->getTilesNumber();
}


int VisualProprioception::getOGLTilesRows()
{
    return si_cad_->getTilesRows();
}


int VisualProprioception::getOGLTilesCols()
{
    return si_cad_->getTilesCols();
}


unsigned int VisualProprioception::getCamWidth()
{
    return cam_width_;
}


unsigned int VisualProprioception::getCamHeight()
{
    return cam_height_;
}


float VisualProprioception::getCamFx()
{
    return cam_fx_;
}


float VisualProprioception::getCamFy()
{
    return cam_fy_;
}


float VisualProprioception::getCamCx()
{
    return cam_cx_;
}


float VisualProprioception::getCamCy()
{
    return cam_cy_;
}


yarp::sig::Matrix VisualProprioception::getInvertedH(const double a, const double d, const double alpha, const double offset, const double q)
{
    /** Table of the DH parameters for the right arm V2.
     *  Link i  Ai (mm)     d_i (mm)    alpha_i (rad)   theta_i (deg)
     *  i = 0	32          0           pi/2               0 + (-22 ->    84)
     *  i = 1	0           -5.5        pi/2             -90 + (-39 ->    39)
     *  i = 2	-23.3647	-143.3      pi/2            -105 + (-59 ->    59)
     *  i = 3	0           -107.74     pi/2             -90 + (  5 ->   -95)
     *  i = 4	0           0           -pi/2            -90 + (  0 -> 160.8)
     *  i = 5	-15.0       -152.28     -pi/2           -105 + (-37 ->   100)
     *  i = 6	15.0        0           pi/2               0 + (5.5 ->   106)
     *  i = 7	0           -141.3      pi/2             -90 + (-50 ->    50)
     *  i = 8	0           0           pi/2              90 + ( 10 ->   -65)
     *  i = 9	62.5        25.98       0                180 + (-25 ->    25)
     **/

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


bool VisualProprioception::openGazeController()
{
    Property opt_gaze;
    opt_gaze.put("device", "gazecontrollerclient");
    opt_gaze.put("local",  "/hand-tracking/VisualProprioception/" + cam_sel_ + "/gaze:i");
    opt_gaze.put("remote", "/iKinGazeCtrl");

    if (drv_gaze_.open(opt_gaze))
    {
        drv_gaze_.view(itf_gaze_);
        if (!itf_gaze_)
        {
            yError() << log_ID_ << "Cannot get head gazecontrollerclient interface!";

            drv_gaze_.close();
            return false;
        }
    }
    else
    {
        yError() << log_ID_ << "Cannot open head gazecontrollerclient!";

        return false;
    }

    return true;
}


bool VisualProprioception::openAnalogs()
{
    if (!analogs_)
    {
        Property opt_right_hand_analog;
        opt_right_hand_analog.put("device", "analogsensorclient");
        opt_right_hand_analog.put("local",  "/hand-tracking/VisualProprioception/" + cam_sel_ + "/" + laterality_ + "_hand/analog:i");
        opt_right_hand_analog.put("remote", "/icub/right_hand/analog:o");

        if (drv_right_hand_analog_.open(opt_right_hand_analog))
        {
            drv_right_hand_analog_.view(itf_right_hand_analog_);
            if (!itf_right_hand_analog_)
            {
                yError() << log_ID_ << "Cannot get right hand analogsensorclient interface!";

                drv_right_hand_analog_.close();
                return false;
            }
        }
        else
        {
            yError() << log_ID_ << "Cannot open right hand analogsensorclient!";

            return false;
        }

        analogs_ = true;

        return true;
    }
    else return false;
}


bool VisualProprioception::closeAnalogs()
{
    if (analogs_)
    {
        drv_right_hand_analog_.close();

        analogs_ = false;

        return true;
    }
    else return false;
}


Vector VisualProprioception::readTorso()
{
    Bottle* b = port_torso_enc_.read();
    if (!b) return Vector(3, 0.0);

    Vector torso_enc(3);
    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector VisualProprioception::readRootToFingers()
{
    Bottle* b = port_arm_enc_.read();
    if (!b) return Vector(19, 0.0);

    Vector root_fingers_enc(19);
    root_fingers_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 16; ++i)
        root_fingers_enc(3 + i) = b->get(i).asDouble();

    return root_fingers_enc;
}


Vector VisualProprioception::readRootToEye(const ConstString cam_sel)
{
    Bottle* b = port_head_enc_.read();
    if (!b) return Vector(8, 0.0);

    Vector root_eye_enc(8);
    root_eye_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 4; ++i)
        root_eye_enc(3 + i) = b->get(i).asDouble();

    if (cam_sel == "left")  root_eye_enc(7) = b->get(4).asDouble() + b->get(5).asDouble()/2.0;
    if (cam_sel == "right") root_eye_enc(7) = b->get(4).asDouble() - b->get(5).asDouble()/2.0;

    return root_eye_enc;
}
