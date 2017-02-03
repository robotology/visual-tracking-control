#include "VisualProprioception.h"

#include <cmath>
#include <iostream>
#include <utility>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <yarp/os/ResourceFinder.h>
#include <yarp/math/Math.h>
#include <yarp/sig/Vector.h>

#define WINDOW_WIDTH  320
#define WINDOW_HEIGHT 240

using namespace cv;
using namespace Eigen;
using namespace iCub::iKin;
using namespace yarp::os;
using namespace yarp::math;
using namespace yarp::sig;
// FIXME: sistemare tutte queste matrici e capire cosa è meglio usare!
typedef typename yarp::sig::Matrix YMatrix;


VisualProprioception::VisualProprioception(GLFWwindow*& window) :
    window_(window), icub_kin_finger_{iCubFinger("right_thumb"), iCubFinger("right_index"), iCubFinger("right_middle")}, icub_arm_(iCubArm("right_v2"))
{
    cam_x_[0] = 0.0;
    cam_x_[1] = 0.0;
    cam_x_[2] = 0.0;

    cam_o_[0] = 0.0;
    cam_o_[1] = 0.0;
    cam_o_[2] = 0.0;
    cam_o_[3] = 0.0;

    // FIXME: middle finger only!
    ResourceFinder rf;
    cad_hand_["palm"] = rf.findFileByName("r_palm.obj");
    if (!file_found(cad_hand_["palm"])) throw std::runtime_error("Runtime error: file r_palm.obj not found!");
//    cad_hand_["thumb1"] = rf.findFileByName("r_tl0.obj");
//    if (!file_found(cad_hand_["thumb1"])) throw std::runtime_error("Runtime error: file r_tl0.obj not found!");
//    cad_hand_["thumb2"] = rf.findFileByName("r_tl1.obj");
//    if (!file_found(cad_hand_["thumb2"])) throw std::runtime_error("Runtime error: file r_tl1.obj not found!");
//    cad_hand_["thumb3"] = rf.findFileByName("r_tl2.obj");
//    if (!file_found(cad_hand_["thumb3"])) throw std::runtime_error("Runtime error: file r_tl2.obj not found!");
//    cad_hand_["thumb4"] = rf.findFileByName("r_tl3.obj");
//    if (!file_found(cad_hand_["thumb4"])) throw std::runtime_error("Runtime error: file r_tl3.obj not found!");
//    cad_hand_["thumb5"] = rf.findFileByName("r_tl4.obj");
//    if (!file_found(cad_hand_["thumb5"])) throw std::runtime_error("Runtime error: file r_tl4.obj not found!");
//    cad_hand_["index0"] = rf.findFileByName("r_indexbase.obj");
//    if (!file_found(cad_hand_["index0"])) throw std::runtime_error("Runtime error: file r_indexbase.obj not found!");
//    cad_hand_["index1"] = rf.findFileByName("r_ail0.obj");
//    if (!file_found(cad_hand_["index1"])) throw std::runtime_error("Runtime error: file r_ail0.obj not found!");
//    cad_hand_["index2"] = rf.findFileByName("r_ail1.obj");
//    if (!file_found(cad_hand_["index2"])) throw std::runtime_error("Runtime error: file r_ail1.obj not found!");
//    cad_hand_["index3"] = rf.findFileByName("r_ail2.obj");
//    if (!file_found(cad_hand_["index3"])) throw std::runtime_error("Runtime error: file r_ail2.obj not found!");
//    cad_hand_["index4"] = rf.findFileByName("r_ail3.obj");
//    if (!file_found(cad_hand_["index4"])) throw std::runtime_error("Runtime error: file r_ail3.obj not found!");
    cad_hand_["medium0"] = rf.findFileByName("r_ml0.obj");
    if (!file_found(cad_hand_["medium0"])) throw std::runtime_error("Runtime error: file r_ml0.obj not found!");
    cad_hand_["medium1"] = rf.findFileByName("r_ml1.obj");
    if (!file_found(cad_hand_["medium1"])) throw std::runtime_error("Runtime error: file r_ml1.obj not found!");
    cad_hand_["medium2"] = rf.findFileByName("r_ml2.obj");
    if (!file_found(cad_hand_["medium2"])) throw std::runtime_error("Runtime error: file r_ml2.obj not found!");
    cad_hand_["medium3"] = rf.findFileByName("r_ml3.obj");
    if (!file_found(cad_hand_["medium3"])) throw std::runtime_error("Runtime error: file r_ml3.obj not found!");
    cad_hand_["forearm"] = rf.findFileByName("r_forearm_scaled.obj");
    if (!file_found(cad_hand_["forearm"])) throw std::runtime_error("Runtime error: file r_forearm.obj not found!");

    si_cad_ = new SICAD(window_, cad_hand_, 232.921, 232.43, 162.202, 125.738);

    icub_kin_finger_[0].setAllConstraints(false);
    icub_kin_finger_[1].setAllConstraints(false);
    icub_kin_finger_[2].setAllConstraints(false);

    icub_arm_.setAllConstraints(false);
    icub_arm_.releaseLink(0);
    icub_arm_.releaseLink(1);
    icub_arm_.releaseLink(2);
}

VisualProprioception::~VisualProprioception() noexcept
{
    delete si_cad_;
}


VisualProprioception::VisualProprioception(const VisualProprioception& proprio) :
    window_(proprio.window_), cad_hand_(proprio.cad_hand_), si_cad_(proprio.si_cad_)
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
    window_(proprio.window_), icub_arm_(std::move(proprio.icub_arm_)), cad_hand_(std::move(proprio.cad_hand_)), si_cad_(std::move(proprio.si_cad_))
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
    window_   = std::move(proprio.window_);
    si_cad_   = std::move(proprio.si_cad_);
    cad_hand_ = std::move(proprio.cad_hand_);

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

    icub_arm_ = std::move(proprio.icub_arm_);

    proprio.cam_x_[0] = 0.0;
    proprio.cam_x_[1] = 0.0;
    proprio.cam_x_[2] = 0.0;

    proprio.cam_o_[0] = 0.0;
    proprio.cam_o_[1] = 0.0;
    proprio.cam_o_[2] = 0.0;
    proprio.cam_o_[3] = 0.0;

    return *this;
}


void VisualProprioception::observe(const Ref<const VectorXf>& cur_state, OutputArray observation)
{
    observation.create(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    Mat hand_ogl = observation.getMat();

    Mat                     hand_ogl_gray;
    SuperImpose::ObjPoseMap hand_pose;
    SuperImpose::ObjPose    pose;
    Vector                  ee_t(4);
    Vector                  ee_o(4);
    float                   ang;


    ee_t(0) = cur_state(0);
    ee_t(1) = cur_state(1);
    ee_t(2) = cur_state(2);
    ee_t(3) =          1.0;
    ang     = cur_state.tail(3).norm();
    ee_o(0) = cur_state(3) / ang;
    ee_o(1) = cur_state(4) / ang;
    ee_o(2) = cur_state(5) / ang;
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

    YMatrix invH6 = Ha *
                    getInvertedH(-0.0625, -0.02598,       0,   -M_PI, -icub_arm_.getAng(9)) *
                    getInvertedH(      0,        0, -M_PI_2, -M_PI_2, -icub_arm_.getAng(8));
    Vector j_x = invH6.getCol(3).subVector(0, 2);
    Vector j_o = dcm2axis(invH6);
    pose.clear();
    pose.assign(j_x.data(), j_x.data()+3);
    pose.insert(pose.end(), j_o.data(), j_o.data()+4);
    hand_pose.emplace("forearm", pose);

    si_cad_->superimpose(hand_pose, cam_x_, cam_o_, hand_ogl);
    glfwPostEmptyEvent();
}


void VisualProprioception::setCamXO(double* cam_x, double* cam_o)
{
    memcpy(cam_x_, cam_x, 3 * sizeof(double));
    memcpy(cam_o_, cam_o, 4 * sizeof(double));
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


void VisualProprioception::superimpose(const SuperImpose::ObjPoseMap& obj2pos_map, Mat& img)
{
    si_cad_->setBackgroundOpt(true);
    si_cad_->setWireframeOpt(true);
    si_cad_->superimpose(obj2pos_map, cam_x_, cam_o_, img);
    glfwPostEmptyEvent();
    si_cad_->setBackgroundOpt(false);
    si_cad_->setWireframeOpt(false);
}


bool VisualProprioception::file_found(const ConstString & file)
{
    if (file.empty())
        return false;
    return true;
}

YMatrix VisualProprioception::getInvertedH(double a, double d, double alpha, double offset, double q)
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
