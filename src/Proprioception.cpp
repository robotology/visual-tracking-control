#include "Proprioception.h"

#include <cmath>
#include <utility>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <yarp/os/ResourceFinder.h>
#include <yarp/math/Math.h>
#include <yarp/sig/Vector.h>


using namespace cv;
using namespace Eigen;
using namespace iCub::iKin;
using namespace yarp::os;
using namespace yarp::math;
using namespace yarp::sig;
// FIXME: sistemare tutte queste matrici e capire cosa è meglio usare!
typedef typename yarp::sig::Matrix YMatrix;


Proprioception::Proprioception(GLFWwindow* window) :
    window_(window)
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
    if (!FileFound(cad_hand_["palm"])) throw std::runtime_error("Runtime error: file r_palm.obj not found!");
    //        cad_hand_["thumb1"] = rf.findFileByName("r_tl0.obj");
    //        if (!FileFound(cad_hand_["thumb1"])) throw std::runtime_error("Runtime error: file r_tl0.obj not found!");
    //        cad_hand_["thumb2"] = rf.findFileByName("r_tl1.obj");
    //        if (!FileFound(cad_hand_["thumb2"])) throw std::runtime_error("Runtime error: file r_tl1.obj not found!");
    //        cad_hand_["thumb3"] = rf.findFileByName("r_tl2.obj");
    //        if (!FileFound(cad_hand_["thumb3"])) throw std::runtime_error("Runtime error: file r_tl2.obj not found!");
    //        cad_hand_["thumb4"] = rf.findFileByName("r_tl3.obj");
    //        if (!FileFound(cad_hand_["thumb4"])) throw std::runtime_error("Runtime error: file r_tl3.obj not found!");
    //        cad_hand_["thumb5"] = rf.findFileByName("r_tl4.obj");
    //        if (!FileFound(cad_hand_["thumb5"])) throw std::runtime_error("Runtime error: file r_tl4.obj not found!");
    //        cad_hand_["index0"] = rf.findFileByName("r_indexbase.obj");
    //        if (!FileFound(cad_hand_["index0"])) throw std::runtime_error("Runtime error: file r_indexbase.obj not found!");
    //        cad_hand_["index1"] = rf.findFileByName("r_ail0.obj");
    //        if (!FileFound(cad_hand_["index1"])) throw std::runtime_error("Runtime error: file r_ail0.obj not found!");
    //        cad_hand_["index2"] = rf.findFileByName("r_ail1.obj");
    //        if (!FileFound(cad_hand_["index2"])) throw std::runtime_error("Runtime error: file r_ail1.obj not found!");
    //        cad_hand_["index3"] = rf.findFileByName("r_ail2.obj");
    //        if (!FileFound(cad_hand_["index3"])) throw std::runtime_error("Runtime error: file r_ail2.obj not found!");
    //        cad_hand_["index4"] = rf.findFileByName("r_ail3.obj");
    //        if (!FileFound(cad_hand_["index4"])) throw std::runtime_error("Runtime error: file r_ail3.obj not found!");
    cad_hand_["medium0"] = rf.findFileByName("r_ml0.obj");
    if (!FileFound(cad_hand_["medium0"])) throw std::runtime_error("Runtime error: file r_ml0.obj not found!");
    cad_hand_["medium1"] = rf.findFileByName("r_ml1.obj");
    if (!FileFound(cad_hand_["medium1"])) throw std::runtime_error("Runtime error: file r_ml1.obj not found!");
    cad_hand_["medium2"] = rf.findFileByName("r_ml2.obj");
    if (!FileFound(cad_hand_["medium2"])) throw std::runtime_error("Runtime error: file r_ml2.obj not found!");
    cad_hand_["medium3"] = rf.findFileByName("r_ml3.obj");
    if (!FileFound(cad_hand_["medium3"])) throw std::runtime_error("Runtime error: file r_ml3.obj not found!");

    si_cad_ = new SICAD();
    si_cad_->Configure(window_, cad_hand_, 232.921, 232.43, 162.202, 125.738);

    // FIXME: non ha senso che siano dei puntatori
    icub_kin_finger_[0] = new iCubFinger("right_thumb");
    icub_kin_finger_[1] = new iCubFinger("right_index");
    icub_kin_finger_[2] = new iCubFinger("right_middle");
    icub_kin_finger_[0]->setAllConstraints(false);
    icub_kin_finger_[1]->setAllConstraints(false);
    icub_kin_finger_[2]->setAllConstraints(false);
}

Proprioception::~Proprioception() noexcept
{
    delete si_cad_;
    for (size_t i = 0; i < 3; ++i) delete icub_kin_finger_[i];
}


Proprioception::Proprioception(const Proprioception& proprio) :
    window_(proprio.window_), si_cad_(proprio.si_cad_), cad_hand_(proprio.cad_hand_), img_back_edge_(proprio.img_back_edge_)
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
}


Proprioception::Proprioception(Proprioception&& proprio) noexcept :
    window_(std::move(proprio.window_)), si_cad_(std::move(proprio.si_cad_)), cad_hand_(std::move(proprio.cad_hand_)), img_back_edge_(std::move(proprio.img_back_edge_))
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

    proprio.cam_x_[0] = 0.0;
    proprio.cam_x_[1] = 0.0;
    proprio.cam_x_[2] = 0.0;

    proprio.cam_o_[0] = 0.0;
    proprio.cam_o_[1] = 0.0;
    proprio.cam_o_[2] = 0.0;
    proprio.cam_o_[3] = 0.0;

    proprio.icub_kin_finger_[0] = nullptr;
    proprio.icub_kin_finger_[1] = nullptr;
    proprio.icub_kin_finger_[2] = nullptr;
}


Proprioception& Proprioception::operator=(const Proprioception& proprio)
{
    Proprioception tmp(proprio);
    *this = std::move(tmp);

    return *this;
}


Proprioception& Proprioception::operator=(Proprioception&& proprio) noexcept
{
    window_        = std::move(proprio.window_);
    si_cad_        = std::move(proprio.si_cad_);
    cad_hand_      = std::move(proprio.cad_hand_);
    img_back_edge_ = std::move(proprio.img_back_edge_);

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

    proprio.cam_x_[0] = 0.0;
    proprio.cam_x_[1] = 0.0;
    proprio.cam_x_[2] = 0.0;

    proprio.cam_o_[0] = 0.0;
    proprio.cam_o_[1] = 0.0;
    proprio.cam_o_[2] = 0.0;
    proprio.cam_o_[3] = 0.0;

    proprio.icub_kin_finger_[0] = nullptr;
    proprio.icub_kin_finger_[1] = nullptr;
    proprio.icub_kin_finger_[2] = nullptr;

    return *this;
}


void Proprioception::observe(const Ref<const VectorXf>& cur_state, OutputArray observation)
{
    observation.create(img_back_edge_.size(), img_back_edge_.type());
    Mat                     hand_ogl = observation.getMat();

    Mat                     hand_ogl_gray;
    SuperImpose::ObjPoseMap hand_pose;
    SuperImpose::ObjPose    pose;
    Vector                  ee_o(4);
    float                   ang;


    ang     = cur_state.tail(3).norm();
    ee_o(0) = cur_state(3) / ang;
    ee_o(1) = cur_state(4) / ang;
    ee_o(2) = cur_state(5) / ang;
    ee_o(3) = ang;

    pose.assign(cur_state.data(), cur_state.data()+3);
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

    si_cad_->Superimpose(hand_pose, cam_x_, cam_o_, hand_ogl);

    /* Debug Only */
    imshow("Superimposed Edges", max(hand_ogl, img_back_edge_));
    /* ********** */
}


void Proprioception::setCamXO(double* cam_x, double* cam_o)
{
    memcpy(cam_x_, cam_x, 3 * sizeof(double));
    memcpy(cam_o_, cam_o, 4 * sizeof(double));
}


void Proprioception::setImgBackEdge(const Mat& img_back_edge)
{
    img_back_edge_ = img_back_edge;
}


void Proprioception::setArmJoints(const Vector& q)
{
    Vector chainjoints;
    for (size_t i = 0; i < 3; ++i)
    {
        icub_kin_finger_[i]->getChainJoints(q, chainjoints);
        icub_kin_finger_[i]->setAng(chainjoints * (M_PI/180.0));
    }
}


void Proprioception::superimpose(const SuperImpose::ObjPoseMap& obj2pos_map, Mat& img)
{
    si_cad_->setBackgroundOpt(true);
    si_cad_->Superimpose(obj2pos_map, cam_x_, cam_o_, img);
    si_cad_->setBackgroundOpt(false);
}


bool Proprioception::FileFound(const ConstString & file)
{
    if (file.empty())
        return false;
    return true;
}
