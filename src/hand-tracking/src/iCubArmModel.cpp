#include <iCubArmModel.h>

#include <iCub/ctrl/math.h>
#include <yarp/math/Math.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/LogStream.h>
#include <yarp/sig/Vector.h>

using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


iCubArmModel::iCubArmModel(const bool use_thumb,
                           const bool use_forearm,
                           const ConstString& laterality,
                           const ConstString& context) :
    use_thumb_(use_thumb),
    use_forearm_(use_forearm),
    laterality_(laterality),
    context_(context),
    icub_arm_(iCubArm(laterality + "_v2")),
    icub_kin_finger_{ iCubFinger(laterality + "_thumb"), iCubFinger(laterality + "_index"), iCubFinger(laterality + "_middle") }
{
    ResourceFinder rf;

    rf.setDefaultContext(context + "/mesh");

    model_path_["palm"] = rf.findFileByName("r_palm.obj");
    if (!file_found(model_path_["palm"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_palm.obj not found!");

    if (use_thumb)
    {
        model_path_["thumb1"] = rf.findFileByName("r_tl0.obj");
        if (!file_found(model_path_["thumb1"]))
            throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl0.obj not found!");
        model_path_["thumb2"] = rf.findFileByName("r_tl1.obj");
        if (!file_found(model_path_["thumb2"]))
            throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl1.obj not found!");
        model_path_["thumb3"] = rf.findFileByName("r_tl2.obj");
        if (!file_found(model_path_["thumb3"]))
            throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl2.obj not found!");
        model_path_["thumb4"] = rf.findFileByName("r_tl3.obj");
        if (!file_found(model_path_["thumb4"]))
            throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl3.obj not found!");
        model_path_["thumb5"] = rf.findFileByName("r_tl4.obj");
        if (!file_found(model_path_["thumb5"]))
            throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_tl4.obj not found!");
    }

    model_path_["index0"] = rf.findFileByName("r_indexbase.obj");
    if (!file_found(model_path_["index0"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_indexbase.obj not found!");
    model_path_["index1"] = rf.findFileByName("r_ail0.obj");
    if (!file_found(model_path_["index1"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ail0.obj not found!");
    model_path_["index2"] = rf.findFileByName("r_ail1.obj");
    if (!file_found(model_path_["index2"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ail1.obj not found!");
    model_path_["index3"] = rf.findFileByName("r_ail2.obj");
    if (!file_found(model_path_["index3"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ail2.obj not found!");
    model_path_["index4"] = rf.findFileByName("r_ail3.obj");
    if (!file_found(model_path_["index4"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ail3.obj not found!");

    model_path_["medium0"] = rf.findFileByName("r_ml0.obj");
    if (!file_found(model_path_["medium0"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ml0.obj not found!");
    model_path_["medium1"] = rf.findFileByName("r_ml1.obj");
    if (!file_found(model_path_["medium1"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ml1.obj not found!");
    model_path_["medium2"] = rf.findFileByName("r_ml2.obj");
    if (!file_found(model_path_["medium2"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ml2.obj not found!");
    model_path_["medium3"] = rf.findFileByName("r_ml3.obj");
    if (!file_found(model_path_["medium3"]))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_ml3.obj not found!");

    if (use_forearm)
    {
        model_path_["forearm"] = rf.findFileByName("r_forearm.obj");
        if (!file_found(model_path_["forearm"]))
            throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::FILE\nERROR: 3D mesh file r_forearm.obj not found!");
    }


    rf.setDefaultContext(context + "/shader");
    shader_path_ = rf.findFileByName("shader_model.vert");
    if (!file_found(shader_path_))
        throw std::runtime_error("ERROR::VISUALICUBPROPRIOCEPTION::CTOR::DIR\nERROR: shader directory not found!");

    size_t rfind_slash = shader_path_.rfind("/");
    if (rfind_slash == std::string::npos)
        rfind_slash = 0;
    size_t rfind_backslash = shader_path_.rfind("\\");
    if (rfind_backslash == std::string::npos)
        rfind_slash = 0;

    shader_path_ = shader_path_.substr(0, rfind_slash > rfind_backslash ? rfind_slash : rfind_backslash);


    icub_kin_finger_[0].setAllConstraints(false);
    icub_kin_finger_[1].setAllConstraints(false);
    icub_kin_finger_[2].setAllConstraints(false);

    icub_arm_.setAllConstraints(false);
    icub_arm_.releaseLink(0);
    icub_arm_.releaseLink(1);
    icub_arm_.releaseLink(2);
}


std::tuple<bool, SICAD::ModelPathContainer> iCubArmModel::readMeshPaths()
{
    return std::make_tuple(true, model_path_);
}


std::tuple<bool, std::string> iCubArmModel::readShaderPaths()
{
    return std::make_tuple(true, shader_path_);
}


std::tuple<bool, std::vector<Superimpose::ModelPoseContainer>> iCubArmModel::getModelPose(const Eigen::Ref<const Eigen::MatrixXf>& cur_states)
{
    bool success = false;
    std::vector<Superimpose::ModelPoseContainer> model_poses(cur_states.cols());


    Vector q;
    std::tie(success, q) = readRootToFingers();

    if (success)
    {
        //q(10) = 32.0;
        //q(11) = 30.0;
        //q(12) = 0.0;
        //q(13) = 0.0;
        //q(14) = 0.0;
        //q(15) = 0.0;
        //q(16) = 0.0;
        //q(17) = 0.0;

        setArmJoints(q);

        for (int i = 0; i < cur_states.cols(); ++i)
        {
            Superimpose::ModelPoseContainer hand_pose;
            Superimpose::ModelPose          pose;
            Vector                          ee_t(4);
            Vector                          ee_o(4);


            ee_t(0) = cur_states(0, i);
            ee_t(1) = cur_states(1, i);
            ee_t(2) = cur_states(2, i);
            ee_t(3) = 1.0;

            ee_o(0) = cur_states(3, i);
            ee_o(1) = cur_states(4, i);
            ee_o(2) = cur_states(5, i);
            ee_o(3) = cur_states(6, i);

            pose.assign(ee_t.data(), ee_t.data() + 3);
            pose.insert(pose.end(), ee_o.data(), ee_o.data() + 4);
            hand_pose.emplace("palm", pose);

            Matrix Ha = axis2dcm(ee_o);
            Ha.setCol(3, ee_t);
            for (unsigned int fng = (use_thumb_ ? 0 : 1); fng < 3; ++fng)
            {
                std::string finger_s;
                pose.clear();
                if (fng != 0)
                {
                    Vector j_x = (Ha * (icub_kin_finger_[fng].getH0().getCol(3))).subVector(0, 2);
                    Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng].getH0());

                    if (fng == 1) finger_s = "index0";
                    else if (fng == 2) finger_s = "medium0";

                    pose.assign(j_x.data(), j_x.data() + 3);
                    pose.insert(pose.end(), j_o.data(), j_o.data() + 4);
                    hand_pose.emplace(finger_s, pose);
                }

                for (size_t i = 0; i < icub_kin_finger_[fng].getN(); ++i)
                {
                    Vector j_x = (Ha * (icub_kin_finger_[fng].getH(i, true).getCol(3))).subVector(0, 2);
                    Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng].getH(i, true));

                    if (fng == 0) finger_s = "thumb";
                    else if (fng == 1) finger_s = "index";
                    else if (fng == 2) finger_s = "medium";
                    finger_s += std::to_string(i + 1);

                    pose.assign(j_x.data(), j_x.data() + 3);
                    pose.insert(pose.end(), j_o.data(), j_o.data() + 4);
                    hand_pose.emplace(finger_s, pose);
                }
            }
            if (use_forearm_)
            {
                yarp::sig::Matrix invH6 = Ha *
                    getInvertedH(-0.0625, -0.02598, 0, -M_PI, -icub_arm_.getAng(9)) *
                    getInvertedH(0, 0, -M_PI_2, -M_PI_2, -icub_arm_.getAng(8)) *
                    getInvertedH(0, 0.1413, -M_PI_2, M_PI_2, 0);

                Vector j_x = invH6.getCol(3).subVector(0, 2);
                Vector j_o = dcm2axis(invH6);

                pose.clear();
                pose.assign(j_x.data(), j_x.data() + 3);
                pose.insert(pose.end(), j_o.data(), j_o.data() + 4);

                hand_pose.emplace("forearm", pose);
            }

            model_poses[i] = hand_pose;
        }
    }

    return std::make_tuple(success, model_poses);
}


bool iCubArmModel::file_found(const ConstString& file)
{
    if (!file.empty())
    {
        yInfo() << log_ID_ << "File " + file.substr(file.rfind("/") + 1) + " found.";
        return true;
    }

    return false;
}


Matrix iCubArmModel::getInvertedH(const double a, const double d, const double alpha, const double offset, const double q)
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
    double c_th = cos(theta);
    double s_th = sin(theta);
    double c_al = cos(alpha);
    double s_al = sin(alpha);

    H(0, 0) = c_th;
    H(0, 1) = -s_th;
    H(0, 2) = 0;
    H(0, 3) = a;

    H(1, 0) = s_th * c_al;
    H(1, 1) = c_th * c_al;
    H(1, 2) = -s_al;
    H(1, 3) = -d * s_al;

    H(2, 0) = s_th * s_al;
    H(2, 1) = c_th * s_al;
    H(2, 2) = c_al;
    H(2, 3) = d * c_al;

    H(3, 0) = 0;
    H(3, 1) = 0;
    H(3, 2) = 0;
    H(3, 3) = 1;

    return H;
}


bool iCubArmModel::setArmJoints(const Vector& q)
{
    icub_arm_.setAng(q.subVector(0, 9) * CTRL_DEG2RAD);

    Vector chainjoints;
    for (size_t i = 0; i < 3; ++i)
    {
        if (!icub_kin_finger_[i].getChainJoints(q.subVector(3, 18), chainjoints))
            return false;
        icub_kin_finger_[i].setAng(chainjoints * CTRL_DEG2RAD);
    }

    return true;
}


std::tuple<bool, Vector> iCubArmModel::readRootToFingers()
{
    Vector root_fingers_enc(19, 0.0);

    Bottle* bottle_torso = port_torso_enc_.read(true);
    Bottle* bottle_head  = port_arm_enc_.read(true);
    if (!bottle_torso || !bottle_head)
        return std::make_tuple(false, root_fingers_enc);

    root_fingers_enc(0) = bottle_torso->get(2).asDouble();
    root_fingers_enc(1) = bottle_torso->get(1).asDouble();
    root_fingers_enc(2) = bottle_torso->get(0).asDouble();

    for (size_t i = 0; i < 16; ++i)
        root_fingers_enc(3 + i) = bottle_torso->get(i).asDouble();

    return std::make_tuple(true, root_fingers_enc);
}