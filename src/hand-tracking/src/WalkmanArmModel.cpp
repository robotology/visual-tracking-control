#include <WalkmanArmModel.h>

#include <array>

#include <iCub/ctrl/math.h>
#include <yarp/math/Math.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/LogStream.h>
#include <yarp/sig/Vector.h>

using namespace iCub::ctrl;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


WalkmanArmModel::WalkmanArmModel(const ConstString& laterality,
                                 const ConstString& context,
                                 const yarp::os::ConstString& port_prefix) :
    port_prefix_(port_prefix),
    laterality_(laterality),
    context_(context)
{
    ResourceFinder rf;

    rf.setDefaultContext(context + "/mesh");

    model_path_["palm"] = rf.findFileByName("soft_hand_open.stl");
    if (!file_found(model_path_["palm"]))
        throw std::runtime_error("ERROR::WALKMANARMMODEL::CTOR::FILE\nERROR: 3D mesh file r_palm.obj not found!");

    rf.setDefaultContext(context + "/shader");
    shader_path_ = rf.findFileByName("shader_model.vert");
    if (!file_found(shader_path_))
        throw std::runtime_error("ERROR::WALKMANARMMODEL::CTOR::DIR\nERROR: shader directory not found!");

    size_t rfind_slash = shader_path_.rfind("/");
    if (rfind_slash == std::string::npos)
        rfind_slash = 0;
    size_t rfind_backslash = shader_path_.rfind("\\");
    if (rfind_backslash == std::string::npos)
        rfind_backslash = 0;

    shader_path_ = shader_path_.substr(0, rfind_slash > rfind_backslash ? rfind_slash : rfind_backslash);
}


std::tuple<bool, SICAD::ModelPathContainer> WalkmanArmModel::getMeshPaths()
{
    return std::make_tuple(true, model_path_);
}


std::tuple<bool, std::string> WalkmanArmModel::getShaderPaths()
{
    return std::make_tuple(true, shader_path_);
}


std::tuple<bool, std::vector<Superimpose::ModelPoseContainer>> WalkmanArmModel::getModelPose(const Eigen::Ref<const Eigen::MatrixXf>& cur_states)
{
    std::vector<Superimpose::ModelPoseContainer> model_poses(cur_states.cols());

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

        model_poses[i] = hand_pose;
    }

    return std::make_tuple(true, model_poses);
}


bool WalkmanArmModel::file_found(const ConstString& file)
{
    if (!file.empty())
    {
        yInfo() << log_ID_ << "File " + file.substr(file.rfind("/") + 1) + " found.";
        return true;
    }

    return false;
}
