#ifndef ICUBARMMODEL_H
#define ICUBARMMODEL_H

#include <MeshModel.h>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Matrix.h>


class iCubArmModel : public bfl::MeshModel
{
public:
    iCubArmModel(const bool use_thumb,
                 const bool use_forearm,
                 const std::string& laterality,
                 const std::string& context,
                 const std::string& port_prefix);

    virtual ~iCubArmModel() noexcept;

    std::tuple<bool, SICAD::ModelPathContainer> getMeshPaths();

    std::tuple<bool, std::string> getShaderPaths();

    std::tuple<bool, std::vector<Superimpose::ModelPoseContainer>> getModelPose(const Eigen::Ref<const Eigen::MatrixXd>& cur_states);

protected:
    bool file_found(const std::string& file);

    yarp::sig::Matrix getInvertedH(const double a, const double d, const double alpha, const double offset, const double q);

    std::tuple<bool, yarp::sig::Vector> readRootToFingers();

    bool setArmJoints(const yarp::sig::Vector& q);

private:
    const std::string log_ID_ = "[iCubArmModel]";

    const std::string port_prefix_ = "iCubArmModel";

    const bool use_thumb_;

    const bool use_forearm_;

    const std::string laterality_;

    const std::string context_;

    SICAD::ModelPathContainer model_path_;

    std::string shader_path_;

    iCub::iKin::iCubArm icub_arm_;

    iCub::iKin::iCubFinger icub_kin_finger_[5];

    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;

    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_enc_;
};

#endif /* ICUBARMMODEL_H */
