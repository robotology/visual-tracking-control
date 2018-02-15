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
                 const yarp::os::ConstString& laterality,
                 const yarp::os::ConstString& context);

    virtual ~iCubArmModel() noexcept;

    std::tuple<bool, SICAD::ModelPathContainer> readMeshPaths();

    std::tuple<bool, std::string> readShaderPaths();

    std::tuple<bool, std::vector<Superimpose::ModelPoseContainer>> getModelPose(const Eigen::Ref<const Eigen::MatrixXf>& cur_states);

protected:
    bool file_found(const yarp::os::ConstString& file);

    yarp::sig::Matrix getInvertedH(const double a, const double d, const double alpha, const double offset, const double q);

    std::tuple<bool, yarp::sig::Vector> readRootToFingers();

    bool setArmJoints(const yarp::sig::Vector& q);

private:
    std::string log_ID_ = "[iCubArmModel]";

    const bool use_thumb_;
    const bool use_forearm_;
    yarp::os::ConstString laterality_;
    yarp::os::ConstString context_;

    SICAD::ModelPathContainer model_path_;
    std::string               shader_path_;

    iCub::iKin::iCubArm    icub_arm_;
    iCub::iKin::iCubFinger icub_kin_finger_[3];

    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_enc_;
};

#endif /* ICUBARMMODEL_H */
