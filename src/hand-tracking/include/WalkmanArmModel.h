#ifndef WALKMANAMMODEL_H
#define WALKMANAMMODEL_H

#include <MeshModel.h>

#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Matrix.h>


class WalkmanArmModel : public bfl::MeshModel
{
public:
    WalkmanArmModel(const yarp::os::ConstString& laterality,
                    const yarp::os::ConstString& context,
                    const yarp::os::ConstString& port_prefix);

    virtual ~WalkmanArmModel() noexcept { };

    std::tuple<bool, SICAD::ModelPathContainer> getMeshPaths();

    std::tuple<bool, std::string> getShaderPaths();

    std::tuple<bool, std::vector<Superimpose::ModelPoseContainer>> getModelPose(const Eigen::Ref<const Eigen::MatrixXf>& cur_states);

protected:
    bool file_found(const yarp::os::ConstString& file);

private:
    const yarp::os::ConstString log_ID_ = "[WalkmanArmModel]";
    yarp::os::ConstString port_prefix_ = "WalkmanArmModel";

    yarp::os::ConstString laterality_;
    yarp::os::ConstString context_;

    SICAD::ModelPathContainer model_path_;
    std::string               shader_path_;
};

#endif /* WALKMANAMMODEL_H */
