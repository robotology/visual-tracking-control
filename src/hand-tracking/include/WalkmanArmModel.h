#ifndef WALKMANAMMODEL_H
#define WALKMANAMMODEL_H

#include <MeshModel.h>

#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Matrix.h>

#include <string>


class WalkmanArmModel : public bfl::MeshModel
{
public:
    WalkmanArmModel(const std::string& laterality,
                    const std::string& context,
                    const std::string& port_prefix);

    virtual ~WalkmanArmModel() noexcept { };

    std::tuple<bool, SICAD::ModelPathContainer> getMeshPaths();

    std::tuple<bool, std::string> getShaderPaths();

    std::tuple<bool, std::vector<Superimpose::ModelPoseContainer>> getModelPose(const Eigen::Ref<const Eigen::MatrixXf>& cur_states);

protected:
    bool file_found(const std::string& file);

private:
    const std::string log_ID_ = "[WalkmanArmModel]";

    const std::string port_prefix_ = "WalkmanArmModel";

    const std::string laterality_;

    const std::string context_;

    SICAD::ModelPathContainer model_path_;

    std::string shader_path_;
};

#endif /* WALKMANAMMODEL_H */
