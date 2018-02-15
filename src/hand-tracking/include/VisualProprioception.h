#ifndef VISUALPROPRIOCEPTION_H
#define VISUALPROPRIOCEPTION_H

#include <Camera.h>
#include <MeshModel.h>

#include <array>
#include <string>
#include <memory>

#include <BayesFilters/VisualObservationModel.h>
#include <opencv2/core/core.hpp>
#include <SuperimposeMesh/SICAD.h>


class VisualProprioception : public bfl::VisualObservationModel
{
public:
    VisualProprioception(const int num_images, std::unique_ptr<bfl::Camera> camera, std::unique_ptr<bfl::MeshModel> mesh_model);

    virtual ~VisualProprioception() noexcept { };

    void observe(const Eigen::Ref<const Eigen::MatrixXf>& cur_states, cv::OutputArray observations) override;

    bool setProperty(const std::string property) override;

    int getOGLTilesNumber();
    int getOGLTilesRows();
    int getOGLTilesCols();

    unsigned int getCamWidth();
    unsigned int getCamHeight();

    float getCamFx();
    float getCamFy();
    float getCamCx();
    float getCamCy();

protected:
    std::string log_ID_ = "[VisualProprioception]";

    std::unique_ptr<bfl::Camera>    camera_;
    std::unique_ptr<bfl::MeshModel> mesh_model_;

    bfl::Camera::CameraParameters cam_params_;

    SICAD::ModelPathContainer mesh_paths_;
    std::string               shader_folder_;

    std::unique_ptr<SICAD> si_cad_;
    const int              num_images_;

    std::array<double, 3> cam_x_{ {0.0, 0.0, 0.0} };
    std::array<double, 4> cam_o_{ {0.0, 0.0, 0.0, 0.0} };
};

#endif /* VISUALPROPRIOCEPTION_H */
