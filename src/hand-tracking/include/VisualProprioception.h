#ifndef VISUALPROPRIOCEPTION_H
#define VISUALPROPRIOCEPTION_H

#include <BayesFilters/MeasurementModel.h>

#include <Camera.h>
#include <MeshModel.h>

#include <array>
#include <string>
#include <memory>

#include <opencv2/core/core.hpp>

#include <SuperimposeMesh/SICAD.h>


class VisualProprioception : public bfl::MeasurementModel
{
public:
    VisualProprioception(std::unique_ptr<bfl::Camera> camera, const int num_requested_images, std::unique_ptr<bfl::MeshModel> mesh_model);

    virtual ~VisualProprioception() noexcept;

    std::pair<bool, bfl::Data> measure(const Eigen::Ref<const Eigen::MatrixXf>& cur_states) const override;

    std::pair<bool, bfl::Data> predictedMeasure(const Eigen::Ref<const Eigen::MatrixXf>& cur_states) const override;

    std::pair<bool, bfl::Data> innovation(const bfl::Data& predicted_measurements, const bfl::Data& measurements) const override;

    bool bufferAgentData() const override;

    std::pair<bool, bfl::Data> getAgentMeasurements() const override;

    /* FIXME
       Find a way to better communicate with the callee. Maybe a struct. */
    int getOGLTilesNumber() const;

    /* For debugging walkman */
    void superimpose(const Superimpose::ModelPoseContainer& obj2pos_map, cv::Mat& img);

protected:
    std::string log_ID_ = "[VisualProprioception]";

    std::unique_ptr<bfl::Camera> camera_ = nullptr;

    mutable std::array<double, 3> camera_position_;

    mutable std::array<double, 4> camera_orientation_;

    std::unique_ptr<bfl::MeshModel> mesh_model_;

    bfl::Camera::CameraIntrinsics cam_params_;

    SICAD::ModelPathContainer mesh_paths_;

    std::string shader_folder_;

    std::unique_ptr<SICAD> si_cad_;

    int num_images_;

    const int block_size_ = 16;

    const int bin_number_ = 9;

    unsigned int feature_dim_;

    /* FIXME
       This is a convenient function over cv2eigen provided by opencv2/core/eigen.hpp. */
    void ocv2eigen(const cv::Mat& src, Eigen::Ref<Eigen::MatrixXf> dst) const;
};

#endif /* VISUALPROPRIOCEPTION_H */
