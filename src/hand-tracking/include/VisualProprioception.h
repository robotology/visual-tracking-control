#ifndef VISUALPROPRIOCEPTION_H
#define VISUALPROPRIOCEPTION_H

#include <BayesFilters/MeasurementModel.h>

#include <Camera.h>
#include <MeshModel.h>

#include <array>
#include <string>
#include <memory>

#include <opencv2/core/core.hpp>
#if HANDTRACKING_USE_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#else
#include <opencv2/objdetect.hpp>
#endif // HANDTRACKING_USE_OPENCV_CUDA
#include <SuperimposeMesh/SICAD.h>

class VisualProprioception : public bfl::MeasurementModel
{
public:
    VisualProprioception(const int num_requested_images, const bfl::Camera::CameraParameters& cam_params, std::unique_ptr<bfl::MeshModel> mesh_model);

    virtual ~VisualProprioception() noexcept;

    std::pair<bool, Eigen::MatrixXf> measure(const Eigen::Ref<const Eigen::MatrixXf>& cur_states) const override;

    std::pair<bool, Eigen::MatrixXf> predictedMeasure(const Eigen::Ref<const Eigen::MatrixXf>& cur_states) const override;

    std::pair<bool, Eigen::MatrixXf> innovation(const Eigen::Ref<const Eigen::MatrixXf>& predicted_measurements, const Eigen::Ref<const Eigen::MatrixXf>& measurements) const override;

    bool registerProcessData(std::shared_ptr<bfl::GenericData> process_data) override;

    std::pair<bool, Eigen::MatrixXf> getProcessMeasurements() const override;

    /* FIXME
       Find a way to better communicate with the callee. Maybe a struct. */
    int getOGLTilesNumber() const;

    /* For debugging walkman */
    void superimpose(const Superimpose::ModelPoseContainer& obj2pos_map, cv::Mat& img);

protected:
    std::string log_ID_ = "[VisualProprioception]";

    std::shared_ptr<cv::Mat> camera_image_ = nullptr;

    std::shared_ptr<std::array<double, 3>> camera_position_ = nullptr;

    std::shared_ptr<std::array<double, 4>> camera_orientation_ = nullptr;

    std::unique_ptr<bfl::MeshModel> mesh_model_;

    bfl::Camera::CameraParameters cam_params_;

    SICAD::ModelPathContainer mesh_paths_;

    std::string shader_folder_;

    std::unique_ptr<SICAD> si_cad_;

    int num_images_;

#if HANDTRACKING_USE_OPENCV_CUDA
    cv::Ptr<cv::cuda::HOG> hog_cuda_;

    const GLuint* pbo_ = nullptr;

    size_t pbo_size_ = 0;

    struct cudaGraphicsResource** pbo_cuda_;
#else
    std::unique_ptr<cv::HOGDescriptor> hog_cpu_;
#endif // HANDTRACKING_USE_OPENCV_CUDA

    const int block_size_ = 16;

    const int bin_number_ = 9;

    unsigned int feature_dim_;

    /* FIXME
       This is a convenient function over cv2eigen provided by opencv2/core/eigen.hpp. */
    void ocv2eigen(const cv::Mat& src, Eigen::Ref<Eigen::MatrixXf> dst) const;
};

#endif /* VISUALPROPRIOCEPTION_H */
