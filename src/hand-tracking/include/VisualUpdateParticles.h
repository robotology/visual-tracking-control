#ifndef VISUALUPDATEPARTICLES_H
#define VISUALUPDATEPARTICLES_H

#include <VisualProprioception.h>

#include <condition_variable>
#include <mutex>
#include <thread>

#include <BayesFilters/PFVisualCorrection.h>
#include <opencv2/core/core.hpp>
#if HANDTRACKING_USE_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#else
#include <opencv2/objdetect.hpp>
#endif // HANDTRACKING_USE_OPENCV_CUDA


class VisualUpdateParticles : public bfl::PFVisualCorrection
{
public:
    VisualUpdateParticles(std::unique_ptr<VisualProprioception> observation_model) noexcept;

    VisualUpdateParticles(std::unique_ptr<VisualProprioception> observation_model, const double likelihood_gain) noexcept;

    VisualUpdateParticles(std::unique_ptr<VisualProprioception> observation_model, const double likelihood_gain, const int num_parallel_processor) noexcept;

    ~VisualUpdateParticles() noexcept;

    void innovation(const Eigen::Ref<const Eigen::MatrixXf>& pred_states, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> innovations) override;

    double likelihood(const Eigen::Ref<const Eigen::MatrixXf>& innovations) override;

    bfl::VisualObservationModel& getVisualObservationModel();

protected:
    void correctStep(const Eigen::Ref<const Eigen::MatrixXf>& pred_states, const Eigen::Ref<const Eigen::VectorXf>& pred_weights, cv::InputArray measurements,
                     Eigen::Ref<Eigen::MatrixXf> cor_states, Eigen::Ref<Eigen::VectorXf> cor_weights) override;

    std::unique_ptr<VisualProprioception> observation_model_;
    double                                likelihood_gain_;

    int num_parallel_processor_;
    int num_rendered_img_;

#if HANDTRACKING_USE_OPENCV_CUDA
    cv::Ptr<cv::cuda::HOG> hog_cuda_;

    std::vector<cv::cuda::Stream> cuda_stream_;
    std::vector<cv::cuda::GpuMat> cuda_img_;
    std::vector<cv::cuda::GpuMat> cuda_img_alpha_;
    std::vector<cv::cuda::GpuMat> cuda_descriptors_;
    std::vector<cv::Mat>          cpu_descriptors_;
#else
    std::unique_ptr<cv::HOGDescriptor> hog_cpu_;

    std::vector<std::vector<float>> cpu_descriptors_;
#endif // HANDTRACKING_USE_OPENCV_CUDA

    std::vector<cv::Mat> hand_rendered_;

    const int    block_size_ = 16;
    const int    bin_number_ = 9;
    unsigned int img_width_;
    unsigned int img_height_;
    unsigned int ogl_tiles_cols_;
    unsigned int ogl_tiles_rows_;
    unsigned int feature_dim_;
};

#endif /* VISUALUPDATEPARTICLES_H */
