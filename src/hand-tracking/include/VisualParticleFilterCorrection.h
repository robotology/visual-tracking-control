#ifndef VISUALPARTICLEFILTERCORRECTION_H
#define VISUALPARTICLEFILTERCORRECTION_H

#include "VisualProprioception.h"

#include <condition_variable>
#include <mutex>
#include <thread>

#include <BayesFiltersLib/VisualCorrection.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>


class VisualParticleFilterCorrection : public bfl::VisualCorrection
{
public:
    /* Default constructor, disabled */
    VisualParticleFilterCorrection() = delete;

    /* VPF correction constructor */
    VisualParticleFilterCorrection(std::unique_ptr<VisualProprioception> observation_model) noexcept;

    /* Detailed VPF correction constructor */
    VisualParticleFilterCorrection(std::unique_ptr<VisualProprioception> observation_model, const int num_cuda_stream) noexcept;

    /* Destructor */
    ~VisualParticleFilterCorrection() noexcept override;

    void correct(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

    void innovation(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> innovation) override;

    void likelihood(const Eigen::Ref<const Eigen::MatrixXf>& innovation, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

    bool setObservationModelProperty(const std::string& property) override;

    /* TO BE DEPRECATED */
    void superimpose(const Eigen::Ref<const Eigen::VectorXf>& state, cv::Mat& img);
    /* **************** */

protected:
    std::unique_ptr<VisualProprioception> measurement_model_;

    cv::Ptr<cv::cuda::HOG>                cuda_hog_;

    const int                             num_cuda_stream_;
    const int                             num_img_stream_;
    std::vector<cv::cuda::Stream>         cuda_stream_;
    std::vector<cv::Mat>                  hand_rendered_;
    std::vector<cv::cuda::GpuMat>         cuda_img_;
    std::vector<cv::cuda::GpuMat>         cuda_img_alpha_;
    std::vector<cv::cuda::GpuMat>         cuda_descriptors_;
    std::vector<cv::Mat>                  cpu_descriptors_;
};

#endif /* VISUALPARTICLEFILTERCORRECTION_H */
