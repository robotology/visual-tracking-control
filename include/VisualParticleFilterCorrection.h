#ifndef VISUALPARTICLEFILTERCORRECTION_H
#define VISUALPARTICLEFILTERCORRECTION_H

#include <condition_variable>
#include <mutex>
#include <thread>

#include <BayesFiltersLib/VisualCorrection.h>
#include <BayesFiltersLib/VisualObservationModel.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>


class VisualParticleFilterCorrection : public bfl::VisualCorrection {
public:
    /* Default constructor, disabled */
    VisualParticleFilterCorrection() = delete;

    /* VPF correction constructor */
    VisualParticleFilterCorrection(std::shared_ptr<bfl::VisualObservationModel> observation_model, const int num_particle) noexcept;

    /* Detailed VPF correction constructor */
    VisualParticleFilterCorrection(std::shared_ptr<bfl::VisualObservationModel> observation_model, const int num_particle, const int num_cuda_stream = 10) noexcept;

    /* Destructor */
    ~VisualParticleFilterCorrection() noexcept override;

    /* Copy constructor */
    VisualParticleFilterCorrection(const VisualParticleFilterCorrection& vpf_correction);

    /* Move constructor */
    VisualParticleFilterCorrection(VisualParticleFilterCorrection&& vpf_correction) noexcept;

    /* Copy assignment operator */
    VisualParticleFilterCorrection& operator=(const VisualParticleFilterCorrection& vpf_correction);

    /* Move assignment operator */
    VisualParticleFilterCorrection& operator=(VisualParticleFilterCorrection&& vpf_correction) noexcept;

    void correct(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

    void innovation(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> innovation) override;

    virtual void parallel_innovation(const int stream_id);

    void likelihood(const Eigen::Ref<const Eigen::MatrixXf>& innovation, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

protected:
    std::shared_ptr<bfl::VisualObservationModel> measurement_model_;

    const int                                    block_size_ = 16;
    const int                                    bin_number_ = 9;
    const int                                    img_width_  = 320;
    const int                                    img_height_ = 240;
    cv::HOGDescriptor                            hog_; // FIXME: confront output before deleting
    cv::Ptr<cv::cuda::HOG>                       cuda_hog_;

    bool                                         is_running_ = true;
    std::vector<bool>                            cuda_go_;
    std::vector<std::mutex>                      cuda_mutex_;
    std::vector<std::unique_lock<std::mutex>>    cuda_ulock_;
    std::vector<std::condition_variable>         cuda_condvar_;
    std::vector<std::thread>                     cuda_thread_;
    const Eigen::Ref<const Eigen::MatrixXf>*     pred_state_;
    const int                                    num_particle_;
    const int                                    num_image_stream_;
    std::vector<cv::Mat>                         hand_rendered_;
    std::vector<cv::cuda::Stream>                cuda_stream_;
    std::vector<cv::cuda::GpuMat>                cuda_img_;
    std::vector<cv::cuda::GpuMat>                cuda_img_alpha_;
    std::vector<cv::cuda::GpuMat>                cuda_descriptors_;
};

#endif /* VISUALPARTICLEFILTERCORRECTION_H */
