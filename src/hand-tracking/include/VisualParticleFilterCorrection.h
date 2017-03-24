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

#include "VisualProprioception.h"


class VisualParticleFilterCorrection : public bfl::VisualCorrection {
public:
    /* Default constructor, disabled */
    VisualParticleFilterCorrection() = delete;

    /* VPF correction constructor */
    VisualParticleFilterCorrection(std::unique_ptr<VisualProprioception> observation_model) noexcept;

    /* Detailed VPF correction constructor */
    VisualParticleFilterCorrection(std::unique_ptr<VisualProprioception> observation_model, const int num_cuda_stream) noexcept;

    /* Destructor */
    ~VisualParticleFilterCorrection() noexcept override;

//    /* Move constructor */
//    VisualParticleFilterCorrection(VisualParticleFilterCorrection&& vpf_correction) noexcept;
//
//    /* Move assignment operator */
//    VisualParticleFilterCorrection& operator=(VisualParticleFilterCorrection&& vpf_correction) noexcept;

    void correct(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

    void innovation(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> innovation) override;

    void likelihood(const Eigen::Ref<const Eigen::MatrixXf>& innovation, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

    void setCamXO(double* cam_x, double* cam_o);

    void setCamIntrinsic(const unsigned int cam_width, const unsigned int cam_height,
                         const float cam_fx, const float cam_cx, const float cam_fy, const float cam_cy);

    void setArmJoints(const yarp::sig::Vector& q);

    void setArmJoints(const yarp::sig::Vector& q, const yarp::sig::Vector& analogs, const yarp::sig::Matrix& analog_bounds);

    void superimpose(const SuperImpose::ObjPoseMap& obj2pos_map, cv::Mat& img);

    bool oglWindowShouldClose();


protected:
    std::unique_ptr<VisualProprioception>  measurement_model_;

    cv::Ptr<cv::cuda::HOG>                 cuda_hog_;

    const int                              num_cuda_stream_;
    const int                              num_img_stream_;
    std::vector<cv::cuda::Stream>          cuda_stream_;
    std::vector<cv::Mat>                   hand_rendered_;
    std::vector<cv::cuda::GpuMat>          cuda_img_;
    std::vector<cv::cuda::GpuMat>          cuda_img_alpha_;
    std::vector<cv::cuda::GpuMat>          cuda_descriptors_;
    std::vector<cv::Mat>                   cpu_descriptors_;
};

#endif /* VISUALPARTICLEFILTERCORRECTION_H */
