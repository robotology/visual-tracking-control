#include <HistogramNormTwoKLD.h>

#include <device_likelihood.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

using namespace bfl;
using namespace Eigen;



struct HistogramNormTwoKLD::ImplData
{
    cublasHandle_t handle_;
    double likelihood_gain_;
    std::size_t histogram_size_;
};


HistogramNormTwoKLD::HistogramNormTwoKLD(const double likelihood_gain, const std::size_t histogram_size) noexcept :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    pImpl_->likelihood_gain_ = likelihood_gain;

    pImpl_->histogram_size_ = histogram_size;

    cublasCreate(&(pImpl_->handle_));
}


HistogramNormTwoKLD::~HistogramNormTwoKLD() noexcept
{
    cublasDestroy(pImpl_->handle_);
}


std::pair<bool, VectorXf> HistogramNormTwoKLD::likelihood
(
    const MeasurementModel& measurement_model,
    const Ref<const MatrixXf>& pred_states
)
{
    bool valid_measurements;
    Data data_measurements;
    std::tie(valid_measurements, data_measurements) = measurement_model.getAgentMeasurements();

    if (!valid_measurements)
        return std::make_pair(false, VectorXf::Zero(1));

    cv::cuda::GpuMat measurements = any::any_cast<cv::cuda::GpuMat>(data_measurements);


    bool valid_predicted_measurements;
    Data data_predicted_measurements;
    std::tie(valid_predicted_measurements, data_predicted_measurements) = measurement_model.predictedMeasure(pred_states);

    cv::cuda::GpuMat predicted_measurements = any::any_cast<cv::cuda::GpuMat&&>(std::move(data_predicted_measurements));

    if (!valid_predicted_measurements)
        return std::make_pair(false, VectorXf::Zero(1));


    thrust::host_vector<float> device_normtwo_kld = bfl::cuda::normtwo_kld(pImpl_->handle_, measurements, predicted_measurements,
                                                                           pImpl_->histogram_size_, predicted_measurements.size().area() / pImpl_->histogram_size_,
                                                                           true);

    Map<VectorXf> likelihood(device_normtwo_kld.data(), device_normtwo_kld.size());
    likelihood = (-static_cast<float>(pImpl_->likelihood_gain_) * likelihood).array().exp();


    return std::make_pair(true, std::move(likelihood));
}
