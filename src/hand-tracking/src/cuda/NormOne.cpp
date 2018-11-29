#include <NormOne.h>

#include <device_likelihood.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

using namespace bfl;
using namespace Eigen;



struct NormOne::ImplData
{
    cublasHandle_t handle_;

    double likelihood_gain_;
};


NormOne::NormOne(const double likelihood_gain) noexcept :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    ImplData& rImpl = *pImpl_;


    rImpl.likelihood_gain_ = likelihood_gain;

    cublasCreate(&(rImpl.handle_));
}


NormOne::~NormOne() noexcept
{
    ImplData& rImpl = *pImpl_;

    cublasDestroy(rImpl.handle_);
}


std::pair<bool, VectorXf> NormOne::likelihood
(
    const MeasurementModel& measurement_model,
    const Ref<const MatrixXf>& pred_states
)
{
    ImplData& rImpl = *pImpl_;


    bool valid_measurements;
    Data data_measurements;
    std::tie(valid_measurements, data_measurements) = measurement_model.getAgentMeasurements();

    if (!valid_measurements)
        return std::make_pair(false, VectorXf::Zero(1));

    cv::cuda::GpuMat measurements = any::any_cast<cv::cuda::GpuMat>(data_measurements);


    bool valid_predicted_measurements;
    Data data_predicted_measurements;
    std::tie(valid_predicted_measurements, data_predicted_measurements) = measurement_model.predictedMeasure(pred_states);

    cv::cuda::GpuMat predicted_measurements = any::any_cast<cv::cuda::GpuMat>(data_predicted_measurements);

    if (!valid_predicted_measurements)
        return std::make_pair(false, VectorXf::Zero(1));

    thrust::host_vector<float> device_normtwo_kld = bfl::cuda::normone(rImpl.handle_,
                                                                       measurements, predicted_measurements);

    Map<VectorXf> likelihood(device_normtwo_kld.data(), device_normtwo_kld.size());
    likelihood = (-static_cast<float>(rImpl.likelihood_gain_) * likelihood).array().exp();


    return std::make_pair(true, std::move(likelihood));
}
