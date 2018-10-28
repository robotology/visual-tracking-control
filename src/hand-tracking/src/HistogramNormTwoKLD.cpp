#include <HistogramNormTwoKLD.h>

#include <devicelikelihood.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

using namespace bfl;
using namespace Eigen;


HistogramNormTwoKLD::HistogramNormTwoKLD(const double likelihood_gain, const int histogram_size) noexcept :
    likelihood_gain_(likelihood_gain),
    histogram_size_(histogram_size)
{ }


HistogramNormTwoKLD::~HistogramNormTwoKLD() noexcept
{ }


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


    thrust::host_vector<float> device_normtwo_kld = bfl::cuda::normtwo_kld(measurements, predicted_measurements, 36, 1131);

    Map<VectorXf> likelihood(device_normtwo_kld.data(), device_normtwo_kld.size());
    likelihood = (-static_cast<float>(likelihood_gain_) * likelihood).array().exp();


    return std::make_pair(true, std::move(likelihood));
}
