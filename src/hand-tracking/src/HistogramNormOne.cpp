#include <HistogramNormOne.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

using namespace bfl;
using namespace Eigen;


HistogramNormOne::HistogramNormOne(const double likelihood_gain, const int histogram_size) noexcept :
    likelihood_gain_(likelihood_gain),
    histogram_size_(histogram_size)
{ }


HistogramNormOne::~HistogramNormOne() noexcept
{ }


std::pair<bool, Eigen::VectorXf> HistogramNormOne::likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states)
{
    bool valid_measurements;
    MatrixXf measurements;
    std::tie(valid_measurements, measurements) = measurement_model.getProcessMeasurements();

    if (!valid_measurements)
        return std::make_pair(false, VectorXf::Zero(1));


    bool valid_predicted_measurements;
    MatrixXf predicted_measurements;
    std::tie(valid_predicted_measurements, predicted_measurements) = measurement_model.predictedMeasure(pred_states);

    if (!valid_predicted_measurements)
        return std::make_pair(false, VectorXf::Zero(1));


    VectorXf likelihood(pred_states.cols());

    for (int i = 0; i < pred_states.cols(); ++i)
    {
        double sum_diff = 0;
        for (int j = 0; j < measurements.cols(); ++j)
            sum_diff += std::abs(measurements(0, j) - predicted_measurements(i, j));

        likelihood(i) = static_cast<float>(sum_diff);
    }
    likelihood = (-static_cast<float>(likelihood_gain_) * likelihood).array().exp();


    return std::make_pair(true, std::move(likelihood));
}
