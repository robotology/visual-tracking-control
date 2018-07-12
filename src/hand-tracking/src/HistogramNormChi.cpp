#include <HistogramNormChi.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

using namespace bfl;
using namespace Eigen;


HistogramNormChi::HistogramNormChi(const double likelihood_gain, const int histogram_size) noexcept :
    likelihood_gain_(likelihood_gain),
    histogram_size_(histogram_size)
{ }


HistogramNormChi::~HistogramNormChi() noexcept
{ }


std::pair<bool, Eigen::VectorXf> HistogramNormChi::likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states)
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
        double norm = 0;
        double chi = 0;

        double sum_normchi = 0;

        size_t histogram_size_counter = 0;
        for (int j = 0; j < measurements.cols(); ++j)
        {
            norm += std::pow(measurements(0, j) - predicted_measurements(i, j), 2.0);

            chi += (std::pow(measurements(0, j) - predicted_measurements(i, j), 2.0)) / (measurements(0, j) + predicted_measurements(i, j) + std::numeric_limits<float>::min());

            histogram_size_counter++;

            if (histogram_size_counter == histogram_size_)
            {
                sum_normchi += std::sqrt(norm) * chi;

                norm = 0;
                chi = 0;

                histogram_size_counter = 0;
            }
        }

        likelihood(i) = exp(-likelihood_gain_ * sum_normchi);
    }

    return std::make_pair(true, likelihood);
}
