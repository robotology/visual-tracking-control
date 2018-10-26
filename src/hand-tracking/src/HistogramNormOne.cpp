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


std::pair<bool, VectorXf> HistogramNormOne::likelihood
(
    const MeasurementModel& measurement_model,
    const Ref<const Eigen::MatrixXf>& pred_states
)
{
    bool valid_agent_measurements;
    Data data_agent_measurements;
    std::tie(valid_agent_measurements, data_agent_measurements) = measurement_model.getAgentMeasurements();

    if (!valid_agent_measurements)
        return std::make_pair(false, VectorXf::Zero(1));

    MatrixXf& measurements = any::any_cast<MatrixXf&>(data_agent_measurements);


    bool valid_predicted_measurements;
    Data data_predicted_measurements;
    std::tie(valid_predicted_measurements, data_predicted_measurements) = measurement_model.predictedMeasure(pred_states);

    MatrixXf predicted_measurements;
    try
    {
        predicted_measurements = any::any_cast<MatrixXf&&>(std::move(data_predicted_measurements));
    }
    catch(const any::bad_any_cast& e)
    {
        std::cerr << e.what() << std::endl;

        valid_predicted_measurements = false;
    }

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
