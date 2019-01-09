#include <NormTwoChiSquare.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

using namespace bfl;
using namespace Eigen;



struct NormTwoChiSquare::ImplData
{
    double likelihood_gain_;

    std::size_t vector_size_;
};


NormTwoChiSquare::NormTwoChiSquare(const double likelihood_gain, const std::size_t vector_size) noexcept :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    pImpl_->likelihood_gain_ = likelihood_gain;

    pImpl_->vector_size_ = vector_size;
}


NormTwoChiSquare::~NormTwoChiSquare() noexcept
{ }


std::pair<bool, VectorXd> NormTwoChiSquare::likelihood(const MeasurementModel& measurement_model, const Ref<const MatrixXd>& pred_states)
{
    bool valid_measurements;
    Data data_measurements;
    std::tie(valid_measurements, data_measurements) = measurement_model.getAgentMeasurements();

    if (!valid_measurements)
        return std::make_pair(false, VectorXd::Zero(1));

    MatrixXf& measurements = any::any_cast<MatrixXf&>(data_measurements);


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
        return std::make_pair(false, VectorXd::Zero(1));


    VectorXd likelihood(pred_states.cols());

    for (int i = 0; i < pred_states.cols(); ++i)
    {
        double norm_two = 0;
        double chi = 0;

        double sum_norm_two_chi = 0;

        size_t histogram_size_counter = 0;
        for (int j = 0; j < measurements.cols(); ++j)
        {
            double squared_diff = std::pow(measurements(0, j) - predicted_measurements(i, j), 2.0);

            norm_two += squared_diff;

            chi += squared_diff / (measurements(0, j) + predicted_measurements(i, j) + std::numeric_limits<float>::min());

            histogram_size_counter++;

            if (histogram_size_counter == pImpl_->vector_size_)
            {
                sum_norm_two_chi += std::sqrt(norm_two) * chi;

                norm_two = 0;
                chi = 0;

                histogram_size_counter = 0;
            }
        }

        likelihood(i) = sum_norm_two_chi;
    }
    likelihood = (-(pImpl_->likelihood_gain_) * likelihood).array().exp();


    return std::make_pair(true, std::move(likelihood));
}
