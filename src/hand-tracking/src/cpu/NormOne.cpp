#include <NormOne.h>

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
    double likelihood_gain_;
};


NormOne::NormOne(const double likelihood_gain) noexcept :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    pImpl_->likelihood_gain_ = likelihood_gain;
}


NormOne::~NormOne() = default;


std::pair<bool, VectorXd> NormOne::likelihood(const MeasurementModel& measurement_model, const Ref<const MatrixXd>& pred_states)
{
    bool valid_measurements;
    Data data_measurements;
    std::tie(valid_measurements, data_measurements) = measurement_model.measure();

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
        double sum_diff = 0;
        for (int j = 0; j < measurements.cols(); ++j)
            sum_diff += std::abs(measurements(0, j) - predicted_measurements(i, j));

        likelihood(i) = sum_diff;
    }

    likelihood = (-(pImpl_->likelihood_gain_) * likelihood).array().exp();


    return std::make_pair(true, std::move(likelihood));
}
