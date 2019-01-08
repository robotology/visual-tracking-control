#include <NormTwoKLD.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

using namespace bfl;
using namespace Eigen;



struct NormTwoKLD::ImplData
{
    double likelihood_gain_;

    std::size_t vector_size_;
};


NormTwoKLD::NormTwoKLD(const double likelihood_gain, const std::size_t vector_size) noexcept :
pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    pImpl_->likelihood_gain_ = likelihood_gain;

    pImpl_->vector_size_ = vector_size;
}


NormTwoKLD::~NormTwoKLD() noexcept
{ }


std::pair<bool, VectorXf> NormTwoKLD::likelihood(const MeasurementModel& measurement_model, const Ref<const MatrixXf>& pred_states)
{
    throw std::runtime_error("[NormTwoKLD][CPU] Unimplemented.");
}
