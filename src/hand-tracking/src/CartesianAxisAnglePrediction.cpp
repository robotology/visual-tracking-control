#include "CartesianAxisAnglePrediction.h"

#include <utility>

using namespace bfl;
using namespace Eigen;


CartesianAxisAnglePrediction::CartesianAxisAnglePrediction(std::unique_ptr<StateModel> transition_model) noexcept :
    state_model_(std::move(transition_model)) { }


CartesianAxisAnglePrediction::~CartesianAxisAnglePrediction() noexcept { }


CartesianAxisAnglePrediction::CartesianAxisAnglePrediction(CartesianAxisAnglePrediction&& pf_prediction) noexcept :
    state_model_(std::move(pf_prediction.state_model_)) { };


CartesianAxisAnglePrediction& CartesianAxisAnglePrediction::operator=(CartesianAxisAnglePrediction&& pf_prediction) noexcept
{
    state_model_ = std::move(pf_prediction.state_model_);

    return *this;
}


void CartesianAxisAnglePrediction::predict(const Eigen::Ref<const Eigen::MatrixXf>& prev_state, Eigen::Ref<Eigen::MatrixXf> pred_state)
{
    state_model_->motion(prev_state, pred_state);
}


StateModel& CartesianAxisAnglePrediction::getStateModel()
{
    return *state_model_;
}
