#include "DrawParticlesPose.h"

#include <utility>

using namespace bfl;
using namespace Eigen;


DrawParticlesPose::DrawParticlesPose(std::unique_ptr<StateModel> transition_model) noexcept :
    state_model_(std::move(transition_model)) { }


DrawParticlesPose::~DrawParticlesPose() noexcept { }


DrawParticlesPose::DrawParticlesPose(DrawParticlesPose&& pf_prediction) noexcept :
    state_model_(std::move(pf_prediction.state_model_)) { };


DrawParticlesPose& DrawParticlesPose::operator=(DrawParticlesPose&& pf_prediction) noexcept
{
    state_model_ = std::move(pf_prediction.state_model_);

    return *this;
}


void DrawParticlesPose::predict(const Ref<const MatrixXf>& prev_states, const Ref<const VectorXf>& prev_weights,
                                           Ref<MatrixXf> pred_states, Ref<VectorXf> pred_weights)
{
    state_model_->motion(prev_states, pred_states);

    pred_weights = prev_weights;
}


StateModel& DrawParticlesPose::getStateModel()
{
    return *state_model_;
}
