#include <DrawParticlesPose.h>

#include <utility>

using namespace bfl;
using namespace Eigen;


DrawParticlesPose::DrawParticlesPose() noexcept { }


DrawParticlesPose::~DrawParticlesPose() noexcept { }


DrawParticlesPose::DrawParticlesPose(DrawParticlesPose&& pf_prediction) noexcept :
    state_model_(std::move(pf_prediction.state_model_)) { };


DrawParticlesPose& DrawParticlesPose::operator=(DrawParticlesPose&& pf_prediction) noexcept
{
    state_model_ = std::move(pf_prediction.state_model_);

    return *this;
}


StateModel& DrawParticlesPose::getStateModel()
{
    return *state_model_;
}


void DrawParticlesPose::setStateModel(std::unique_ptr<StateModel> state_model)
{
    state_model_ = std::move(state_model);
}


void DrawParticlesPose::predictStep(const Ref<const MatrixXf>& prev_states, const Ref<const VectorXf>& prev_weights,
                                    Ref<MatrixXf> pred_states, Ref<VectorXf> pred_weights)
{
    state_model_->motion(prev_states, pred_states);

    pred_weights = prev_weights;
}
