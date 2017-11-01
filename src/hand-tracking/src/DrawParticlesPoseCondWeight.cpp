#include "DrawParticlesPoseCondWeight.h"

#include <utility>

using namespace bfl;
using namespace Eigen;


DrawParticlesPoseCondWeight::DrawParticlesPoseCondWeight() noexcept { }


DrawParticlesPoseCondWeight::~DrawParticlesPoseCondWeight() noexcept { }


DrawParticlesPoseCondWeight::DrawParticlesPoseCondWeight(DrawParticlesPoseCondWeight&& pf_prediction) noexcept :
    state_model_(std::move(pf_prediction.state_model_)) { };


DrawParticlesPoseCondWeight& DrawParticlesPoseCondWeight::operator=(DrawParticlesPoseCondWeight&& pf_prediction) noexcept
{
    state_model_ = std::move(pf_prediction.state_model_);

    return *this;
}


StateModel& DrawParticlesPoseCondWeight::getStateModel()
{
    return *state_model_;
}


void DrawParticlesPoseCondWeight::setStateModel(std::unique_ptr<StateModel> state_model)
{
    state_model_ = std::move(state_model);
}


void DrawParticlesPoseCondWeight::predictStep(const Ref<const MatrixXf>& prev_states, const Ref<const VectorXf>& prev_weights,
                                              Ref<MatrixXf> pred_states, Ref<VectorXf> pred_weights)
{
    VectorXf sorted_cor = prev_weights;
    std::sort(sorted_cor.data(), sorted_cor.data() + sorted_cor.size());
    float threshold = sorted_cor.tail(6)(0);

    state_model_->setProperty("ICFW_DELTA");

    for (int j = 0; j < prev_weights.rows(); ++j)
    {
        if (prev_weights(j) <= threshold)
            state_model_->motion(prev_states.col(j), pred_states.col(j));
        else
            state_model_->propagate(prev_states.col(j), pred_states.col(j));
    }

    pred_weights = prev_weights;
}
