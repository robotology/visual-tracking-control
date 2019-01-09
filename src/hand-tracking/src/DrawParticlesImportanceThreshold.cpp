#include <DrawParticlesImportanceThreshold.h>

#include <utility>

using namespace bfl;
using namespace Eigen;


DrawParticlesImportanceThreshold::DrawParticlesImportanceThreshold() noexcept { }


DrawParticlesImportanceThreshold::~DrawParticlesImportanceThreshold() noexcept { }


DrawParticlesImportanceThreshold::DrawParticlesImportanceThreshold(DrawParticlesImportanceThreshold&& pf_prediction) noexcept :
    state_model_(std::move(pf_prediction.state_model_)),
    exogenous_model_(std::move(pf_prediction.exogenous_model_)) { };


DrawParticlesImportanceThreshold& DrawParticlesImportanceThreshold::operator=(DrawParticlesImportanceThreshold&& pf_prediction) noexcept
{
    state_model_ = std::move(pf_prediction.state_model_);

    exogenous_model_ = std::move(pf_prediction.exogenous_model_);

    return *this;
}


StateModel& DrawParticlesImportanceThreshold::getStateModel()
{
    return *state_model_;
}


void DrawParticlesImportanceThreshold::setStateModel(std::unique_ptr<StateModel> state_model)
{
    state_model_ = std::move(state_model);
}


ExogenousModel& DrawParticlesImportanceThreshold::getExogenousModel()
{
    return *exogenous_model_;
}


void DrawParticlesImportanceThreshold::setExogenousModel(std::unique_ptr<ExogenousModel> exogenous_model)
{
    exogenous_model_ = std::move(exogenous_model);
}


void DrawParticlesImportanceThreshold::predictStep(const ParticleSet& prev_particles, ParticleSet& pred_particles)
{
    VectorXd sorted_cor = prev_particles.weight();

    std::sort(sorted_cor.data(), sorted_cor.data() + sorted_cor.size());
    double threshold = sorted_cor.tail(6)(0);

    /* FIXME
       There should be no set property method here.
       There is a coupling with the prediction and pose-related exogenous models. */
    exogenous_model_->setProperty("kin_pose_delta");

    for (int j = 0; j < prev_particles.weight().size(); ++j)
    {
        VectorXd tmp_state = VectorXd::Zero(prev_particles.state().rows());

        if (!getSkipExogenous())
            exogenous_model_->propagate(prev_particles.state(j), tmp_state);
        else
            tmp_state = prev_particles.state(j);

        if (!getSkipState() && prev_particles.weight(j) <= threshold)
            state_model_->motion(tmp_state, pred_particles.state(j));
        else
            pred_particles.state(j) = tmp_state;
    }

    pred_particles.weight() = prev_particles.weight();
}
