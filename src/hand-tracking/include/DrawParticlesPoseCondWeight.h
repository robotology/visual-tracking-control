#ifndef DRAWPARTICLESPOSECONDWEIGHT_H
#define DRAWPARTICLESPOSECONDWEIGHT_H

#include <memory>
#include <random>

#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/StateModel.h>

namespace bfl {
    class DrawParticlesPoseCondWeight;
}


class bfl::DrawParticlesPoseCondWeight : public bfl::PFPrediction
{
public:
    DrawParticlesPoseCondWeight() noexcept;

    DrawParticlesPoseCondWeight(DrawParticlesPoseCondWeight&& pf_prediction) noexcept;

    ~DrawParticlesPoseCondWeight() noexcept;

    DrawParticlesPoseCondWeight& operator=(DrawParticlesPoseCondWeight&& pf_prediction) noexcept;


    StateModel& getStateModel() override;

    void setStateModel(std::unique_ptr<StateModel> state_model) override;

protected:
    void predictStep(const Eigen::Ref<const Eigen::MatrixXf>& prev_states, const Eigen::Ref<const Eigen::VectorXf>& prev_weights,
                     Eigen::Ref<Eigen::MatrixXf> pred_states, Eigen::Ref<Eigen::VectorXf> pred_weights) override;

    std::unique_ptr<StateModel> state_model_;
};

#endif /* DRAWPARTICLESPOSECONDWEIGHT_H */
