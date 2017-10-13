#ifndef DRAWPARTICLESPOSE_H
#define DRAWPARTICLESPOSE_H

#include <memory>
#include <random>

#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/StateModel.h>

namespace bfl {
    class DrawParticlesPose;
}


class bfl::DrawParticlesPose : public bfl::PFPrediction
{
public:
    DrawParticlesPose(std::unique_ptr<StateModel> state_model_) noexcept;

    DrawParticlesPose(DrawParticlesPose&& pf_prediction) noexcept;

    ~DrawParticlesPose() noexcept;

    DrawParticlesPose& operator=(DrawParticlesPose&& pf_prediction) noexcept;

    void predict(const Eigen::Ref<const Eigen::MatrixXf>& prev_states, const Eigen::Ref<const Eigen::VectorXf>& prev_weights,
                 Eigen::Ref<Eigen::MatrixXf> pred_states, Eigen::Ref<Eigen::VectorXf> pred_weights) override;

    StateModel& getStateModel() override;

protected:
    std::unique_ptr<StateModel> state_model_;
};

#endif /* DRAWPARTICLESPOSE_H */
