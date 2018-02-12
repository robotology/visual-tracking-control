#ifndef DRAWPARTICLESIMPORTANCETHRESHOLD_H
#define DRAWPARTICLESIMPORTANCETHRESHOLD_H

#include <memory>
#include <random>

#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/StateModel.h>

namespace bfl {
    class DrawParticlesImportanceThreshold;
}


class bfl::DrawParticlesImportanceThreshold : public bfl::PFPrediction
{
public:
    DrawParticlesImportanceThreshold() noexcept;

    DrawParticlesImportanceThreshold(DrawParticlesImportanceThreshold&& pf_prediction) noexcept;

    ~DrawParticlesImportanceThreshold() noexcept;

    DrawParticlesImportanceThreshold& operator=(DrawParticlesImportanceThreshold&& pf_prediction) noexcept;


    StateModel& getStateModel() override;

    void setStateModel(std::unique_ptr<StateModel> state_model) override;

    ExogenousModel& getExogenousModel() override;

    void setExogenousModel(std::unique_ptr<ExogenousModel> exogenous_model) override;

protected:
    void predictStep(const Eigen::Ref<const Eigen::MatrixXf>& prev_states, const Eigen::Ref<const Eigen::VectorXf>& prev_weights,
                     Eigen::Ref<Eigen::MatrixXf> pred_states, Eigen::Ref<Eigen::VectorXf> pred_weights) override;

    std::unique_ptr<StateModel> state_model_;

    std::unique_ptr<ExogenousModel> exogenous_model_;
};

#endif /* DRAWPARTICLESIMPORTANCETHRESHOLD_H */
