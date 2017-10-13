#ifndef DRAWPOSE_H
#define DRAWPOSE_H

#include <memory>
#include <random>

#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/StateModel.h>

namespace bfl {
    class CartesianAxisAnglePrediction;
}


class bfl::CartesianAxisAnglePrediction : public bfl::PFPrediction
{
public:
    CartesianAxisAnglePrediction(std::unique_ptr<StateModel> state_model_) noexcept;

    CartesianAxisAnglePrediction(CartesianAxisAnglePrediction&& pf_prediction) noexcept;

    ~CartesianAxisAnglePrediction() noexcept;

    CartesianAxisAnglePrediction& operator=(CartesianAxisAnglePrediction&& pf_prediction) noexcept;

    void predict(const Eigen::Ref<const Eigen::MatrixXf>& prev_states, const Eigen::Ref<const Eigen::VectorXf>& prev_weights,
                 Eigen::Ref<Eigen::MatrixXf> pred_states, Eigen::Ref<Eigen::VectorXf> pred_weights) override;

    StateModel& getStateModel() override;

protected:
    std::unique_ptr<StateModel> state_model_;
};

#endif /* DRAWPOSE_H */
