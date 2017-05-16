#ifndef DRAWPOSE_H
#define DRAWPOSE_H

#include <memory>
#include <random>

#include <BayesFiltersLib/ParticleFilterPrediction.h>
#include <BayesFiltersLib/StateModel.h>

namespace bfl {
    class CartesianAxisAnglePrediction;
}


class bfl::CartesianAxisAnglePrediction : public bfl::ParticleFilterPrediction
{
public:
    /* Default constructor, disabled */
    CartesianAxisAnglePrediction() = delete;

    /* PF prediction constructor */
    CartesianAxisAnglePrediction(std::unique_ptr<StateModel> transition_model) noexcept;

    /* Destructor */
    ~CartesianAxisAnglePrediction() noexcept override;

    /* Move constructor */
    CartesianAxisAnglePrediction(CartesianAxisAnglePrediction&& pf_prediction) noexcept;

    /* Move assignment operator */
    CartesianAxisAnglePrediction& operator=(CartesianAxisAnglePrediction&& pf_prediction) noexcept;

    void predict(const Eigen::Ref<const Eigen::VectorXf>& prev_state, Eigen::Ref<Eigen::VectorXf> pred_state) override;

    void motion(const Eigen::Ref<const Eigen::VectorXf>& cur_state, Eigen::Ref<Eigen::VectorXf> prop_state) override;

    void motionDisturbance(Eigen::Ref<Eigen::VectorXf> sample) override;

    bool setStateModelProperty(const std::string& property) override;

protected:
    std::unique_ptr<StateModel> state_model_;

    void addAxisangleDisturbance(const Eigen::Ref<const Eigen::Vector3f>& current_vec, float disturbance_angle, const Eigen::Ref<const Eigen::Vector3f>& disturbance_vec, Eigen::Ref<Eigen::Vector3f> rotated_vec);
};

#endif /* DRAWPOSE_H */
