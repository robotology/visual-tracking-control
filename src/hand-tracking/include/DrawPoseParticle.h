#ifndef DRAWPOSEPARTICLE_H
#define DRAWPOSEPARTICLE_H

#include <memory>
#include <random>

#include <BayesFiltersLib/ParticleFilterPrediction.h>
#include <BayesFiltersLib/StateModel.h>

namespace bfl {
    class DrawPoseParticle;
}


class bfl::DrawPoseParticle : public ParticleFilterPrediction
{
public:
    /* Default constructor, disabled */
    DrawPoseParticle() = delete;

    /* PF prediction constructor */
    DrawPoseParticle(std::unique_ptr<StateModel> transition_model) noexcept;

    /* Destructor */
    ~DrawPoseParticle() noexcept override;

    /* Move constructor */
    DrawPoseParticle(DrawPoseParticle&& pf_prediction) noexcept;

    /* Move assignment operator */
    DrawPoseParticle& operator=(DrawPoseParticle&& pf_prediction) noexcept;

    void predict(const Eigen::Ref<const Eigen::VectorXf>& prev_state, Eigen::Ref<Eigen::VectorXf> pred_state) override;

    void motion(const Eigen::Ref<const Eigen::VectorXf>& cur_state, Eigen::Ref<Eigen::VectorXf> prop_state) override;

    void motionDisturbance(Eigen::Ref<Eigen::VectorXf> sample) override;

    bool setMotionModelProperty(const std::string& property) override;

protected:
    std::unique_ptr<StateModel> state_model_;

    void addAxisangleDisturbance(const Eigen::Ref<const Eigen::Vector3f>& current_vec, const Eigen::Ref<const Eigen::Vector3f>& disturbance_vec, Eigen::Ref<Eigen::Vector3f> rotated_vec);
};

#endif /* DRAWPOSEPARTICLE_H */
