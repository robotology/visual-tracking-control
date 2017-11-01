#ifndef BROWNIANMOTIONPOSE_H
#define BROWNIANMOTIONPOSE_H

#include <functional>
#include <random>

#include <BayesFilters/StateModel.h>


class BrownianMotionPose : public bfl::StateModel
{
public:
    BrownianMotionPose(const float q_xy, const float q_z, const float theta, const float cone_angle, const unsigned int seed) noexcept;

    BrownianMotionPose(const float q_xy, const float q_z, const float theta, const float cone_angle) noexcept;

    BrownianMotionPose() noexcept;

    BrownianMotionPose(const BrownianMotionPose& bm);

    BrownianMotionPose(BrownianMotionPose&& bm) noexcept;

    ~BrownianMotionPose() noexcept;

    BrownianMotionPose& operator=(const BrownianMotionPose& bm);

    BrownianMotionPose& operator=(BrownianMotionPose&& bm) noexcept;

    void propagate(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> prop_state) override;

    void motion(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> mot_state) override;

    Eigen::MatrixXf getNoiseSample(const int num) override;

    Eigen::MatrixXf getNoiseCovariance() override { return Eigen::MatrixXf::Zero(1, 1); };

    bool setProperty(const std::string& property) override { return false; };

protected:
    float                                 q_xy_;        /* Noise standard deviation for z   3D position */
    float                                 q_z_;         /* Noise standard deviation for x-y 3D position */
    float                                 theta_;       /* Noise standard deviation for axis-angle rotation */
    float                                 cone_angle_;  /* Noise standard deviation for axis-angle axis cone */

    Eigen::Vector4f                       cone_dir_;    /* Cone direction of rotation. Fixed, left here for future implementation. */

    std::mt19937_64                       generator_;
    std::normal_distribution<float>       distribution_pos_xy_;
    std::normal_distribution<float>       distribution_pos_z_;
    std::normal_distribution<float>       distribution_theta_;
    std::uniform_real_distribution<float> distribution_cone_;
    std::function<float()>                gaussian_random_pos_xy_;
    std::function<float()>                gaussian_random_pos_z_;
    std::function<float()>                gaussian_random_theta_;
    std::function<float()>                gaussian_random_cone_;

    void addAxisangleDisturbance(const Eigen::Ref<const Eigen::MatrixXf>& disturbance_vec, Eigen::Ref<Eigen::MatrixXf> current_vec);
};

#endif /* BROWNIANMOTIONPOSE_H */
