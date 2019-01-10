#ifndef BROWNIANMOTIONPOSE_H
#define BROWNIANMOTIONPOSE_H

#include <functional>
#include <random>

#include <BayesFilters/StateModel.h>


class BrownianMotionPose : public bfl::StateModel
{
public:
    BrownianMotionPose(const double q_xy, const double q_z, const double theta, const double cone_angle, const unsigned int seed) noexcept;

    BrownianMotionPose(const double q_xy, const double q_z, const double theta, const double cone_angle) noexcept;

    BrownianMotionPose() noexcept;

    BrownianMotionPose(const BrownianMotionPose& bm);

    BrownianMotionPose(BrownianMotionPose&& bm) noexcept;

    ~BrownianMotionPose() noexcept;

    BrownianMotionPose& operator=(const BrownianMotionPose& bm);

    BrownianMotionPose& operator=(BrownianMotionPose&& bm) noexcept;

    void propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> prop_state) override;

    void motion(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> mot_state) override;

    Eigen::MatrixXd getNoiseSample(const int num);

    Eigen::MatrixXd getNoiseCovarianceMatrix() override { return Eigen::MatrixXd::Zero(1, 1); };

    bool setProperty(const std::string& property) override { return false; };

    std::pair<std::size_t, std::size_t> getOutputSize() const override;

protected:
    double                                 q_xy_;        /* Noise standard deviation for z   3D position */
    double                                 q_z_;         /* Noise standard deviation for x-y 3D position */
    double                                 theta_;       /* Noise standard deviation for axis-angle rotation */
    double                                 cone_angle_;  /* Noise standard deviation for axis-angle axis cone */

    Eigen::Vector4f                        cone_dir_;    /* Cone direction of rotation. Fixed, left here for future implementation. */

    std::mt19937_64                        generator_;
    std::normal_distribution<double>       distribution_pos_xy_;
    std::normal_distribution<double>       distribution_pos_z_;
    std::normal_distribution<double>       distribution_theta_;
    std::uniform_real_distribution<double> distribution_cone_;
    std::function<double()>                gaussian_random_pos_xy_;
    std::function<double()>                gaussian_random_pos_z_;
    std::function<double()>                gaussian_random_theta_;
    std::function<double()>                gaussian_random_cone_;

    void addAxisangleDisturbance(const Eigen::Ref<const Eigen::MatrixXd>& disturbance_vec, Eigen::Ref<Eigen::MatrixXd> current_vec);
};

#endif /* BROWNIANMOTIONPOSE_H */
