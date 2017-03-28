#ifndef BROWNIANMOTION_H
#define BROWNIANMOTION_H

#include <functional>
#include <random>

#include <BayesFiltersLib/StateModel.h>


class BrownianMotion : public bfl::StateModel {
public:
    /* BM complete constructor */
    BrownianMotion(const float q_xy, const float q_z, const float theta, const float cone_angle, const unsigned int seed) noexcept;

    /* BM constructor, no rnd seed */
    BrownianMotion(const float q_xy, const float q_z, const float theta, const float cone_angle) noexcept;

    /* Default constructor */
    BrownianMotion() noexcept;

    /* Destructor */
    ~BrownianMotion() noexcept override;

    /* Copy constructor */
    BrownianMotion(const BrownianMotion& bm);

    /* Move constructor */
    BrownianMotion(BrownianMotion&& bm) noexcept;

    /* Copy assignment operator */
    BrownianMotion& operator=(const BrownianMotion& bm);

    /* Move assignment operator */
    BrownianMotion& operator=(BrownianMotion&& bm) noexcept;

    void propagate(const Eigen::Ref<const Eigen::VectorXf>& cur_state, Eigen::Ref<Eigen::VectorXf> prop_state) override;

    void noiseSample(Eigen::Ref<Eigen::VectorXf> sample) override;

    void motion(const Eigen::Ref<const Eigen::VectorXf>& cur_state, Eigen::Ref<Eigen::VectorXf> next_state) override;

protected:
    Eigen::MatrixXf                       F_;           /* State transition matrix */
    float                                 q_xy_;        /* Noise standard deviation for z   3D position */
    float                                 q_z_;         /* Noise standard deviation for x-y 3D position */
    float                                 theta_;       /* Noise standard deviation for axis-angle rotation */
    float                                 cone_angle_;  /* Noise standard deviation for axis-angle axis cone */
    Eigen::Vector4f                       cone_dir_;    /* Cone direction of rotation */

    std::mt19937_64                       generator_;
    std::normal_distribution<float>       distribution_pos_xy_;
    std::normal_distribution<float>       distribution_pos_z_;
    std::normal_distribution<float>       distribution_theta_;
    std::uniform_real_distribution<float> distribution_cone_;
    std::function<float()>                gaussian_random_pos_xy_;
    std::function<float()>                gaussian_random_pos_z_;
    std::function<float()>                gaussian_random_theta_;
    std::function<float()>                gaussian_random_cone_;

    void setConeDirection(const Eigen::Ref<const Eigen::Vector3f>& cur_state);
};

#endif /* BROWNIANMOTION_H */
