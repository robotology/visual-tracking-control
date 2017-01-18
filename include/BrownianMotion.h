#ifndef BROWNIANMOTION_H
#define BROWNIANMOTION_H

#include <functional>
#include <random>

#include <BayesFiltersLib/StateModel.h>


class BrownianMotion : public bfl::StateModel {
public:
    /* BM complete constructor */
    BrownianMotion(float q, float theta, float cone_angle, unsigned int seed) noexcept;

    /* BM constructor, no rnd seed */
    BrownianMotion(float q, float theta, float cone_angle) noexcept;

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
    Eigen::MatrixXf                       F_;                /* State transition matrix */
    Eigen::Matrix3f                       Q_;                /* Process white noise convariance matrix */
    float                                 q_;                /* Noise standard deviation for 3D position */
    float                                 theta_;                /* Noise standard deviation for axis-angle rotation */
    float                                 cone_angle_;                /* Noise standard deviation for axis-angle axis cone */

    Eigen::Matrix3f                       sqrt_Q_;                /* Square root matrix of the process white noise convariance matrix */
    std::mt19937_64                       generator_;
    std::normal_distribution<float>       distribution_pos_;
    std::normal_distribution<float>       distribution_theta_;
    std::uniform_real_distribution<float> distribution_cone_;
    std::function<float()>                gaussian_random_pos_;
    std::function<float()>                gaussian_random_theta_;
    std::function<float()>                gaussian_random_cone_;
};

#endif /* BROWNIANMOTION_H */
