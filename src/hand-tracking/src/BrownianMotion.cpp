#include "BrownianMotion.h"

#include <cmath>
#include <iostream>
#include <utility>

using namespace Eigen;


BrownianMotion::BrownianMotion(const float q_xy, const float q_z, const float theta, const float cone_angle, const unsigned int seed) noexcept :
    F_(MatrixXf::Identity(6, 6)),
    q_xy_(q_xy),
    q_z_(q_z),
    theta_(theta * (M_PI/180.0)),
    cone_angle_(cone_angle * (M_PI/180.0)),
    cone_dir_(Vector4f(0.0, 0.0, 1.0, 0.0)),
    generator_(std::mt19937_64(seed)),
    distribution_pos_xy_(std::normal_distribution<float>(0.0, q_xy)),
    distribution_pos_z_(std::normal_distribution<float>(0.0, q_z)),
    distribution_theta_(std::normal_distribution<float>(0.0, theta_)),
    distribution_cone_(std::uniform_real_distribution<float>(0.0, 1.0)),
    gaussian_random_pos_xy_([&] { return (distribution_pos_xy_)(generator_); }),
    gaussian_random_pos_z_([&] { return (distribution_pos_z_)(generator_); }),
    gaussian_random_theta_([&] { return (distribution_theta_)(generator_); }),
    gaussian_random_cone_([&] { return (distribution_cone_)(generator_); }) { }


BrownianMotion::BrownianMotion(const float q_xy, const float q_z, const float theta, const float cone_angle) noexcept :
    BrownianMotion(q_xy, q_z, theta, cone_angle, 1) { }


BrownianMotion::BrownianMotion() noexcept :
    BrownianMotion(0.005, 0.005, 3.0, 2.5, 1) { }


BrownianMotion::~BrownianMotion() noexcept { }


BrownianMotion::BrownianMotion(const BrownianMotion& brown) :
    F_(brown.F_),
    q_xy_(brown.q_xy_),
    q_z_(brown.q_z_),
    theta_(brown.theta_),
    cone_angle_(brown.cone_angle_),
    cone_dir_(brown.cone_dir_),
    generator_(brown.generator_),
    distribution_pos_xy_(brown.distribution_pos_xy_),
    distribution_pos_z_(brown.distribution_pos_z_),
    distribution_theta_(brown.distribution_theta_),
    distribution_cone_(brown.distribution_cone_),
    gaussian_random_pos_xy_(brown.gaussian_random_pos_xy_),
    gaussian_random_pos_z_(brown.gaussian_random_pos_z_),
    gaussian_random_theta_(brown.gaussian_random_theta_),
    gaussian_random_cone_(brown.gaussian_random_cone_) { }


BrownianMotion::BrownianMotion(BrownianMotion&& brown) noexcept :
    F_(std::move(brown.F_)),
    q_xy_(brown.q_xy_),
    theta_(brown.theta_),
    cone_angle_(brown.cone_angle_),
    cone_dir_(std::move(brown.cone_dir_)),
    generator_(std::move(brown.generator_)),
    distribution_pos_xy_(std::move(brown.distribution_pos_xy_)),
    distribution_pos_z_(std::move(brown.distribution_pos_z_)),
    distribution_theta_(std::move(brown.distribution_theta_)),
    distribution_cone_(std::move(brown.distribution_cone_)),
    gaussian_random_pos_xy_(std::move(brown.gaussian_random_pos_xy_)),
    gaussian_random_pos_z_(std::move(brown.gaussian_random_pos_z_)),
    gaussian_random_theta_(std::move(brown.gaussian_random_theta_)),
    gaussian_random_cone_(std::move(brown.gaussian_random_cone_))
{
    brown.q_xy_       = 0.0;
    brown.q_z_        = 0.0;
    brown.theta_      = 0.0;
    brown.cone_angle_ = 0.0;
}


BrownianMotion& BrownianMotion::operator=(const BrownianMotion& brown)
{
    BrownianMotion tmp(brown);
    *this = std::move(tmp);

    return *this;
}


BrownianMotion& BrownianMotion::operator=(BrownianMotion&& brown) noexcept
{
    F_          = std::move(brown.F_);
    q_xy_       = brown.q_xy_;
    q_z_        = brown.q_z_;
    theta_      = brown.theta_;
    cone_angle_ = brown.cone_angle_;
    cone_dir_   = std::move(brown.cone_dir_);

    generator_              = std::move(brown.generator_);
    distribution_pos_xy_    = std::move(brown.distribution_pos_xy_);
    distribution_pos_z_     = std::move(brown.distribution_pos_z_);
    distribution_theta_     = std::move(brown.distribution_theta_);
    distribution_cone_      = std::move(brown.distribution_cone_);
    gaussian_random_pos_xy_ = std::move(brown.gaussian_random_pos_xy_);
    gaussian_random_pos_z_  = std::move(brown.gaussian_random_pos_z_);
    gaussian_random_theta_  = std::move(brown.gaussian_random_theta_);
    gaussian_random_cone_   = std::move(brown.gaussian_random_cone_);

    brown.q_xy_       = 0.0;
    brown.q_z_        = 0.0;
    brown.theta_      = 0.0;
    brown.cone_angle_ = 0.0;

    return *this;
}


void BrownianMotion::propagate(const Ref<const VectorXf> & cur_state, Ref<VectorXf> prop_state)
{
    prop_state = F_ * cur_state;
}


void BrownianMotion::noiseSample(Ref<VectorXf> sample)
{
    /* Position */
    sample.head<2>() = VectorXf::NullaryExpr(2, gaussian_random_pos_xy_);
    sample(2)        = VectorXf::NullaryExpr(1, gaussian_random_pos_z_).coeff(0);

    /* Axis-angle */
    /* Generate points on the spherical cap around the north pole [1]. */
    /* [1] http://math.stackexchange.com/a/205589/81266 */
    float z   = gaussian_random_cone_() * (1 - cos(cone_angle_)) + cos(cone_angle_);
    float phi = gaussian_random_cone_() * (2.0 * M_PI);
    float x   = sqrt(1 - (z * z)) * cos(phi);
    float y   = sqrt(1 - (z * z)) * sin(phi);

    /* Generate random rotation angle */
    sample.middleRows<3>(3) = Vector3f(x, y, z);
    sample(6) = gaussian_random_theta_();
}
