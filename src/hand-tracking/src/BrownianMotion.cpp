#include "BrownianMotion.h"

#include <cmath>
#include <iostream>
#include <utility>

using namespace Eigen;


BrownianMotion::BrownianMotion(const float q_xy, const float q_z, const float theta, const float cone_angle, const unsigned int seed) noexcept :
    F_(MatrixXf::Identity(6, 6)), q_xy_(q_xy), q_z_(q_z), theta_(theta * (M_PI/180.0)), cone_angle_(cone_angle * (M_PI/180.0)), cone_dir_(Vector4f(0.0, 0.0, 1.0, 0.0)),
    generator_(std::mt19937_64(seed)),
    distribution_pos_xy_(std::normal_distribution<float>(0.0, q_xy)), distribution_pos_z_(std::normal_distribution<float>(0.0, q_z)),
    distribution_theta_(std::normal_distribution<float>(0.0, theta_)), distribution_cone_(std::uniform_real_distribution<float>(0.0, 1.0)),
    gaussian_random_pos_xy_([&] { return (distribution_pos_xy_)(generator_); }), gaussian_random_pos_z_([&] { return (distribution_pos_z_)(generator_); }),
    gaussian_random_theta_([&] { return (distribution_theta_)(generator_); }), gaussian_random_cone_([&] { return (distribution_cone_)(generator_); }) { }


BrownianMotion::BrownianMotion(const float q_xy, const float q_z, const float theta, const float cone_angle) noexcept :
    BrownianMotion(q_xy, q_z, theta, cone_angle, 1) { }


BrownianMotion::BrownianMotion() noexcept :
    BrownianMotion(0.005, 0.005, 3.0, 2.5, 1) { }


BrownianMotion::~BrownianMotion() noexcept { }


BrownianMotion::BrownianMotion(const BrownianMotion& brown) :
    F_(brown.F_), q_xy_(brown.q_xy_), q_z_(brown.q_z_), theta_(brown.theta_), cone_angle_(brown.cone_angle_),  cone_dir_(brown.cone_dir_),
    generator_(brown.generator_), distribution_pos_xy_(brown.distribution_pos_xy_), distribution_pos_z_(brown.distribution_pos_z_), distribution_theta_(brown.distribution_theta_), distribution_cone_(brown.distribution_cone_), gaussian_random_pos_xy_(brown.gaussian_random_pos_xy_), gaussian_random_pos_z_(brown.gaussian_random_pos_z_), gaussian_random_theta_(brown.gaussian_random_theta_), gaussian_random_cone_(brown.gaussian_random_cone_) { }


BrownianMotion::BrownianMotion(BrownianMotion&& brown) noexcept :
F_(std::move(brown.F_)), q_xy_(brown.q_xy_), theta_(brown.theta_), cone_angle_(brown.cone_angle_), cone_dir_(std::move(brown.cone_dir_)),
    generator_(std::move(brown.generator_)), distribution_pos_xy_(std::move(brown.distribution_pos_xy_)), distribution_pos_z_(std::move(brown.distribution_pos_z_)), distribution_theta_(std::move(brown.distribution_theta_)), distribution_cone_(std::move(brown.distribution_cone_)), gaussian_random_pos_xy_(std::move(brown.gaussian_random_pos_xy_)), gaussian_random_pos_z_(std::move(brown.gaussian_random_pos_z_)), gaussian_random_theta_(std::move(brown.gaussian_random_theta_)), gaussian_random_cone_(std::move(brown.gaussian_random_cone_))
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
//    setConeDirection(cur_state.tail<3>());

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
    float phi = gaussian_random_cone_() * (2 * M_PI);
    float x   = sqrt(1 - (z * z)) * cos(phi);
    float y   = sqrt(1 - (z * z)) * sin(phi);

//    /* Find the rotation axis 'u' and rotation angle 'rot' [1] */
//    Vector3f def_dir(0.0, 0.0, 1.0);
//    Vector3f u = def_dir.cross(cone_dir_.head<3>()).normalized();
//    float rot = static_cast<float>(acos(cone_dir_.head<3>().dot(def_dir)));
//
//    /* Convert rotation axis and angle to 3x3 rotation matrix [2] */
//    /* [2] https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle */
//    Matrix3f cross_matrix;
//    cross_matrix <<     0,  -u(2),   u(1),
//                     u(2),      0,  -u(0),
//                    -u(1),   u(0),      0;
//    Matrix3f R = cos(rot) * Matrix3f::Identity() + sin(rot) * cross_matrix + (1 - cos(rot)) * (u * u.transpose());
//
//    /* Rotate [x y z]' from north pole to 'cone_dir_' */
//    Vector3f r_to_rotate(x, y, z);
//    Vector3f r = R * r_to_rotate;

    /* Generate random rotation angle */
//    float ang = static_cast<float>(cone_dir_(3)) + gaussian_random_theta_();

//    sample.tail<3>() = ang * r;

    /* Generate random rotation angle */
    float ang = gaussian_random_theta_();

    sample.middleRows<3>(3) = Vector3f(x, y, z);
    sample(6) = ang;
}


void BrownianMotion::setConeDirection(const Eigen::Ref<const Eigen::Vector3f>& cone_dir)
{
    cone_dir_(3)        = cone_dir.norm();
    cone_dir_.head<3>() = cone_dir.normalized();
}
