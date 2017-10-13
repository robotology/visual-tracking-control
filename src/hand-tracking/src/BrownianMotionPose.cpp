#include "BrownianMotionPose.h"

#include <cmath>
#include <iostream>
#include <utility>

using namespace Eigen;


BrownianMotionPose::BrownianMotionPose(const float q_xy, const float q_z, const float theta, const float cone_angle, const unsigned int seed) noexcept :
    F_(MatrixXf::Identity(7, 7)),
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


BrownianMotionPose::BrownianMotionPose(const float q_xy, const float q_z, const float theta, const float cone_angle) noexcept :
    BrownianMotionPose(q_xy, q_z, theta, cone_angle, 1) { }


BrownianMotionPose::BrownianMotionPose() noexcept :
    BrownianMotionPose(0.005, 0.005, 3.0, 2.5, 1) { }


BrownianMotionPose::BrownianMotionPose(const BrownianMotionPose& brown) :
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


BrownianMotionPose::BrownianMotionPose(BrownianMotionPose&& brown) noexcept :
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


BrownianMotionPose::~BrownianMotionPose() noexcept { }


BrownianMotionPose& BrownianMotionPose::operator=(const BrownianMotionPose& brown)
{
    BrownianMotionPose tmp(brown);
    *this = std::move(tmp);

    return *this;
}


BrownianMotionPose& BrownianMotionPose::operator=(BrownianMotionPose&& brown) noexcept
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


void BrownianMotionPose::propagate(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> prop_state)
{
    prop_state = F_ * cur_state;
}


void BrownianMotionPose::motion(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> mot_state)
{
    propagate(cur_state, mot_state);

    MatrixXf sample(7, mot_state.cols());
    sample = getNoiseSample(mot_state.cols());

    mot_state.topRows<3>() += sample.topRows<3>();
}


Eigen::MatrixXf BrownianMotionPose::getNoiseSample(const int num)
{
    MatrixXf sample(7, num);

    /* Position */
    sample.topRows<2>() = MatrixXf::NullaryExpr(2, num, gaussian_random_pos_xy_);
    sample.row(2)       = MatrixXf::NullaryExpr(1, num, gaussian_random_pos_z_);

    /* Axis-angle */
    /* Generate points on the spherical cap around the north pole [1]. */
    /* [1] http://math.stackexchange.com/a/205589/81266 */
    for(unsigned int i = 0; i < num; ++i)
    {
        float z   = gaussian_random_cone_() * (1 - cos(cone_angle_)) + cos(cone_angle_);
        float phi = gaussian_random_cone_() * (2.0 * M_PI);
        float x   = sqrt(1 - (z * z)) * cos(phi);
        float y   = sqrt(1 - (z * z)) * sin(phi);

        sample.col(i).middleRows<3>(3) = Vector3f(x, y, z);
    }

    /* Generate random rotation angle */
    sample.row(6) = MatrixXf::NullaryExpr(1, num, gaussian_random_theta_);

    return sample;
}


void BrownianMotionPose::addAxisangleDisturbance(const Ref<const MatrixXf>& disturbance_vec, Ref<MatrixXf> current_vec)
{
    for (unsigned int i = 0; i < current_vec.cols(); ++i)
    {
        float ang = current_vec(i, 3) + disturbance_vec(i, 3);

        if      (ang >   M_PI) ang -= 2.0 * M_PI;
        else if (ang <= -M_PI) ang += 2.0 * M_PI;


        /* Find the rotation axis 'u' and rotation angle 'rot' [1] */
        Vector3f def_dir(0.0, 0.0, 1.0);

        Vector3f u = def_dir.cross(current_vec.col(i).head<3>()).normalized();

        float rot  = static_cast<float>(std::acos(current_vec.col(i).head<3>().dot(def_dir)));


        /* Convert rotation axis and angle to 3x3 rotation matrix [2] */
        /* [2] https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle */
        Matrix3f cross_matrix;
        cross_matrix <<     0,  -u(2),   u(1),
                         u(2),      0,  -u(0),
                        -u(1),   u(0),      0;

        Matrix3f R = std::cos(rot) * Matrix3f::Identity() + std::sin(rot) * cross_matrix + (1 - std::cos(rot)) * (u * u.transpose());


        current_vec.col(i).head<3>() = (R * disturbance_vec.col(i).head<3>()).normalized();
        current_vec(i, 3)            = ang;
    }
}
