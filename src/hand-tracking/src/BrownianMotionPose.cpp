#include <BrownianMotionPose.h>

#include <cmath>
#include <iostream>
#include <utility>

using namespace Eigen;


BrownianMotionPose::BrownianMotionPose(const double q_x, const double q_y, const double q_z, const double theta, const double cone_angle, const unsigned int seed) noexcept :
    q_x_(q_x),
    q_y_(q_y),
    q_z_(q_z),
    theta_(theta * (M_PI/180.0)),
    cone_angle_(cone_angle * (M_PI/180.0)),
    cone_dir_(Vector4f(0.0, 0.0, 1.0, 0.0)),
    generator_(std::mt19937_64(seed)),
    distribution_pos_x_(std::normal_distribution<double>(0.0, q_x)),
    distribution_pos_y_(std::normal_distribution<double>(0.0, q_y)),
    distribution_pos_z_(std::normal_distribution<double>(0.0, q_z)),
    distribution_theta_(std::normal_distribution<double>(0.0, theta_)),
    distribution_cone_(std::uniform_real_distribution<double>(0.0, 1.0)),
    gaussian_random_pos_x_([&] { return (distribution_pos_x_)(generator_); }),
    gaussian_random_pos_y_([&] { return (distribution_pos_y_)(generator_); }),
    gaussian_random_pos_z_([&] { return (distribution_pos_z_)(generator_); }),
    gaussian_random_theta_([&] { return (distribution_theta_)(generator_); }),
    gaussian_random_cone_([&] { return (distribution_cone_)(generator_); }) { }


BrownianMotionPose::BrownianMotionPose(const double q_x, const double q_y, const double q_z, const double theta, const double cone_angle) noexcept :
    BrownianMotionPose(q_x, q_y, q_z, theta, cone_angle, 1) { }


BrownianMotionPose::BrownianMotionPose() noexcept :
    BrownianMotionPose(0.005, 0.005, 0.005, 3.0, 2.5, 1) { }


BrownianMotionPose::BrownianMotionPose(const BrownianMotionPose& brown) :
    q_x_(brown.q_x_),
    q_y_(brown.q_y_),
    q_z_(brown.q_z_),
    theta_(brown.theta_),
    cone_angle_(brown.cone_angle_),
    cone_dir_(brown.cone_dir_),
    generator_(brown.generator_),
    distribution_pos_x_(brown.distribution_pos_x_),
    distribution_pos_y_(brown.distribution_pos_y_),
    distribution_pos_z_(brown.distribution_pos_z_),
    distribution_theta_(brown.distribution_theta_),
    distribution_cone_(brown.distribution_cone_),
    gaussian_random_pos_x_(brown.gaussian_random_pos_x_),
    gaussian_random_pos_y_(brown.gaussian_random_pos_y_),
    gaussian_random_pos_z_(brown.gaussian_random_pos_z_),
    gaussian_random_theta_(brown.gaussian_random_theta_),
    gaussian_random_cone_(brown.gaussian_random_cone_) { }


BrownianMotionPose::BrownianMotionPose(BrownianMotionPose&& brown) noexcept :
    q_x_(brown.q_x_),
    q_y_(brown.q_y_),
    q_z_(brown.q_z_),
    theta_(brown.theta_),
    cone_angle_(brown.cone_angle_),
    cone_dir_(std::move(brown.cone_dir_)),
    generator_(std::move(brown.generator_)),
    distribution_pos_x_(std::move(brown.distribution_pos_x_)),
    distribution_pos_y_(std::move(brown.distribution_pos_y_)),
    distribution_pos_z_(std::move(brown.distribution_pos_z_)),
    distribution_theta_(std::move(brown.distribution_theta_)),
    distribution_cone_(std::move(brown.distribution_cone_)),
    gaussian_random_pos_x_(std::move(brown.gaussian_random_pos_x_)),
    gaussian_random_pos_y_(std::move(brown.gaussian_random_pos_y_)),
    gaussian_random_pos_z_(std::move(brown.gaussian_random_pos_z_)),
    gaussian_random_theta_(std::move(brown.gaussian_random_theta_)),
    gaussian_random_cone_(std::move(brown.gaussian_random_cone_))
{
    brown.q_x_        = 0.0;
    brown.q_y_        = 0.0;
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
    q_x_        = brown.q_x_;
    q_y_        = brown.q_y_;
    q_z_        = brown.q_z_;
    theta_      = brown.theta_;
    cone_angle_ = brown.cone_angle_;
    cone_dir_   = std::move(brown.cone_dir_);

    generator_              = std::move(brown.generator_);
    distribution_pos_x_     = std::move(brown.distribution_pos_x_);
    distribution_pos_y_     = std::move(brown.distribution_pos_y_);
    distribution_pos_z_     = std::move(brown.distribution_pos_z_);
    distribution_theta_     = std::move(brown.distribution_theta_);
    distribution_cone_      = std::move(brown.distribution_cone_);
    gaussian_random_pos_x_  = std::move(brown.gaussian_random_pos_x_);
    gaussian_random_pos_y_  = std::move(brown.gaussian_random_pos_y_);
    gaussian_random_pos_z_  = std::move(brown.gaussian_random_pos_z_);
    gaussian_random_theta_  = std::move(brown.gaussian_random_theta_);
    gaussian_random_cone_   = std::move(brown.gaussian_random_cone_);

    brown.q_x_        = 0.0;
    brown.q_y_        = 0.0;
    brown.q_z_        = 0.0;
    brown.theta_      = 0.0;
    brown.cone_angle_ = 0.0;

    return *this;
}


void BrownianMotionPose::propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> prop_state)
{
    prop_state = cur_state;
}


void BrownianMotionPose::motion(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> mot_state)
{
    propagate(cur_state, mot_state);

    MatrixXd sample(7, mot_state.cols());
    sample = getNoiseSample(mot_state.cols());

    mot_state.topRows<3>() += sample.topRows<3>();
    addAxisangleDisturbance(sample.bottomRows<4>(), mot_state.bottomRows<4>());
}


Eigen::MatrixXd BrownianMotionPose::getNoiseSample(const int num)
{
    MatrixXd sample(7, num);

    /* Position */
    for (unsigned int i = 0; i < num; ++i)
    {
        sample(0, i) = gaussian_random_pos_x_();
        sample(1, i) = gaussian_random_pos_y_();
        sample(2, i) = gaussian_random_pos_z_();
    }

    /* Axis-angle vector */
    /* Generate points on the spherical cap around the north pole [1]. */
    /* [1] http://math.stackexchange.com/a/205589/81266 */
    for(unsigned int i = 0; i < num; ++i)
    {
        double z   = gaussian_random_cone_() * (1 - cos(cone_angle_)) + cos(cone_angle_);
        double phi = gaussian_random_cone_() * (2.0 * M_PI);
        double x   = sqrt(1 - (z * z)) * cos(phi);
        double y   = sqrt(1 - (z * z)) * sin(phi);

        sample(3, i) = x;
        sample(4, i) = y;
        sample(5, i) = z;
        sample(6, i) = gaussian_random_theta_();
    }

    return sample;
}


std::pair<std::size_t, std::size_t> BrownianMotionPose::getOutputSize() const
{
    return std::make_pair(7, 0);
}


void BrownianMotionPose::addAxisangleDisturbance(const Ref<const MatrixXd>& disturbance_vec, Ref<MatrixXd> current_vec)
{
    for (unsigned int i = 0; i < current_vec.cols(); ++i)
    {
        double ang = current_vec(3, i) + disturbance_vec(3, i);

        if      (ang >   M_PI) ang -= 2.0 * M_PI;
        else if (ang <= -M_PI) ang += 2.0 * M_PI;


        /* Find the rotation axis 'u' and rotation angle 'rot' [1] */
        Vector3d def_dir(0.0, 0.0, 1.0);

        Vector3d u = def_dir.cross(current_vec.col(i).head<3>()).normalized();

        double rot  = static_cast<double>(std::acos(current_vec.col(i).head<3>().dot(def_dir)));


        /* Convert rotation axis and angle to 3x3 rotation matrix [2] */
        /* [2] https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle */
        Matrix3d cross_matrix;
        cross_matrix <<     0,  -u(2),   u(1),
                         u(2),      0,  -u(0),
                        -u(1),   u(0),      0;

        Matrix3d R = std::cos(rot) * Matrix3d::Identity() + std::sin(rot) * cross_matrix + (1 - std::cos(rot)) * (u * u.transpose());


        current_vec.col(i).head<3>() = (R * disturbance_vec.col(i).head<3>()).normalized();
        current_vec(3, i)            = ang;
    }
}
