#include "BrownianMotion.h"

#include <cmath>
#include <utility>

using namespace Eigen;


BrownianMotion::BrownianMotion(float q, float theta, float cone_angle, unsigned int seed) noexcept :
    F_(MatrixXf::Identity(6, 6)), q_(q), theta_(theta * (M_PI/180.0)), cone_angle_(cone_angle * (M_PI/180.0)),
    generator_(std::mt19937_64(seed)), distribution_pos_(std::normal_distribution<float>(0.0, q_)), distribution_theta_(std::normal_distribution<float>(0.0, theta_)), distribution_cone_(std::uniform_real_distribution<float>(0.0, 1.0)), gaussian_random_pos_([&] { return (distribution_pos_)(generator_); }), gaussian_random_theta_([&] { return (distribution_theta_)(generator_); }), gaussian_random_cone_([&] { return (distribution_cone_)(generator_); }) { }


BrownianMotion::BrownianMotion(float q, float theta, float cone_angle) noexcept :
    BrownianMotion(q, theta, cone_angle, 1) { }


BrownianMotion::BrownianMotion() noexcept :
    BrownianMotion(0.005, 3.0, 2.5, 1) { }


BrownianMotion::~BrownianMotion() noexcept { }


BrownianMotion::BrownianMotion(const BrownianMotion& wna) :
    F_(wna.F_), q_(wna.q_), theta_(wna.theta_), cone_angle_(wna.cone_angle_),
    generator_(wna.generator_), distribution_pos_(wna.distribution_pos_), distribution_theta_(wna.distribution_theta_), distribution_cone_(wna.distribution_cone_), gaussian_random_pos_(wna.gaussian_random_pos_), gaussian_random_theta_(wna.gaussian_random_theta_), gaussian_random_cone_(wna.gaussian_random_cone_) { }


BrownianMotion::BrownianMotion(BrownianMotion&& wna) noexcept :
    F_(std::move(wna.F_)), q_(wna.q_), theta_(wna.theta_), cone_angle_(wna.cone_angle_),
    generator_(std::move(wna.generator_)), distribution_pos_(std::move(wna.distribution_pos_)), distribution_theta_(std::move(wna.distribution_theta_)), distribution_cone_(std::move(wna.distribution_cone_)), gaussian_random_pos_(std::move(wna.gaussian_random_pos_)), gaussian_random_theta_(std::move(wna.gaussian_random_theta_)), gaussian_random_cone_(std::move(wna.gaussian_random_cone_))
{
    wna.q_          = 0.0;
    wna.theta_      = 0.0;
    wna.cone_angle_ = 0.0;
}


BrownianMotion& BrownianMotion::operator=(const BrownianMotion& wna)
{
    BrownianMotion tmp(wna);
    *this = std::move(tmp);

    return *this;
}


BrownianMotion& BrownianMotion::operator=(BrownianMotion&& wna) noexcept
{
    F_          = std::move(wna.F_);
    q_          = wna.q_;
    theta_      = wna.theta_;
    cone_angle_ = wna.cone_angle_;

    generator_             = std::move(wna.generator_);
    distribution_pos_      = std::move(wna.distribution_pos_);
    distribution_theta_    = std::move(wna.distribution_theta_);
    distribution_cone_     = std::move(wna.distribution_cone_);
    gaussian_random_pos_   = std::move(wna.gaussian_random_pos_);
    gaussian_random_theta_ = std::move(wna.gaussian_random_theta_);
    gaussian_random_cone_  = std::move(wna.gaussian_random_cone_);

    wna.q_          = 0.0;
    wna.theta_      = 0.0;
    wna.cone_angle_ = 0.0;

    return *this;
}


void BrownianMotion::propagate(const Ref<const VectorXf> & cur_state, Ref<VectorXf> prop_state)
{
    prop_state = F_ * cur_state;
}


// FIXME: sample.tail(3) è usato come direzione iniziale dell'asse di rotazione casuale
void BrownianMotion::noiseSample(Ref<VectorXf> sample)
{
    /* Position */
    sample.head(3) = VectorXf::NullaryExpr(3, gaussian_random_pos_);

    /* Axis-angle */
    /* Generate points on the spherical cap around the north pole [1]. */
    /* [1] http://math.stackexchange.com/a/205589/81266 */
    float z   = gaussian_random_cone_() * (1 - cos(cone_angle_)) + cos(cone_angle_);
    float phi = gaussian_random_cone_() * (2 * M_PI);
    float x   = sqrt(1 - (z * z)) * cos(phi);
    float y   = sqrt(1 - (z * z)) * sin(phi);

    /* Find the rotation axis 'u' and rotation angle 'rot' [1] */
    Vector3f def_dir(0.0, 0.0, 1.0);
    Vector3f cone_dir;
    if (sample.tail(3).isZero())
        cone_dir << 0.0, 0.0, 1.0;
    else
        cone_dir = sample.tail(3).normalized();
    Vector3f u = def_dir.cross(cone_dir).normalized();
    float rot = static_cast<float>(acos(cone_dir.dot(def_dir)));

    /* Convert rotation axis and angle to 3x3 rotation matrix [2] */
    /* [2] https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle */
    Matrix3f cross_matrix;
    cross_matrix <<     0,  -u(2),   u(1),
                     u(2),      0,  -u(0),
                    -u(1),   u(0),      0;
    Matrix3f R = cos(rot) * Matrix3f::Identity() + sin(rot) * cross_matrix + (1 - cos(rot)) * (u * u.transpose());

    /* Rotate [x y z]' from north pole to 'cone_dir' */
    Vector3f r_to_rotate(x, y, z);
    Vector3f r = R * r_to_rotate;

    float ang      =  static_cast<float>(sample.tail(3).norm()) + gaussian_random_theta_();
    sample.tail(3) =  r;
    sample.tail(3) *= ang;
}


void BrownianMotion::motion(const Ref<const VectorXf>& cur_state, Ref<VectorXf> next_state)
{
    propagate(cur_state, next_state);

    VectorXf sample(VectorXf::Zero(6, 1));
    sample.tail(3) = next_state.tail(3);
    noiseSample(sample);

    next_state.head(3) += sample.head(3);
    next_state.tail(3) =  sample.tail(3);
}
