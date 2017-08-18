#include <utility>

#include "CartesianAxisAnglePrediction.h"

using namespace bfl;
using namespace Eigen;


CartesianAxisAnglePrediction::CartesianAxisAnglePrediction(std::unique_ptr<StateModel> transition_model) noexcept :
    state_model_(std::move(transition_model)) { }


CartesianAxisAnglePrediction::~CartesianAxisAnglePrediction() noexcept { }


CartesianAxisAnglePrediction::CartesianAxisAnglePrediction(CartesianAxisAnglePrediction&& pf_prediction) noexcept :
    state_model_(std::move(pf_prediction.state_model_)) { };


CartesianAxisAnglePrediction& CartesianAxisAnglePrediction::operator=(CartesianAxisAnglePrediction&& pf_prediction) noexcept
{
    state_model_ = std::move(pf_prediction.state_model_);

    return *this;
}


void CartesianAxisAnglePrediction::motion(const Ref<const VectorXf>& cur_state, Ref<VectorXf> prop_state)
{
    state_model_->propagate(cur_state, prop_state);
}


void CartesianAxisAnglePrediction::motionDisturbance(Ref<VectorXf> sample)
{
    state_model_->noiseSample(sample);
}


void CartesianAxisAnglePrediction::predict(const Ref<const VectorXf>& prev_state, Ref<VectorXf> pred_state)
{
    motion(prev_state, pred_state);


    VectorXf sample(VectorXf::Zero(7, 1));
    motionDisturbance(sample);

    pred_state.head<3>() += sample.head<3>();
    addAxisangleDisturbance(pred_state.tail<4>(), sample.middleRows<4>(3));
}

bool CartesianAxisAnglePrediction::setStateModelProperty(const std::string& property)
{
    return state_model_->setProperty(property);
}


void CartesianAxisAnglePrediction::addAxisangleDisturbance(Ref<Vector4f> current_vec, const Ref<const Vector4f>& disturbance_vec)
{
    float ang = current_vec(3) + disturbance_vec(3);

    if      (ang >   M_PI) ang -= 2.0 * M_PI;
    else if (ang <= -M_PI) ang += 2.0 * M_PI;


    /* Find the rotation axis 'u' and rotation angle 'rot' [1] */
    Vector3f def_dir(0.0, 0.0, 1.0);

    Vector3f u = def_dir.cross(current_vec.head<3>()).normalized();

    float rot  = static_cast<float>(std::acos(current_vec.head<3>().dot(def_dir)));


    /* Convert rotation axis and angle to 3x3 rotation matrix [2] */
    /* [2] https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle */
    Matrix3f cross_matrix;
    cross_matrix <<     0,  -u(2),   u(1),
                     u(2),      0,  -u(0),
                    -u(1),   u(0),      0;

    Matrix3f R = std::cos(rot) * Matrix3f::Identity() + std::sin(rot) * cross_matrix + (1 - std::cos(rot)) * (u * u.transpose());


    current_vec.head<3>() = (R * disturbance_vec.head<3>()).normalized();
    current_vec(3)        = ang;
}
