#include "GatePose.h"

#include <cmath>

using namespace bfl;
using namespace cv;
using namespace Eigen;


GatePose::GatePose(std::unique_ptr<PFVisualCorrection> visual_correction,
                   const double gate_x, const double gate_y, const double gate_z,
                   const double gate_rotation,
                   const double gate_aperture) noexcept :
    PFVisualCorrectionDecorator(std::move(visual_correction)),
    gate_x_(gate_x), gate_y_(gate_y), gate_z_(gate_z),
    gate_aperture_((M_PI / 180.0) * gate_aperture),
    gate_rotation_((M_PI / 180.0) * gate_rotation)
{
    ee_pose_ = VectorXd::Zero(7);
}


GatePose::GatePose(std::unique_ptr<PFVisualCorrection> visual_correction) noexcept :
    GatePose(std::move(visual_correction), 0.1, 0.1, 0.1, 5, 30) { }


GatePose::~GatePose() noexcept { }


void GatePose::correct(const Ref<const MatrixXf>& pred_states, const Ref<const VectorXf>& pred_weights, cv::InputArray measurements,
                       Ref<MatrixXf> cor_states, Ref<VectorXf> cor_weights)
{
    PFVisualCorrectionDecorator::correct(pred_states, pred_weights, measurements,
                                         cor_states, cor_weights);

    ee_pose_ = readPose();

    for (int i = 0; i < cor_states.cols(); ++i)
    {
        if (!isInsideEllipsoid(cor_states.col(i).head<3>())        ||
            !isInsideCone     (cor_states.col(i).middleRows<3>(3)) ||
            !isWithinRotation (cor_states(6, i))                     )
            cor_weights(i) = std::numeric_limits<float>::min();
    }
}


void GatePose::innovation(const Ref<const MatrixXf>& pred_states, cv::InputArray measurements, Ref<MatrixXf> innovations)
{
    PFVisualCorrectionDecorator::innovation(pred_states, measurements, innovations);
}


double GatePose::likelihood(const Ref<const MatrixXf>& innovations)
{
    return PFVisualCorrectionDecorator::likelihood(innovations);
}


VisualObservationModel& GatePose::getVisualObservationModel()
{
    return PFVisualCorrectionDecorator::getVisualObservationModel();
}


bool GatePose::isInsideEllipsoid(const Ref<const VectorXf>& state)
{
    return ( (abs(state(0) - ee_pose_(0)) <= gate_x_) &&
             (abs(state(1) - ee_pose_(1)) <= gate_y_) &&
             (abs(state(2) - ee_pose_(2)) <= gate_z_) );
}


bool GatePose::isWithinRotation(float rot_angle)
{
    float ang_diff = abs(rot_angle - ee_pose_(6));

    return (ang_diff <= gate_rotation_);
}


bool GatePose::isInsideCone(const Ref<const VectorXf>& state)
{
    /* See: http://stackoverflow.com/questions/10768142/verify-if-point-is-inside-a-cone-in-3d-space#10772759 */

    double   half_aperture    =  gate_aperture_ / 2.0;

    VectorXd test_direction   = -state.cast<double>();

    VectorXd fwdkin_direction = -ee_pose_.middleRows<3>(3);

    return ( (test_direction.dot(fwdkin_direction) / test_direction.norm() / fwdkin_direction.norm()) >= cos(half_aperture) );
}
