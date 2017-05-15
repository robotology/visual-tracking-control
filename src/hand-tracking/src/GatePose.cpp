#include "GatePose.h"

#include <cmath>

using namespace bfl;
using namespace cv;
using namespace Eigen;


GatePose::GatePose(std::unique_ptr<VisualCorrection> visual_correction,
                   double gate_x, double gate_y, double gate_z,
                   double gate_rotation, double gate_aperture) noexcept :
    VisualCorrectionDecorator(std::move(visual_correction)),
    gate_x_(gate_x), gate_y_(gate_y), gate_z_(gate_z), gate_aperture_((M_PI / 180.0) * gate_aperture), gate_rotation_((M_PI / 180.0) * gate_rotation)
{
    ee_pose_ = VectorXd::Zero(6);
}


GatePose::GatePose(std::unique_ptr<VisualCorrection> visual_correction) noexcept :
    GatePose(std::move(visual_correction), 0.1, 0.1, 0.1, 5, 30) { }


GatePose::~GatePose() noexcept { }


void GatePose::correct(const Ref<const MatrixXf>& pred_state, InputArray measurements, Ref<MatrixXf> cor_state)
{
    VisualCorrectionDecorator::correct(pred_state, measurements, cor_state);

    ee_pose_ = readPose();

    for (int i = 0; i < pred_state.cols(); ++i)
    {
        if (!isInsideEllipsoid(pred_state.block<3, 1>(0, i))              ||
            !isWithinRotation (pred_state.block<3, 1>(3, i).norm())       ||
            !isInsideCone     (pred_state.block<3, 1>(3, i).normalized())   )
            cor_state(i, 0) = std::numeric_limits<float>::min();
    }
}


void GatePose::innovation(const Ref<const MatrixXf>& pred_state, InputArray measurements, Ref<MatrixXf> innovation)
{
    VisualCorrectionDecorator::innovation(pred_state, measurements, innovation);
}


void GatePose::likelihood(const Ref<const MatrixXf>& innovation, Ref<MatrixXf> cor_state)
{
    VisualCorrectionDecorator::likelihood(innovation, cor_state);
}


bool GatePose::setObservationModelProperty(const std::string& property)
{
    return VisualCorrectionDecorator::setObservationModelProperty(property);
}


bool GatePose::isInsideEllipsoid(const Eigen::Ref<const Eigen::VectorXf>& state)
{
    return ( (abs(state(0) - ee_pose_(0)) < gate_x_) &&
             (abs(state(1) - ee_pose_(1)) < gate_y_) &&
             (abs(state(2) - ee_pose_(2)) < gate_z_) );
}


bool GatePose::isWithinRotation(float rot_angle)
{
    float ang_diff = rot_angle - ee_pose_(6);
    if (ang_diff >  2.0 * M_PI) ang_diff -= 2.0 * M_PI;
    if (ang_diff <=        0.0) ang_diff += 2.0 * M_PI;

    return (ang_diff <= gate_rotation_);
}


bool GatePose::isInsideCone(const Eigen::Ref<const Eigen::VectorXf>& state)
{
    double   half_aperture    =  gate_aperture_ / 2.0;

    VectorXd test_direction   = -state.cast<double>();

    VectorXd fwdkin_direction = -ee_pose_.middleRows<3>(3);

    return ( (test_direction.dot(fwdkin_direction) / test_direction.norm() / fwdkin_direction.norm()) >= cos(half_aperture) );
}
