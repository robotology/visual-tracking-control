#include <GatePose.h>

#include <cmath>

using namespace bfl;
using namespace Eigen;


GatePose::GatePose(std::unique_ptr<PFCorrection> visual_correction,
                   const double gate_x, const double gate_y, const double gate_z,
                   const double gate_rotation,
                   const double gate_aperture) noexcept :
    PFCorrectionDecorator(std::move(visual_correction)),
    gate_x_(gate_x), gate_y_(gate_y), gate_z_(gate_z),
    gate_aperture_((M_PI / 180.0) * gate_aperture),
    gate_rotation_((M_PI / 180.0) * gate_rotation)
{
    ee_pose_ = VectorXd::Zero(7);
}


GatePose::GatePose(std::unique_ptr<PFCorrection> visual_correction) noexcept :
    GatePose(std::move(visual_correction), 0.1, 0.1, 0.1, 5, 30) { }


GatePose::~GatePose() noexcept { }


void GatePose::correctStep(const ParticleSet& pred_particles, ParticleSet& corr_particles)
{
    PFCorrectionDecorator::correctStep(pred_particles, corr_particles);

    ee_pose_ = readPose();

    for (int i = 0; i < corr_particles.state().cols(); ++i)
    {
        if (!isInsideEllipsoid(corr_particles.state(i).topRows<3>())     ||
            !isInsideCone     (corr_particles.state(i).middleRows<3>(3)) ||
            !isWithinRotation (corr_particles.state(i, 6))                 )
            corr_particles.weight(i) = std::numeric_limits<double>::min();
    }
}


bool GatePose::isInsideEllipsoid(const Ref<const VectorXd>& state)
{
    return ( (abs(state(0) - ee_pose_(0)) <= gate_x_) &&
             (abs(state(1) - ee_pose_(1)) <= gate_y_) &&
             (abs(state(2) - ee_pose_(2)) <= gate_z_) );
}


bool GatePose::isWithinRotation(double rot_angle)
{
    double ang_diff = abs(rot_angle - ee_pose_(6));

    return (ang_diff <= gate_rotation_);
}


bool GatePose::isInsideCone(const Ref<const VectorXd>& state)
{
    /* See: http://stackoverflow.com/questions/10768142/verify-if-point-is-inside-a-cone-in-3d-space#10772759 */

    double   half_aperture    =  gate_aperture_ / 2.0;

    VectorXd fwdkin_direction = -ee_pose_.middleRows<3>(3);

    return ( (state.dot(fwdkin_direction) / state.norm() / fwdkin_direction.norm()) >= cos(half_aperture) );
}
