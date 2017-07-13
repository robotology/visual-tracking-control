#ifndef ICUBGATEPOSE_H
#define ICUBGATEPOSE_H

#include "GatePose.h"

#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class iCubGatePose : public GatePose
{
public:
    /* Constructor */
    iCubGatePose(std::unique_ptr<VisualCorrection> visual_correction,
                 const double gate_x, const double gate_y, const double gate_z,
                 const double gate_aperture,
                 const double gate_rotation,
                 const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    iCubGatePose(std::unique_ptr<VisualCorrection> visual_correction,
                 const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    /* Destructor */
    ~iCubGatePose() noexcept override;

protected:
    yarp::dev::PolyDriver  drv_arm_enc_;
    yarp::dev::IEncoders * itf_arm_enc_;
    yarp::dev::PolyDriver  drv_torso_enc_;
    yarp::dev::IEncoders * itf_torso_enc_;

    iCub::iKin::iCubArm    icub_kin_arm_;

    Eigen::VectorXd   readPose() override;

    yarp::sig::Vector readRootToEE();

    yarp::sig::Vector readTorso();

private:
    const yarp::os::ConstString ID_     = "iCubGatePose";
    const yarp::os::ConstString log_ID_ = "[" + ID_ + "]";

    yarp::os::ConstString robot_;
    yarp::os::ConstString laterality_;
    yarp::os::ConstString port_prefix_;
};

#endif /* ICUBGATEPOSE_H */
