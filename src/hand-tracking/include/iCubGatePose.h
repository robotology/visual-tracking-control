#ifndef ICUBGATEPOSE_H
#define ICUBGATEPOSE_H

#include "GatePose.h"


class iCubGatePose : public GatePose
{
public:
    /* Constructor */
    iCubGatePose(std::unique_ptr<VisualCorrection> visual_correction,
                 double gate_x, double gate_y, double gate_z,
                 double gate_aperture, double gate_rotation,
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
    yarp::os::ConstString  ID_     = "iCubGatePose";
    yarp::os::ConstString  log_ID_ = "[" + ID_ + "]";

    yarp::os::ConstString  robot_;
    yarp::os::ConstString  laterality_;
    yarp::os::ConstString  port_prefix_;
};

#endif /* ICUBGATEPOSE_H */
