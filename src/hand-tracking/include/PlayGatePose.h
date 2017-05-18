#ifndef PLAYGATEPOSE_H
#define PLAYGATEPOSE_H

#include "GatePose.h"

#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Vector.h>


class PlayGatePose : public GatePose
{
public:
    /* Constructor */
    PlayGatePose(std::unique_ptr<VisualCorrection> visual_correction,
                 double gate_x, double gate_y, double gate_z,
                 double gate_aperture, double gate_rotation,
                 const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    PlayGatePose(std::unique_ptr<VisualCorrection> visual_correction,
                 const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    /* Destructor */
    ~PlayGatePose() noexcept override;

protected:
    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_enc_;
    iCub::iKin::iCubArm                      icub_kin_arm_;

    Eigen::VectorXd   readPose() override;

    yarp::sig::Vector readRootToEE();

    yarp::sig::Vector readTorso();

private:
    yarp::os::ConstString  ID_     = "PlayGatePose";
    yarp::os::ConstString  log_ID_ = "[" + ID_ + "]";

    yarp::os::ConstString  robot_;
    yarp::os::ConstString  laterality_;
    yarp::os::ConstString  port_prefix_;
};

#endif /* PLAYGATEPOSE_H */
