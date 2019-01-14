#ifndef INITICUBARM_H
#define INITICUBARM_H

#include <InitPoseParticlesAxisAngle.h>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Vector.h>

#include <string>


class InitiCubArm : public InitPoseParticlesAxisAngle
{
public:
    InitiCubArm(const std::string& laterality, const std::string& port_prefix) noexcept;

    InitiCubArm(const std::string& laterality) noexcept;

    ~InitiCubArm() noexcept;

protected:
    Eigen::VectorXd readPoseAxisAngle() override;

private:
    const std::string  log_ID_ = "[InitiCubArm]";

    const std::string port_prefix_;

    iCub::iKin::iCubArm                      icub_kin_arm_;

    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;

    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_enc_;

    yarp::sig::Vector                        readTorso();

    yarp::sig::Vector                        readRootToEE();
};

#endif /* INITICUBARM_H */
