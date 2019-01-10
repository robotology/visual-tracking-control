#ifndef INITWALKMANBARM_H
#define INITWALKMANBARM_H

#include <InitPoseParticles.h>

#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Vector.h>

#include <string>


class InitWalkmanArm : public InitPoseParticles
{
public:
    InitWalkmanArm(const std::string& laterality, const std::string& port_prefix) noexcept;

    InitWalkmanArm(const std::string& laterality) noexcept;

    ~InitWalkmanArm() noexcept;

protected:
    Eigen::VectorXd readPose() override;

private:
    const std::string  log_ID_ = "[InitWalkmanArm]";

    const std::string port_prefix_;

    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_pose_;

    yarp::sig::Vector readRootToEE();
};

#endif /* INITWALKMANBARM_H */
