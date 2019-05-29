/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef INITWALKMANBARM_H
#define INITWALKMANBARM_H

#include <InitPoseParticlesAxisAngle.h>

#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Vector.h>

#include <string>


class InitWalkmanArm : public InitPoseParticlesAxisAngle
{
public:
    InitWalkmanArm(const std::string& laterality, const std::string& port_prefix) noexcept;

    InitWalkmanArm(const std::string& laterality) noexcept;

    ~InitWalkmanArm() noexcept;

protected:
    Eigen::VectorXd readPoseAxisAngle() override;

private:
    const std::string  log_ID_ = "[InitWalkmanArm]";

    const std::string port_prefix_;

    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_pose_;

    yarp::sig::Vector readRootToEE();
};

#endif /* INITWALKMANBARM_H */
