/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef PLAYWALKMANPOSEMODEL_H
#define PLAYWALKMANPOSEMODEL_H

#include <KinPoseModelAxisAngle.h>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Vector.h>

#include <string>


class PlayWalkmanPoseModel : public KinPoseModelAxisAngle
{
public:
    PlayWalkmanPoseModel(const std::string& robot, const std::string& laterality, const std::string& port_prefix) noexcept;

    ~PlayWalkmanPoseModel() noexcept;

    bool setProperty(const std::string& property) override;

protected:
    Eigen::VectorXd readPoseAxisAngle() override;

    yarp::sig::Vector readRootToEE();

    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_pose_;

private:
    const std::string log_ID_ = "[PlayWalkmanPoseModel]";

    const std::string port_prefix_ = "PlayWalkmanPoseModel";

    const std::string robot_;

    const std::string laterality_;
};

#endif /* PLAYWALKMANPOSEMODEL_H */
