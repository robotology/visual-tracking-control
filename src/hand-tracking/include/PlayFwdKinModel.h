/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef PLAYFWDKINMODEL_H
#define PLAYFWDKINMODEL_H

#include <FwdKinModel.h>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class PlayFwdKinModel : public FwdKinModel
{
public:
    PlayFwdKinModel(const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    ~PlayFwdKinModel() noexcept;

    bool setProperty(const std::string& property) override;

protected:
    Eigen::VectorXd   readPose() override;

    yarp::sig::Vector readTorso();

    yarp::sig::Vector readRootToEE();

    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_enc_;
    iCub::iKin::iCubArm                      icub_kin_arm_;

private:
    yarp::os::ConstString ID_     = "PlayFwdKinModel";
    yarp::os::ConstString log_ID_ = "[" + ID_ + "]";

    yarp::os::ConstString robot_;
    yarp::os::ConstString laterality_;
    yarp::os::ConstString port_prefix_;
};

#endif /* PLAYFWDKINMODEL_H */
