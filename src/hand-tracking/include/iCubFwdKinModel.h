/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef ICUBFWDKINMODEL_H
#define ICUBFWDKINMODEL_H

#include <KinPoseModelAxisAngle.h>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/sig/Vector.h>

#include <string>


class iCubFwdKinModel : public KinPoseModelAxisAngle
{
public:
    iCubFwdKinModel(const std::string& robot, const std::string& laterality, const std::string& port_prefix);

    ~iCubFwdKinModel() noexcept;

    bool setProperty(const std::string& property) override;

protected:
    Eigen::VectorXd   readPoseAxisAngle() override;

    yarp::sig::Vector readTorso();

    yarp::sig::Vector readRootToEE();

    yarp::dev::PolyDriver  drv_arm_enc_;

    yarp::dev::PolyDriver  drv_torso_enc_;

    yarp::dev::IEncoders * itf_arm_enc_;

    yarp::dev::IEncoders * itf_torso_enc_;

    iCub::iKin::iCubArm    icub_kin_arm_;

private:
    const std::string  log_ID_ = "[iCubFwdKinModel]";

    const std::string  port_prefix_ = "iCubFwdKinModel";

    const std::string  robot_;

    const std::string  laterality_;
};

#endif /* ICUBFWDKINMODEL_H */
