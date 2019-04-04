/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <PlayiCubFwdKinModel.h>

#include <exception>
#include <functional>

#include <iCub/ctrl/math.h>
#include <yarp/math/Math.h>
#include <yarp/eigen/Eigen.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/Property.h>
#include <yarp/os/LogStream.h>

using namespace bfl;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::eigen;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


PlayiCubFwdKinModel::PlayiCubFwdKinModel(const std::string& robot, const std::string& laterality, const std::string& port_prefix) noexcept :
    icub_kin_arm_(iCubArm(laterality+"_v2")),
    port_prefix_(port_prefix),
    robot_(robot),
    laterality_(laterality)
{
    port_arm_enc_.open  ("/" + port_prefix_ + "/" + laterality_ + "_arm:i");
    port_torso_enc_.open("/" + port_prefix_ + "/torso:i");

    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    yInfo() << log_ID_ << "Succesfully initialized.";
}


PlayiCubFwdKinModel::~PlayiCubFwdKinModel() noexcept
{
    port_torso_enc_.interrupt();
    port_torso_enc_.close();

    port_arm_enc_.interrupt();
    port_arm_enc_.close();
}


bool PlayiCubFwdKinModel::setProperty(const std::string& property)
{
    return KinPoseModel::setProperty(property);
}


VectorXd PlayiCubFwdKinModel::readPoseAxisAngle()
{
    return toEigen(icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE()));
}


Vector PlayiCubFwdKinModel::readTorso()
{
    Bottle* b = port_torso_enc_.read(true);
    if (!b) return Vector(3, 0.0);

    Vector torso_enc(3);
    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector PlayiCubFwdKinModel::readRootToEE()
{
    Bottle* b = port_arm_enc_.read(true);
    if (!b) return Vector(10, 0.0);

    Vector root_ee_enc(10);
    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i + 3) = b->get(i).asDouble();

    return root_ee_enc;
}
