/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <InitWalkmanArm.h>

#include <iCub/ctrl/math.h>
#include <yarp/eigen/Eigen.h>

using namespace Eigen;
using namespace iCub::iKin;
using namespace iCub::ctrl;
using namespace yarp::eigen;
using namespace yarp::math;
using namespace yarp::sig;
using namespace yarp::os;


InitWalkmanArm::InitWalkmanArm(const std::string& laterality, const std::string& port_prefix) noexcept :
    port_prefix_(port_prefix)
{
    port_arm_pose_.open("/" + port_prefix_ + "/" + laterality + "_arm:i");
}


InitWalkmanArm::InitWalkmanArm(const std::string& laterality) noexcept :
    InitWalkmanArm("InitWalkmanArm", laterality) { }


InitWalkmanArm::~InitWalkmanArm() noexcept
{
    port_arm_pose_.interrupt();
    port_arm_pose_.close();
}


VectorXd InitWalkmanArm::readPoseAxisAngle()
{
    return toEigen(readRootToEE());
}


Vector InitWalkmanArm::readRootToEE()
{
    Bottle* b = port_arm_pose_.read(true);
    if (!b) return Vector(7, 0.0);

    Vector root_ee_enc(7);

    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i) = b->get(i).asDouble();

    return root_ee_enc;
}
