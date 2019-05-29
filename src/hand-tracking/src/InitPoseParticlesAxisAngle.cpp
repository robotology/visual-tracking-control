/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <InitPoseParticlesAxisAngle.h>
#include <utils.h>

using namespace Eigen;
using namespace hand_tracking::utils;

InitPoseParticlesAxisAngle::~InitPoseParticlesAxisAngle()
{ }

Eigen::VectorXd InitPoseParticlesAxisAngle::readPose()
{
    VectorXd pose_axis_angle = readPoseAxisAngle();

    VectorXd pose(6);
    pose.head<3>() = pose_axis_angle.head<3>();
    pose.tail<3>() = axis_angle_to_euler(pose_axis_angle.segment<3>(3), pose_axis_angle(6), AxisOfRotation::UnitZ, AxisOfRotation::UnitY, AxisOfRotation::UnitX);

    return pose;
}
