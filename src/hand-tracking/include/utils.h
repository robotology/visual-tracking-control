/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef HANDTRACKINGUTILS_H
#define HANDTRACKINGUTILS_H

#include <Eigen/Dense>

namespace hand_tracking
{
namespace utils
{

enum class AxisOfRotation
{
    UnitX,
    UnitY,
    UnitZ
};

std::size_t axis_of_rotation_to_index(const AxisOfRotation axis);

Eigen::Vector3d axis_of_rotation_to_vector(const AxisOfRotation axis);

Eigen::VectorXd axis_angle_to_euler(const Eigen::Ref<const Eigen::VectorXd>& axis, const double angle, const AxisOfRotation axis_1, const AxisOfRotation axis_2, const AxisOfRotation axis_3);

Eigen::VectorXd euler_to_axis_angle(const Eigen::Ref<const Eigen::VectorXd>& euler_angles, const AxisOfRotation axis_1, const AxisOfRotation axis_2, const AxisOfRotation axis_3);
}
}
#endif /* HANDTRACKINGUTILS_H */
