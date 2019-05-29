/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <utils.h>

#include <exception>

using namespace Eigen;
using namespace hand_tracking::utils;


std::size_t hand_tracking::utils::axis_of_rotation_to_index(const AxisOfRotation axis)
{
   if (axis == AxisOfRotation::UnitX)
       return 0;
   else if (axis == AxisOfRotation::UnitY)
       return 1;
   else if (axis == AxisOfRotation::UnitZ)
       return 2;

   throw std::runtime_error("ERROR::UTILS::AXIS_OF_ROTATION_TO_INDEX\nERROR: Wrong input value of enum class type hand_tracking::utils::AxisOfRotation.");
}


Eigen::Vector3d hand_tracking::utils::axis_of_rotation_to_vector(const AxisOfRotation axis)
{
   if (axis == AxisOfRotation::UnitX)
       return Vector3d::UnitX();
   else if (axis == AxisOfRotation::UnitY)
       return Vector3d::UnitY();
   else if (axis == AxisOfRotation::UnitZ)
       return Vector3d::UnitZ();

   throw std::runtime_error("ERROR::UTILS::AXIS_OF_ROTATION_TO_VECTOR\nERROR: Wrong input value of enum class type hand_tracking::utils::AxisOfRotation.");
}


VectorXd hand_tracking::utils::axis_angle_to_euler
(
    const Ref<const VectorXd>& axis,
    const double angle,
    const AxisOfRotation axis_1,
    const AxisOfRotation axis_2,
    const AxisOfRotation axis_3
)
{
    AngleAxisd angle_axis(angle, axis);
    return angle_axis.toRotationMatrix().eulerAngles(axis_of_rotation_to_index(axis_1),
                                                     axis_of_rotation_to_index(axis_2),
                                                     axis_of_rotation_to_index(axis_3));
}


VectorXd hand_tracking::utils::euler_to_axis_angle
(
    const Ref<const VectorXd>& euler_angles,
    const AxisOfRotation axis_1,
    const AxisOfRotation axis_2,
    const AxisOfRotation axis_3
)
{
    AngleAxisd angle_axis(AngleAxisd(euler_angles(0), axis_of_rotation_to_vector(axis_1)) *
                          AngleAxisd(euler_angles(1), axis_of_rotation_to_vector(axis_2)) *
                          AngleAxisd(euler_angles(2), axis_of_rotation_to_vector(axis_3)));

    VectorXd axis_angle(4);
    axis_angle.head<3>() = angle_axis.axis();
    axis_angle(3) = angle_axis.angle();

    return axis_angle;
}
