#include <utils.h>

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
}


Eigen::Vector3d hand_tracking::utils::axis_of_rotation_to_vector(const AxisOfRotation axis)
{
   if (axis == AxisOfRotation::UnitX)
       return Vector3d::UnitX();
   else if (axis == AxisOfRotation::UnitY)
       return Vector3d::UnitY();
   else if (axis == AxisOfRotation::UnitZ)
       return Vector3d::UnitZ();
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
