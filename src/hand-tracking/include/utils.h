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
}
}
#endif /* HANDTRACKINGUTILS_H */
