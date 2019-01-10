#include <PlayWalkmanPoseModel.h>

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


PlayWalkmanPoseModel::PlayWalkmanPoseModel(const std::string& robot, const std::string& laterality, const std::string& port_prefix) noexcept :
    port_prefix_(port_prefix),
    robot_(robot),
    laterality_(laterality)
{
    port_arm_pose_.open  ("/" + port_prefix_ + "/" + laterality_ + "_arm:i");

    yInfo() << log_ID_ << "Succesfully initialized.";
}


PlayWalkmanPoseModel::~PlayWalkmanPoseModel() noexcept
{
    port_arm_pose_.interrupt();
    port_arm_pose_.close();
}


bool PlayWalkmanPoseModel::setProperty(const std::string& property)
{
    return KinPoseModel::setProperty(property);
}


VectorXd PlayWalkmanPoseModel::readPose()
{
    return toEigen(readRootToEE());
}


Vector PlayWalkmanPoseModel::readRootToEE()
{
    Bottle* b = port_arm_pose_.read(true);
    if (!b) return Vector(7, 0.0);

    Vector root_ee_enc(7);
    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i) = b->get(i).asDouble();

    return root_ee_enc;
}
