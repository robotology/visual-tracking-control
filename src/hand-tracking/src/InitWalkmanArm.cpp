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


InitWalkmanArm::InitWalkmanArm(const ConstString& port_prefix, const ConstString& cam_sel, const ConstString& laterality) noexcept
    port_prefix_(port_prefix)
{
    port_arm_pose_.open("/" + port_prefix_ + "/" + laterality + "_arm:i");
}


InitWalkmanArm::InitWalkmanArm(const ConstString& cam_sel, const ConstString& laterality) noexcept :
    InitWalkmanArm("InitWalkmanArm", cam_sel, laterality) { }


InitWalkmanArm::~InitWalkmanArm() noexcept
{
    port_arm_pose_.interrupt();
    port_arm_pose_.close();
}


VectorXd InitWalkmanArm::readPose()
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
