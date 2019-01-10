#include <InitiCubArm.h>

#include <iCub/ctrl/math.h>
#include <yarp/eigen/Eigen.h>

using namespace Eigen;
using namespace iCub::iKin;
using namespace iCub::ctrl;
using namespace yarp::eigen;
using namespace yarp::math;
using namespace yarp::sig;
using namespace yarp::os;


InitiCubArm::InitiCubArm(const std::string& laterality, const std::string& port_prefix) noexcept :
    port_prefix_(port_prefix),
    icub_kin_arm_(iCubArm(laterality + "_v2"))
{
    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    port_arm_enc_.open  ("/" + port_prefix_ + "/" + laterality + "_arm:i");
    port_torso_enc_.open("/" + port_prefix_ + "/torso:i");
}


InitiCubArm::InitiCubArm(const std::string& laterality) noexcept :
    InitiCubArm("InitiCubArm", laterality) { }


InitiCubArm::~InitiCubArm() noexcept
{
    port_arm_enc_.interrupt();
    port_arm_enc_.close();

    port_torso_enc_.interrupt();
    port_torso_enc_.close();
}


VectorXd InitiCubArm::readPose()
{
    return toEigen(icub_kin_arm_.EndEffPose(readRootToEE() * CTRL_DEG2RAD));
}


Vector InitiCubArm::readTorso()
{
    Bottle* b = port_torso_enc_.read(true);
    if (!b) return Vector(3, 0.0);

    Vector torso_enc(3);
    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector InitiCubArm::readRootToEE()
{
    Bottle* b = port_arm_enc_.read(true);
    if (!b) return Vector(10, 0.0);

    Vector root_ee_enc(10);

    root_ee_enc.setSubvector(0, readTorso());

    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i + 3) = b->get(i).asDouble();

    return root_ee_enc;
}
