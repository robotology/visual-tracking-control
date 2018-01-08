#include <PlayFwdKinModel.h>

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


PlayFwdKinModel::PlayFwdKinModel(const ConstString& robot, const ConstString& laterality, const ConstString& port_prefix) noexcept :
    icub_kin_arm_(iCubArm(laterality+"_v2")),
    robot_(robot), laterality_(laterality), port_prefix_(port_prefix)
{
    port_arm_enc_.open  ("/hand-tracking/" + ID_ + "/" + port_prefix_ + "/" + laterality_ + "_arm:i");
    port_torso_enc_.open("/hand-tracking/" + ID_ + "/" + port_prefix_ + "/torso:i");

    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    yInfo() << log_ID_ << "Succesfully initialized.";
}

PlayFwdKinModel::~PlayFwdKinModel() noexcept
{
    port_torso_enc_.interrupt();
    port_torso_enc_.close();

    port_arm_enc_.interrupt();
    port_arm_enc_.close();
}


bool PlayFwdKinModel::setProperty(const std::string& property)
{
    return FwdKinModel::setProperty(property);
}


VectorXd PlayFwdKinModel::readPose()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    return toEigen(ee_pose);
}


Vector PlayFwdKinModel::readTorso()
{
    Bottle* b = port_torso_enc_.read(true);
    if (!b) return Vector(3, 0.0);

    Vector torso_enc(3);
    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector PlayFwdKinModel::readRootToEE()
{
    Bottle* b = port_arm_enc_.read(true);
    if (!b) return Vector(10, 0.0);

    Vector root_ee_enc(10);
    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i + 3) = b->get(i).asDouble();

    return root_ee_enc;
}
