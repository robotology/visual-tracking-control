#include "PlayFwdKinMotion.h"

#include <exception>
#include <functional>
#include <iCub/ctrl/math.h>
#include <yarp/math/Math.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/Property.h>
#include <yarp/os/LogStream.h>

using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


PlayFwdKinMotion::PlayFwdKinMotion(std::unique_ptr<StateModel> state_model, const ConstString& robot, const ConstString& laterality, const ConstString& port_prefix) noexcept :
    bfl::StateModelDecorator(std::move(state_model)),
    icub_kin_arm_(iCubArm(laterality+"_v2")),
    robot_(robot), laterality_(laterality), port_prefix_(port_prefix),
    delta_hand_pose_(VectorXd::Zero(7)), delta_angle_(0.0)
{
    port_arm_enc_.open  ("/hand-tracking/" + ID_ + "/" + port_prefix_ + "/" + laterality_ + "_arm:i");
    port_torso_enc_.open("/hand-tracking/" + ID_ + "/" + port_prefix_ + "/torso:i");

    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    yInfo() << log_ID_ << "Succesfully initialized.";
}


PlayFwdKinMotion::PlayFwdKinMotion(std::unique_ptr<StateModel> state_model, const ConstString& robot, const ConstString& laterality, const ConstString& port_prefix, bool init_pose) noexcept :
    PlayFwdKinMotion(std::move(state_model), robot, laterality, port_prefix)
{
    if (init_pose)
    {
        Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
        Map<VectorXd> cur_ee_pose(ee_pose.data(), 7, 1);
        prev_ee_pose_ = cur_ee_pose;
    }

    yInfo() << log_ID_ << "Succesfully initialized initial pose.";
}


PlayFwdKinMotion::~PlayFwdKinMotion() noexcept
{
    port_torso_enc_.interrupt();
    port_torso_enc_.close();

    port_arm_enc_.interrupt();
    port_arm_enc_.close();
}


PlayFwdKinMotion::PlayFwdKinMotion(PlayFwdKinMotion&& state_model) noexcept :
    bfl::StateModelDecorator(std::move(state_model)) { }


PlayFwdKinMotion& PlayFwdKinMotion::operator=(PlayFwdKinMotion&& state_model) noexcept
{
    bfl::StateModelDecorator::operator=(std::move(state_model));

    return *this;
}


void PlayFwdKinMotion::propagate(const Ref<const VectorXf>& cur_state, Ref<VectorXf> prop_state)
{
    prop_state.head<3>() = cur_state.head<3>() + delta_hand_pose_.head<3>().cast<float>();

    prop_state.middleRows<3>(3) = (cur_state.middleRows<3>(3) + delta_hand_pose_.middleRows<3>(3).cast<float>()).normalized();

    float   ang = cur_state(6) + static_cast<float>(delta_angle_);
    if      (ang >  2.0 * M_PI) ang -= 2.0 * M_PI;
    else if (ang <=        0.0) ang += 2.0 * M_PI;

    prop_state(6) = ang;

    bfl::StateModelDecorator::propagate(prop_state, prop_state);
}


void PlayFwdKinMotion::noiseSample(Ref<VectorXf> sample)
{
    bfl::StateModelDecorator::noiseSample(sample);
}


bool PlayFwdKinMotion::setProperty(const std::string& property)
{
    if (!bfl::StateModelDecorator::setProperty(property))
    {
        if (property == "ICFW_DELTA")
            return setDeltaMotion();

        if (property == "ICFW_PLAY_INIT")
            return setInitialPose();
    }

    return false;
}


Vector PlayFwdKinMotion::readTorso()
{
    Bottle* b = port_torso_enc_.read();
    if (!b) return Vector(3, 0.0);

    yAssert(b->size() == 3);

    Vector torso_enc(3);
    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector PlayFwdKinMotion::readRootToEE()
{
    Bottle* b = port_arm_enc_.read();
    if (!b) return Vector(10, 0.0);

    yAssert(b->size() == 16);

    Vector root_ee_enc(10);
    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
    {
        root_ee_enc(i+3) = b->get(i).asDouble();
    }

    return root_ee_enc;
}


bool PlayFwdKinMotion::setInitialPose()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> cur_ee_pose(ee_pose.data(), 7, 1);
    prev_ee_pose_ = cur_ee_pose;

    return true;
}


bool PlayFwdKinMotion::setDeltaMotion()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> cur_ee_pose(ee_pose.data(), 7, 1);

    delta_hand_pose_.head<3>() = cur_ee_pose.head<3>() - prev_ee_pose_.head<3>();

    delta_hand_pose_.middleRows<3>(3) = (cur_ee_pose.middleRows<3>(3) - prev_ee_pose_.middleRows<3>(3)).normalized();

    delta_angle_ = cur_ee_pose(6) - prev_ee_pose_(6);
    if      (delta_angle_ >  2.0 * M_PI) delta_angle_ -= 2.0 * M_PI;
    else if (delta_angle_ <=        0.0) delta_angle_ += 2.0 * M_PI;

    prev_ee_pose_ = cur_ee_pose;

    return true;
}
