#include "playFwdKinMotion.h"

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


playFwdKinMotion::playFwdKinMotion(std::unique_ptr<StateModel> state_model, const ConstString& robot, const ConstString& laterality, const ConstString& port_prefix) noexcept :
    bfl::StateModelDecorator(std::move(state_model)),
    robot_(robot), laterality_(laterality), port_prefix_(port_prefix),
    icub_kin_arm_(iCubArm(laterality+"_v2")),
    delta_hand_pose_(VectorXd::Zero(6)), delta_angle_(0.0)
{
    /* Arm encoders:   /icub/right_arm/state:o
       Torso encoders: /icub/torso/state:o     */
    port_arm_enc_.open  ("/hand-tracking/playFwdKinMotion/" + port_prefix_ + "/" + laterality_ + "_arm:i");
    port_torso_enc_.open("/hand-tracking/playFwdKinMotion/" + port_prefix_ + "/torso:i");

    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    yInfo() << log_ID_ << "Succesfully initialized.";
}


playFwdKinMotion::playFwdKinMotion(std::unique_ptr<StateModel> state_model, const ConstString& robot, const ConstString& laterality, const ConstString& port_prefix, bool init_pose) noexcept :
    playFwdKinMotion(std::move(state_model), robot, laterality, port_prefix)
{
    if (init_pose)
    {
        Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
        Map<VectorXd> cur_ee_pose(ee_pose.data(), 6, 1);
        cur_ee_pose.tail<3>() *= ee_pose(6);
        prev_ee_pose_ = cur_ee_pose;
    }

    yInfo() << log_ID_ << "Succesfully initialized initial pose.";
}


playFwdKinMotion::~playFwdKinMotion() noexcept
{
    port_torso_enc_.interrupt();
    port_torso_enc_.close();

    port_arm_enc_.interrupt();
    port_arm_enc_.close();
}


playFwdKinMotion::playFwdKinMotion(playFwdKinMotion&& state_model) noexcept :
    bfl::StateModelDecorator(std::move(state_model)) { }


playFwdKinMotion& playFwdKinMotion::operator=(playFwdKinMotion&& state_model) noexcept
{
    bfl::StateModelDecorator::operator=(std::move(state_model));

    return *this;
}


void playFwdKinMotion::propagate(const Ref<const VectorXf>& cur_state, Ref<VectorXf> prop_state)
{
    float ang;
    ang        = cur_state.tail<3>().norm();
    prop_state = cur_state;

    prop_state.head<3>() += delta_hand_pose_.head<3>().cast<float>();

    prop_state.tail<3>() = prop_state.tail<3>().normalized() + delta_hand_pose_.tail<3>().cast<float>();
    prop_state.tail<3>() = prop_state.tail<3>().normalized();

    ang += static_cast<float>(delta_angle_);
    if (ang >   M_PI) ang -= 2.0 * M_PI;
    if (ang <= -M_PI) ang += 2.0 * M_PI;
    prop_state.tail<3>() *= ang;

    bfl::StateModelDecorator::propagate(prop_state, prop_state);
}


void playFwdKinMotion::noiseSample(Ref<VectorXf> sample)
{
    bfl::StateModelDecorator::noiseSample(sample);
}


bool playFwdKinMotion::setProperty(const std::string& property)
{
    if (!bfl::StateModelDecorator::setProperty(property))
    {
        if (property == "ICFW_DELTA")
            return setDeltaMotion();

        if (property == "ICFW_INIT")
            return setInitialPose();
    }

    return false;
}


Vector playFwdKinMotion::readTorso()
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


Vector playFwdKinMotion::readRootToEE()
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


bool playFwdKinMotion::setInitialPose()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> cur_ee_pose(ee_pose.data(), 6, 1);
    cur_ee_pose.tail<3>() *= ee_pose(6);
    prev_ee_pose_ = cur_ee_pose;

    return true;
}


bool playFwdKinMotion::setDeltaMotion()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> cur_ee_pose(ee_pose.data(), 6, 1);
    cur_ee_pose.tail<3>() *= ee_pose(6);

    delta_hand_pose_.head<3>() = cur_ee_pose.head<3>() - prev_ee_pose_.head<3>();
    delta_angle_               = cur_ee_pose.tail<3>().norm() - prev_ee_pose_.tail<3>().norm();
    if (delta_angle_ >   M_PI) delta_angle_ -= 2.0 * M_PI;
    if (delta_angle_ <= -M_PI) delta_angle_ += 2.0 * M_PI;
    delta_hand_pose_.tail<3>() = cur_ee_pose.tail<3>().normalized() - prev_ee_pose_.tail<3>().normalized();

    prev_ee_pose_ = cur_ee_pose;

    return true;
}
