#include "iCubFwdKinMotion.h"

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


iCubFwdKinMotion::iCubFwdKinMotion(std::unique_ptr<StateModel> state_model, const ConstString& robot, const ConstString& laterality, const ConstString& port_prefix) noexcept :
    bfl::StateModelDecorator(std::move(state_model)),
    robot_(robot), laterality_(laterality), port_prefix_(port_prefix),
    icub_kin_arm_(iCubArm(laterality+"_v2")),
    delta_hand_pose_(VectorXd::Zero(6)), delta_angle_(0.0)
{
    Property opt_arm_enc;
    opt_arm_enc.put("device", "remote_controlboard");
    opt_arm_enc.put("local",  "/hand-tracking/iCubFwdKinMotion/" + port_prefix + "/control_" + laterality_ + "_arm");
    opt_arm_enc.put("remote", "/" + robot_ + "/right_arm");

    yInfo() << log_ID_ << "Opening " + laterality_ + " arm remote_controlboard driver...";
    if (drv_arm_enc_.open(opt_arm_enc))
    {
        yInfo() << log_ID_ << "Succesfully opened " + laterality_ + " arm remote_controlboard interface.";

        yInfo() << log_ID_ << "Getting " + laterality_ + " arm encoder interface...";
        drv_arm_enc_.view(itf_arm_enc_);
        if (!itf_arm_enc_)
        {
            yError() << log_ID_ << "Cannot get " + laterality_ + " arm encoder interface!";
            drv_arm_enc_.close();
            throw std::runtime_error("ERROR::iCubFwdKinMotion::CTOR::INTERFACE\nERROR: cannot get " + laterality_ + " arm encoder interface!");
        }
        yInfo() << log_ID_ << "Succesfully got " + laterality_ + " arm encoder interface.";
    }
    else
    {
        yError() << log_ID_ << "Cannot open " + laterality_ + " arm remote_controlboard!";
        throw std::runtime_error("ERROR::iCubFwdKinMotion::CTOR::DRIVER\nERROR: cannot open " + laterality_ + " arm remote_controlboard!");
    }

    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    Property opt_torso_enc;
    opt_torso_enc.put("device", "remote_controlboard");
    opt_torso_enc.put("local",  "/hand-tracking/iCubFwdKinMotion/" + port_prefix + "/control_torso");
    opt_torso_enc.put("remote", "/" + robot_ + "/torso");

    yInfo() << log_ID_ << "Opening torso remote_controlboard driver...";
    if (drv_torso_enc_.open(opt_torso_enc))
    {
        yInfo() << log_ID_ << "Succesfully opened torso remote_controlboard driver.";

        yInfo() << log_ID_ << "Getting torso encoder interface...";
        drv_torso_enc_.view(itf_torso_enc_);
        if (!itf_torso_enc_)
        {
            yError() << log_ID_ << "Cannot get torso encoder interface!";
            drv_torso_enc_.close();
            throw std::runtime_error("ERROR::iCubFwdKinMotion::CTOR::INTERFACE\nERROR: cannot get torso encoder interface!");
        }
        yInfo() << log_ID_ << "Succesfully got torso encoder interface.";
    }
    else
    {
        yError() << log_ID_ << "Cannot open torso remote_controlboard!";
        throw std::runtime_error("ERROR::iCubFwdKinMotion::CTOR::DRIVER\nERROR: cannot open torso remote_controlboard!");
    }

    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> cur_ee_pose(ee_pose.data(), 6, 1);
    cur_ee_pose.tail<3>() *= ee_pose(6);
    prev_ee_pose_ = cur_ee_pose;

    yInfo() << log_ID_ << "Succesfully initialized.";
}


iCubFwdKinMotion::~iCubFwdKinMotion() noexcept
{
    drv_arm_enc_.close();
    drv_torso_enc_.close();
}


iCubFwdKinMotion::iCubFwdKinMotion(iCubFwdKinMotion&& state_model) noexcept :
    bfl::StateModelDecorator(std::move(state_model)) { }


iCubFwdKinMotion& iCubFwdKinMotion::operator=(iCubFwdKinMotion&& state_model) noexcept
{
    bfl::StateModelDecorator::operator=(std::move(state_model));

    return *this;
}


void iCubFwdKinMotion::propagate(const Ref<const VectorXf>& cur_state, Ref<VectorXf> prop_state)
{
    prop_state = cur_state;

    prop_state.head<3>() += delta_hand_pose_.head<3>().cast<float>();

    float ang;
    ang = cur_state.tail<3>().norm();
    prop_state.tail<3>() /= ang;

    prop_state.tail<3>() += delta_hand_pose_.tail<3>().cast<float>();
    prop_state.tail<3>() /= prop_state.tail<3>().norm();

    ang += static_cast<float>(delta_angle_);
    if (ang >  M_PI) ang -= 2.0 * M_PI;
    if (ang < -M_PI) ang += 2.0 * M_PI;
    prop_state.tail<3>() *= ang;

    bfl::StateModelDecorator::propagate(prop_state, prop_state);
}


void iCubFwdKinMotion::noiseSample(Ref<VectorXf> sample)
{
    bfl::StateModelDecorator::noiseSample(sample);
}


bool iCubFwdKinMotion::setProperty(const std::string& property)
{
    if (!bfl::StateModelDecorator::setProperty(property))
        if (property == "ICFW_DELTA")
            return setDeltaMotion();

    return false;
}


Vector iCubFwdKinMotion::readTorso()
{
    int torso_enc_num;
    itf_arm_enc_->getAxes(&torso_enc_num);
    Vector enc_torso(torso_enc_num);

    while (!itf_torso_enc_->getEncoders(enc_torso.data()));

    std::swap(enc_torso(0), enc_torso(2));

    return enc_torso;
}


Vector iCubFwdKinMotion::readRootToEE()
{
    int arm_enc_num;
    itf_arm_enc_->getAxes(&arm_enc_num);
    Vector enc_arm(arm_enc_num);

    Vector root_ee_enc(10);

    root_ee_enc.setSubvector(0, readTorso());

    while (!itf_arm_enc_->getEncoders(enc_arm.data()));
    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i+3) = enc_arm(i);

    return root_ee_enc;
}


bool iCubFwdKinMotion::setDeltaMotion()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> cur_ee_pose(ee_pose.data(), 6, 1);
    cur_ee_pose.tail<3>() *= ee_pose(6);

    delta_hand_pose_.head<3>() = cur_ee_pose.head<3>() - prev_ee_pose_.head<3>();
    delta_angle_               = cur_ee_pose.tail<3>().norm() - prev_ee_pose_.tail<3>().norm();
    delta_hand_pose_.tail<3>() = (cur_ee_pose.tail<3>() / cur_ee_pose.tail<3>().norm()) - (prev_ee_pose_.tail<3>() / prev_ee_pose_.tail<3>().norm());
    if (delta_angle_ >  M_PI) delta_angle_ -= 2.0 * M_PI;
    if (delta_angle_ < -M_PI) delta_angle_ += 2.0 * M_PI;

    prev_ee_pose_ = cur_ee_pose;

    return true;
}
