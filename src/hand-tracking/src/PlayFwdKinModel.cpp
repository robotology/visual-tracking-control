#include "PlayFwdKinModel.h"

#include <exception>
#include <functional>
#include <iCub/ctrl/math.h>
#include <yarp/math/Math.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/Property.h>
#include <yarp/os/LogStream.h>

using namespace bfl;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


PlayFwdKinModel::PlayFwdKinModel(std::unique_ptr<StateModel> state_model, const ConstString& robot, const ConstString& laterality, const ConstString& port_prefix) noexcept :
    StateModelDecorator(std::move(state_model)),
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


PlayFwdKinModel::PlayFwdKinModel(std::unique_ptr<StateModel> state_model, const ConstString& robot, const ConstString& laterality, const ConstString& port_prefix, bool init_pose) noexcept :
    PlayFwdKinModel(std::move(state_model), robot, laterality, port_prefix)
{
    if (init_pose)
    {
        Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
        Map<VectorXd> cur_ee_pose(ee_pose.data(), 7, 1);
        prev_ee_pose_ = cur_ee_pose;
    }

    yInfo() << log_ID_ << "Succesfully initialized initial pose.";
}


PlayFwdKinModel::PlayFwdKinModel(PlayFwdKinModel&& state_model) noexcept :
    StateModelDecorator(std::move(state_model)) { }


PlayFwdKinModel::~PlayFwdKinModel() noexcept
{
    port_torso_enc_.interrupt();
    port_torso_enc_.close();

    port_arm_enc_.interrupt();
    port_arm_enc_.close();
}


PlayFwdKinModel& PlayFwdKinModel::operator=(PlayFwdKinModel&& state_model) noexcept
{
    StateModelDecorator::operator=(std::move(state_model));

    return *this;
}


void PlayFwdKinModel::propagate(const Ref<const MatrixXf>& cur_state, Ref<MatrixXf> prop_state)
{
    MatrixXf temp_state(cur_state.rows(), cur_state.cols());
    PlayFwdKinPropagate(cur_state, temp_state);

    StateModelDecorator::propagate(temp_state, prop_state);
}


void PlayFwdKinModel::motion(const Ref<const MatrixXf>& cur_state, Ref<MatrixXf> mot_state)
{
    MatrixXf temp_state(cur_state.rows(), cur_state.cols());
    PlayFwdKinPropagate(cur_state, temp_state);

    StateModelDecorator::motion(temp_state, mot_state);
}


MatrixXf PlayFwdKinModel::getNoiseSample(const int num)
{
    return StateModelDecorator::getNoiseSample(num);
}


MatrixXf PlayFwdKinModel::getNoiseCovariance()
{
    return StateModelDecorator::getNoiseCovariance();
}


bool PlayFwdKinModel::setProperty(const std::string& property)
{
    if (!StateModelDecorator::setProperty(property))
    {
        if (property == "ICFW_DELTA")
            return setDeltaMotion();

        if (property == "ICFW_PLAY_INIT")
            return setInitialPose();
    }

    return false;
}


Vector PlayFwdKinModel::readTorso()
{
    Bottle* b = port_torso_enc_.read();
    if (!b) return Vector(3, 0.0);

    Vector torso_enc(3);
    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector PlayFwdKinModel::readRootToEE()
{
    Bottle* b = port_arm_enc_.read();
    if (!b) return Vector(10, 0.0);

    Vector root_ee_enc(10);
    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i + 3) = b->get(i).asDouble();

    return root_ee_enc;
}


bool PlayFwdKinModel::setInitialPose()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> cur_ee_pose(ee_pose.data(), 7, 1);
    prev_ee_pose_ = cur_ee_pose;

    return true;
}


bool PlayFwdKinModel::setDeltaMotion()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> cur_ee_pose(ee_pose.data(), 7, 1);

    delta_hand_pose_.head<3>() = cur_ee_pose.head<3>() - prev_ee_pose_.head<3>();

    delta_hand_pose_.middleRows<3>(3) = cur_ee_pose.middleRows<3>(3) - prev_ee_pose_.middleRows<3>(3);

    delta_angle_ = cur_ee_pose(6) - prev_ee_pose_(6);
    if      (delta_angle_ >   M_PI) delta_angle_ -= 2.0 * M_PI;
    else if (delta_angle_ <= -M_PI) delta_angle_ += 2.0 * M_PI;

    prev_ee_pose_ = cur_ee_pose;

    return true;
}


void PlayFwdKinModel::PlayFwdKinPropagate(const Ref<const MatrixXf>& cur_state, Ref<MatrixXf> prop_state)
{
    prop_state.topRows<3>() = cur_state.topRows<3>().colwise() + delta_hand_pose_.head<3>().cast<float>();

    prop_state.middleRows<3>(3) = (cur_state.middleRows<3>(3).colwise() + delta_hand_pose_.middleRows<3>(3).cast<float>());
    prop_state.middleRows<3>(3).colwise().normalize();

    RowVectorXf ang = cur_state.bottomRows<1>().array() + static_cast<float>(delta_angle_);

    for (unsigned int i = 0; i < ang.cols(); ++i)
    {
        if      (ang(i) >   M_PI) ang(i) -= 2.0 * M_PI;
        else if (ang(i) <= -M_PI) ang(i) += 2.0 * M_PI;
    }

    prop_state.bottomRows<1>() = ang;
}
