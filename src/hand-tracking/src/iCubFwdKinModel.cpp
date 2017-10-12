#include "iCubFwdKinModel.h"

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


iCubFwdKinModel::iCubFwdKinModel(std::unique_ptr<StateModel> state_model, const ConstString& robot, const ConstString& laterality, const ConstString& port_prefix) noexcept :
    StateModelDecorator(std::move(state_model)),
    icub_kin_arm_(iCubArm(laterality+"_v2")),
    robot_(robot), laterality_(laterality), port_prefix_(port_prefix),
    delta_hand_pose_(VectorXd::Zero(7)), delta_angle_(0.0)
{
    Property opt_arm_enc;
    opt_arm_enc.put("device", "remote_controlboard");
    opt_arm_enc.put("local",  "/hand-tracking/" + ID_ + "/" + port_prefix + "/control_" + laterality_ + "_arm");
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
            throw std::runtime_error("ERROR::" + ID_ + "::CTOR::INTERFACE\nERROR: cannot get " + laterality_ + " arm encoder interface!");
        }

        yInfo() << log_ID_ << "Succesfully got " + laterality_ + " arm encoder interface.";
    }
    else
    {
        yError() << log_ID_ << "Cannot open " + laterality_ + " arm remote_controlboard!";

        throw std::runtime_error("ERROR::" + ID_ + "::CTOR::DRIVER\nERROR: cannot open " + laterality_ + " arm remote_controlboard!");
    }

    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    Property opt_torso_enc;
    opt_torso_enc.put("device", "remote_controlboard");
    opt_torso_enc.put("local",  "/hand-tracking/" + ID_ + "/" + port_prefix + "/control_torso");
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
            throw std::runtime_error("ERROR::" + ID_ + "::CTOR::INTERFACE\nERROR: cannot get torso encoder interface!");
        }

        yInfo() << log_ID_ << "Succesfully got torso encoder interface.";
    }
    else
    {
        yError() << log_ID_ << "Cannot open torso remote_controlboard!";

        throw std::runtime_error("ERROR::" + ID_ + "::CTOR::DRIVER\nERROR: cannot open torso remote_controlboard!");
    }

    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    Map<VectorXd> cur_ee_pose(ee_pose.data(), 7, 1);
    prev_ee_pose_ = cur_ee_pose;

    yInfo() << log_ID_ << "Succesfully initialized.";
}


iCubFwdKinModel::~iCubFwdKinModel() noexcept
{
    drv_arm_enc_.close();
    drv_torso_enc_.close();
}


void iCubFwdKinModel::propagate(const Ref<const MatrixXf>& cur_state, Ref<MatrixXf> prop_state)
{
    MatrixXf temp_state(cur_state.rows(), cur_state.cols());
    iCubFwdKinPropagate(cur_state, temp_state);

    StateModelDecorator::propagate(temp_state, prop_state);
}


void iCubFwdKinModel::motion(const Ref<const MatrixXf>& cur_state, Ref<MatrixXf> mot_state)
{
    MatrixXf temp_state(cur_state.rows(), cur_state.cols());
    iCubFwdKinPropagate(cur_state, temp_state);

    StateModelDecorator::motion(temp_state, mot_state);
}


MatrixXf iCubFwdKinModel::getNoiseSample(const int num)
{
    return StateModelDecorator::getNoiseSample(num);
}


MatrixXf iCubFwdKinModel::getNoiseCovariance()
{
    return StateModelDecorator::getNoiseCovariance();
}


bool iCubFwdKinModel::setProperty(const std::string& property)
{
    if (!StateModelDecorator::setProperty(property))
        if (property == "ICFW_DELTA")
            return setDeltaMotion();

    return false;
}


Vector iCubFwdKinModel::readTorso()
{
    int torso_enc_num;
    itf_torso_enc_->getAxes(&torso_enc_num);
    Vector enc_torso(torso_enc_num);

    while (!itf_torso_enc_->getEncoders(enc_torso.data()));

    std::swap(enc_torso(0), enc_torso(2));

    return enc_torso;
}


Vector iCubFwdKinModel::readRootToEE()
{
    int arm_enc_num;
    itf_arm_enc_->getAxes(&arm_enc_num);
    Vector enc_arm(arm_enc_num);

    Vector root_ee_enc(10);

    root_ee_enc.setSubvector(0, readTorso());

    while (!itf_arm_enc_->getEncoders(enc_arm.data()));
    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i + 3) = enc_arm(i);

    return root_ee_enc;
}


bool iCubFwdKinModel::setDeltaMotion()
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


void iCubFwdKinModel::iCubFwdKinPropagate(const Ref<const MatrixXf>& cur_state, Ref<MatrixXf> prop_state)
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
