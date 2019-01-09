#include <KinPoseModel.h>

#include <exception>
#include <iostream>
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


KinPoseModel::KinPoseModel() noexcept { }


KinPoseModel::~KinPoseModel() noexcept { }


void KinPoseModel::propagate(const Ref<const MatrixXd>& cur_state, Ref<MatrixXd> prop_state)
{
    prop_state.topRows<3>() = cur_state.topRows<3>().colwise() + delta_hand_pose_.head<3>().cast<float>();

    prop_state.middleRows<3>(3) = (cur_state.middleRows<3>(3).colwise() + delta_hand_pose_.middleRows<3>(3).cast<float>());
    prop_state.middleRows<3>(3).colwise().normalize();

    RowVectorXd ang = cur_state.bottomRows<1>().array() + static_cast<float>(delta_angle_);

    for (unsigned int i = 0; i < ang.cols(); ++i)
    {
        if      (ang(i) >   M_PI) ang(i) -= 2.0 * M_PI;
        else if (ang(i) <= -M_PI) ang(i) += 2.0 * M_PI;
    }

    prop_state.bottomRows<1>() = ang;
}


MatrixXd KinPoseModel::getExogenousMatrix()
{
    std::cerr << "ERROR::PFPREDICTION::SETEXOGENOUSMODEL\n";
    std::cerr << "ERROR:\n\tCall to unimplemented base class method." << std::endl;

    return MatrixXd::Zero(1, 1);
}


bool KinPoseModel::setProperty(const std::string& property)
{
    if (property == "kin_pose_delta")
        return setDeltaMotion();

    if (property == "init")
    {
        initialize_delta_ = true;
        return setDeltaMotion();
    }

    return false;
}


bool KinPoseModel::setDeltaMotion()
{
    VectorXd ee_pose = readPose();

    if (!initialize_delta_)
    {
        delta_hand_pose_.head<3>() = ee_pose.head<3>() - prev_ee_pose_.head<3>();

        delta_hand_pose_.middleRows<3>(3) = ee_pose.middleRows<3>(3) - prev_ee_pose_.middleRows<3>(3);

        delta_angle_ = ee_pose(6) - prev_ee_pose_(6);
        if      (delta_angle_ >   M_PI) delta_angle_ -= 2.0 * M_PI;
        else if (delta_angle_ <= -M_PI) delta_angle_ += 2.0 * M_PI;
    }
    else
    {
        delta_hand_pose_  = VectorXd::Zero(6);
        delta_angle_      = 0.0;
        initialize_delta_ = false;
    }

    prev_ee_pose_ = ee_pose;

    return true;
}
