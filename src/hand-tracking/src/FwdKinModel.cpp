/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <FwdKinModel.h>

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


FwdKinModel::FwdKinModel() noexcept { }


FwdKinModel::~FwdKinModel() noexcept { }


void FwdKinModel::propagate(const Ref<const MatrixXf>& cur_state, Ref<MatrixXf> prop_state)
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


MatrixXf FwdKinModel::getExogenousMatrix()
{
    std::cerr << "ERROR::PFPREDICTION::SETEXOGENOUSMODEL" << std::endl;
    std::cerr << "ERROR:\n\tCall to unimplemented base class method.";

    return MatrixXf::Zero(1, 1);
}


bool FwdKinModel::setProperty(const std::string& property)
{
    if (property == "ICFW_DELTA")
        return setDeltaMotion();

    if (property == "ICFW_INIT")
    {
        initialize_delta_ = true;
        return setDeltaMotion();
    }

    return false;
}


bool FwdKinModel::setDeltaMotion()
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
