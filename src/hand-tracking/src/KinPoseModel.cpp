/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

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
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


KinPoseModel::KinPoseModel() noexcept
{ }


KinPoseModel::~KinPoseModel() noexcept
{ }


void KinPoseModel::propagate(const Ref<const MatrixXd>& cur_state, Ref<MatrixXd> prop_state)
{
    prop_state.topRows<3>() = cur_state.topRows<3>().colwise() + delta_hand_pos_;
    prop_state.bottomRows<3>() = perturbOrientation(cur_state.bottomRows<3>(), delta_hand_rot_);
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


std::pair<std::size_t, std::size_t> KinPoseModel::getOutputSize() const
{
    return std::make_pair(3, 3);
}


bool KinPoseModel::setDeltaMotion()
{
    VectorXd ee_pose = readPose();

    if (!initialize_delta_)
    {
        delta_hand_pos_ = ee_pose.head<3>() - prev_ee_pose_.head<3>();

        delta_hand_rot_.noalias() = relativeOrientation(prev_ee_pose_, ee_pose);
    }
    else
    {
        delta_hand_pos_  = VectorXd::Zero(3);
        delta_hand_rot_   = Matrix3d::Identity();
        initialize_delta_ = false;
    }

    prev_ee_pose_ = ee_pose;

    return true;
}


Matrix3d KinPoseModel::relativeOrientation(const Ref<const VectorXd>& prev_pose, const Ref<VectorXd>& curr_pose)
{
    /*
     * Evaluate rotation matrix due to previous pose.
     */
    Matrix3d prev_rot = (AngleAxisd(prev_pose(3), Vector3d::UnitZ())
                       * AngleAxisd(prev_pose(4), Vector3d::UnitY())
                       * AngleAxisd(prev_pose(5), Vector3d::UnitX())).toRotationMatrix();

    // Evaluate rotation matrix due to previous pose
    Matrix3d curr_rot = (AngleAxisd(curr_pose(3), Vector3d::UnitZ())
                       * AngleAxisd(curr_pose(4), Vector3d::UnitY())
                       * AngleAxisd(curr_pose(5), Vector3d::UnitX())).toRotationMatrix();

    // Evaluate relative rotation R s.t. current_rot = previous_rot * R;
    return prev_rot.transpose() * curr_rot;
}


MatrixXd KinPoseModel::perturbOrientation(const Ref<const MatrixXd>& state, const Ref<const MatrixXd>& perturbation)
{
    MatrixXd perturbed_orientations(3, state.cols());

    /*
     * Evaluate rotation matrix due to current pose.
     */
    for (int i = 0; i < state.cols(); i++)
    {
        Matrix3d state_rot = (AngleAxisd(state(0, i), Vector3d::UnitZ())
                            * AngleAxisd(state(1, i), Vector3d::UnitY())
                            * AngleAxisd(state(2, i), Vector3d::UnitX())).toRotationMatrix();

        /*
         * Perturb rotation using the evaluated relative rotation.
         */
        Matrix3d perturbed_rot = state_rot * perturbation;

        /*
         * Extract ZYX Euler angles.
         */
        perturbed_orientations.col(i) = perturbed_rot.eulerAngles(2, 1, 0);
    }

    return perturbed_orientations;
}
