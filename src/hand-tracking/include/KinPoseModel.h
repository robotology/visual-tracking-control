/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef FWDPOSEMODEL_H
#define FWDPOSEMODEL_H

#include <BayesFilters/ExogenousModel.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/sig/Vector.h>


class KinPoseModel : public bfl::ExogenousModel
{
public:
    KinPoseModel() noexcept;

    ~KinPoseModel() noexcept;

    void propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> prop_state) override;

    Eigen::MatrixXd getExogenousMatrix() override;

    bool setProperty(const std::string& property) override;

    std::pair<std::size_t, std::size_t> getOutputSize() const override;

protected:
    virtual Eigen::VectorXd readPose() = 0;

    bool initialize_delta_ = true;

    bool setDeltaMotion();

    Eigen::Matrix3d relativeOrientation(const Eigen::Ref<const Eigen::VectorXd>& prev_pose, const Eigen::Ref<Eigen::VectorXd>& curr_pose);

    Eigen::MatrixXd perturbOrientation(const Eigen::Ref<const Eigen::MatrixXd>& state, const Eigen::Ref<const Eigen::MatrixXd>& perturbation);

private:
    Eigen::VectorXd prev_ee_pose_   = Eigen::VectorXd::Zero(6);

    Eigen::Vector3d delta_hand_pos_ = Eigen::Vector3d::Zero();

    Eigen::Matrix3d delta_hand_rot_ = Eigen::Matrix3d::Zero();
};

#endif /* FWDPOSEMODEL_H */
