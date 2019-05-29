/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef KINPOSEMODELAXISANGLE_H
#define KINPOSEMODELAXISANGLE_H

#include <KinPoseModel.h>

#include <Eigen/Dense>


class KinPoseModelAxisAngle : public KinPoseModel
{
public:
    virtual ~KinPoseModelAxisAngle() noexcept;

protected:
    Eigen::VectorXd readPose() override;

    virtual Eigen::VectorXd readPoseAxisAngle() = 0;
};

#endif /* KINPOSEMODELAXISANGLE_H */
