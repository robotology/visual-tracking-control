/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef INITPOSEPARTICLESAXISANGLE_H
#define INITPOSEPARTICLESAXISANGLE_H

#include <InitPoseParticles.h>


class InitPoseParticlesAxisAngle : public InitPoseParticles
{
public:
    virtual ~InitPoseParticlesAxisAngle() noexcept;

protected:
    Eigen::VectorXd readPose() override;

    virtual Eigen::VectorXd readPoseAxisAngle() = 0;
};


#endif /* INITPOSEPARTICLESAXISANGLE_H */
