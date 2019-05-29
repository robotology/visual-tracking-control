/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <InitPoseParticles.h>

#include <iCub/ctrl/math.h>

using namespace bfl;
using namespace Eigen;


bool InitPoseParticles::initialize(ParticleSet& particles)
{
    VectorXd pose = readPose();

    for (int i = 0; i < particles.state().cols(); ++i)
        particles.state(i) << pose;

    particles.weight().fill(-std::log(particles.state().cols()));

    return true;
}
