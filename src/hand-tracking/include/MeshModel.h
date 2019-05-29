/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef MESHMODEL_H
#define MESHMODEL_H

#include <tuple>

#include <Eigen/Dense>

#include <SuperimposeMesh/SICAD.h>

namespace bfl {
    class MeshModel;
}


class bfl::MeshModel
{
public:
    virtual ~MeshModel() noexcept { };

    virtual std::tuple<bool, SICAD::ModelPathContainer> getMeshPaths() = 0;

    virtual std::tuple<bool, std::string> getShaderPaths() = 0;

    virtual std::tuple<bool, std::vector<Superimpose::ModelPoseContainer>> getModelPose(const Eigen::Ref<const Eigen::MatrixXd>& cur_states) = 0;
};

#endif /* MESHMODEL_H */
