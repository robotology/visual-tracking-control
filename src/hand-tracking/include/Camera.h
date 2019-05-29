/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef CAMERA_H
#define CAMERA_H

#include <BayesFilters/Agent.h>

#include <array>
#include <tuple>

#include <opencv2/core/core.hpp>

namespace bfl {
    class Camera;
}


class bfl::Camera : public bfl::Agent
{
public:
    virtual ~Camera() noexcept { };

    struct CameraIntrinsics
    {
        unsigned int width;
        unsigned int height;
        double fx;
        double fy;
        double cx;
        double cy;
    };

    using CameraData = std::tuple<cv::Mat, std::array<double, 3>, std::array<double, 4>>;

    virtual CameraIntrinsics getCameraParameters() const = 0;
};

#endif /* CAMERA_H */
