/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef WALKMANHEAD_H
#define WALKMANHEAD_H

#include <Camera.h>
#include <BayesFilters/Data.h>

#include <string>

#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Image.h>


class WalkmanCamera : public bfl::Camera
{
public:
    WalkmanCamera(const std::string& cam_sel,
                  const double resolution_ratio,
                  const std::string& context,
                  const std::string& port_prefix);

    virtual ~WalkmanCamera() noexcept;

    bool bufferData() override;

    bfl::Data getData() const override;

    CameraIntrinsics getCameraParameters() const override;
    
protected:
    CameraIntrinsics params_;

    cv::Mat image_;

    std::array<double, 3> position_;

    std::array<double, 4> orientation_;

private:
    const std::string log_ID_ = "[WalkmanCamera]";

    std::string port_prefix_ = "WalkmanCamera";

    std::string cam_sel_;

    const double resolution_ratio_;

    std::string context_;

    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_in_;

    bool init_img_in_ = false;

    yarp::os::BufferedPort<yarp::os::Bottle> port_camera_pose_;
};

#endif /* WALKMANHEAD_H */
