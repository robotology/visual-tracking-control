/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef ICUBHEAD_H
#define ICUBHEAD_H

#include <Camera.h>

#include <string>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Vector.h>


class iCubCamera : public bfl::Camera
{
public:
    iCubCamera(const std::string& cam_sel,
               const double resolution_ratio,
               const std::string& context,
               const std::string& port_prefix);

    virtual ~iCubCamera() noexcept;

    bool bufferData() override;

    bfl::Data getData() const override;

    CameraIntrinsics getCameraParameters() const override;

protected:
    CameraIntrinsics params_;

    cv::Mat image_;

    std::array<double, 3> position_;

    std::array<double, 4> orientation_;

    bool openGazeController();

    std::tuple<bool, yarp::sig::Vector> readRootToEye();

private:
    const std::string log_ID_ = "[iCubCamera]";

    const std::string port_prefix_ = "iCubCamera";

    const std::string cam_sel_;

    const double resolution_ratio_;

    const std::string context_;

    yarp::dev::PolyDriver drv_gaze_;

    yarp::dev::IGazeControl* itf_gaze_ = YARP_NULLPTR;

    iCub::iKin::iCubEye icub_kin_eye_;

    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_in_;

    bool init_img_in_ = false;

    yarp::os::BufferedPort<yarp::os::Bottle> port_head_enc_;

    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;
};

#endif /* ICUBHEAD_H */
