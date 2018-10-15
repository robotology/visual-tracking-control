#ifndef ICUBHEAD_H
#define ICUBHEAD_H

#include <Camera.h>
#include <CameraData.h>

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
    iCubCamera(const yarp::os::ConstString& cam_sel,
               const double resolution_ratio,
               const yarp::os::ConstString& context,
               const yarp::os::ConstString& port_prefix);

    virtual ~iCubCamera() noexcept;

    bool bufferData() override;

    bfl::Data getData() const override;

    CameraParameters getCameraParameters() const override;

protected:
    CameraParameters params_;

    std::shared_ptr<bfl::CameraData> camera_data_ = nullptr;

    bool openGazeController();

    std::tuple<bool, yarp::sig::Vector> readRootToEye();

private:
    const yarp::os::ConstString log_ID_ = "[iCubCamera]";

    yarp::os::ConstString port_prefix_ = "iCubCamera";

    yarp::os::ConstString cam_sel_;

    const double resolution_ratio_;

    yarp::os::ConstString context_;

    yarp::dev::PolyDriver drv_gaze_;

    yarp::dev::IGazeControl* itf_gaze_ = YARP_NULLPTR;

    iCub::iKin::iCubEye icub_kin_eye_;

    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_in_;

    bool init_img_in_ = false;

    yarp::os::BufferedPort<yarp::os::Bottle> port_head_enc_;

    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;
};

#endif /* ICUBHEAD_H */
