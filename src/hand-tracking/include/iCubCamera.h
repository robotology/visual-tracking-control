#ifndef ICUBHEAD_H
#define ICUBHEAD_H

#include <Camera.h>

#include <string>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class iCubCamera : public bfl::Camera
{
public:
    iCubCamera(const yarp::os::ConstString& cam_sel, const double resolution_ratio, const yarp::os::ConstString& context);

    virtual ~iCubCamera() noexcept { };

    std::tuple<bool, CameraParameters> readCameraParameters() override;

    std::tuple<bool, std::array<double, 3>, std::array<double, 4>> readCameraPose() override;

protected:
    CameraParameters params_;

    bool openGazeController();

    std::tuple<bool, yarp::sig::Vector> readRootToEye();

private:
    std::string log_ID_ = "[iCubCamera]";

    yarp::os::ConstString cam_sel_;
    const double          resolution_ratio_;
    yarp::os::ConstString context_;

    yarp::dev::PolyDriver    drv_gaze_;
    yarp::dev::IGazeControl* itf_gaze_ = YARP_NULLPTR;

    iCub::iKin::iCubEye icub_kin_eye_;
    yarp::os::BufferedPort<yarp::os::Bottle> port_head_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;
};

#endif /* ICUBHEAD_H */
