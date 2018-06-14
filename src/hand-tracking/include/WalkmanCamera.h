#ifndef WALKMANHEAD_H
#define WALKMANHEAD_H

#include <Camera.h>

#include <string>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class WalkmanCamera : public bfl::Camera
{
public:
    WalkmanCamera(const yarp::os::ConstString& cam_sel,
                  const double resolution_ratio,
                  const yarp::os::ConstString& context,
                  const yarp::os::ConstString& port_prefix);

    virtual ~WalkmanCamera() noexcept;

    std::tuple<bool, CameraParameters> getCameraParameters() override;

    std::tuple<bool, std::array<double, 3>, std::array<double, 4>> getCameraPose() override;

protected:
    CameraParameters params_;

private:
    const yarp::os::ConstString log_ID_ = "[WalkmanCamera]";
    yarp::os::ConstString port_prefix_ = "WalkmanCamera";

    yarp::os::ConstString cam_sel_;
    const double          resolution_ratio_;
    yarp::os::ConstString context_;

    yarp::os::BufferedPort<yarp::os::Bottle> port_camera_pose_;
};

#endif /* WALKMANHEAD_H */
