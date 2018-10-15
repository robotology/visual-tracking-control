#ifndef WALKMANHEAD_H
#define WALKMANHEAD_H

#include <Camera.h>
#include <CameraData.h>

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

    CameraParameters getCameraParameters() const override;

protected:
    CameraParameters params_;

    std::shared_ptr<bfl::CameraData> camera_data_ = nullptr;

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
