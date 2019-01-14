#include <WalkmanCamera.h>

#include <opencv2/core/core_c.h>
#include <yarp/cv/Cv.h>
#include <yarp/math/Math.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Property.h>
#include <yarp/os/ResourceFinder.h>

using namespace bfl;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


WalkmanCamera::WalkmanCamera
(
    const std::string& cam_sel,
    const double resolution_ratio,
    const std::string& context,
    const std::string& port_prefix
 ) :
    port_prefix_(port_prefix),
    cam_sel_(cam_sel),
    resolution_ratio_(resolution_ratio),
    context_(context)
{
    ResourceFinder rf;

    unsigned int cam_width = 800;
    unsigned int cam_height = 800;
    double cam_fx = 476.7030836014194;
    double cam_cx = 400.5;
    double cam_fy = 476.7030836014194;
    double cam_cy = 400.5;

    yInfo() << log_ID_ << "Found camera information:";
    yInfo() << log_ID_ << " - width:" << cam_width;
    yInfo() << log_ID_ << " - height:" << cam_height;
    yInfo() << log_ID_ << " - fx:" << cam_fx;
    yInfo() << log_ID_ << " - fy:" << cam_fy;
    yInfo() << log_ID_ << " - cx:" << cam_cx;
    yInfo() << log_ID_ << " - cy:" << cam_cy;

    params_.width = cam_width / resolution_ratio_;
    params_.height = cam_height / resolution_ratio_;
    params_.fx = cam_fx / resolution_ratio_;
    params_.cx = cam_cx / resolution_ratio_;
    params_.fy = cam_fy / resolution_ratio_;
    params_.cy = cam_cy / resolution_ratio_;

    yInfo() << log_ID_ << "Running with:";
    yInfo() << log_ID_ << " - resolution_ratio:" << resolution_ratio_;
    yInfo() << log_ID_ << " - width:" << params_.width;
    yInfo() << log_ID_ << " - height:" << params_.height;
    yInfo() << log_ID_ << " - fx:" << params_.fx;
    yInfo() << log_ID_ << " - cx:" << params_.cx;
    yInfo() << log_ID_ << " - fy:" << params_.fy;
    yInfo() << log_ID_ << " - cy:" << params_.cy;


    port_image_in_.open("/" + port_prefix_ + "/img:i");


    port_camera_pose_.open("/" + port_prefix_ + "/camera_pose:i");
}


WalkmanCamera::~WalkmanCamera() noexcept
{
    port_camera_pose_.interrupt();
    port_camera_pose_.close();
}


bool WalkmanCamera::bufferData()
{
    bool success = false;

    ImageOf<PixelRgb>* tmp_imgin = YARP_NULLPTR;
    tmp_imgin = port_image_in_.read(false);
    if (tmp_imgin != YARP_NULLPTR)
    {
        init_img_in_ = true;
        image_ = yarp::cv::toCvMat(*tmp_imgin).clone();
    }

    if (init_img_in_)
    {
        Bottle* bottle_camera_pose = port_camera_pose_.read(true);

        if (!bottle_camera_pose)
        {
            position_[0] = bottle_camera_pose->get(0).asDouble();
            position_[1] = bottle_camera_pose->get(1).asDouble();
            position_[2] = bottle_camera_pose->get(2).asDouble();

            orientation_[0] = bottle_camera_pose->get(3).asDouble();
            orientation_[1] = bottle_camera_pose->get(4).asDouble();
            orientation_[2] = bottle_camera_pose->get(5).asDouble();
            orientation_[3] = bottle_camera_pose->get(6).asDouble();

            success = true;
        }

    }

    return success;
}


Data WalkmanCamera::getData() const
{
    return Data(CameraData(image_, position_, orientation_));
}


Camera::CameraIntrinsics WalkmanCamera::getCameraParameters() const
{
    return params_;
}
