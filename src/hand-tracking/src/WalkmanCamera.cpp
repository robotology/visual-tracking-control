#include <WalkmanCamera.h>

#include <exception>

#include <iCub/ctrl/math.h>
#include <yarp/math/Math.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Property.h>
#include <yarp/os/ResourceFinder.h>

using namespace bfl;
using namespace iCub::iKin;
using namespace iCub::ctrl;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


WalkmanCamera::WalkmanCamera(const yarp::os::ConstString& cam_sel,
                             const double resolution_ratio,
                             const yarp::os::ConstString& context,
                             const yarp::os::ConstString& port_prefix) :
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


    port_camera_pose_.open("/" + port_prefix_ + "/camera_pose:i");
}


WalkmanCamera::~WalkmanCamera() noexcept
{
    port_camera_pose_.interrupt();
    port_camera_pose_.close();
}


std::tuple<bool, Camera::CameraParameters> WalkmanCamera::getCameraParameters()
{
    return std::make_tuple(true, params_);
}


std::tuple<bool, std::array<double, 3>, std::array<double, 4>> WalkmanCamera::getCameraPose()
{
    bool success = false;
    std::array<double, 3> position{ { 0, 0, 0 } };
    std::array<double, 4> orientation{ { 0, 0, 0, 0 } };

    Vector camera_pose(7, 0.0);

    Bottle* bottle_camera_pose = port_camera_pose_.read(true);
    if (!bottle_camera_pose)
    {
        position[0] = bottle_camera_pose->get(0).asDouble();
        position[1] = bottle_camera_pose->get(1).asDouble();
        position[2] = bottle_camera_pose->get(2).asDouble();

        position[3] = bottle_camera_pose->get(3).asDouble();
        position[4] = bottle_camera_pose->get(4).asDouble();
        position[5] = bottle_camera_pose->get(5).asDouble();
        position[6] = bottle_camera_pose->get(6).asDouble();

        success = true;
    }

    return std::make_tuple(success, position, orientation);
}
