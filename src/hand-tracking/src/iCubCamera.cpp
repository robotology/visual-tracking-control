#include <iCubCamera.h>

#include <exception>

#include <iCub/ctrl/math.h>
#include <opencv2/core/core_c.h>
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


iCubCamera::iCubCamera(const yarp::os::ConstString& cam_sel,
                       const double resolution_ratio,
                       const yarp::os::ConstString& context,
                       const yarp::os::ConstString& port_prefix) :
    camera_data_(std::make_shared<CameraData>()),
    port_prefix_(port_prefix),
    cam_sel_(cam_sel),
    resolution_ratio_(resolution_ratio),
    context_(context)
{
    ResourceFinder rf;

    unsigned int cam_width;
    unsigned int cam_height;
    double cam_fx;
    double cam_cx;
    double cam_fy;
    double cam_cy;

    if (openGazeController())
    {
        Bottle cam_info;
        itf_gaze_->getInfo(cam_info);

        yInfo() << log_ID_ << cam_info.toString();

        Bottle* cam_sel_intrinsic = cam_info.findGroup("camera_intrinsics_" + cam_sel_).get(1).asList();

        cam_width  = static_cast<unsigned int>(cam_info.findGroup("camera_width_" + cam_sel_).get(1).asInt());
        cam_height = static_cast<unsigned int>(cam_info.findGroup("camera_height_" + cam_sel_).get(1).asInt());
        cam_fx     = cam_sel_intrinsic->get(0).asDouble();
        cam_cx     = cam_sel_intrinsic->get(2).asDouble();
        cam_fy     = cam_sel_intrinsic->get(5).asDouble();
        cam_cy     = cam_sel_intrinsic->get(6).asDouble();
    }
    else
    {
        yWarning() << log_ID_ << "No intrinisc camera information could be found by the ctor. Looking for fallback values in config.ini.";

        rf.setVerbose();
        rf.setDefaultContext(context_);
        rf.setDefaultConfigFile("config.ini");
        rf.configure(0, YARP_NULLPTR);

        Bottle* fallback_intrinsic = rf.findGroup("FALLBACK").find("intrinsic_" + cam_sel_).asList();
        if (fallback_intrinsic)
        {
            yInfo() << log_ID_ << fallback_intrinsic->toString();

            cam_width  = static_cast<unsigned int>(fallback_intrinsic->get(0).asInt());
            cam_height = static_cast<unsigned int>(fallback_intrinsic->get(1).asInt());
            cam_fx     = fallback_intrinsic->get(2).asDouble();
            cam_cx     = fallback_intrinsic->get(3).asDouble();
            cam_fy     = fallback_intrinsic->get(4).asDouble();
            cam_cy     = fallback_intrinsic->get(5).asDouble();
        }
        else
            throw std::runtime_error("ERROR::ICUBCAMERA::CTOR\nERROR: No camera intrinsic parameters could be found.");

        icub_kin_eye_ = iCubEye(cam_sel_ + "_v2");
        icub_kin_eye_.setAllConstraints(false);
        icub_kin_eye_.releaseLink(0);
        icub_kin_eye_.releaseLink(1);
        icub_kin_eye_.releaseLink(2);

        port_head_enc_.open("/" + port_prefix_ + "/head:i");
        port_torso_enc_.open("/" + port_prefix_ + "/torso:i");
    }

    yInfo() << log_ID_ << "Found camera information:";
    yInfo() << log_ID_ << " - width:"  << cam_width;
    yInfo() << log_ID_ << " - height:" << cam_height;
    yInfo() << log_ID_ << " - fx:"     << cam_fx;
    yInfo() << log_ID_ << " - fy:"     << cam_fy;
    yInfo() << log_ID_ << " - cx:"     << cam_cx;
    yInfo() << log_ID_ << " - cy:"     << cam_cy;

    params_.width  = cam_width  / resolution_ratio_;
    params_.height = cam_height / resolution_ratio_;
    params_.fx     = cam_fx     / resolution_ratio_;
    params_.cx     = cam_cx     / resolution_ratio_;
    params_.fy     = cam_fy     / resolution_ratio_;
    params_.cy     = cam_cy     / resolution_ratio_;

    yInfo() << log_ID_ << "Running with:";
    yInfo() << log_ID_ << " - resolution_ratio:" << resolution_ratio_;
    yInfo() << log_ID_ << " - width:"  << params_.width;
    yInfo() << log_ID_ << " - height:" << params_.height;
    yInfo() << log_ID_ << " - fx:"     << params_.fx;
    yInfo() << log_ID_ << " - cx:"     << params_.cx;
    yInfo() << log_ID_ << " - fy:"     << params_.fy;
    yInfo() << log_ID_ << " - cy:"     << params_.cy;


    port_image_in_.open("/" + port_prefix_ + "/img:i");
    

    port_head_enc_.open("/" + port_prefix_ + "/head:i");
    port_torso_enc_.open("/" + port_prefix_ + "/torso:i");
}


iCubCamera::~iCubCamera() noexcept
{
    drv_gaze_.close();

    port_image_in_.close();

    port_head_enc_.interrupt();
    port_head_enc_.close();

    port_torso_enc_.interrupt();
    port_torso_enc_.close();
}


bool iCubCamera::bufferProcessData()
{
    bool success = false;

    ImageOf<PixelRgb>* tmp_imgin = YARP_NULLPTR;
    tmp_imgin = port_image_in_.read(false);
    if (tmp_imgin != YARP_NULLPTR)
    {
        init_img_in_ = true;
        (*camera_data_->image_) = cv::cvarrToMat(tmp_imgin->getIplImage()).clone();
    }

    if (init_img_in_)
    {
        if (itf_gaze_)
        {
            Vector x(3);
            Vector o(4);

            if (cam_sel_ == "left")
                success = itf_gaze_->getLeftEyePose(x, o);
            else if (cam_sel_ == "right")
                success = itf_gaze_->getRightEyePose(x, o);

            if (success)
            {
                (*camera_data_->position_)[0] = x[0];
                (*camera_data_->position_)[1] = x[1];
                (*camera_data_->position_)[2] = x[2];

                (*camera_data_->orientation_)[0] = o[0];
                (*camera_data_->orientation_)[1] = o[1];
                (*camera_data_->orientation_)[2] = o[2];
                (*camera_data_->orientation_)[3] = o[3];
            }
        }
        else
        {
            Vector root_to_eye_enc;
            std::tie(success, root_to_eye_enc) = readRootToEye();

            if (success)
            {
                Vector eye_pose = icub_kin_eye_.EndEffPose(CTRL_DEG2RAD * root_to_eye_enc);

                (*camera_data_->position_)[0] = eye_pose[0];
                (*camera_data_->position_)[1] = eye_pose[1];
                (*camera_data_->position_)[2] = eye_pose[2];

                (*camera_data_->orientation_)[0] = eye_pose[3];
                (*camera_data_->orientation_)[1] = eye_pose[4];
                (*camera_data_->orientation_)[2] = eye_pose[5];
                (*camera_data_->orientation_)[3] = eye_pose[6];
            }
        }
    }

    return success;
}


std::shared_ptr<GenericData> iCubCamera::getProcessData()
{
    return camera_data_;
}


Camera::CameraParameters iCubCamera::getCameraParameters() const
{
    return params_;
}


bool iCubCamera::openGazeController()
{
    Property opt_gaze;
    opt_gaze.put("device", "gazecontrollerclient");
    opt_gaze.put("local", "/" + port_prefix_ + "/gaze:i");
    opt_gaze.put("remote", "/iKinGazeCtrl");

    if (drv_gaze_.open(opt_gaze))
    {
        drv_gaze_.view(itf_gaze_);
        if (!itf_gaze_)
        {
            yError() << log_ID_ << "Cannot get head gazecontrollerclient interface!";

            drv_gaze_.close();
            return false;
        }
    }
    else
    {
        yError() << log_ID_ << "Cannot open head gazecontrollerclient!";

        return false;
    }

    return true;
}


std::tuple<bool, Vector> iCubCamera::readRootToEye()
{
    Vector root_eye_enc(8, 0.0);

    Bottle* bottle_torso = port_torso_enc_.read(true);
    Bottle* bottle_head  = port_head_enc_.read(true);
    if (!bottle_torso || !bottle_head)
        return std::make_tuple(false, root_eye_enc);


    root_eye_enc(0) = bottle_torso->get(2).asDouble();
    root_eye_enc(1) = bottle_torso->get(1).asDouble();
    root_eye_enc(2) = bottle_torso->get(0).asDouble();

    for (size_t i = 0; i < 4; ++i)
        root_eye_enc(3 + i) = bottle_head->get(i).asDouble();

    if (cam_sel_ == "left")  root_eye_enc(7) = bottle_head->get(4).asDouble() + bottle_head->get(5).asDouble() / 2.0;
    if (cam_sel_ == "right") root_eye_enc(7) = bottle_head->get(4).asDouble() - bottle_head->get(5).asDouble() / 2.0;

    return std::make_tuple(true, root_eye_enc);
}
