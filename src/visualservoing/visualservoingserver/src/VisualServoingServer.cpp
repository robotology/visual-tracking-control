#include "VisualServoingServer.h"

#include <iCub/ctrl/minJerkCtrl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <yarp/math/Math.h>
#include <yarp/math/SVD.h>
#include <yarp/os/Network.h>
#include <yarp/os/Property.h>
#include <yarp/os/Time.h>

using namespace yarp::dev;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;
using namespace iCub::ctrl;


/* Ctors and Dtors */
VisualServoingServer::VisualServoingServer()
{
    yInfo() << "*** Invoked VisualServoingServer ctor ***";
    yInfo() << "*** VisualServoingServer constructed ***";
}


VisualServoingServer::~VisualServoingServer()
{
    yInfo() << "*** Invoked VisualServoingServer dtor ***";
    yInfo() << "*** VisualServoingServer destructed ***";
}


/* DeviceDriver overrides */
bool VisualServoingServer::open(Searchable &config)
{
    verbosity_ = config.check("verbosity", Value(false)).asBool();
    yInfo() << "|> Verbosity: " + ConstString(verbosity_? "ON" : "OFF");

    sim_ = config.check("simulate", Value(false)).asBool();
    yInfo() << "|> Simulation: " + ConstString(sim_? "TRUE" : "FALSE");


    yInfoVerbose("*** Configuring VisualServoingServer ***");


    robot_name_ = config.check("robot", Value("icub")).asString();
    if (robot_name_.empty())
    {
        yErrorVerbose("Robot name not provided!");
        return false;
    }
    else
        yInfoVerbose("|> Robot name: " + robot_name_);


    if (!port_pose_left_in_.open("/visualservoing/pose/left:i"))
    {
        yErrorVerbose("Could not open /visualservoing/pose/left:i port!");
        return false;
    }
    if (!port_pose_right_in_.open("/visualservoing/pose/right:i"))
    {
        yErrorVerbose("Could not open /visualservoing/pose/right:i port!");
        return false;
    }


    if (!port_image_left_in_.open("/visualservoing/cam_left/img:i"))
    {
        yErrorVerbose("Could not open /visualservoing/cam_left/img:i port!");
        return false;
    }
    if (!port_image_left_out_.open("/visualservoing/cam_left/img:o"))
    {
        yErrorVerbose("Could not open /visualservoing/cam_left/img:o port!");
        return false;
    }


    if (!port_image_right_in_.open("/visualservoing/cam_right/img:i"))
    {
        yErrorVerbose("Could not open /visualservoing/cam_right/img:i port!");
        return false;
    }
    if (!port_image_right_out_.open("/visualservoing/cam_right/img:o"))
    {
        yErrorVerbose("Could not open /visualservoing/cam_right/img:o port!");
        return false;
    }


    if (!port_click_left_.open("/visualservoing/cam_left/click:i"))
    {
        yErrorVerbose("Could not open /visualservoing/cam_left/click:in port!");
        return false;
    }
    if (!port_click_right_.open("/visualservoing/cam_right/click:i"))
    {
        yErrorVerbose("Could not open /visualservoing/cam_right/click:i port!");
        return false;
    }


    if (!setGazeController()) return false;

    if (!setRightArmCartesianController()) return false;


    Bottle btl_cam_info;
    itf_gaze_->getInfo(btl_cam_info);
    yInfoVerbose("[CAM]" + btl_cam_info.toString());
    Bottle* cam_left_info = btl_cam_info.findGroup("camera_intrinsics_left").get(1).asList();
    Bottle* cam_right_info = btl_cam_info.findGroup("camera_intrinsics_right").get(1).asList();

    double left_fx = cam_left_info->get(0).asDouble();
    double left_cx = cam_left_info->get(2).asDouble();
    double left_fy = cam_left_info->get(5).asDouble();
    double left_cy = cam_left_info->get(6).asDouble();

    yInfoVerbose("[CAM] Left camera intrinsic parameters:");
    yInfoVerbose("[CAM]  - fx: " + std::to_string(left_fx));
    yInfoVerbose("[CAM]  - fy: " + std::to_string(left_fy));
    yInfoVerbose("[CAM]  - cx: " + std::to_string(left_cx));
    yInfoVerbose("[CAM]  - cy: " + std::to_string(left_cy));

    l_proj_       = zeros(3, 4);
    l_proj_(0, 0) = left_fx;
    l_proj_(0, 2) = left_cx;
    l_proj_(1, 1) = left_fy;
    l_proj_(1, 2) = left_cy;
    l_proj_(2, 2) = 1.0;

    yInfoVerbose("Left projection matrix = [\n" + l_proj_.toString() + "\n]");

    double right_fx = cam_right_info->get(0).asDouble();
    double right_cx = cam_right_info->get(2).asDouble();
    double right_fy = cam_right_info->get(5).asDouble();
    double right_cy = cam_right_info->get(6).asDouble();

    yInfoVerbose("[CAM] Right camera intrinsic paramenter:");
    yInfoVerbose("[CAM]  - fx: " + std::to_string(right_fx));
    yInfoVerbose("[CAM]  - fy: " + std::to_string(right_fy));
    yInfoVerbose("[CAM]  - cx: " + std::to_string(right_cx));
    yInfoVerbose("[CAM]  - cy: " + std::to_string(right_cy));

    r_proj_       = zeros(3, 4);
    r_proj_(0, 0) = right_fx;
    r_proj_(0, 2) = right_cx;
    r_proj_(1, 1) = right_fy;
    r_proj_(1, 2) = right_cy;
    r_proj_(2, 2) = 1.0;

    yInfoVerbose("Right projection matrix = [\n" + r_proj_.toString() + "\n]");


    if (!setCommandPort())
    {
        yErrorVerbose("Could not open /visualservoing/cmd:i port!");
        return false;
    }


    yInfoVerbose("*** VisualServoingServer configured! ***");

    return true;
}


bool VisualServoingServer::close()
{
    yInfoVerbose("*** Interrupting VisualServoingServer ***");

    yInfoVerbose("Ensure controllers are stopped...");
    itf_rightarm_cart_->stopControl();
    itf_gaze_->stopControl();

    yInfoVerbose("...interrupting ports.");
    port_pose_left_in_.interrupt();
    port_pose_right_in_.interrupt();
    port_image_left_in_.interrupt();
    port_image_left_out_.interrupt();
    port_click_left_.interrupt();
    port_image_right_in_.interrupt();
    port_image_right_out_.interrupt();
    port_click_right_.interrupt();

    yInfoVerbose("*** Interrupting VisualServoingServer done! ***");


    yInfoVerbose("*** Closing VisualServoingServer ***");

    yInfoVerbose("Closing ports...");
    port_pose_left_in_.close();
    port_pose_right_in_.close();
    port_image_left_in_.close();
    port_image_left_out_.close();
    port_click_left_.close();
    port_image_right_in_.close();
    port_image_right_out_.close();
    port_click_right_.close();

    yInfoVerbose("...removing frames...");
    itf_rightarm_cart_->removeTipFrame();

    yInfoVerbose("...closing drivers.");
    if (rightarm_cartesian_driver_.isValid()) rightarm_cartesian_driver_.close();
    if (gaze_driver_.isValid())               gaze_driver_.close();

    yInfoVerbose("*** Closing VisualServoingServer done! ***");


    return true;
}


/* IVisualServoing overrides */
bool VisualServoingServer::initFacilities(const bool use_direct_kin)
{
    if (use_direct_kin)
    {
        yInfoVerbose("Connecting to Cartesian controller right arm output state.");

        if (!Network::connect("/" + robot_name_ + "/cartesianController/right_arm/state:o", port_pose_left_in_.getName(), "tcp", !verbosity_))
            return false;

        if (!Network::connect("/" + robot_name_ + "/cartesianController/right_arm/state:o", port_pose_right_in_.getName(), "tcp", !verbosity_))
            return false;

        yInfoVerbose("Using direct kinematics information for visual servoing.");
    }
    else
    {
        yInfoVerbose("Connecting to external pose trackers command ports.");

        if (!Network::connect(port_rpc_tracker_left_.getName(),  "/hand-tracking/left/cmd:i", "tcp", !verbosity_))
            return false;

        if (!Network::connect(port_rpc_tracker_right_.getName(), "/hand-tracking/right/cmd:i", "tcp", !verbosity_))
            return false;


        yInfoVerbose("Sending commands to external pose trackers.");

        Bottle cmd;
        cmd.addString("run_filter");

        Bottle response_left;
        if (!port_rpc_tracker_left_.write(cmd, response_left))
            return false;

        if (!response_left.get(0).asBool())
            return false;

        yInfoVerbose("Left camera external pose tracker running.");


        Bottle response_right;
        if (!port_rpc_tracker_right_.write(cmd, response_right))
            return false;

        if (!response_right.get(0).asBool())
            return false;

        yInfoVerbose("Right camera external pose tracker running.");

        yInfoVerbose("Using external pose trackers information for visual servoing.");


        yInfoVerbose("Waiting for the filter to provide good estimates...");
        yarp::os::Time::delay(10);
        yInfoVerbose("...done!");


        yInfoVerbose("Connecting to external pose trackers output ports.");

        if (!Network::connect("/hand-tracking/left/result/estimates:o",   port_pose_left_in_.getName(),  "tcp", !verbosity_))
            return false;

        if (!Network::connect("/hand-tracking/right/result/estimates:o ", port_pose_right_in_.getName(), "tcp", !verbosity_))
            return false;

        yInfoVerbose("Receiving end-effector pose from external trackers.");
    }

    return true;
}


bool VisualServoingServer::resetFacilities()
{
    if (port_rpc_tracker_left_.getOutputCount() > 0 && port_rpc_tracker_right_.getOutputCount() > 0)
    {
        yInfoVerbose("Sending commands to external pose trackers.");

        Bottle cmd;
        cmd.addString("rest_filter");

        Bottle response_left;
        if (!port_rpc_tracker_left_.write(cmd, response_left))
            return false;

        if (!response_left.get(0).asBool())
            return false;

        yInfoVerbose("Left camera external pose tracker reset.");


        Bottle response_right;
        if (!port_rpc_tracker_right_.write(cmd, response_right))
            return false;

        if (!response_right.get(0).asBool())
            return false;

        yInfoVerbose("Right camera external pose tracker reset.");

        return true;
    }
    else
        return false;
}


bool VisualServoingServer::stopFacilities()
{
    if (port_rpc_tracker_left_.getOutputCount() > 0 && port_rpc_tracker_right_.getOutputCount() > 0)
    {
        yInfoVerbose("Sending commands to external pose trackers.");

        Bottle cmd;
        cmd.addString("stop_filter");

        Bottle response_left;
        if (!port_rpc_tracker_left_.write(cmd, response_left))
            return false;

        if (!response_left.get(0).asBool())
            return false;

        yInfoVerbose("Left camera external pose tracker stopped.");

        if (!Network::disconnect(port_rpc_tracker_left_.getName(),  "/hand-tracking/left/cmd:i", !verbosity_))
            return false;


        Bottle response_right;
        if (!port_rpc_tracker_right_.write(cmd, response_right))
            return false;

        if (!response_right.get(0).asBool())
            return false;

        yInfoVerbose("Right camera external pose tracker stopped.");

        if (!Network::disconnect(port_rpc_tracker_right_.getName(), "/hand-tracking/right/cmd:i", !verbosity_))
            return false;


        yInfoVerbose("Disconnecting from external pose trackers output ports.");

        if (!Network::disconnect("/hand-tracking/left/result/estimates:o",   port_pose_left_in_.getName(), !verbosity_))
            return false;

        if (!Network::disconnect("/hand-tracking/right/result/estimates:o ", port_pose_right_in_.getName(), !verbosity_))
            return false;

        yInfoVerbose("Disconnected from external trackers.");


        return true;
    }
    else
        return false;
}


bool VisualServoingServer::goToGoal(const Vector& vec_x, const Vector& vec_o)
{
    yInfoVerbose("*** VisualServoingServer::goToGoal with pose goal invoked ***");

    std::vector<Vector> vec_px_l = getGoalPixelsFrom3DPose(vec_x, vec_o, CamSel::left);
    if (vec_px_l.size() == 0)
        return false;

    std::vector<Vector> vec_px_r = getGoalPixelsFrom3DPose(vec_x, vec_o, CamSel::right);
    if (vec_px_r.size() == 0)
        return false;

    setPoseGoal(vec_x, vec_o);
    setPixelGoal(vec_px_l, vec_px_r);

    return start();
}


bool VisualServoingServer::goToGoal(const std::vector<Vector>& vec_px_l, const std::vector<Vector>& vec_px_r)
{
    yInfoVerbose("*** VisualServoingServer::goToGoal with pixel goal invoked ***");

    yWarningVerbose("<!-- If you did not invoke either one of:");
    yWarningVerbose("<!--  1. get3DPositionGoalFrom3DPose()");
    yWarningVerbose("<!--  2. getGoalPixelsFrom3DPose");
    yWarningVerbose("<!-- visual servoing will just not work, and the server may also close.");
    yWarningVerbose("<!-- To the current implementation, this behaviour is intentional.");
    yWarningVerbose("<!-- Upcoming releases will change the behaviour of this method.");

    setPixelGoal(vec_px_l, vec_px_r);

    return start();
}


bool VisualServoingServer::setModality(const std::string& mode)
{
    if (mode == "position")
        op_mode_ = OperatingMode::position;
    else if (mode == "orientation")
        op_mode_ = OperatingMode::orientation;
    else if (mode == "pose")
        op_mode_ = OperatingMode::pose;
    else
        return false;

    return true;
}


bool VisualServoingServer::setVisualServoControl(const std::string& control)
{
    if (control == "decoupled")
        vs_control_ = VisualServoControl::decoupled;
    else if (control == "robust")
        vs_control_ = VisualServoControl::robust;
    else
        return false;

    return true;
}


bool VisualServoingServer::setControlPoint(const yarp::os::ConstString& point)
{
    yWarningVerbose("*** Service setControlPoint is unimplemented. ***");

    return false;
}


bool VisualServoingServer::getVisualServoingInfo(yarp::os::Bottle& info)
{
    yWarningVerbose("*** Service getVisualServoingInfo is unimplemented. ***");

    return false;
}


bool VisualServoingServer::setGoToGoalTolerance(const double tol)
{
    px_tol_ = tol;

    return true;
}


bool VisualServoingServer::checkVisualServoingController()
{
    yInfoVerbose("");
    yInfoVerbose("*** Checking visual servoing controller ***");
    yInfoVerbose(" |> Controller: " + ConstString(vs_control_running_ ? "running." : "idle."));
    yInfoVerbose(" |> Goal: "       + ConstString(vs_goal_reached_    ? "" : "not ") + "reached.");
    yInfoVerbose("");

    return vs_control_running_;
}


bool VisualServoingServer::waitVisualServoingDone(const double period, const double timeout)
{
    yInfoVerbose("");
    yInfoVerbose("*** Joining visual servoing control thread ***");
    yInfoVerbose(" |> Controller: " + ConstString(vs_control_running_ ? "running." : "idle."));
    yInfoVerbose(" |> Goal: "       + ConstString(vs_goal_reached_    ? "" : "not ") + "reached.");
    yInfoVerbose("");

    return join();
}


bool VisualServoingServer::stopController()
{
    yInfoVerbose("");
    yInfoVerbose("*** Stopping visual servoing controller ***");
    yInfoVerbose(" |> Controller: " + ConstString(vs_control_running_ ? "running." : "idle."));
    yInfoVerbose(" |> Goal: "       + ConstString(vs_goal_reached_    ? "" : "not ") + "reached.");
    yInfoVerbose("");

    return stop();
}


bool VisualServoingServer::setTranslationGain(const double K_x_1, const double K_x_2)
{
    K_x_[0] = K_x_1;
    K_x_[1] = K_x_2;

    return true;
}


bool VisualServoingServer::setMaxTranslationVelocity(const double max_x_dot)
{
    max_x_dot_ = max_x_dot;

    return true;
}


bool VisualServoingServer::setTranslationGainSwitchTolerance(const double K_x_tol)
{
    K_x_tol_ = K_x_tol;

    return true;
}


bool VisualServoingServer::setOrientationGain(const double K_o_1, const double K_o_2)
{
    K_o_[0] = K_o_1;
    K_o_[1] = K_o_2;

    return true;
}


bool VisualServoingServer::setMaxOrientationVelocity(const double max_o_dot)
{
    max_o_dot_ = max_o_dot;

    return true;
}


bool VisualServoingServer::setOrientationGainSwitchTolerance(const double K_o_tol)
{
    K_o_tol_ = K_o_tol;

    return true;
}


std::vector<Vector> VisualServoingServer::get3DGoalPositionsFrom3DPose(const Vector& x, const Vector& o)
{
    if (x.length() != 3 || o.length() != 4)
        return std::vector<Vector>();


    Vector pose(7);
    pose.setSubvector(0, x);
    pose.setSubvector(3, o);

    yWarningVerbose("<!-- Invoking setPoseGoal() to set the goal pose for visual servoing.");
    yWarningVerbose("<!-- Be warned that this specific invokation will be removed in upcoming releases.");
    setPoseGoal(x, o);

    std::vector<Vector> vec_goal_points = getControlPointsFromPose(pose);

    return vec_goal_points;
}


std::vector<Vector> VisualServoingServer::getGoalPixelsFrom3DPose(const Vector& x, const Vector& o, const CamSel& cam)
{
    if (x.length() != 3 || o.length() != 4)
        return std::vector<Vector>();


    Vector pose(7);
    pose.setSubvector(0, x);
    pose.setSubvector(3, o);

    yWarningVerbose("<!-- Invoking setPoseGoal() to set the goal pose for visual servoing.");
    yWarningVerbose("<!-- Be warned that this specific invokation will be removed in upcoming releases.");
    setPoseGoal(x, o);

    setCameraTransformations();

    std::vector<Vector> vec_goal_points = getPixelsFromPose(pose, cam);

    return vec_goal_points;
}


/* VisualServoingServerIDL overrides */
bool VisualServoingServer::storedInit(const std::string& label)
{
    itf_rightarm_cart_->storeContext(&ctx_remote_cart_);
    itf_rightarm_cart_->restoreContext(ctx_local_cart_);


    Vector xd       = zeros(3);
    Vector od       = zeros(4);
    Vector gaze_loc = zeros(3);

    if (label == "t170427")
    {
        /* -0.346 0.133 0.162 0.140 -0.989 0.026 2.693 */
        xd[0] = -0.346;
        xd[1] =  0.133;
        xd[2] =  0.162;

        od[0] =  0.140;
        od[1] = -0.989;
        od[2] =  0.026;
        od[3] =  2.693;

        /* -6.706 1.394 -3.618 */
        gaze_loc[0] = -6.706;
        gaze_loc[1] =  1.394;
        gaze_loc[2] = -3.618;
    }
    else if (label == "t170517")
    {
        /* -0.300 0.088 0.080 -0.245 0.845 -0.473 2.896 */
        xd[0] = -0.300;
        xd[1] =  0.088;
        xd[2] =  0.080;

        od[0] = -0.245;
        od[1] =  0.845;
        od[2] = -0.473;
        od[3] =  2.896;

        /* -5.938 1.385 -4.724 */
        gaze_loc[0] = -5.938;
        gaze_loc[1] =  1.385;
        gaze_loc[2] = -4.724;
    }
    else if (label == "t170713")
    {
        /* -0.288 0.15 0.118 0.131 -0.492 0.86 2.962 */
        xd[0] = -0.288;
        xd[1] =  0.15;
        xd[2] =  0.118;

        od[0] =  0.131;
        od[1] = -0.492;
        od[2] =  0.86;
        od[3] =  2.962;

        /* -5.938 1.385 -4.724 */
        gaze_loc[0] = -5.938;
        gaze_loc[1] =  1.385;
        gaze_loc[2] = -4.724;
    }
    else if (label == "sfm300517")
    {
        /* -0.333 0.203 -0.053 0.094 0.937 -0.335 3.111 */
        xd[0] = -0.333;
        xd[1] =  0.203;
        xd[2] = -0.053;

        od[0] =  0.094;
        od[1] =  0.937;
        od[2] = -0.335;
        od[3] =  3.111;

        /* -0.589 0.252 -0.409 */
        gaze_loc[0] = -0.589;
        gaze_loc[1] =  0.252;
        gaze_loc[2] = -0.409;

        itf_rightarm_cart_->setLimits(0,  25.0,  25.0);
    }
    else
        return false;

    yInfoVerbose("Init position: "    + xd.toString());
    yInfoVerbose("Init orientation: " + od.toString());


    unsetTorsoDOF();

    itf_rightarm_cart_->goToPoseSync(xd, od);
    itf_rightarm_cart_->waitMotionDone(0.1, 10.0);
    itf_rightarm_cart_->stopControl();

    itf_rightarm_cart_->removeTipFrame();

    setTorsoDOF();


    yInfoVerbose("Fixation point: " + gaze_loc.toString());

    itf_gaze_->lookAtFixationPointSync(gaze_loc);
    itf_gaze_->waitMotionDone(0.1, 10.0);
    itf_gaze_->stopControl();


    itf_rightarm_cart_->restoreContext(ctx_remote_cart_);

    return true;
}


/* Set a fixed goal in pixel coordinates */
bool VisualServoingServer::storedGoToGoal(const std::string& label)
{
    if (label == "t170427")
    {
        /* -0.323 0.018 0.121 0.310 -0.873 0.374 3.008 */
        goal_pose_[0] = -0.323;
        goal_pose_[1] =  0.018;
        goal_pose_[2] =  0.121;

        goal_pose_[3] =  0.310;
        goal_pose_[4] = -0.873;
        goal_pose_[5] =  0.374;
        goal_pose_[6] =  3.008;
    }
    else if (label == "t170517")
    {
        /* -0.284 0.013 0.104 -0.370 0.799 -0.471 2.781 */
        goal_pose_[0] = -0.284;
        goal_pose_[1] =  0.013;
        goal_pose_[2] =  0.104;

        goal_pose_[3] = -0.370;
        goal_pose_[4] =  0.799;
        goal_pose_[5] = -0.471;
        goal_pose_[6] =  2.781;
    }
    else if (label == "t170711")
    {
        /* -0.356 0.024 -0.053 0.057 0.98 -0.189 2.525 */
        goal_pose_[0] = -0.356;
        goal_pose_[1] =  0.024;
        goal_pose_[2] = -0.053;

        goal_pose_[3] =  0.057;
        goal_pose_[4] =  0.980;
        goal_pose_[5] = -0.189;
        goal_pose_[6] =  2.525;
    }
    else if (label == "t170713")
    {
        /* -0.282 0.061 0.068 0.213 -0.94 0.265 2.911 */
        goal_pose_[0] = -0.282;
        goal_pose_[1] =  0.061;
        goal_pose_[2] =  0.068;

        goal_pose_[3] =  0.213;
        goal_pose_[4] = -0.94 ;
        goal_pose_[5] =  0.265;
        goal_pose_[6] =  2.911;
    }
    else
        return false;

    yInfoVerbose("6D goal: " + goal_pose_.toString());

    setCameraTransformations();

    std::vector<Vector> l_px_from_pose = getPixelsFromPose(goal_pose_, CamSel::left);
    std::vector<Vector> r_px_from_pose = getPixelsFromPose(goal_pose_, CamSel::right);

    setPixelGoal(l_px_from_pose, r_px_from_pose);

    return start();
}


/* Get 3D point from Structure From Motion clicking on the left camera image */
bool VisualServoingServer::goToSFMGoal()
{
    Bottle cmd;
    Bottle rep;

    Bottle* click_left = port_click_left_.read(true);
    Vector l_click = zeros(2);
    l_click[0] = click_left->get(0).asDouble();
    l_click[1] = click_left->get(1).asDouble();

    RpcClient port_sfm;
    port_sfm.open("/visualservoing/tosfm");
    Network::connect("/visualservoing/tosfm", "/SFM/rpc");

    cmd.clear();

    cmd.addInt(l_click[0]);
    cmd.addInt(l_click[1]);

    Bottle reply_pos;
    port_sfm.write(cmd, reply_pos);
    if (reply_pos.size() == 5)
    {
        Matrix R_ee = zeros(3, 3);
        R_ee(0, 0) = -1.0;
//        R_ee(1, 1) =  1.0;
        R_ee(1, 2) = -1.0;
//        R_ee(2, 2) = -1.0;
        R_ee(2, 1) = -1.0;
        Vector ee_o = dcm2axis(R_ee);

        Vector sfm_pos = zeros(3);
        sfm_pos[0] = reply_pos.get(0).asDouble();
        sfm_pos[1] = reply_pos.get(1).asDouble();
        sfm_pos[2] = reply_pos.get(2).asDouble();

        Vector p = zeros(7);
        p.setSubvector(0, sfm_pos.subVector(0, 2));
        p.setSubvector(3, ee_o.subVector(0, 3));

        goal_pose_ = p;
        yInfoVerbose("6D goal: " + goal_pose_.toString());

        setCameraTransformations();

        std::vector<Vector> l_px_from_pose = getPixelsFromPose(goal_pose_, CamSel::left);
        std::vector<Vector> r_px_from_pose = getPixelsFromPose(goal_pose_, CamSel::right);

        setPixelGoal(l_px_from_pose, r_px_from_pose);
    }
    else
        return false;

    Network::disconnect("/visualservoing/tosfm", "/SFM/rpc");
    port_sfm.close();

    return start();
}


/* Thread overrides */
void VisualServoingServer::beforeStart()
{
    yInfoVerbose("*** Thread::beforeStart invoked ***");
}


bool VisualServoingServer::threadInit()
{
    yInfoVerbose("*** Thread::threadInit invoked ***");


    /* SETTING STATUS */
    vs_control_running_ = true;
    vs_goal_reached_    = false;


    /* RESTORING CARTESIAN AND GAZE CONTEXT */
    itf_rightarm_cart_->storeContext(&ctx_remote_cart_);
    itf_rightarm_cart_->restoreContext(ctx_local_cart_);


    /* SETTING BACKGROUND THREAD */
    yInfoVerbose("*** Launching background process UpdateVisualServoingParamters ***");
    is_stopping_backproc_update_vs_params = false;
    thr_background_update_params_ = new std::thread(&VisualServoingServer::backproc_UpdateVisualServoingParamters, this);


    yInfoVerbose("");
    yInfoVerbose("*** Running visual servoing! ***");
    yInfoVerbose("");

    return true;
}


void VisualServoingServer::run()
{
    if (vs_control_ == VisualServoControl::decoupled)
        decoupledImageBasedVisualServoControl();
    else if (vs_control_ == VisualServoControl::robust)
        robustImageBasedVisualServoControl();
}


void VisualServoingServer::afterStart(bool success)
{
    yInfoVerbose("*** Thread::afterStart invoked ***");

    if (success)
    {
        yInfoVerbose("Visual servoing controller status report:");
        yInfoVerbose(" |> Controller: " + ConstString(vs_control_running_ ? "running." : "idle."));
        yInfoVerbose(" |> Goal: "       + ConstString(vs_goal_reached_    ? "" : "not") + "reached.");
    }
    else
    {
        yInfoVerbose("Visual servoing controller failed to start!");
        vs_control_running_ = false;
        vs_goal_reached_    = false;
    }
}


void VisualServoingServer::onStop()
{
    yInfoVerbose("*** Thread::onStop invoked ***");
}


void VisualServoingServer::threadRelease()
{
    yInfoVerbose("*** Thread::threadRelease invoked ***");


    /* ENSURE CONTROLLERS ARE STOPPED */
    itf_rightarm_cart_->stopControl();
    itf_gaze_->stopControl();


    /* STOPPING AND JOINING BACKGROUND THREAD */
    yInfoVerbose("*** Stopping background process UpdateVisualServoingParamters ***");
    is_stopping_backproc_update_vs_params = true;
    thr_background_update_params_->join();
    delete thr_background_update_params_;


    /* RESTORING REMOTE CARTESIAN AND GAZE CONTEXTS */
    itf_rightarm_cart_->restoreContext(ctx_remote_cart_);


    vs_control_running_ = false;

    yInfoVerbose("");
    yInfoVerbose("*** Visual servoing terminated! ***");
    yInfoVerbose("");
}


/* VisualServoingServerIDL overrides */
bool VisualServoingServer::stored_init(const std::string& label)
{
    return storedInit(label);
}


bool VisualServoingServer::stored_go_to_goal(const std::string& label)
{
    return storedGoToGoal(label);
}


bool VisualServoingServer::get_goal_from_sfm()
{
    return goToSFMGoal();
}


bool VisualServoingServer::quit()
{
    yInfoVerbose("*** Quitting visual servoing server ***");

    bool is_stopping_controller = stopController();
    if (!is_stopping_controller)
    {
        yWarningVerbose("Could not stop visual servoing controller!");
        return false;
    }

    bool is_closing = close();
    if (!is_closing)
    {
        yWarningVerbose("Could not close visual servoing server!");
        return false;
    }

    return true;
}


bool VisualServoingServer::init_facilities(const bool use_direct_kin)
{
    return initFacilities(use_direct_kin);
}


bool VisualServoingServer::reset_facilities()
{
    return resetFacilities();
}


bool VisualServoingServer::stop_facilities()
{
    return stopFacilities();
}


bool VisualServoingServer::go_to_px_goal(const std::vector<std::vector<double>>& vec_px_l, const std::vector<std::vector<double>>& vec_px_r)
{
    if (vec_px_l.size() != 4 || vec_px_l.size() != 4)
        return false;

    std::vector<Vector> yvec_px_l;
    for (const std::vector<double>& vec : vec_px_l)
    {
        if (vec.size() != 2)
            return false;

        yvec_px_l.emplace_back(Vector(vec.size(), vec.data()));
    }

    std::vector<Vector> yvec_px_r;
    for (const std::vector<double>& vec : vec_px_r)
    {
        if (vec.size() != 2)
            return false;

        yvec_px_r.emplace_back(Vector(vec.size(), vec.data()));
    }

    return goToGoal(yvec_px_l, yvec_px_r);
}


bool VisualServoingServer::go_to_pose_goal(const std::vector<double>& vec_x, const std::vector<double>& vec_o)
{
    if (vec_x.size() != 3 || vec_o.size() != 4)
        return false;

    Vector yvec_x(vec_x.size(), vec_x.data());
    Vector yvec_o(vec_o.size(), vec_o.data());

    return goToGoal(yvec_x, yvec_o);
}


bool VisualServoingServer::set_modality(const std::string& mode)
{
    return setModality(mode);
}


bool VisualServoingServer::set_visual_servo_control(const std::string& control)
{
    return setVisualServoControl(control);
}


bool VisualServoingServer::set_control_point(const std::string& point)
{
    return setControlPoint(point);
}


std::vector<std::string> VisualServoingServer::get_visual_servoing_info()
{
    Bottle info;
    getVisualServoingInfo(info);

    std::vector<std::string> info_str;
    info_str.emplace_back(info.toString());

    return info_str;
}


bool VisualServoingServer::set_go_to_goal_tolerance(const double tol)
{
    return setGoToGoalTolerance(tol);
}


bool VisualServoingServer::check_visual_servoing_controller()
{
    return checkVisualServoingController();
}


bool VisualServoingServer::wait_visual_servoing_done(const double period, const double timeout)
{
    return waitVisualServoingDone(period, timeout);
}


bool VisualServoingServer::stop_controller()
{
    return stopController();
}


bool VisualServoingServer::set_translation_gain(const double K_x_1, const double K_x_2)
{
    return setTranslationGain(K_x_1, K_x_2);
}


bool VisualServoingServer::set_max_translation_velocity(const double max_x_dot)
{
    return setMaxTranslationVelocity(max_x_dot);
}


bool VisualServoingServer::set_translation_gain_switch_tolerance(const double K_x_tol)
{
    return setTranslationGainSwitchTolerance(K_x_tol);
}


bool VisualServoingServer::set_orientation_gain(const double K_o_1, const double K_o_2)
{
    return setOrientationGain(K_o_1, K_o_2);
}


bool VisualServoingServer::set_max_orientation_velocity(const double max_o_dot)
{
    return setMaxOrientationVelocity(max_o_dot);
}


bool VisualServoingServer::set_orientation_gain_switch_tolerance(const double K_o_tol)
{
    return setOrientationGainSwitchTolerance(K_o_tol);
}


std::vector<std::vector<double>> VisualServoingServer::get_3D_goal_positions_from_3D_pose(const std::vector<double>& x, const std::vector<double>& o)
{
    if (x.size() != 3 || o.size() != 4)
        return std::vector<std::vector<double>>();


    Vector yx(x.size(), x.data());
    Vector yo(o.size(), o.data());
    std::vector<Vector> yvec_3d_goal_points = get3DGoalPositionsFrom3DPose(yx, yo);

    std::vector<std::vector<double>> vec_3d_goal_points;
    for (const Vector& yvec_3d_goal : yvec_3d_goal_points)
        vec_3d_goal_points.emplace_back(std::vector<double>(yvec_3d_goal.data(), yvec_3d_goal.data() + yvec_3d_goal.size()));

    return vec_3d_goal_points;
}


std::vector<std::vector<double>> VisualServoingServer::get_goal_pixels_from_3D_pose(const std::vector<double>& x, const std::vector<double>& o, const std::string& cam)
{
    if (x.size() != 3 || o.size() != 4 || (cam != "left" && cam != "right"))
        return std::vector<std::vector<double>>();


    Vector yx(x.size(), x.data());
    Vector yo(o.size(), o.data());
    CamSel e_cam = cam == "left" ? CamSel::left : CamSel::right;

    std::vector<Vector> yvec_px_goal_points = getGoalPixelsFrom3DPose(yx, yo, e_cam);
    if (yvec_px_goal_points.size() == 0)
        return std::vector<std::vector<double>>();

    std::vector<std::vector<double>> vec_px_goal_points;
    for (const Vector& yvec_px_goal : yvec_px_goal_points)
        vec_px_goal_points.emplace_back(std::vector<double>(yvec_px_goal.data(), yvec_px_goal.data() + yvec_px_goal.size()));

    return vec_px_goal_points;
}


/* Private class methods */
void VisualServoingServer::decoupledImageBasedVisualServoControl()
{
    /* VARIABLE DEFINITIONS */
    Vector* endeffector_pose;
    Vector  eepose_copy_left(7);
    Vector  eepose_copy_right(7);

    std::vector<Vector> l_px_position;
    std::vector<Vector> l_px_orientation;

    std::vector<Vector> r_px_position;
    std::vector<Vector> r_px_orientation;

    Vector px_ee_cur_position    = zeros(12);
    Matrix jacobian_position     = zeros(12, 6);
    Vector px_ee_cur_orientation = zeros(12);
    Matrix jacobian_orientation  = zeros(12, 6);


    /* GET THE INITIAL END-EFFECTOR POSE FOR THE LEFT EYE */
    endeffector_pose = port_pose_left_in_.read(true);
    eepose_copy_left = *endeffector_pose;

    yInfoVerbose("Got [" + eepose_copy_left.toString() + "] end-effector pose for the left eye.");

    /* GET THE INITIAL END-EFFECTOR POSE FOR THE RIGHT EYE */
    endeffector_pose = port_pose_right_in_.read(true);
    eepose_copy_right = *endeffector_pose;

    yInfoVerbose("Got [" + eepose_copy_right.toString() + "] end-effector pose for the right eye.");

    while (!isStopping() && !vs_goal_reached_)
    {
        yInfoVerbose("Desired goal pixels = [" + px_des_.toString() + "]");


        /* EVALUATING CONTROL POINTS */
        l_px_position    = getControlPixelsFromPose(eepose_copy_left, CamSel::left, PixelControlMode::x);
        l_px_orientation = getControlPixelsFromPose(eepose_copy_left, CamSel::left, PixelControlMode::o);

        r_px_position    = getControlPixelsFromPose(eepose_copy_right, CamSel::right, PixelControlMode::x);
        r_px_orientation = getControlPixelsFromPose(eepose_copy_right, CamSel::right, PixelControlMode::o);


        /* POSITION: FEATURES AND JACOBIAN */
        getCurrentStereoFeaturesAndJacobian(l_px_position, r_px_position,
                                            px_ee_cur_position, jacobian_position);

        yInfoVerbose("Position controlled pixels = ["   + px_ee_cur_position.toString() + "]");
        yInfoVerbose("Position image Jacobian    = [\n" + jacobian_position.toString()  + "]");


        /* ORIENTATION: FEATURES AND JACOBIAN */
        getCurrentStereoFeaturesAndJacobian(l_px_orientation, r_px_orientation,
                                            px_ee_cur_orientation, jacobian_orientation);

        yInfoVerbose("Orientation controlled pixels = ["   + px_ee_cur_orientation.toString() + "]");
        yInfoVerbose("Orientation image Jacobian    = [\n" + jacobian_orientation.toString()  + "]");


        /* *** *** *** DEBUG OUTPUT - TO BE DELETED *** *** *** */
        std::vector<cv::Scalar> color;
        color.emplace_back(cv::Scalar(255,   0,   0));
        color.emplace_back(cv::Scalar(  0, 255,   0));
        color.emplace_back(cv::Scalar(  0,   0, 255));
        color.emplace_back(cv::Scalar(255, 127,  51));


        /* Left eye end-effector superimposition */
        ImageOf<PixelRgb>* l_imgin  = port_image_left_in_.read(true);
        ImageOf<PixelRgb>& l_imgout = port_image_left_out_.prepare();
        l_imgout = *l_imgin;
        cv::Mat l_img = cv::cvarrToMat(l_imgout.getIplImage());

        /* Right eye end-effector superimposition */
        ImageOf<PixelRgb>* r_imgin  = port_image_right_in_.read(true);
        ImageOf<PixelRgb>& r_imgout = port_image_right_out_.prepare();
        r_imgout = *r_imgin;
        cv::Mat r_img = cv::cvarrToMat(r_imgout.getIplImage());


        /* Plot points and lines */
        std::vector<Vector> l_px_pose = getPixelsFromPose(eepose_copy_left,  CamSel::left);
        std::vector<Vector> r_px_pose = getPixelsFromPose(eepose_copy_right, CamSel::right);

        for (int i = 0; i < l_px_pose.size(); ++i)
        {
            const Vector& l_cur_px_p = l_px_pose[i];
            const Vector& l_pre_px_p = l_px_pose[(3 + i) % 4];

            const Vector& r_cur_px_p = r_px_pose[i];
            const Vector& r_pre_px_p = r_px_pose[(3 + i) % 4];

            const Vector& l_cur_px_g = l_px_goal_[i];
            const Vector& l_pre_px_g = l_px_goal_[(3 + i) % 4];

            const Vector& r_cur_px_g = r_px_goal_[i];
            const Vector& r_pre_px_g = r_px_goal_[(3 + i) % 4];


            cv::line(l_img, cv::Point(l_pre_px_p[0], l_pre_px_p[1]), cv::Point(l_cur_px_p[0], l_cur_px_p[1]), color[3], 4, cv::LineTypes::LINE_AA);
            cv::line(r_img, cv::Point(r_pre_px_p[0], r_pre_px_p[1]), cv::Point(r_cur_px_p[0], r_cur_px_p[1]), color[3], 4, cv::LineTypes::LINE_AA);

            cv::line(l_img, cv::Point(l_pre_px_g[0], l_pre_px_g[1]), cv::Point(l_cur_px_g[0], l_cur_px_g[1]), color[3], 4, cv::LineTypes::LINE_AA);
            cv::line(r_img, cv::Point(r_pre_px_g[0], r_pre_px_g[1]), cv::Point(r_cur_px_g[0], r_cur_px_g[1]), color[3], 4, cv::LineTypes::LINE_AA);

            if (i == 2 || i == 3)
            {
                cv::circle(l_img, cv::Point(l_cur_px_p[0], l_cur_px_p[1]), 4, color[i - 1], 4);
                cv::circle(r_img, cv::Point(r_cur_px_p[0], r_cur_px_p[1]), 4, color[i - 1], 4);

                cv::circle(l_img, cv::Point(l_cur_px_g[0], l_cur_px_g[1]), 4, color[i - 1], 4);
                cv::circle(r_img, cv::Point(r_cur_px_g[0], r_cur_px_g[1]), 4, color[i - 1], 4);
            }
        }


        Vector l_cur_px_ee = getPixelFromPoint(CamSel::left,  cat(eepose_copy_left.subVector(0, 2), 1.0));
        Vector r_cur_px_ee = getPixelFromPoint(CamSel::right, cat(eepose_copy_right.subVector(0, 2), 1.0));

        cv::circle(l_img, cv::Point(l_cur_px_ee[0], l_cur_px_ee[1]), 4, color[0], 4);
        cv::circle(r_img, cv::Point(r_cur_px_ee[0], r_cur_px_ee[1]), 4, color[0], 4);


        Vector l_cur_px_g = getPixelFromPoint(CamSel::left,  cat(goal_pose_.subVector(0, 2), 1.0));
        Vector r_cur_px_g = getPixelFromPoint(CamSel::right, cat(goal_pose_.subVector(0, 2), 1.0));

        cv::circle(l_img, cv::Point(l_cur_px_g[0], l_cur_px_g[1]), 4, color[0], 4);
        cv::circle(r_img, cv::Point(r_cur_px_g[0], r_cur_px_g[1]), 4, color[0], 4);
        
        port_image_left_out_.write();
        port_image_right_out_.write();
        /* *** *** ***  *** *** *** *** *** *** *** *** *** *** */


        /* CHECK FOR GOAL */
        mtx_px_des_.lock();

        bool is_pos_done    = checkVisualServoingStatus(px_ee_cur_position,    px_tol_);
        bool is_orient_done = checkVisualServoingStatus(px_ee_cur_orientation, px_tol_);

        mtx_px_des_.unlock();


        if (op_mode_ == OperatingMode::position)
            vs_goal_reached_ = is_pos_done;
        else if (op_mode_ == OperatingMode::orientation)
            vs_goal_reached_ = is_orient_done;
        else if (op_mode_ == OperatingMode::pose)
            vs_goal_reached_ = is_pos_done && is_orient_done;


        if (vs_goal_reached_)
        {
            yInfoVerbose("");
            yInfoVerbose("*** Goal reached! ***");
            yInfoVerbose("Desired goal pixels = "                   + px_des_.toString());
            yInfoVerbose("Current position controlled pixels = "    + px_ee_cur_position.toString());
            yInfoVerbose("Current orientation controlled pixels = " + px_ee_cur_orientation.toString());
            yInfoVerbose("*** ------------- ***");
            yInfoVerbose("");
        }
        else
        {
            /* EVALUATING ERRORS */
            mtx_px_des_.lock();

            Vector e_position               = px_des_ - px_ee_cur_position;
            Matrix inv_jacobian_position    = pinv(jacobian_position);

            Vector e_orientation            = px_des_ - px_ee_cur_orientation;
            Matrix inv_jacobian_orientation = pinv(jacobian_orientation);

            yInfoVerbose("Position error in pixels    = [" + e_position.toString()    + "]");
            yInfoVerbose("Orientation error in pixels = [" + e_orientation.toString() + "]");

            mtx_px_des_.unlock();


            /* EVALUATING CONTROL VELOCITIES */
            mtx_H_eye_cam_.lock();

            Vector vel_x = zeros(3);
            Vector vel_o = zeros(3);
            for (int i = 0; i < inv_jacobian_position.cols(); ++i)
            {
                Vector delta_vel_position    = inv_jacobian_position.getCol(i)    * e_position(i);
                Vector delta_vel_orientation = inv_jacobian_orientation.getCol(i) * e_orientation(i);

                if (i == 1 || i == 4 || i == 7 || i == 10)
                {
                    vel_x += r_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel_position.subVector(0, 2);
                    vel_o += r_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel_orientation.subVector(3, 5);
                }
                else
                {
                    vel_x += l_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel_position.subVector(0, 2);
                    vel_o += l_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel_orientation.subVector(3, 5);
                }
            }

            mtx_H_eye_cam_.unlock();

            double ang = norm(vel_o);
            vel_o /= ang;
            vel_o.push_back(ang);

            yInfoVerbose("Translational velocity = [" + vel_x.toString() + "]");
            yInfoVerbose("Orientation velocity   = [" + vel_o.toString() + "]");


            /* ENFORCE TRANSLATIONAL VELOCITY BOUNDS */
            for (size_t i = 0; i < vel_x.length(); ++i)
            {
                if (!std::isnan(vel_x[i]) && !std::isinf(vel_x[i]))
                    vel_x[i] = sign(vel_x[i]) * std::min(max_x_dot_, std::fabs(vel_x[i]));
                else
                {
                    vel_x = Vector(3, 0.0);
                    break;
                }
            }
            yInfoVerbose("Bounded translational velocity = [" + vel_x.toString() + "]");


            /* ENFORCE ROTATIONAL VELOCITY BOUNDS */
            if (!std::isnan(vel_o[0]) && !std::isinf(vel_x[0]) &&
                !std::isnan(vel_o[1]) && !std::isinf(vel_x[1]) &&
                !std::isnan(vel_o[2]) && !std::isinf(vel_x[2]) &&
                !std::isnan(vel_o[3]) && !std::isinf(vel_x[3]))
                vel_o[3] = sign(vel_o[3]) * std::min(max_o_dot_, std::fabs(vel_o[3]));
            else
                vel_o = Vector(4, 0.0);
            yInfoVerbose("Bounded orientation velocity = [" + vel_o.toString() + "]");


            /* VISUAL CONTROL LAW */
            if (!checkVisualServoingStatus(px_ee_cur_position, K_x_tol_))
            {
                vel_x *= K_x_[0];
                if (K_x_hysteresis_)
                {
                    K_x_hysteresis_ = false;

                    K_x_tol_ -= 5.0;
                }
            }
            else
            {
                vel_x *= K_x_[1];
                if (!K_x_hysteresis_)
                {
                    K_x_hysteresis_ = true;

                    K_x_tol_ += 5.0;
                }
            }

            if (!checkVisualServoingStatus(px_ee_cur_orientation, K_o_tol_))
            {
                vel_o(3) *= K_o_[0];
                if (K_o_hysteresis_)
                {
                    K_o_hysteresis_ = false;

                    K_o_tol_ -= 5.0;
                }
            }
            else
            {
                vel_o(3) *= K_o_[1];
                if (!K_o_hysteresis_)
                {
                    K_o_hysteresis_ = true;

                    K_o_tol_ += 5.0;
                }
            }


            if (!sim_)
            {
                /* COMMAND END-EFFECTOR WITH THE CARTESIAN CONTROLLER */
                if (op_mode_ == OperatingMode::position)
                    itf_rightarm_cart_->setTaskVelocities(vel_x, Vector(4, 0.0));
                else if (op_mode_ == OperatingMode::orientation)
                    itf_rightarm_cart_->setTaskVelocities(Vector(3, 0.0), vel_o);
                else if (op_mode_ == OperatingMode::pose)
                    itf_rightarm_cart_->setTaskVelocities(vel_x, vel_o);


                /* WAIT FOR SOME MOTION */
                Time::delay(Ts_);


                /* UPDATE END-EFFECTOR POSE FOR THE LEFT EYE */
                endeffector_pose = port_pose_left_in_.read(true);
                eepose_copy_left = *endeffector_pose;

                /* UPDATE END-EFFECTOR POSE FOR THE RIGHT EYE */
                endeffector_pose = port_pose_right_in_.read(true);
                eepose_copy_right = *endeffector_pose;
            }
            else
            {
                /* SIMULATE END-EFFECTOR COMMAND AND UPDATE ITS POSE */
                Matrix l_R = axis2dcm(eepose_copy_left.subVector(3, 6));
                Matrix r_R = axis2dcm(eepose_copy_right.subVector(3, 6));


                vel_o[3] *= Ts_;
                l_R = axis2dcm(vel_o) * l_R;
                r_R = axis2dcm(vel_o) * r_R;


                Vector l_new_o = dcm2axis(l_R);
                Vector r_new_o = dcm2axis(r_R);


                if (op_mode_ == OperatingMode::position)
                {
                    eepose_copy_left.setSubvector(0, eepose_copy_left.subVector(0, 2)  + vel_x * Ts_);
                    eepose_copy_right.setSubvector(0, eepose_copy_right.subVector(0, 2)  + vel_x * Ts_);
                }
                else if (op_mode_ == OperatingMode::orientation)
                {
                    eepose_copy_left.setSubvector(3, l_new_o);
                    eepose_copy_right.setSubvector(3, r_new_o);
                }
                else if (op_mode_ == OperatingMode::pose)
                {
                    eepose_copy_left.setSubvector(0, eepose_copy_left.subVector(0, 2)  + vel_x * Ts_);
                    eepose_copy_right.setSubvector(0, eepose_copy_right.subVector(0, 2)  + vel_x * Ts_);
                    eepose_copy_left.setSubvector(3, l_new_o);
                    eepose_copy_right.setSubvector(3, r_new_o);
                }
            }
        }
    }
}


void VisualServoingServer::robustImageBasedVisualServoControl()
{
    /* VARIABLE DEFINITIONS */
    Vector* endeffector_pose;
    Vector  eepose_copy_left(7);
    Vector  eepose_copy_right(7);

    std::vector<Vector> l_px_pose;
    std::vector<Vector> r_px_pose;

    Vector px_ee_cur_pose = zeros(12);
    Matrix jacobian_pose  = zeros(12, 6);
    Vector px_ee_cur_goal = zeros(12);
    Matrix jacobian_goal  = zeros(12, 6);


    /* GET THE INITIAL END-EFFECTOR POSE FOR THE LEFT EYE */
    endeffector_pose = port_pose_left_in_.read(true);
    eepose_copy_left = *endeffector_pose;

    yInfoVerbose("Got [" + eepose_copy_left.toString() + "] end-effector pose for the left eye.");

    /* GET THE INITIAL END-EFFECTOR POSE FOR THE RIGHT EYE */
    endeffector_pose = port_pose_right_in_.read(true);
    eepose_copy_right = *endeffector_pose;

    yInfoVerbose("Got [" + eepose_copy_right.toString() + "] end-effector pose for the right eye.");

    while (!isStopping() && !vs_goal_reached_)
    {
        yInfoVerbose("Desired goal pixels = [" + px_des_.toString() + "]");


        /* EVALUATING CONTROL POINTS */
        l_px_pose = getControlPixelsFromPose(eepose_copy_left,  CamSel::left,  PixelControlMode::all);
        r_px_pose = getControlPixelsFromPose(eepose_copy_right, CamSel::right, PixelControlMode::all);


        /* POSE: FEATURES AND JACOBIAN */
        getCurrentStereoFeaturesAndJacobian(l_px_pose, r_px_pose,
                                            px_ee_cur_pose, jacobian_pose);

        yInfoVerbose("Pose controlled pixels = ["   + px_ee_cur_pose.toString() + "]");
        yInfoVerbose("Pose image Jacobian    = [\n" + jacobian_pose.toString()  + "]");

        /* GOAL: FEATURES AND JACOBIAN */
        getCurrentStereoFeaturesAndJacobian(l_px_goal_, r_px_goal_,
                                            px_ee_cur_goal, jacobian_goal);

        yInfoVerbose("Goal 'controlled' pixels = ["   + px_ee_cur_goal.toString() + "]");
        yInfoVerbose("Desired goal pixels      = ["   + px_des_.toString()        + "]");
        yInfoVerbose("Goal image Jacobian      = [\n" + jacobian_goal.toString()  + "]");


        /* *** *** *** DEBUG OUTPUT - TO BE DELETED *** *** *** */
        std::vector<cv::Scalar> color;
        color.emplace_back(cv::Scalar(255,   0,   0));
        color.emplace_back(cv::Scalar(  0, 255,   0));
        color.emplace_back(cv::Scalar(  0,   0, 255));
        color.emplace_back(cv::Scalar(255, 127,  51));


        /* Left eye end-effector superimposition */
        ImageOf<PixelRgb>* l_imgin  = port_image_left_in_.read(true);
        ImageOf<PixelRgb>& l_imgout = port_image_left_out_.prepare();
        l_imgout = *l_imgin;
        cv::Mat l_img = cv::cvarrToMat(l_imgout.getIplImage());

        /* Right eye end-effector superimposition */
        ImageOf<PixelRgb>* r_imgin  = port_image_right_in_.read(true);
        ImageOf<PixelRgb>& r_imgout = port_image_right_out_.prepare();
        r_imgout = *r_imgin;
        cv::Mat r_img = cv::cvarrToMat(r_imgout.getIplImage());


        /* Plot points and lines */
        for (int i = 0; i < l_px_pose.size(); ++i)
        {
            const Vector& l_cur_px_p = l_px_pose[i];
            const Vector& l_pre_px_p = l_px_pose[(3 + i) % 4];

            const Vector& r_cur_px_p = r_px_pose[i];
            const Vector& r_pre_px_p = r_px_pose[(3 + i) % 4];

            const Vector& l_cur_px_g = l_px_goal_[i];
            const Vector& l_pre_px_g = l_px_goal_[(3 + i) % 4];

            const Vector& r_cur_px_g = r_px_goal_[i];
            const Vector& r_pre_px_g = r_px_goal_[(3 + i) % 4];


            cv::line(l_img, cv::Point(l_pre_px_p[0], l_pre_px_p[1]), cv::Point(l_cur_px_p[0], l_cur_px_p[1]), color[3], 4, cv::LineTypes::LINE_AA);
            cv::line(r_img, cv::Point(r_pre_px_p[0], r_pre_px_p[1]), cv::Point(r_cur_px_p[0], r_cur_px_p[1]), color[3], 4, cv::LineTypes::LINE_AA);

            cv::line(l_img, cv::Point(l_pre_px_g[0], l_pre_px_g[1]), cv::Point(l_cur_px_g[0], l_cur_px_g[1]), color[3], 4, cv::LineTypes::LINE_AA);
            cv::line(r_img, cv::Point(r_pre_px_g[0], r_pre_px_g[1]), cv::Point(r_cur_px_g[0], r_cur_px_g[1]), color[3], 4, cv::LineTypes::LINE_AA);

            if (i == 2 || i == 3)
            {
                cv::circle(l_img, cv::Point(l_cur_px_p[0], l_cur_px_p[1]), 4, color[i - 1], 4);
                cv::circle(r_img, cv::Point(r_cur_px_p[0], r_cur_px_p[1]), 4, color[i - 1], 4);

                cv::circle(l_img, cv::Point(l_cur_px_g[0], l_cur_px_g[1]), 4, color[i - 1], 4);
                cv::circle(r_img, cv::Point(r_cur_px_g[0], r_cur_px_g[1]), 4, color[i - 1], 4);
            }
        }


        Vector l_cur_px_ee = getPixelFromPoint(CamSel::left,  cat(eepose_copy_left.subVector(0, 2), 1.0));
        Vector r_cur_px_ee = getPixelFromPoint(CamSel::right, cat(eepose_copy_right.subVector(0, 2), 1.0));

        cv::circle(l_img, cv::Point(l_cur_px_ee[0], l_cur_px_ee[1]), 4, color[0], 4);
        cv::circle(r_img, cv::Point(r_cur_px_ee[0], r_cur_px_ee[1]), 4, color[0], 4);


        Vector l_cur_px_g = getPixelFromPoint(CamSel::left,  cat(goal_pose_.subVector(0, 2), 1.0));
        Vector r_cur_px_g = getPixelFromPoint(CamSel::right, cat(goal_pose_.subVector(0, 2), 1.0));

        cv::circle(l_img, cv::Point(l_cur_px_g[0], l_cur_px_g[1]), 4, color[0], 4);
        cv::circle(r_img, cv::Point(r_cur_px_g[0], r_cur_px_g[1]), 4, color[0], 4);

        port_image_left_out_.write();
        port_image_right_out_.write();
        /* *** *** ***  *** *** *** *** *** *** *** *** *** *** */


        /* CHECK FOR GOAL */
        mtx_px_des_.lock();

        vs_goal_reached_ = checkVisualServoingStatus(px_ee_cur_pose, px_tol_);

        mtx_px_des_.unlock();


        if (vs_goal_reached_)
        {
            yInfoVerbose("");
            yInfoVerbose("*** Goal reached! ***");
            yInfoVerbose("Desired goal pixels = "                + px_des_.toString());
            yInfoVerbose("Current position controlled pixels = " + px_ee_cur_pose.toString());
            yInfoVerbose("*** ------------- ***");
            yInfoVerbose("");
        }
        else
        {
            /* EVALUATING ERRORS */
            mtx_px_des_.lock();

            Vector e            = px_des_ - px_ee_cur_pose;
            Matrix inv_jacobian = pinv(0.5 * (jacobian_pose + jacobian_goal));

            yInfoVerbose("Position error in pixels = [" + e.toString() + "]");

            mtx_px_des_.unlock();


            /* EVALUATING CONTROL VELOCITIES */
            mtx_H_eye_cam_.lock();

            Vector vel_x = zeros(3);
            Vector vel_o = zeros(3);
            for (int i = 0; i < inv_jacobian.cols(); ++i)
            {
                Vector delta_vel = inv_jacobian.getCol(i) * e(i);

                if (i == 1 || i == 4 || i == 7 || i == 10)
                {
                    vel_x += r_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel.subVector(0, 2);
                    vel_o += r_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel.subVector(3, 5);
                }
                else
                {
                    vel_x += l_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel.subVector(0, 2);
                    vel_o += l_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel.subVector(3, 5);
                }
            }

            mtx_H_eye_cam_.unlock();

            double ang = norm(vel_o);
            vel_o /= ang;
            vel_o.push_back(ang);

            yInfoVerbose("Translational velocity = [" + vel_x.toString() + "]");
            yInfoVerbose("Orientation velocity = [" + vel_o.toString() + "]");


            /* ENFORCE TRANSLATIONAL VELOCITY BOUNDS */
            for (size_t i = 0; i < vel_x.length(); ++i)
            {
                if (!std::isnan(vel_x[i]) && !std::isinf(vel_x[i]))
                    vel_x[i] = sign(vel_x[i]) * std::min(max_x_dot_, std::fabs(vel_x[i]));
                else
                {
                    vel_x = Vector(3, 0.0);
                    break;
                }
            }
            yInfoVerbose("Bounded translational velocity = [" + vel_x.toString() + "]");


            /* ENFORCE ROTATIONAL VELOCITY BOUNDS */
            if (!std::isnan(vel_o[0]) && !std::isinf(vel_x[0]) &&
                !std::isnan(vel_o[1]) && !std::isinf(vel_x[1]) &&
                !std::isnan(vel_o[2]) && !std::isinf(vel_x[2]) &&
                !std::isnan(vel_o[3]) && !std::isinf(vel_x[3]))
                vel_o[3] = sign(vel_o[3]) * std::min(max_o_dot_, std::fabs(vel_o[3]));
            else
                vel_o = Vector(4, 0.0);
            yInfoVerbose("Bounded orientation velocity = [" + vel_o.toString() + "]");


            /* VISUAL CONTROL LAW */
            if (!checkVisualServoingStatus(px_ee_cur_pose, K_x_tol_))
            {
                vel_x *= K_x_[0];
                if (K_x_hysteresis_)
                {
                    K_x_hysteresis_ = false;

                    K_x_tol_ -= 5.0;
                }
            }
            else
            {
                vel_x *= K_x_[1];
                if (!K_x_hysteresis_)
                {
                    K_x_hysteresis_ = true;

                    K_x_tol_ += 5.0;
                }
            }

            if (!checkVisualServoingStatus(px_ee_cur_goal, K_o_tol_))
            {
                vel_o(3) *= K_o_[0];
                if (K_o_hysteresis_)
                {
                    K_o_hysteresis_ = false;

                    K_o_tol_ -= 5.0;
                }
            }
            else
            {
                vel_o(3) *= K_o_[1];
                if (!K_o_hysteresis_)
                {
                    K_o_hysteresis_ = true;

                    K_o_tol_ += 5.0;
                }
            }


            if (!sim_)
            {
                /* COMMAND END-EFFECTOR WITH THE CARTESIAN CONTROLLER */
                itf_rightarm_cart_->setTaskVelocities(vel_x, vel_o);


                /* WAIT FOR SOME MOTION */
                Time::delay(Ts_);


                /* UPDATE END-EFFECTOR POSE FOR THE LEFT EYE */
                endeffector_pose = port_pose_left_in_.read(true);
                eepose_copy_left = *endeffector_pose;

                /* UPDATE END-EFFECTOR POSE FOR THE RIGHT EYE */
                endeffector_pose = port_pose_right_in_.read(true);
                eepose_copy_right = *endeffector_pose;
            }
            else
            {
                Matrix l_R = axis2dcm(eepose_copy_left.subVector(3, 6));
                Matrix r_R = axis2dcm(eepose_copy_right.subVector(3, 6));


                vel_o[3] *= Ts_;
                l_R = axis2dcm(vel_o) * l_R;
                r_R = axis2dcm(vel_o) * r_R;


                Vector l_new_o = dcm2axis(l_R);
                Vector r_new_o = dcm2axis(r_R);


                /* SIMULATE END-EFFECTOR COMMAND AND UPDATE ITS POSE */
                eepose_copy_left.setSubvector(0, eepose_copy_left.subVector(0, 2)  + vel_x * Ts_);
                eepose_copy_right.setSubvector(0, eepose_copy_right.subVector(0, 2)  + vel_x * Ts_);
                eepose_copy_left.setSubvector(3, l_new_o);
                eepose_copy_right.setSubvector(3, r_new_o);
            }
        }
    }
}


bool VisualServoingServer::setRightArmCartesianController()
{
    Property rightarm_cartesian_options;
    rightarm_cartesian_options.put("device", "cartesiancontrollerclient");
    rightarm_cartesian_options.put("local",  "/visualservoing/cart_right_arm");
    rightarm_cartesian_options.put("remote", "/" + robot_name_ + "/cartesianController/right_arm");

    rightarm_cartesian_driver_.open(rightarm_cartesian_options);
    if (rightarm_cartesian_driver_.isValid())
    {
        rightarm_cartesian_driver_.view(itf_rightarm_cart_);
        if (!itf_rightarm_cart_)
        {
            yErrorVerbose("Error getting ICartesianControl interface!");
            return false;
        }
        yInfoVerbose("cartesiancontrollerclient succefully opened.");
    }
    else
    {
        yErrorVerbose("Error opening cartesiancontrollerclient device!");
        return false;
    }


    if (!itf_rightarm_cart_->storeContext(&ctx_remote_cart_))
    {
        yErrorVerbose("Error storing remote ICartesianControl context!");
        return false;
    }
    yInfoVerbose("Remote ICartesianControl context stored.");


    if (!itf_rightarm_cart_->setTrajTime(traj_time_))
    {
        yErrorVerbose("Error setting ICartesianControl trajectory time!");
        return false;
    }
    yInfoVerbose("Succesfully set ICartesianControl trajectory time.");


    if (!itf_rightarm_cart_->setInTargetTol(0.01))
    {
        yErrorVerbose("Error setting ICartesianControl target tolerance!");
        return false;
    }
    yInfoVerbose("Succesfully set ICartesianControl target tolerance.");


    if (!setTorsoDOF())
    {
        yErrorVerbose("Unable to change torso DOF!");
        return false;
    }
    yInfoVerbose("Succesfully changed torso DOF.");


    if (!itf_rightarm_cart_->storeContext(&ctx_local_cart_))
    {
        yErrorVerbose("Error storing local ICartesianControl context!");
        return false;
    }
    yInfoVerbose("Local ICartesianControl context stored.");


    if (!itf_rightarm_cart_->restoreContext(ctx_remote_cart_))
    {
        yErrorVerbose("Error restoring remote ICartesianControl context!");
        return false;
    }
    yInfoVerbose("Remote ICartesianControl context restored.");


    return true;
}


bool VisualServoingServer::setGazeController()
{
    Property gaze_option;
    gaze_option.put("device", "gazecontrollerclient");
    gaze_option.put("local",  "/visualservoing/gaze");
    gaze_option.put("remote", "/iKinGazeCtrl");

    gaze_driver_.open(gaze_option);
    if (gaze_driver_.isValid())
    {
        gaze_driver_.view(itf_gaze_);
        if (!itf_gaze_)
        {
            yErrorVerbose("Error getting IGazeControl interface!");
            return false;
        }
    }
    else
    {
        yErrorVerbose("Gaze control device not available!");
        return false;
    }

    return true;
}


bool VisualServoingServer::setCommandPort()
{
    yInfoVerbose("Opening RPC command port.");

    if (!port_rpc_command_.open("/visualservoing/cmd:i"))
    {
        yErrorVerbose("Cannot open the RPC server command port!");
        return false;
    }
    if (!yarp().attachAsServer(port_rpc_command_))
    {
        yErrorVerbose("Cannot attach the RPC server command port!");
        return false;
    }


    if (!port_rpc_tracker_left_.open("/visualservoing/toTracker/left/cmd:o"))
    {
        yErrorVerbose("Cannot open the RPC command port to left camera tracker!");
        return false;
    }
    if (!port_rpc_tracker_right_.open("/visualservoing/toTracker/right/cmd:o"))
    {
        yErrorVerbose("Cannot open the RPC command port to right camera tracker!");
        return false;
    }


    yInfoVerbose("RPC command port opened and attached. Ready to recieve commands.");

    return true;
}


bool VisualServoingServer::setTorsoDOF()
{
    Vector curDOF;
    itf_rightarm_cart_->getDOF(curDOF);
    yInfoVerbose("Old DOF: [" + curDOF.toString(0) + "].");

    yInfoVerbose("Setting iCub to use torso DOF.");

    Vector newDOF(curDOF);
    newDOF[0] = 1;
    newDOF[1] = 0;
    newDOF[2] = 1;
    if (!itf_rightarm_cart_->setDOF(newDOF, curDOF))
        return false;

    yInfoVerbose("New DOF: Yaw " + ConstString(curDOF[0] == 0 ? "blocked, " : "actuated, ") +
                 "Roll "         + ConstString(curDOF[1] == 0 ? "blocked, " : "actuated, ") +
                 "Pitch "        + ConstString(curDOF[2] == 0 ? "blocked."  : "actuated." ));

    return true;
}


bool VisualServoingServer::unsetTorsoDOF()
{
    Vector curDOF;
    itf_rightarm_cart_->getDOF(curDOF);
    yInfoVerbose("Old DOF: [" + curDOF.toString(0) + "].");

    yInfoVerbose("Setting iCub to block torso DOF.");

    Vector newDOF(curDOF);
    newDOF[0] = 0;
    newDOF[1] = 0;
    newDOF[2] = 0;
    if (!itf_rightarm_cart_->setDOF(newDOF, curDOF))
        return false;

    yInfoVerbose("New DOF: Yaw " + ConstString(curDOF[0] == 0 ? "blocked, " : "actuated, ") +
                 "Roll "         + ConstString(curDOF[1] == 0 ? "blocked, " : "actuated, ") +
                 "Pitch "        + ConstString(curDOF[2] == 0 ? "blocked."  : "actuated." ));

    return true;
}


std::vector<Vector> VisualServoingServer::getPixelsFromPose(const Vector& pose, const CamSel& cam)
{
    yAssert(pose.length() == 7);
    yAssert(cam == CamSel::left || cam == CamSel::right);


    std::vector<Vector> pt_from_pose = getControlPointsFromPose(pose);

    std::vector<Vector> px_from_pose;
    for (const Vector& v : pt_from_pose)
        px_from_pose.emplace_back(getPixelFromPoint(cam, v));

    return px_from_pose;
}


std::vector<Vector> VisualServoingServer::getControlPixelsFromPose(const Vector& pose, const CamSel& cam, const PixelControlMode& mode)
{
    yAssert(pose.length() == 7);
    yAssert(cam == CamSel::left || cam == CamSel::right);
    yAssert(mode == PixelControlMode::all || mode == PixelControlMode::x || mode == PixelControlMode::o);


    Vector control_pose = pose;
    if (mode == PixelControlMode::x)
        control_pose.setSubvector(3, goal_pose_.subVector(3, 6));
    else if (mode == PixelControlMode::o)
        control_pose.setSubvector(0, goal_pose_.subVector(0, 2));

    std::vector<Vector> control_pt_from_pose = getControlPointsFromPose(control_pose);

    std::vector<Vector> control_px_from_pose;
    for (const Vector& v : control_pt_from_pose)
        control_px_from_pose.emplace_back(getControlPixelFromPoint(cam, v));

    return control_px_from_pose;
}


std::vector<Vector> VisualServoingServer::getControlPointsFromPose(const Vector& pose)
{
    Vector ee_x = pose.subVector(0, 2);
    ee_x.push_back(1.0);

    Vector ee_o = pose.subVector(3, 6);


    Matrix H_ee_to_root = axis2dcm(ee_o);
    H_ee_to_root.setCol(3, ee_x);


    Vector p = zeros(4);
    std::vector<Vector> control_pt_from_pose;

    p(0) =  0;
    p(1) = -0.015;
    p(2) =  0;
    p(3) =  1.0;
    control_pt_from_pose.emplace_back(H_ee_to_root * p);

    p(0) = 0;
    p(1) = 0.015;
    p(2) = 0;
    p(3) = 1.0;
    control_pt_from_pose.emplace_back(H_ee_to_root * p);

    p(0) = -0.035;
    p(1) =  0.015;
    p(2) =  0;
    p(3) =  1.0;
    control_pt_from_pose.emplace_back(H_ee_to_root * p);

    p(0) = -0.035;
    p(1) = -0.015;
    p(2) =  0;
    p(3) =  1.0;
    control_pt_from_pose.emplace_back(H_ee_to_root * p);


    return control_pt_from_pose;
}


Vector VisualServoingServer::getPixelFromPoint(const CamSel& cam, const Vector& p) const
{
    return getControlPixelFromPoint(cam, p).subVector(0, 1);
}


Vector VisualServoingServer::getControlPixelFromPoint(const CamSel& cam, const Vector& p) const
{
    yAssert(cam == CamSel::left || cam == CamSel::right);
    yAssert(p.size() == 4);


    Vector px;

    if (cam == CamSel::left)
        px = l_H_r_to_cam_ * p;
    else if (cam == CamSel::right)
        px = r_H_r_to_cam_ * p;

    px[0] /= px[2];
    px[1] /= px[2];

    return px;
}


void VisualServoingServer::getCurrentStereoFeaturesAndJacobian(const std::vector<Vector>& left_px,  const std::vector<Vector>& right_px,
                                                               Vector& features, Matrix& jacobian)
{
    yAssert(left_px.size() == right_px.size());

    if (features.length() != 12)
        features.resize(12);

    if (jacobian.rows() != 12 || jacobian.cols() != 6)
        jacobian.resize(12, 6);


    auto iter_left_px   = left_px.cbegin();
    auto iter_right_px  = right_px.cbegin();
    unsigned int offset = 0;
    while (iter_left_px != left_px.cend() && iter_right_px != right_px.cend())
    {
        const Vector& l_v = (*iter_left_px);
        const Vector& r_v = (*iter_right_px);

        /* FEATURES */
        features[3 * offset]     = l_v[0];  /* l_u_xi */
        features[3 * offset + 1] = r_v[0];  /* r_u_xi */
        features[3 * offset + 2] = l_v[1];  /* l_v_xi */

        /* JACOBIAN */
        jacobian.setRow(3 * offset,      getJacobianU(CamSel::left,  l_v));
        jacobian.setRow(3 * offset + 1,  getJacobianU(CamSel::right, r_v));
        jacobian.setRow(3 * offset + 2,  getJacobianV(CamSel::left,  l_v));

        ++iter_left_px;
        ++iter_right_px;
        ++offset;
    }
}


Vector VisualServoingServer::getJacobianU(const CamSel& cam, const Vector& px)
{
    Vector jacobian = zeros(6);

    if (cam == CamSel::left)
    {
        jacobian(0) = l_proj_(0, 0) / px(2);
        jacobian(2) = - (px(0) - l_proj_(0, 2)) / px(2);
        jacobian(3) = - ((px(0) - l_proj_(0, 2)) * (px(1) - l_proj_(1, 2))) / l_proj_(1, 1);
        jacobian(4) = (pow(l_proj_(0, 0), 2.0) + pow(px(0) - l_proj_(0, 2), 2.0)) / l_proj_(0, 0);
        jacobian(5) = - l_proj_(0, 0) / l_proj_(1, 1) * (px(1) - l_proj_(1, 2));
    }
    else if (cam == CamSel::right)
    {
        jacobian(0) = r_proj_(0, 0) / px(2);
        jacobian(2) = - (px(0) - r_proj_(0, 2)) / px(2);
        jacobian(3) = - ((px(0) - r_proj_(0, 2)) * (px(1) - r_proj_(1, 2))) / r_proj_(1, 1);
        jacobian(4) = (pow(r_proj_(0, 0), 2.0) + pow(px(0) - r_proj_(0, 2), 2.0)) / r_proj_(0, 0);
        jacobian(5) = - r_proj_(0, 0) / r_proj_(1, 1) * (px(1) - r_proj_(1, 2));
    }

    return jacobian;
}


Vector VisualServoingServer::getJacobianV(const CamSel& cam, const Vector& px)
{
    Vector jacobian = zeros(6);

    if (cam == CamSel::left)
    {
        jacobian(1) = l_proj_(1, 1) / px(2);
        jacobian(2) = - (px(1) - l_proj_(1, 2)) / px(2);
        jacobian(3) = - (pow(l_proj_(1, 1), 2.0) + pow(px(1) - l_proj_(1, 2), 2.0)) / l_proj_(1, 1);
        jacobian(4) = ((px(0) - l_proj_(0, 2)) * (px(1) - l_proj_(1, 2))) / l_proj_(0, 0);
        jacobian(5) = l_proj_(1, 1) / l_proj_(0, 0) * (px(0) - l_proj_(0, 2));
    }
    else if (cam == CamSel::right)
    {
        jacobian(1) = r_proj_(1, 1) / px(2);
        jacobian(2) = - (px(1) - r_proj_(1, 2)) / px(2);
        jacobian(3) = - (pow(r_proj_(1, 1), 2.0) + pow(px(1) - r_proj_(1, 2), 2.0)) / r_proj_(1, 1);
        jacobian(4) = ((px(0) - r_proj_(0, 2)) * (px(1) - r_proj_(1, 2))) / r_proj_(0, 0);
        jacobian(5) = r_proj_(1, 1) / r_proj_(0, 0) * (px(0) - r_proj_(0, 2));
    }

    return jacobian;
}


bool VisualServoingServer::setCameraTransformations()
{
    std::lock_guard<std::mutex> lock(mtx_H_eye_cam_);

    Vector left_eye_x;
    Vector left_eye_o;
    if (!itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o))
        return false;

    Vector right_eye_x;
    Vector right_eye_o;
    if (!itf_gaze_->getRightEyePose(right_eye_x, right_eye_o))
        return false;

    yInfoVerbose("left_eye_o = ["  + left_eye_o.toString()  + "]");
    yInfoVerbose("right_eye_o = [" + right_eye_o.toString() + "]");


    l_H_eye_to_r_ = axis2dcm(left_eye_o);
    left_eye_x.push_back(1.0);
    l_H_eye_to_r_.setCol(3, left_eye_x);
    Matrix l_H_r_to_eye = SE3inv(l_H_eye_to_r_);

    r_H_eye_to_r_ = axis2dcm(right_eye_o);
    right_eye_x.push_back(1.0);
    r_H_eye_to_r_.setCol(3, right_eye_x);
    Matrix r_H_r_to_eye = SE3inv(r_H_eye_to_r_);

    l_H_r_to_cam_ = l_proj_ * l_H_r_to_eye;
    r_H_r_to_cam_ = r_proj_ * r_H_r_to_eye;

    return true;
}


bool VisualServoingServer::setPoseGoal(const yarp::sig::Vector& goal_x, const yarp::sig::Vector& goal_o)
{
    goal_pose_.setSubvector(0, goal_x);
    goal_pose_.setSubvector(3, goal_o);

    return true;
}


bool VisualServoingServer::setPixelGoal(const std::vector<Vector>& l_px_goal, const std::vector<Vector>& r_px_goal)
{
    std::lock_guard<std::mutex> lock(mtx_px_des_);


    l_px_goal_ = l_px_goal;
    r_px_goal_ = r_px_goal;

    for (unsigned int i = 0; i < l_px_goal_.size(); ++i)
    {
        yInfoVerbose("Left goal pixels #"  + std::to_string(i) + " = [" + l_px_goal_[i].toString() + "]");
        yInfoVerbose("Right goal pixels #" + std::to_string(i) + " = [" + r_px_goal_[i].toString() + "]");

        px_des_[3 * i]     = l_px_goal_[i][0];   /* l_u_xi */
        px_des_[3 * i + 1] = r_px_goal_[i][0];   /* r_u_xi */
        px_des_[3 * i + 2] = l_px_goal_[i][1];   /* l_v_xi */
    }

    yInfoVerbose("Desired goal pixels = ["  + px_des_.toString() + "]");

    return true;
}


void VisualServoingServer::backproc_UpdateVisualServoingParamters()
{
    yInfoVerbose("*** Started background process UpdateVisualServoingParamters ***");

    while (!is_stopping_backproc_update_vs_params)
    {
        Vector vec_x = goal_pose_.subVector(0, 2);
        Vector vec_o = goal_pose_.subVector(3, 6);

        std::vector<Vector> vec_px_l = getGoalPixelsFrom3DPose(vec_x, vec_o, CamSel::left);
        std::vector<Vector> vec_px_r = getGoalPixelsFrom3DPose(vec_x, vec_o, CamSel::right);

        if (vec_px_l.size() != 0 && vec_px_r.size() != 0)
            setPixelGoal(vec_px_l, vec_px_r);
        else
            yErrorVerbose("[BACKPROC] Could not update goal pixels from 3D pose.");
    }

    yInfoVerbose("*** Stopped background process UpdateVisualServoingParamters ***");
}


bool VisualServoingServer::checkVisualServoingStatus(const Vector& px_cur, const double tol)
{
    yAssert(px_cur.size() == 12);
    yAssert(tol > 0);

    return ((std::abs(px_des_(0) - px_cur(0)) < tol) && (std::abs(px_des_(1)  - px_cur(1))  < tol) && (std::abs(px_des_(2)  - px_cur(2))  < tol) &&
            (std::abs(px_des_(3) - px_cur(3)) < tol) && (std::abs(px_des_(4)  - px_cur(4))  < tol) && (std::abs(px_des_(5)  - px_cur(5))  < tol) &&
            (std::abs(px_des_(6) - px_cur(6)) < tol) && (std::abs(px_des_(7)  - px_cur(7))  < tol) && (std::abs(px_des_(8)  - px_cur(8))  < tol) &&
            (std::abs(px_des_(9) - px_cur(9)) < tol) && (std::abs(px_des_(10) - px_cur(10)) < tol) && (std::abs(px_des_(11) - px_cur(11)) < tol));
}
