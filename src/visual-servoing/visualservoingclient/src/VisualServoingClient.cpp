#include "VisualServoingClient.h"

#include <iCub/ctrl/minJerkCtrl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <yarp/math/Math.h>
#include <yarp/math/SVD.h>
#include <yarp/os/Network.h>
#include <yarp/os/Property.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/Time.h>

using namespace yarp::dev;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;
using namespace iCub::ctrl;


/* Ctors and Dtors */
VisualServoingClient::VisualServoingClient()
{
    yInfo("*** Invoked VisualServoingClient ctor ***");
    yInfo("*** VisualServoingClient constructed ***");
}


VisualServoingClient::~VisualServoingClient()
{
    yInfo("*** Invoked VisualServoingClient dtor ***");
    yInfo("*** VisualServoingClient destructed ***");
}


/* DeviceDriver overrides */
bool VisualServoingClient::open(Searchable &config)
{
    verbosity_ = config.check("verbosity", Value(false)).asBool();
    yInfo("|> Verbosity: " + ConstString(verbosity_? "ON" : "OFF"));


    yInfoVerbose("*** Configuring VisualServoingClient ***");

    local_ = config.check("local", Value("")).asString();
    yInfoVerbose("|> Local port prefix: " + local_);
    if (local_ == "")
    {
        yErrorVerbose("Invalid local port prefix name.");
        return false;
    }
    local_ += "/cmd:o";
    yInfoVerbose("|> Local port command name: " + local_);

    remote_ = config.check("remote", Value("")).asString();
    yInfoVerbose("|> Remote port name: " + remote_);
    if (remote_ == "")
    {
        yErrorVerbose("Invalid remote port prefix name.");
        return false;
    }
    remote_ += "/cmd:i";
    yInfoVerbose("|> Remote port command name: " + remote_);

    if (!port_rpc_command_.open(local_))
    {
        yErrorVerbose("Could not open " + local_ + " port.");
        return false;
    }

    if (!Network::connect(local_, remote_, "tcp", verbosity_))
    {
        yErrorVerbose("Could not connect to " + local_ + " remote port.");
        return false;
    }

    if (!visualservoing_control.yarp().attachAsClient(port_rpc_command_))
    {
        yErrorVerbose("Cannot attach the RPC client command port.");
        return false;
    }

    yInfoVerbose("*** VisualServoingClient configured! ***");

    return true;
}


bool VisualServoingClient::close()
{
    yInfoVerbose("*** Interrupting VisualServoingClient ***");

    yInfoVerbose("Interrupting ports.");
    port_rpc_command_.interrupt();

    yInfoVerbose("*** Interrupting VisualServoingClient done! ***");


    yInfoVerbose("*** Closing VisualServoingClient ***");

    yInfoVerbose("Closing ports.");
    port_rpc_command_.close();

    yInfoVerbose("*** Closing VisualServoingClient done! ***");


    return true;
}


/* IVisualServoing overrides */
bool VisualServoingClient::goToGoal(const Vector& vec_x, const Vector& vec_o)
{
    if (vec_x.size() != 3 || vec_o.size() != 4)
        return false;

    std::vector<double> std_vec_x(vec_x.data(), vec_x.data() + vec_x.size());
    std::vector<double> std_vec_o(vec_o.data(), vec_o.data() + vec_o.size());

    return visualservoing_control.go_to_pose_goal(std_vec_x, std_vec_o);
}


bool VisualServoingClient::goToGoal(const std::vector<Vector>& vec_px_l, const std::vector<Vector>& vec_px_r)
{
    if (vec_px_l.size() != vec_px_r.size())
        return false;

    const size_t num_points = vec_px_l.size();

    std::vector<std::vector<double>> std_vec_px_l(num_points);
    std::vector<std::vector<double>> std_vec_px_r(num_points);

    for (size_t i = 0; i < num_points; ++i)
    {
        const Vector&        yv_l = vec_px_l[i];
        std::vector<double>& v_l  = std_vec_px_l[i];

        const Vector&        yv_r = vec_px_r[i];
        std::vector<double>& v_r  = std_vec_px_r[i];

        v_l = std::vector<double>(yv_l.data(), yv_l.data() + yv_l.size());
        v_r = std::vector<double>(yv_r.data(), yv_r.data() + yv_r.size());
    }

    return visualservoing_control.go_to_px_goal(std_vec_px_l, std_vec_px_r);
}


bool VisualServoingClient::setModality(const std::string& mode)
{
    return visualservoing_control.set_modality(mode);
}


bool VisualServoingClient::setControlPoint(const yarp::os::ConstString& point)
{
    yWarningVerbose("*** Service setControlPoint is unimplemented. ***");

    return false;
}


bool VisualServoingClient::getVisualServoingInfo(yarp::os::Bottle& info)
{
    yWarningVerbose("*** Service getVisualServoingInfo is unimplemented. ***");

    return false;
}


bool VisualServoingClient::setGoToGoalTolerance(const double tol)
{
    return visualservoing_control.set_go_to_goal_tolerance(tol);
}


bool VisualServoingClient::checkVisualServoingController()
{
    return visualservoing_control.check_visual_servoing_controller();
}


bool VisualServoingClient::waitVisualServoingDone(const double period, const double timeout)
{
    return visualservoing_control.wait_visual_servoing_done(period, timeout);
}


bool VisualServoingClient::stopController()
{
    return visualservoing_control.stop_controller();
}


bool VisualServoingClient::setTranslationGain(const float K_x)
{
    return visualservoing_control.set_translation_gain(K_x);
}


bool VisualServoingClient::setMaxTranslationVelocity(const float max_x_dot)
{
    return visualservoing_control.set_max_translation_velocity(max_x_dot);
}


bool VisualServoingClient::setOrientationGain(const float K_o)
{
    return visualservoing_control.set_orientation_gain(K_o);
}


bool VisualServoingClient::setMaxOrientationVelocity(const float max_o_dot)
{
    return visualservoing_control.set_max_orientation_velocity(max_o_dot);
}


std::vector<Vector> VisualServoingClient::get3DPositionGoalFrom3DPose(const Vector& x, const Vector& o)
{
    std::vector<double> std_x(x.data(), x.data() + x.size());
    std::vector<double> std_o(o.data(), o.data() + o.size());

    std::vector<std::vector<double>> std_vec_goal_points = visualservoing_control.get_3D_position_goal_from_3D_pose(std_x, std_o);

    size_t num_points = std_vec_goal_points.size();
    std::vector<Vector> vec_goal_points(num_points);
    for (size_t i = 0; i < num_points; ++i)
        vec_goal_points[i] = Vector(std_vec_goal_points[i].size(), std_vec_goal_points[i].data());

    return vec_goal_points;
}


std::vector<Vector> VisualServoingClient::getPixelPositionGoalFrom3DPose(const Vector& x, const Vector& o, const CamSel& cam)
{
    std::vector<double> std_x(x.data(), x.data() + x.size());
    std::vector<double> std_o(o.data(), o.data() + o.size());

    std::vector<std::vector<double>> std_vec_goal_points = visualservoing_control.get_pixel_position_goal_from_3D_pose(std_x, std_o, (cam == CamSel::left ? "left" : "right"));

    size_t num_points = std_vec_goal_points.size();
    std::vector<Vector> vec_goal_points(num_points);
    for (size_t i = 0; i < num_points; ++i)
        vec_goal_points[i] = Vector(std_vec_goal_points[i].size(), std_vec_goal_points[i].data());

    return vec_goal_points;
}


bool VisualServoingClient::storedInit(const std::string& label)
{
    return visualservoing_control.stored_init(label);
}


bool VisualServoingClient::storedGoToGoal(const std::string& label)
{
    return visualservoing_control.stored_go_to_goal(label);
}


bool VisualServoingClient::goToSFMGoal()
{
    return visualservoing_control.get_goal_from_sfm();
}
