#ifndef VISUALSERVOINGSERVER_H
#define VISUALSERVOINGSERVER_H

#include "thrift/VisualServoingIDL.h"

#include <array>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#include <yarp/dev/CartesianControl.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/DeviceDriver.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/dev/IVisualServoing.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/math/Math.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Port.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/Thread.h>
#include <yarp/os/Searchable.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>


class VisualServoingServer : public    yarp::dev::DeviceDriver,
                             public    yarp::dev::IVisualServoing,
                             protected yarp::os::Thread,
                             public    VisualServoingIDL
{
public:
    /* Ctors and Dtors */
    VisualServoingServer();

    ~VisualServoingServer();


    /* DeviceDriver overrides */
    bool open(yarp::os::Searchable &config) override;

    bool close() override;


    /* IVisualServoing overrides */
    bool initFacilities(const bool use_direct_kin) override;

    bool resetFacilities() override;

    bool stopFacilities() override;

    bool goToGoal(const yarp::sig::Vector& vec_x, const yarp::sig::Vector& vec_o) override;

    bool goToGoal(const std::vector<yarp::sig::Vector>& vec_px_l, const std::vector<yarp::sig::Vector>& vec_px_r) override;

    bool setModality(const std::string& mode) override;

    bool setVisualServoControl(const std::string& control) override;

    bool setControlPoint(const yarp::os::ConstString& point) override;

    bool getVisualServoingInfo(yarp::os::Bottle& info) override;

    bool setGoToGoalTolerance(const double tol = 15.0) override;

    bool checkVisualServoingController() override;

    bool waitVisualServoingDone(const double period = 0.1, const double timeout = 0.0) override;

    bool stopController() override;

    bool setTranslationGain(const double K_x_1 = 1.0, const double K_x_2 = 0.25) override;

    bool setMaxTranslationVelocity(const double max_x_dot) override;

    bool setTranslationGainSwitchTolerance(const double K_x_tol = 30.0) override;

    bool setOrientationGain(const double K_x_1 = 1.5, const double K_x_2 = 0.375) override;

    bool setMaxOrientationVelocity(const double max_o_dot) override;

    bool setOrientationGainSwitchTolerance(const double K_o_tol = 30.0) override;

    std::vector<yarp::sig::Vector> get3DGoalPositionsFrom3DPose(const yarp::sig::Vector& x, const yarp::sig::Vector& o) override;

    std::vector<yarp::sig::Vector> getGoalPixelsFrom3DPose(const yarp::sig::Vector& x, const yarp::sig::Vector& o, const CamSel& cam) override;


    /* TO BE DEPRECATED */
    bool storedInit(const std::string& label) override;

    bool storedGoToGoal(const std::string& label) override;

    bool goToSFMGoal() override;

protected:
    /* Thread overrides */
    void beforeStart() override;

    bool threadInit() override;

    void run() override;

    void afterStart(bool success) override;

    void onStop() override;

    void threadRelease() override;


    /* VisualServoingIDL overrides */
    bool init_facilities(const bool use_direct_kin) override;

    bool reset_facilities() override;

    bool stop_facilities() override;

    bool go_to_px_goal(const std::vector<std::vector<double>>& vec_px_l, const std::vector<std::vector<double>>& vec_px_r) override;

    bool go_to_pose_goal(const std::vector<double>& vec_x, const std::vector<double>& vec_o) override;

    bool set_modality(const std::string& mode) override;

    bool set_visual_servo_control(const std::string& control) override;

    bool set_control_point(const std::string& point) override;

    std::vector<std::string> get_visual_servoing_info() override;

    bool set_go_to_goal_tolerance(const double tol) override;

    bool check_visual_servoing_controller() override;

    bool wait_visual_servoing_done(const double period, const double timeout) override;

    bool stop_controller() override;

    bool set_translation_gain(const double K_x_1, const double K_x_2) override;

    bool set_max_translation_velocity(const double max_x_dot) override;

    bool set_translation_gain_switch_tolerance(const double K_x_tol) override;

    bool set_orientation_gain(const double K_o_1, const double K_o_2) override;

    bool set_max_orientation_velocity(const double max_o_dot) override;

    bool set_orientation_gain_switch_tolerance(const double K_o_tol) override;

    std::vector<std::vector<double>> get_3D_goal_positions_from_3D_pose(const std::vector<double>& x, const std::vector<double>& o) override;

    std::vector<std::vector<double>> get_goal_pixels_from_3D_pose(const std::vector<double> & x, const std::vector<double> & o, const std::string& cam) override;

    bool quit() override;

    /* TO BE DEPRECATED */
    bool stored_init(const std::string& label) override;

    bool stored_go_to_goal(const std::string& label) override;

    bool get_goal_from_sfm() override;


    /* Enum helpers */
    enum class VisualServoControl { decoupled, robust, cartesian };

    enum class PixelControlMode { all, x, o };

    enum class OperatingMode { position, orientation, pose };

private:
    bool                           verbosity_  = false;
    bool                           sim_        = false;
    yarp::os::ConstString          robot_name_ = "icub";

    VisualServoControl             vs_control_ = VisualServoControl::decoupled;
    OperatingMode                  op_mode_ = OperatingMode::pose;

    yarp::dev::PolyDriver          rightarm_cartesian_driver_;
    yarp::dev::ICartesianControl*  itf_rightarm_cart_;

    yarp::dev::PolyDriver          gaze_driver_;
    yarp::dev::IGazeControl*       itf_gaze_;

    bool                           vs_control_running_ = false;
    bool                           vs_goal_reached_    = false;
    const double                   Ts_                 = 0.1; /* [s] */
    std::array<double, 2>          K_x_                = {{0.5, 0.25}};
    std::array<double, 2>          K_o_                = {{0.5, 0.25}};
    double                         max_x_dot_          = 0.025; /* [m/s] */
    double                         max_o_dot_          = 5.0 * M_PI / 180.0; /* [rad/s] */
    double                         K_x_tol_            = 20.0; /* [pixel] */
    double                         K_o_tol_            = 20.0; /* [pixel] */
    double                         K_position_tol_     = 0.03; /* [m] */
    double                         K_orientation_tol_  = 5.0 * M_PI / 180.0; /* [rad] */
    bool                           K_x_hysteresis_     = false;
    bool                           K_o_hysteresis_     = false;
    bool                           K_pose_hysteresis_  = false;
    double                         tol_px_             = 5.0; /* [pixel] */
    double                         tol_position_       = 0.01; /* [m] */
    double                         tol_angle_          = 1.0 * M_PI / 180.0; /* [rad] */
    double                         traj_time_          = 1.0; /* [s] */

    yarp::sig::Matrix              l_proj_;
    yarp::sig::Matrix              r_proj_;

    yarp::sig::Vector              goal_pose_    = yarp::math::zeros(7);
    yarp::sig::Vector              px_des_       = yarp::math::zeros(12);
    yarp::sig::Matrix              l_H_eye_to_r_ = yarp::math::zeros(4, 4);
    yarp::sig::Matrix              r_H_eye_to_r_ = yarp::math::zeros(4, 4);
    yarp::sig::Matrix              l_H_r_to_cam_ = yarp::math::zeros(4, 4);
    yarp::sig::Matrix              r_H_r_to_cam_ = yarp::math::zeros(4, 4);

    std::mutex                     mtx_px_des_;
    std::mutex                     mtx_H_eye_cam_;

    std::thread*                   thr_background_update_params_;

    std::vector<yarp::sig::Vector> l_px_goal_ = std::vector<yarp::sig::Vector>(4);
    std::vector<yarp::sig::Vector> r_px_goal_ = std::vector<yarp::sig::Vector>(4);

    int                            ctx_local_cart_;
    int                            ctx_remote_cart_;

    yarp::os::BufferedPort<yarp::sig::Vector>                       port_pose_left_in_;
    yarp::os::BufferedPort<yarp::sig::Vector>                       port_pose_right_in_;

    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_left_in_;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_left_out_;
    yarp::os::BufferedPort<yarp::os::Bottle>                        port_click_left_;

    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_right_in_;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_right_out_;
    yarp::os::BufferedPort<yarp::os::Bottle>                        port_click_right_;

    yarp::os::Port                                                  port_rpc_command_;
    yarp::os::RpcClient                                             port_rpc_tracker_left_;
    yarp::os::RpcClient                                             port_rpc_tracker_right_;

    /* Private class methods */
    void decoupledImageBasedVisualServoControl();

    void robustImageBasedVisualServoControl();

    void cartesianPositionBasedVisualServoControl();

    bool setRightArmCartesianController();

    bool setGazeController();

    bool setCommandPort();

    bool setTorsoDOF();

    bool unsetTorsoDOF();

    std::vector<yarp::sig::Vector> getPixelsFromPose(const yarp::sig::Vector& pose, const CamSel& cam);

    std::vector<yarp::sig::Vector> getControlPixelsFromPose(const yarp::sig::Vector& pose, const CamSel& cam, const PixelControlMode& mode);

    std::vector<yarp::sig::Vector> getControlPointsFromPose(const yarp::sig::Vector& pose);

    yarp::sig::Vector getPixelFromPoint(const CamSel& cam, const yarp::sig::Vector& p) const;

    yarp::sig::Vector getControlPixelFromPoint(const CamSel& cam, const yarp::sig::Vector& p) const;

    void getCurrentStereoFeaturesAndJacobian(const std::vector<yarp::sig::Vector>& left_px,  const std::vector<yarp::sig::Vector>& right_px,
                                             yarp::sig::Vector& features, yarp::sig::Matrix& jacobian);

    yarp::sig::Vector getJacobianU(const CamSel& cam, const yarp::sig::Vector& px);

    yarp::sig::Vector getJacobianV(const CamSel& cam, const yarp::sig::Vector& px);

    bool setCameraTransformations();

    bool setPoseGoal(const yarp::sig::Vector& goal_x, const yarp::sig::Vector& goal_o);

    bool setPixelGoal(const std::vector<yarp::sig::Vector>& l_px_goal, const std::vector<yarp::sig::Vector>& r_px_goal);

    void backproc_UpdateVisualServoingParamters();
    bool is_stopping_backproc_update_vs_params = true;

    bool checkVisualServoingStatus(const yarp::sig::Vector& px_cur, const double tol);

    void yInfoVerbose   (const yarp::os::ConstString& str) const { if(verbosity_) yInfo()    << str; };
    void yWarningVerbose(const yarp::os::ConstString& str) const { if(verbosity_) yWarning() << str; };
    void yErrorVerbose  (const yarp::os::ConstString& str) const { if(verbosity_) yError()   << str; };

    /* EXPERIMENTAL */
    yarp::sig::Vector averagePose(const yarp::sig::Vector& l_pose, const yarp::sig::Vector& r_pose) const;

    bool checkVisualServoingStatus(const yarp::sig::Vector& pose_cur, const double tol_position, const double tol_angle);
};

#endif /* VISUALSERVOINGSERVER_H */
