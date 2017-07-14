#ifndef VISUALSERVOINGSERVER_H
#define VISUALSERVOINGSERVER_H

#include "thrift/VisualServoingIDL.h"

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
    bool goToGoal(const yarp::sig::Vector& vec_x, const yarp::sig::Vector& vec_o) override;

    bool goToGoal(const std::vector<yarp::sig::Vector>& vec_px_l, const std::vector<yarp::sig::Vector>& vec_px_r) override;

    bool setModality(const std::string& mode) override;

    bool setControlPoint(const yarp::os::ConstString& point) override;

    bool getVisualServoingInfo(yarp::os::Bottle& info) override;

    bool setGoToGoalTolerance(const double tol = 10.0) override;

    bool checkVisualServoingController() override;

    bool waitVisualServoingDone(const double period = 0.1, const double timeout = 0.0) override;

    bool stopController() override;

    bool setTranslationGain(const float K_x = 0.5) override;

    bool setMaxTranslationVelocity(const float max_x_dot) override;

    bool setOrientationGain(const float K_o = 0.5) override;

    bool setMaxOrientationVelocity(const float max_o_dot) override;

    std::vector<yarp::sig::Vector> get3DPositionGoalFrom3DPose(const yarp::sig::Vector& x, const yarp::sig::Vector& o) override;

    std::vector<yarp::sig::Vector> getPixelPositionGoalFrom3DPose(const yarp::sig::Vector& x, const yarp::sig::Vector& o, const CamSel& cam) override;

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
    /* EXPERIMENTAL */
    bool stored_init(const std::string& label) override;

    bool stored_go_to_goal(const std::string& label) override;

    bool get_goal_from_sfm() override;

    /* TO CONSIDER */
    bool quit() override;

    /* FROM INTERFACE */
    bool go_to_px_goal(const std::vector<std::vector<double>>& vec_px_l, const std::vector<std::vector<double>>& vec_px_r) override;

    bool go_to_pose_goal(const std::vector<double>& vec_x, const std::vector<double>& vec_o) override;

    bool set_modality(const std::string& mode) override;

    bool set_control_point(const std::string& point) override;

    std::vector<std::string> get_visual_servoing_info() override;

    bool set_go_to_goal_tolerance(const double tol) override;

    bool check_visual_servoing_controller() override;

    bool wait_visual_servoing_done(const double period, const double timeout) override;

    bool stop_controller() override;

    bool set_translation_gain(const double K_x) override;

    bool set_max_translation_velocity(const double max_x_dot) override;

    bool set_orientation_gain(const double K_o) override;

    bool set_max_orientation_velocity(const double max_o_dot) override;

    std::vector<std::vector<double>> get_3D_position_goal_from_3D_pose(const std::vector<double>& x, const std::vector<double>& o) override;

    std::vector<std::vector<double>> get_pixel_position_goal_from_3D_pose(const std::vector<double> & x, const std::vector<double> & o, const std::string& cam) override;


    /* Enum helpers */
    enum class PixelControlMode { all, x, o };

    enum class OperatingMode { position, orientation, pose };

private:
    bool                           verbosity_  = false;
    //!!!: rimuovere o gestire meglio la simulazione
    bool                           sim_        = false;
    yarp::os::ConstString          robot_name_ = "icub";

    OperatingMode                  op_mode_ = OperatingMode::pose;

    yarp::dev::PolyDriver          rightarm_cartesian_driver_;
    yarp::dev::ICartesianControl*  itf_rightarm_cart_;

    yarp::dev::PolyDriver          gaze_driver_;
    yarp::dev::IGazeControl     *  itf_gaze_;

    bool                           vs_control_running_ = false;
    bool                           vs_goal_reached_    = false;
    const double                   Ts_                 = 0.1; /* [s] */
    double                         K_x_                = 0.75;
    double                         K_o_                = 1.5;
    double                         max_x_dot_          = 0.025; /* [m/s] */
    double                         max_o_dot_          = 5 * M_PI / 180.0; /* [rad/s] */
    double                         px_tol_             = 10.0;
    double                         traj_time_          = 3.0;

    yarp::sig::Matrix              l_proj_;
    yarp::sig::Matrix              r_proj_;

    yarp::sig::Vector              goal_pose_    = yarp::math::zeros(6);
    yarp::sig::Vector              px_des_       = yarp::math::zeros(12);
    yarp::sig::Matrix              l_H_eye_to_r_ = yarp::math::zeros(4, 4);
    yarp::sig::Matrix              r_H_eye_to_r_ = yarp::math::zeros(4, 4);
    yarp::sig::Matrix              l_H_r_to_cam_ = yarp::math::zeros(4, 4);
    yarp::sig::Matrix              r_H_r_to_cam_ = yarp::math::zeros(4, 4);

    std::mutex                     mtx_px_des_;
    std::mutex                     mtx_H_eye_cam_;

    std::thread                    thr_background_update_params_;

    std::vector<yarp::sig::Vector> l_px_goal_ = std::vector<yarp::sig::Vector>(4);
    std::vector<yarp::sig::Vector> r_px_goal_ = std::vector<yarp::sig::Vector>(4);

    int                            ctx_local_cart_;
    int                            ctx_remote_cart_;
    int                            ctx_local_gaze_;
    int                            ctx_remote_gaze_;

    yarp::os::BufferedPort<yarp::sig::Vector>                       port_pose_left_in_;
    yarp::os::BufferedPort<yarp::sig::Vector>                       port_pose_right_in_;

    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_left_in_;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_left_out_;
    yarp::os::BufferedPort<yarp::os::Bottle>                        port_click_left_;

    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_right_in_;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_right_out_;
    yarp::os::BufferedPort<yarp::os::Bottle>                        port_click_right_;

    yarp::os::Port                                                  port_rpc_command_;

    /* Private class methods */
    bool setRightArmCartesianController();

    bool setGazeController();

    bool setCommandPort();

    bool setTorsoDOF();

    bool unsetTorsoDOF();

    std::vector<yarp::sig::Vector> getControlPixelsFromPose(const yarp::sig::Vector& pose, const CamSel& cam, const PixelControlMode& mode);

    std::vector<yarp::sig::Vector> getPixelsFromPose(const yarp::sig::Vector& pose, const CamSel& cam);

    std::vector<yarp::sig::Vector> getControlPointsFromPose(const yarp::sig::Vector& pose);

    yarp::sig::Vector getPixelFromPoint(const CamSel& cam, const yarp::sig::Vector& p) const;

    yarp::sig::Vector getControlPixelFromPoint(const CamSel& cam, const yarp::sig::Vector& p) const;

    void getCurrentStereoFeaturesAndJacobian(const std::vector<yarp::sig::Vector>& left_px,  const std::vector<yarp::sig::Vector>& right_px,
                                             yarp::sig::Vector& features, yarp::sig::Matrix& jacobian);

    yarp::sig::Vector getJacobianU(const CamSel& cam, const yarp::sig::Vector& px);

    yarp::sig::Vector getJacobianV(const CamSel& cam, const yarp::sig::Vector& px);
    
    yarp::sig::Vector getAxisAngle(const yarp::sig::Vector& v);

    bool setCameraTransformations();

    bool setPoseGoal(const yarp::sig::Vector& goal_x, const yarp::sig::Vector& goal_o);

    bool setPixelGoal(const std::vector<yarp::sig::Vector>& l_px_goal, const std::vector<yarp::sig::Vector>& r_px_goal);

    void backproc_UpdateVisualServoingParamters();
    bool is_stopping_backproc_update_vs_params = true;

    void yInfoVerbose   (const yarp::os::ConstString& str) const { if(verbosity_) yInfo()    << str; };
    void yWarningVerbose(const yarp::os::ConstString& str) const { if(verbosity_) yWarning() << str; };
    void yErrorVerbose  (const yarp::os::ConstString& str) const { if(verbosity_) yError()   << str; };
};

#endif /* VISUALSERVOINGSERVER_H */
