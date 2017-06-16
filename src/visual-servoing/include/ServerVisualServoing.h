#ifndef SERVERVISUALSERVOING_H
#define SERVERVISUALSERVOING_H

#include <cmath>
#include <vector>

#include <thrift/ServerVisualServoingIDL.h>
#include <yarp/dev/CartesianControl.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/DeviceDriver.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/dev/IVisualServoing.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Port.h>
#include <yarp/os/RateThread.h>
#include <yarp/os/Searchable.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>


class ServerVisualServoing : public    yarp::dev::DeviceDriver,
                             public    yarp::dev::IVisualServoing,
                             protected yarp::os::Thread,
                             public    ServerVisualServoingIDL
{
public:
    /* Ctors and Dtors */
    ServerVisualServoing();

    ~ServerVisualServoing();


    /* DeviceDriver overrides */
    bool open(yarp::os::Searchable &config) override;

    bool close() override;


    /* IVisualServoing overrides */
    bool goToGoal(const yarp::sig::Vector& px_l, const yarp::sig::Vector& px_r) override;

    bool goToGoal(const std::vector<yarp::sig::Vector>& vec_px_l, const std::vector<yarp::sig::Vector>& vec_px_r) override;

    bool setModality(const bool mode) override;

    bool setControlPoint(const yarp::os::ConstString& point) override;

    bool getVisualServoingInfo(yarp::os::Bottle& info) override;

    bool setGoToGoalTolerance(const double tol) override;

    bool checkVisualServoingController(bool& is_running) override;

    bool waitVisualServoingDone(const double period = 0.1, const double timeout = 0.0) override;

    bool stopController() override;

    bool setTranslationGain(const float k_x = 0.5) override;

    bool setMaxTranslationVelocity(const float max_x_dot) override;

    bool setOrientationGain(const float k_o) override;

    bool setMaxOrientationVelocity(const float max_o_dot) override;

    bool get3DPositionGoalFrom3DPose(const yarp::sig::Vector& x, const yarp::sig::Vector& o,
                                     std::vector<yarp::sig::Vector> vec_goal_points) override;

protected:
    /* Thread overrides */
    void beforeStart() override;

    bool threadInit() override;

    void run() override;

    void afterStart(bool success) override;

    void onStop() override;

    void threadRelease() override;


    /* ServerVisualServoingIDL overrides */
    std::vector<std::string> get_info() override;

    bool init(const std::string& label) override;

    bool set_goal(const std::string& label) override;

    bool get_sfm_points() override;

    bool set_modality(const std::string& mode) override;

    bool set_position_gain(const double k) override;

    bool set_orientation_gain(const double k) override;

    bool set_position_bound(const double b) override;

    bool set_orientation_bound(const double b) override;

    bool set_goal_tol(const double px) override;

    bool go() override;

    bool quit() override;


    /* Protected class methods */
    bool interrupt();


    /* Enum helpers */
    enum class CamSel { left, right };

    enum class ControlPixelMode { origin, origin_x, origin_o };

    enum class OperatingMode { position, orientation, pose };

private:
    bool                          verbosity_ = false;
    yarp::os::ConstString         robot_name_;

    bool                          should_stop_ = false;
    OperatingMode                 op_mode_     = OperatingMode::pose;

    yarp::dev::PolyDriver         rightarm_cartesian_driver_;
    yarp::dev::ICartesianControl* itf_rightarm_cart_;

    yarp::dev::PolyDriver         gaze_driver_;
    yarp::dev::IGazeControl     * itf_gaze_;

    const double                  Ts_     = 0.1;
    double                        K_x_    = 0.5;
    double                        K_o_    = 0.5;
    double                        vx_max_ = 0.025; /* [m/s] */
    double                        vo_max_ = 5 * M_PI / 180.0; /* [rad/s] */
    double                        px_tol_ = 10.0;

    yarp::sig::Vector             goal_pose_;
    yarp::sig::Matrix             l_proj_;
    yarp::sig::Matrix             r_proj_;
    yarp::sig::Matrix             l_H_r_to_eye_;
    yarp::sig::Matrix             r_H_r_to_eye_;
    yarp::sig::Matrix             l_H_eye_to_r_;
    yarp::sig::Matrix             r_H_eye_to_r_;
    yarp::sig::Matrix             l_H_r_to_cam_;
    yarp::sig::Matrix             r_H_r_to_cam_;
    yarp::sig::Matrix             px_to_cartesian_;

    double                        traj_time_ = 3.0;
    yarp::sig::Vector             l_px_goal_;
    yarp::sig::Vector             r_px_goal_;
    yarp::sig::Vector             px_des_;

    int                           ctx_cart_;
    int                           ctx_gaze_;

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


    bool           attach(yarp::os::Port &source);
    bool           setCommandPort();


    bool setTorsoDOF();

    bool unsetTorsoDOF();

    void getControlPixelsFromPose(const yarp::sig::Vector& pose, const CamSel cam, const ControlPixelMode mode, yarp::sig::Vector& px0, yarp::sig::Vector& px1, yarp::sig::Vector& px2, yarp::sig::Vector& px3);

    void getControlPointsFromPose(const yarp::sig::Vector& pose, yarp::sig::Vector& p0, yarp::sig::Vector& p1, yarp::sig::Vector& p2, yarp::sig::Vector& p3);

    yarp::sig::Vector getPixelFromPoint(const CamSel cam, const yarp::sig::Vector& p) const;

    void getCurrentStereoFeaturesAndJacobian(const yarp::sig::Vector& left_px0,  const yarp::sig::Vector& left_px1,  const yarp::sig::Vector& left_px2,  const yarp::sig::Vector& left_px3,
                                             const yarp::sig::Vector& right_px0, const yarp::sig::Vector& right_px1, const yarp::sig::Vector& right_px2, const yarp::sig::Vector& right_px3,
                                             yarp::sig::Vector& features, yarp::sig::Matrix& jacobian);

    yarp::sig::Vector getJacobianU(const CamSel cam, const yarp::sig::Vector& px);

    yarp::sig::Vector getJacobianV(const CamSel cam, const yarp::sig::Vector& px);
    
    yarp::sig::Vector getAxisAngle(const yarp::sig::Vector& v);

    void yInfoVerbose   (const yarp::os::ConstString& str) const { if(verbosity_) yInfo(str);    };
    void yWarningVerbose(const yarp::os::ConstString& str) const { if(verbosity_) yWarning(str); };
    void yErrorVerbose  (const yarp::os::ConstString& str) const { if(verbosity_) yError(str);   };
};

#endif /* SERVERVISUALSERVOING_H */
