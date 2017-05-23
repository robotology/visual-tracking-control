#ifndef SERVERVISUALSERVOING_H
#define SERVERVISUALSERVOING_H

#include <vector>

#include <thrift/ServerVisualServoingIDL.h>
#include <yarp/dev/CartesianControl.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/Port.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/RFModule.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>


class ServerVisualServoing : public yarp::os::RFModule,
                             public ServerVisualServoingIDL
{
public:
    bool configure(yarp::os::ResourceFinder &rf);


    double getPeriod() { return 0; }


    bool updateModule();


    bool interruptModule();


    bool close();

protected:
    std::vector<std::string> get_info();


    bool init(const std::string& label);


    bool set_goal(const std::string& label);


    bool get_sfm_points();


    bool go();


    bool quit();


    enum class CamSel { left, right };


    enum class ControlPixelMode { origin, origin_x, origin_o };

private:
    yarp::os::ConstString         robot_name_;

    bool                          should_stop_ = false;

    yarp::dev::PolyDriver         rightarm_cartesian_driver_;
    yarp::dev::ICartesianControl* itf_rightarm_cart_;

    yarp::dev::PolyDriver         gaze_driver_;
    yarp::dev::IGazeControl     * itf_gaze_;

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
    bool                          take_estimates_ = false;

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


    bool setRightArmCartesianController();


    bool setGazeController();


    yarp::os::Port port_rpc_command_;
    bool           attach(yarp::os::Port &source);
    bool           setCommandPort();


    bool setTorsoDOF();


    bool unsetTorsoDOF();


    void getPalmPoints(const yarp::sig::Vector& endeffector, yarp::sig::Vector& p0, yarp::sig::Vector& p1, yarp::sig::Vector& p2, yarp::sig::Vector& p3);

    
    yarp::sig::Vector setJacobianU(const int cam, const yarp::sig::Vector& px);

    
    yarp::sig::Vector setJacobianV(const int cam, const yarp::sig::Vector& px);

    
    yarp::sig::Vector getAxisAngle(const yarp::sig::Vector& v);
};

#endif
