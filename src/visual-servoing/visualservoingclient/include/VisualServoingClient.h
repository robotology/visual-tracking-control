#ifndef VISUALSERVOINGCLIENT_H
#define VISUALSERVOINGCLIENT_H

#include "thrift/VisualServoingIDL.h"

#include <vector>

#include <yarp/dev/DeviceDriver.h>
#include <yarp/dev/IVisualServoing.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Port.h>
#include <yarp/os/Searchable.h>
#include <yarp/sig/Vector.h>


class VisualServoingClient : public yarp::dev::DeviceDriver,
                             public yarp::dev::IVisualServoing
{
public:
    /* Ctors and Dtors */
    VisualServoingClient();

    ~VisualServoingClient();


    /* DeviceDriver overrides */
    bool open(yarp::os::Searchable &config) override;

    bool close() override;


    /* IVisualServoing overrides */
    bool goToGoal(const yarp::sig::Vector& vec_x, const yarp::sig::Vector& vec_o) override;

    bool goToGoal(const std::vector<yarp::sig::Vector>& vec_px_l, const std::vector<yarp::sig::Vector>& vec_px_r) override;

    bool setModality(const std::string& mode) override;

    bool setControlPoint(const yarp::os::ConstString& point) override;

    bool getVisualServoingInfo(yarp::os::Bottle& info) override;

    bool setGoToGoalTolerance(const double tol) override;

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

    
private:
    bool                  verbosity_ = false;
    yarp::os::ConstString local_     = "";
    yarp::os::ConstString remote_    = "";

    VisualServoingIDL     visualservoing_control;
    yarp::os::Port        port_rpc_command_;

    void yInfoVerbose   (const yarp::os::ConstString& str) const { if(verbosity_) yInfo()    << str; };
    void yWarningVerbose(const yarp::os::ConstString& str) const { if(verbosity_) yWarning() << str; };
    void yErrorVerbose  (const yarp::os::ConstString& str) const { if(verbosity_) yError()   << str; };
};

#endif /* VISUALSERVOINGCLIENT_H */
