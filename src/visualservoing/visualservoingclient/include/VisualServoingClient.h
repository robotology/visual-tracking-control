/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

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

    virtual ~VisualServoingClient();


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

    bool setOrientationGain(const double K_o_1 = 1.5, const double K_o_2 = 0.375) override;

    bool setMaxOrientationVelocity(const double max_o_dot) override;

    bool setOrientationGainSwitchTolerance(const double K_o_tol = 30.0) override;

    std::vector<yarp::sig::Vector> get3DGoalPositionsFrom3DPose(const yarp::sig::Vector& x, const yarp::sig::Vector& o) override;

    std::vector<yarp::sig::Vector> getGoalPixelsFrom3DPose(const yarp::sig::Vector& x, const yarp::sig::Vector& o, const CamSel& cam) override;

    bool storedInit(const std::string& label) override;

    bool storedGoToGoal(const std::string& label) override;

    bool goToSFMGoal() override;


private:
    bool verbosity_ = false;

    yarp::os::ConstString local_ = "";

    yarp::os::ConstString remote_ = "";

    VisualServoingIDL visualservoing_control;

    yarp::os::Port port_rpc_command_;

    void yInfoVerbose(const yarp::os::ConstString& str) const { if(verbosity_) yInfo() << str; };

    void yWarningVerbose(const yarp::os::ConstString& str) const { if(verbosity_) yWarning() << str; };

    void yErrorVerbose(const yarp::os::ConstString& str) const { if(verbosity_) yError() << str; };
};

#endif /* VISUALSERVOINGCLIENT_H */
