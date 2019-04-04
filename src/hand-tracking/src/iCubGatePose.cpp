/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <iCubGatePose.h>

#include <iCub/ctrl/math.h>
#include <yarp/eigen/Eigen.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Property.h>

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::eigen;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


iCubGatePose::iCubGatePose(std::unique_ptr<PFVisualCorrection> visual_correction,
                           const double gate_x, const double gate_y, const double gate_z,
                           const double gate_rotation,
                           const double gate_aperture,
                           const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) :
    GatePose(std::move(visual_correction),
             gate_x, gate_y, gate_z,
             gate_rotation,
             gate_aperture),
    icub_kin_arm_(iCubArm(laterality + "_v2")), robot_(robot), laterality_(laterality), port_prefix_(port_prefix)
{
    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);


    Property opt_arm_enc;
    opt_arm_enc.put("device", "remote_controlboard");
    opt_arm_enc.put("local",  "/hand-tracking/" + ID_ + "/" + port_prefix + "/control_" + laterality_ + "_arm");
    opt_arm_enc.put("remote", "/" + robot_ + "/" + laterality_ + "_arm");

    yInfo() << log_ID_ << "Opening " + laterality_ + " arm remote_controlboard driver...";
    if (drv_arm_enc_.open(opt_arm_enc))
    {
        yInfo() << log_ID_ << "Succesfully opened " + laterality_ + " arm remote_controlboard interface.";

        yInfo() << log_ID_ << "Getting " + laterality_ + " arm encoder interface...";

        drv_arm_enc_.view(itf_arm_enc_);
        if (!itf_arm_enc_)
        {
            yError() << log_ID_ << "Cannot get " + laterality_ + " arm encoder interface!";

            drv_arm_enc_.close();
            throw std::runtime_error("ERROR::" + ID_ + "::CTOR::INTERFACE\nERROR: cannot get " + laterality_ + " arm encoder interface!");
        }

        yInfo() << log_ID_ << "Succesfully got " + laterality_ + " arm encoder interface.";
    }
    else
    {
        yError() << log_ID_ << "Cannot open " + laterality_ + " arm remote_controlboard!";

        throw std::runtime_error("ERROR::" + ID_ + "::CTOR::DRIVER\nERROR: cannot open " + laterality_ + " arm remote_controlboard!");
    }


    Property opt_torso_enc;
    opt_torso_enc.put("device", "remote_controlboard");
    opt_torso_enc.put("local",  "/hand-tracking/" + ID_ + "/" + port_prefix + "/control_torso");
    opt_torso_enc.put("remote", "/" + robot_ + "/torso");

    yInfo() << log_ID_ << "Opening torso remote_controlboard driver...";

    if (drv_torso_enc_.open(opt_torso_enc))
    {
        yInfo() << log_ID_ << "Succesfully opened torso remote_controlboard driver.";

        yInfo() << log_ID_ << "Getting torso encoder interface...";

        drv_torso_enc_.view(itf_torso_enc_);
        if (!itf_torso_enc_)
        {
            yError() << log_ID_ << "Cannot get torso encoder interface!";

            drv_torso_enc_.close();
            throw std::runtime_error("ERROR::" + ID_ + "::CTOR::INTERFACE\nERROR: cannot get torso encoder interface!");
        }

        yInfo() << log_ID_ << "Succesfully got torso encoder interface.";
    }
    else
    {
        yError() << log_ID_ << "Cannot open torso remote_controlboard!";

        throw std::runtime_error("ERROR::" + ID_ + "::CTOR::DRIVER\nERROR: cannot open torso remote_controlboard!");
    }

    yInfo() << log_ID_ << "Succesfully initialized.";
}


iCubGatePose::iCubGatePose(std::unique_ptr<PFVisualCorrection> visual_correction,
                           const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) :
    iCubGatePose(std::move(visual_correction), 0.1, 0.1, 0.1, 5, 30, robot, laterality, port_prefix) { }


iCubGatePose::~iCubGatePose() noexcept { }


Vector iCubGatePose::readTorso()
{
    int torso_enc_num;
    itf_torso_enc_->getAxes(&torso_enc_num);
    Vector enc_torso(torso_enc_num);

    while (!itf_torso_enc_->getEncoders(enc_torso.data()));

    std::swap(enc_torso(0), enc_torso(2));

    return enc_torso;
}


Vector iCubGatePose::readRootToEE()
{
    int arm_enc_num;
    itf_arm_enc_->getAxes(&arm_enc_num);
    Vector enc_arm(arm_enc_num);

    Vector root_ee_enc(10);

    root_ee_enc.setSubvector(0, readTorso());

    while (!itf_arm_enc_->getEncoders(enc_arm.data()));

    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i + 3) = enc_arm(i);

    return root_ee_enc;
}


VectorXd iCubGatePose::readPose()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());

    return toEigen(ee_pose);
}
