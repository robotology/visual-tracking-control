#include "GatePoseParticle.h"

#include <iCub/ctrl/math.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Property.h>

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


GatePoseParticle::GatePoseParticle(std::unique_ptr<VisualCorrection> visual_correction,
                                   double gate_x, double gate_y, double gate_z,
                                   double gate_aperture, double gate_rotation,
                                   const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept :
    VisualCorrectionDecorator(std::move(visual_correction)),
    gate_x_(gate_x), gate_y_(gate_y), gate_z_(gate_z), gate_aperture_(gate_aperture), gate_rotation_(gate_rotation)
{
    Property opt_arm_enc;
    opt_arm_enc.put("device", "remote_controlboard");
    opt_arm_enc.put("local",  "/hand-tracking/" + ID_ + "/" + port_prefix + "/control_" + laterality_ + "_arm");
    opt_arm_enc.put("remote", "/" + robot_ + "/right_arm");

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

    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

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

    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    ee_pose_ = Map<VectorXd>(ee_pose.data(), 7, 1);

    yInfo() << log_ID_ << "Succesfully initialized.";
}


GatePoseParticle::GatePoseParticle(std::unique_ptr<VisualCorrection> visual_correction,
                                   const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept :
    GatePoseParticle(std::move(visual_correction), 0.1, 0.1, 0.1, 30, 5, robot, laterality, port_prefix) { }


GatePoseParticle::~GatePoseParticle() noexcept { }


void GatePoseParticle::correct(const Ref<const MatrixXf>& pred_state, InputArray measurements, Ref<MatrixXf> cor_state)
{
    VisualCorrectionDecorator::correct(pred_state, measurements, cor_state);

    for (int i = 0; i < pred_state.cols(); ++i)
    {
        if (!isInsideEllipsoid(pred_state.block<3, 1>(0, i)) ||
            !isWithinRotation (pred_state(6, i))             ||
            !isInsideCone     (pred_state.block<3, 1>(3, i))   )
            cor_state(i, 0) = std::numeric_limits<float>::min();
    }
}


void GatePoseParticle::innovation(const Ref<const MatrixXf>& pred_state, InputArray measurements, Ref<MatrixXf> innovation)
{
    VisualCorrectionDecorator::innovation(pred_state, measurements, innovation);
}


void GatePoseParticle::likelihood(const Ref<const MatrixXf>& innovation, Ref<MatrixXf> cor_state)
{
    VisualCorrectionDecorator::likelihood(innovation, cor_state);
}


bool GatePoseParticle::setObservationModelProperty(const std::string& property)
{
    if (!VisualCorrectionDecorator::setObservationModelProperty(property))
        if (property == "ICGPP_POSE")
            return setPose();

    return false;
}


Vector GatePoseParticle::readTorso()
{
    int torso_enc_num;
    itf_arm_enc_->getAxes(&torso_enc_num);
    Vector enc_torso(torso_enc_num);

    while (!itf_torso_enc_->getEncoders(enc_torso.data()));

    std::swap(enc_torso(0), enc_torso(2));

    return enc_torso;
}


Vector GatePoseParticle::readRootToEE()
{
    int arm_enc_num;
    itf_arm_enc_->getAxes(&arm_enc_num);
    Vector enc_arm(arm_enc_num);

    Vector root_ee_enc(10);

    root_ee_enc.setSubvector(0, readTorso());

    while (!itf_arm_enc_->getEncoders(enc_arm.data()));
    for (size_t i = 0; i < 7; ++i)
        root_ee_enc(i+3) = enc_arm(i);

    return root_ee_enc;
}


bool GatePoseParticle::setPose()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    ee_pose_ = Map<VectorXd>(ee_pose.data(), 7, 1);

    return true;
}


bool GatePoseParticle::isInsideEllipsoid(const Eigen::Ref<const Eigen::VectorXf>& state)
{
    return ( (abs(state(0) - ee_pose_(0)) < gate_x_) &&
             (abs(state(1) - ee_pose_(1)) < gate_y_) &&
             (abs(state(2) - ee_pose_(2)) < gate_z_) );
}


bool GatePoseParticle::isWithinRotation(float rot_angle)
{
    float ang_diff = rot_angle - ee_pose_(6);
    if (ang_diff >  2.0 * M_PI) ang_diff -= 2.0 * M_PI;
    if (ang_diff <=        0.0) ang_diff += 2.0 * M_PI;

    return (ang_diff <= gate_rotation_);
}


bool GatePoseParticle::isInsideCone(const Eigen::Ref<const Eigen::VectorXf>& state)
{
    double   half_aperture    = CTRL_DEG2RAD * (gate_aperture_ / 2.0);

    VectorXd test_direction   = -state.cast<double>();

    VectorXd fwdkin_direction = -ee_pose_.middleRows<3>(3);

    return ( (test_direction.dot(fwdkin_direction) / test_direction.norm() / fwdkin_direction.norm()) >= cos(half_aperture) );
}
