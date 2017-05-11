#ifndef GATEPOSEPARTICLE_H
#define GATEPOSEPARTICLE_H

#include <BayesFiltersLib/VisualCorrectionDecorator.h>
#include <BayesFiltersLib/StateModel.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class GatePoseParticle : public bfl::VisualCorrectionDecorator
{
public:
    /* Constructor */
    GatePoseParticle(std::unique_ptr<VisualCorrection> visual_correction,
                     double gate_x, double gate_y, double gate_z,
                     double gate_aperture, double gate_rotation,
                     const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    GatePoseParticle(std::unique_ptr<VisualCorrection> visual_correction,
                     const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    /* Default constructor, disabled */
    GatePoseParticle() = delete;

    /* Destructor */
    ~GatePoseParticle() noexcept override;

    /* Move constructor, disabled */
    GatePoseParticle(GatePoseParticle&& visual_correction) = delete;

    /* Move assignment operator, disabled */
    GatePoseParticle& operator=(GatePoseParticle&& visual_correction) = delete;

    void correct(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

    void innovation(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> innovation) override;

    void likelihood(const Eigen::Ref<const Eigen::MatrixXf>& innovation, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

    bool setObservationModelProperty(const std::string& property) override;

protected:
    yarp::dev::PolyDriver  drv_arm_enc_;
    yarp::dev::IEncoders * itf_arm_enc_;
    iCub::iKin::iCubArm    icub_kin_arm_;
    yarp::dev::PolyDriver  drv_torso_enc_;
    yarp::dev::IEncoders * itf_torso_enc_;

    yarp::sig::Vector      readTorso();

    yarp::sig::Vector      readRootToEE();

    bool                   setPose();

    bool                   isInsideEllipsoid(const Eigen::Ref<const Eigen::VectorXf>& state);

    bool                   isWithinRotation(float rot_angle);

    bool                   isInsideCone(const Eigen::Ref<const Eigen::VectorXf>& state);

private:
    yarp::os::ConstString  ID_     = "GatePoseParticle";
    yarp::os::ConstString  log_ID_ = "[" + ID_ + "]";

    double gate_x_;
    double gate_y_;
    double gate_z_;
    double gate_aperture_;
    double gate_rotation_;

    yarp::os::ConstString  robot_;
    yarp::os::ConstString  laterality_;
    yarp::os::ConstString  port_prefix_;

    Eigen::VectorXd        ee_pose_;
};

#endif /* GATEPOSEPARTICLE_H */
