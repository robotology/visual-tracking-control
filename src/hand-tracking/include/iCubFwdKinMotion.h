#ifndef ICUBFWDKINMOTION_H
#define ICUBFWDKINMOTION_H

#include <BayesFilters/StateModelDecorator.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class iCubFwdKinMotion : public bfl::StateModelDecorator
{
public:
    /* Constructor */
    iCubFwdKinMotion(std::unique_ptr<StateModel> state_model, const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    /* Default constructor, disabled */
    iCubFwdKinMotion() = delete;

    /* Destructor */
    ~iCubFwdKinMotion() noexcept override;

    // FIXME: il move constructor e assignment non può essere implementato perchè non è implementato nelle classi YARP ed iCub
    /* Move constructor */
    iCubFwdKinMotion(iCubFwdKinMotion&& state_model) noexcept;

    /* Move assignment operator */
    iCubFwdKinMotion& operator=(iCubFwdKinMotion&& state_model) noexcept;

    void propagate(const Eigen::Ref<const Eigen::VectorXf>& cur_state, Eigen::Ref<Eigen::VectorXf> prop_state) override;

    void noiseSample(Eigen::Ref<Eigen::VectorXf> sample) override;

    bool setProperty(const std::string& property) override;

protected:
    yarp::dev::PolyDriver  drv_arm_enc_;
    yarp::dev::PolyDriver  drv_torso_enc_;
    yarp::dev::IEncoders * itf_arm_enc_;
    yarp::dev::IEncoders * itf_torso_enc_;
    iCub::iKin::iCubArm    icub_kin_arm_;

    yarp::sig::Vector      readTorso();

    yarp::sig::Vector      readRootToEE();

    bool                   setDeltaMotion();

private:
    yarp::os::ConstString  ID_     = "iCubFwdKinMotion";
    yarp::os::ConstString  log_ID_ = "[" + ID_ + "]";
    
    yarp::os::ConstString  robot_;
    yarp::os::ConstString  laterality_;
    yarp::os::ConstString  port_prefix_;

    Eigen::VectorXd        prev_ee_pose_;
    Eigen::VectorXd        delta_hand_pose_;
    double                 delta_angle_;
};

#endif /* ICUBFWDKINMOTION_H */
