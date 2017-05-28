#ifndef PLAYFWDKINMOTION_H
#define PLAYFWDKINMOTION_H

#include <BayesFilters/StateModelDecorator.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class PlayFwdKinMotion : public bfl::StateModelDecorator
{
public:
    /* Constructor */
    PlayFwdKinMotion(std::unique_ptr<StateModel> state_model, const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    PlayFwdKinMotion(std::unique_ptr<StateModel> state_model, const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix, bool init_pose) noexcept;

    /* Default constructor, disabled */
    PlayFwdKinMotion() = delete;

    /* Destructor */
    ~PlayFwdKinMotion() noexcept override;

    /* Move constructor */
    PlayFwdKinMotion(PlayFwdKinMotion&& state_model) noexcept;

    /* Move assignment operator */
    PlayFwdKinMotion& operator=(PlayFwdKinMotion&& state_model) noexcept;

    void propagate(const Eigen::Ref<const Eigen::VectorXf>& cur_state, Eigen::Ref<Eigen::VectorXf> prop_state) override;

    void noiseSample(Eigen::Ref<Eigen::VectorXf> sample) override;

    bool setProperty(const std::string& property) override;

protected:
    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_enc_;
    iCub::iKin::iCubArm                      icub_kin_arm_;

    yarp::sig::Vector readTorso();

    yarp::sig::Vector readRootToEE();

    bool              setInitialPose();

    bool              setDeltaMotion();

private:
    yarp::os::ConstString ID_     = "PlayFwdKinMotion";
    yarp::os::ConstString log_ID_ = "[" + ID_ + "]";

    yarp::os::ConstString robot_;
    yarp::os::ConstString laterality_;
    yarp::os::ConstString port_prefix_;

    Eigen::VectorXd       prev_ee_pose_;
    Eigen::VectorXd       delta_hand_pose_;
    double                delta_angle_;
};

#endif /* PLAYFWDKINMOTION_H */
