#ifndef PLAYFWDKINMODEL_H
#define PLAYFWDKINMODEL_H

#include <BayesFilters/StateModelDecorator.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class PlayFwdKinModel : public bfl::StateModelDecorator
{
public:
    PlayFwdKinModel(std::unique_ptr<StateModel> state_model, const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    PlayFwdKinModel(std::unique_ptr<StateModel> state_model, const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix, bool init_pose) noexcept;

    PlayFwdKinModel(PlayFwdKinModel&& state_model) noexcept;

    ~PlayFwdKinModel() noexcept;

    PlayFwdKinModel& operator=(PlayFwdKinModel&& state_model) noexcept;

    void propagate(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> prop_state) override;

    void motion(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> mot_state) override;

    Eigen::MatrixXf getNoiseSample(const int num) override;

    Eigen::MatrixXf getNoiseCovarianceMatrix() override;

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
    yarp::os::ConstString ID_     = "PlayFwdKinModel";
    yarp::os::ConstString log_ID_ = "[" + ID_ + "]";

    yarp::os::ConstString robot_;
    yarp::os::ConstString laterality_;
    yarp::os::ConstString port_prefix_;

    Eigen::VectorXd       prev_ee_pose_;
    Eigen::VectorXd       delta_hand_pose_;
    double                delta_angle_;

    void PlayFwdKinPropagate(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> prop_state);
};

#endif /* PLAYFWDKINMODEL_H */
