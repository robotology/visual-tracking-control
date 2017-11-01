#ifndef ICUBFWDKINMODEL_H
#define ICUBFWDKINMODEL_H

#include <BayesFilters/StateModelDecorator.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class iCubFwdKinModel : public bfl::StateModelDecorator
{
public:
    iCubFwdKinModel(std::unique_ptr<StateModel> state_model, const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    ~iCubFwdKinModel() noexcept;

    void propagate(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> prop_state) override;

    void motion(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> mot_state) override;

    Eigen::MatrixXf getNoiseSample(const int num) override;

    Eigen::MatrixXf getNoiseCovarianceMatrix() override;

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
    yarp::os::ConstString  ID_     = "iCubFwdKinModel";
    yarp::os::ConstString  log_ID_ = "[" + ID_ + "]";
    
    yarp::os::ConstString  robot_;
    yarp::os::ConstString  laterality_;
    yarp::os::ConstString  port_prefix_;

    Eigen::VectorXd        prev_ee_pose_;
    Eigen::VectorXd        delta_hand_pose_;
    double                 delta_angle_;

    void iCubFwdKinPropagate(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> prop_state);
};

#endif /* ICUBFWDKINMODEL_H */
