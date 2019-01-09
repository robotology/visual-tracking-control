#ifndef FWDPOSEMODEL_H
#define FWDPOSEMODEL_H

#include <BayesFilters/ExogenousModel.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/sig/Vector.h>


class KinPoseModel : public bfl::ExogenousModel
{
public:
    KinPoseModel() noexcept;

    ~KinPoseModel() noexcept;

    void propagate(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, Eigen::Ref<Eigen::MatrixXf> prop_state) override;

    Eigen::MatrixXf getExogenousMatrix() override;

    bool setProperty(const std::string& property) override;

protected:
    virtual Eigen::VectorXd readPose() = 0;

    bool initialize_delta_ = true;

    bool setDeltaMotion();

private:
    Eigen::VectorXd prev_ee_pose_    = Eigen::VectorXd::Zero(7);

    Eigen::VectorXd delta_hand_pose_ = Eigen::VectorXd::Zero(6);

    double          delta_angle_     = 0.0;
};

#endif /* FWDPOSEMODEL_H */
