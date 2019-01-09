#ifndef PLAYICUBFWDKINMODEL_H
#define PLAYICUBFWDKINMODEL_H

#include <KinPoseModel.h>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Vector.h>

#include <string>


class PlayiCubFwdKinModel : public KinPoseModel
{
public:
    PlayiCubFwdKinModel(const std::string& robot, const std::string& laterality, const std::string& port_prefix) noexcept;

    ~PlayiCubFwdKinModel() noexcept;

    bool setProperty(const std::string& property) override;

protected:
    Eigen::VectorXd readPose() override;

    yarp::sig::Vector readTorso();

    yarp::sig::Vector readRootToEE();

    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;

    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_enc_;

    iCub::iKin::iCubArm icub_kin_arm_;

private:
    const std::string log_ID_ = "[PlayiCubFwdKinModel]";

    const std::string port_prefix_ = "PlayiCubFwdKinModel";

    const std::string robot_;

    const std::string laterality_;
};

#endif /* PLAYICUBFWDKINMODEL_H */
