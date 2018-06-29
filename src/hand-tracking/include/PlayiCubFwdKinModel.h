#ifndef PLAYICUBFWDKINMODEL_H
#define PLAYICUBFWDKINMODEL_H

#include <KinPoseModel.h>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class PlayiCubFwdKinModel : public KinPoseModel
{
public:
    PlayiCubFwdKinModel(const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

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
    const yarp::os::ConstString log_ID_ = "[PlayiCubFwdKinModel]";

    yarp::os::ConstString port_prefix_ = "PlayiCubFwdKinModel";

    yarp::os::ConstString robot_;

    yarp::os::ConstString laterality_;
};

#endif /* PLAYICUBFWDKINMODEL_H */
