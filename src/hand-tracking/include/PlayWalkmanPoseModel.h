#ifndef PLAYWALKMANPOSEMODEL_H
#define PLAYWALKMANPOSEMODEL_H

#include <KinPoseModel.h>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class PlayWalkmanPoseModel : public KinPoseModel
{
public:
    PlayWalkmanPoseModel(const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept;

    ~PlayWalkmanPoseModel() noexcept;

    bool setProperty(const std::string& property) override;

protected:
    Eigen::VectorXd   readPose() override;

    yarp::sig::Vector readRootToEE();

    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_pose_;

private:
    const yarp::os::ConstString log_ID_ = "[PlayWalkmanPoseModel]";

    yarp::os::ConstString port_prefix_ = "PlayWalkmanPoseModel";

    yarp::os::ConstString robot_;

    yarp::os::ConstString laterality_;
};

#endif /* PLAYWALKMANPOSEMODEL_H */
