#ifndef INITICUBARM_H
#define INITICUBARM_H

#include <BayesFiltersLib/Initialization.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class InitiCubArm : public bfl::Initialization
{
public:
    InitiCubArm(yarp::os::ConstString cam_sel, yarp::os::ConstString laterality) noexcept;

    ~InitiCubArm() noexcept;

    void initialize(Eigen::Ref<Eigen::MatrixXf> state, Eigen::Ref<Eigen::VectorXf> weight) override;

private:
    iCub::iKin::iCubArm                      icub_kin_arm_;
    iCub::iKin::iCubFinger                   icub_kin_finger_[3];
    yarp::os::BufferedPort<yarp::os::Bottle> port_torso_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle> port_arm_enc_;

    yarp::sig::Vector                        readTorso();

    yarp::sig::Vector                        readRootToEE();
};

#endif /* INITICUBARM_H */
