#ifndef INITPOSEPARTICLES_H
#define INITPOSEPARTICLES_H

#include <BayesFilters/Initialization.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class InitPoseParticles : public bfl::Initialization
{
public:
    virtual ~InitPoseParticles() noexcept { };

    void initialize(Eigen::Ref<Eigen::MatrixXf> state, Eigen::Ref<Eigen::VectorXf> weight) override;

protected:
    virtual Eigen::VectorXd readPose() = 0;
};


#endif /* INITPOSEPARTICLES_H */
