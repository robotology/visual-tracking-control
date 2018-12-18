#ifndef INITPOSEPARTICLES_H
#define INITPOSEPARTICLES_H

#include <BayesFilters/ParticleSetInitialization.h>

#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Vector.h>


class InitPoseParticles : public bfl::ParticleSetInitialization
{
public:
    virtual ~InitPoseParticles() noexcept { };

    bool initialize(Eigen::Ref<Eigen::MatrixXf> state, Eigen::Ref<Eigen::VectorXf> weight) override;

protected:
    virtual Eigen::VectorXd readPose() = 0;
};


#endif /* INITPOSEPARTICLES_H */
