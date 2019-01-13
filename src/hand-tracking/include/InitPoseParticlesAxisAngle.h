#ifndef INITPOSEPARTICLESAXISANGLE_H
#define INITPOSEPARTICLESAXISANGLE_H

#include <InitPoseParticles.h>


class InitPoseParticlesAxisAngle : public InitPoseParticles
{
public:
    virtual ~InitPoseParticlesAxisAngle() noexcept;

protected:
    Eigen::VectorXd readPose() override;

    virtual Eigen::VectorXd readPoseAxisAngle() = 0;
};


#endif /* INITPOSEPARTICLESAXISANGLE_H */
