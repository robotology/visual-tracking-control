#include <InitPoseParticles.h>

#include <iCub/ctrl/math.h>

using namespace bfl;
using namespace Eigen;


bool InitPoseParticles::initialize(ParticleSet& particles)
{
    VectorXd pose = readPose();

    for (int i = 0; i < particles.state().cols(); ++i)
        particles.state(i) << pose;

    particles.weight().fill(1.0 / particles.state().cols());

    return true;
}
