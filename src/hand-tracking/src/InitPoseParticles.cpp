#include <InitPoseParticles.h>

#include <iCub/ctrl/math.h>

using namespace Eigen;


bool InitPoseParticles::initialize(Eigen::Ref<Eigen::MatrixXf> state, Eigen::Ref<Eigen::VectorXf> weight)
{
    VectorXd pose = readPose();

    for (int i = 0; i < state.cols(); ++i)
        state.col(i) << pose.cast<float>();

    weight.fill(1.0 / state.cols());

    return true;
}
