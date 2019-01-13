#ifndef KINPOSEMODELAXISANGLE_H
#define KINPOSEMODELAXISANGLE_H

#include <KinPoseModel.h>

#include <Eigen/Dense>

class KinPoseModelAxisAngle : public KinPoseModel
{
public:
    virtual ~KinPoseModelAxisAngle() noexcept;

protected:
    Eigen::VectorXd readPose() override;

    virtual Eigen::VectorXd readPoseAxisAngle() = 0;
};

#endif /* KINPOSEMODELAXISANGLE_H */
