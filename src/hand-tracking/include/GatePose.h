#ifndef GATEPOSE_H
#define GATEPOSE_H

#include <string>

#include <BayesFilters/PFCorrectionDecorator.h>
#include <BayesFilters/StateModel.h>


class GatePose : public bfl::PFCorrectionDecorator
{
public:
    GatePose(std::unique_ptr<PFCorrection> visual_correction,
             const double gate_x, const double gate_y, const double gate_z,
             const double gate_rotation,
             const double gate_aperture) noexcept;

    GatePose(std::unique_ptr<PFCorrection> visual_correction) noexcept;

    ~GatePose() noexcept;

protected:
    void correctStep(const Eigen::Ref<const Eigen::MatrixXf>& pred_states, const Eigen::Ref<const Eigen::VectorXf>& pred_weights,
                     Eigen::Ref<Eigen::MatrixXf> cor_states, Eigen::Ref<Eigen::VectorXf> cor_weights) override;

    virtual Eigen::VectorXd readPose() = 0;

    bool isInsideEllipsoid(const Eigen::Ref<const Eigen::VectorXf>& state);

    bool isWithinRotation(float rot_angle);

    bool isInsideCone(const Eigen::Ref<const Eigen::VectorXf>& state);

private:
    double gate_x_;

    double gate_y_;

    double gate_z_;

    double gate_aperture_;
    
    double gate_rotation_;

    Eigen::VectorXd ee_pose_;
};

#endif /* GATEPOSE_H */
