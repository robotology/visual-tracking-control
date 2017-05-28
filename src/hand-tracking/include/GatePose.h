#ifndef GATEPOSE_H
#define GATEPOSE_H

#include <string>

#include <BayesFilters/VisualCorrectionDecorator.h>
#include <BayesFilters/StateModel.h>


class GatePose : public bfl::VisualCorrectionDecorator
{
public:
    /* Constructor */
    GatePose(std::unique_ptr<VisualCorrection> visual_correction,
             const double gate_x, const double gate_y, const double gate_z,
             const double gate_rotation, const double gate_aperture) noexcept;

    GatePose(std::unique_ptr<VisualCorrection> visual_correction) noexcept;

    /* Default constructor, disabled */
    GatePose() = delete;

    /* Destructor */
    ~GatePose() noexcept override;

    /* Move constructor, disabled */
    GatePose(GatePose&& visual_correction) = delete;

    /* Move assignment operator, disabled */
    GatePose& operator=(GatePose&& visual_correction) = delete;

    void correct(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

    void innovation(const Eigen::Ref<const Eigen::MatrixXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> innovation) override;

    void likelihood(const Eigen::Ref<const Eigen::MatrixXf>& innovation, Eigen::Ref<Eigen::MatrixXf> cor_state) override;

    bool setObservationModelProperty(const std::string& property) override;

protected:
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
