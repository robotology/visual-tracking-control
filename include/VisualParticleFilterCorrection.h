#ifndef VISUALPARTICLEFILTERCORRECTION_H
#define VISUALPARTICLEFILTERCORRECTION_H

#include <BayesFiltersLib/VisualCorrection.h>
#include <BayesFiltersLib/VisualObservationModel.h>


class VisualParticleFilterCorrection : public bfl::VisualCorrection {
public:
    /* Default constructor, disabled */
    VisualParticleFilterCorrection() = delete;

    /* VPF correction constructor */
    VisualParticleFilterCorrection(std::shared_ptr<bfl::VisualObservationModel> observation_model) noexcept;

    /* Destructor */
    ~VisualParticleFilterCorrection() noexcept override;

    /* Copy constructor */
    VisualParticleFilterCorrection(const VisualParticleFilterCorrection& vpf_correction);

    /* Move constructor */
    VisualParticleFilterCorrection(VisualParticleFilterCorrection&& vpf_correction) noexcept;

    /* Copy assignment operator */
    VisualParticleFilterCorrection& operator=(const VisualParticleFilterCorrection& vpf_correction);

    /* Move assignment operator */
    VisualParticleFilterCorrection& operator=(VisualParticleFilterCorrection&& vpf_correction) noexcept;

    void correct(const Eigen::Ref<const Eigen::VectorXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::VectorXf> cor_state) override;

    void innovation(const Eigen::Ref<const Eigen::VectorXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> innovation) override;

    void likelihood(const Eigen::Ref<const Eigen::MatrixXf>& innovation, Eigen::Ref<Eigen::VectorXf> cor_state) override;

protected:
    std::shared_ptr<bfl::VisualObservationModel> measurement_model_;
};

#endif /* VISUALPARTICLEFILTERCORRECTION_H */
