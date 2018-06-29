#ifndef VISUALUPDATEPARTICLES_H
#define VISUALUPDATEPARTICLES_H

#include <BayesFilters/LikelihoodModel.h>


class HistogramNormChi : public bfl::LikelihoodModel
{
public:
    HistogramNormChi(const double likelihood_gain, const int histogram_size) noexcept;

    ~HistogramNormChi() noexcept;

    std::pair<bool, Eigen::VectorXf> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states) override;

protected:
    double likelihood_gain_;

    int histogram_size_;
};

#endif /* VISUALUPDATEPARTICLES_H */
