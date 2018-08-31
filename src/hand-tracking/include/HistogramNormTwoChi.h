#ifndef HISTOGRAMNORMTWOCHI_H
#define HISTOGRAMNORMTWOCHI_H

#include <BayesFilters/LikelihoodModel.h>


class HistogramNormTwoChi : public bfl::LikelihoodModel
{
public:
    HistogramNormTwoChi(const double likelihood_gain, const int histogram_size) noexcept;

    ~HistogramNormTwoChi() noexcept;

    std::pair<bool, Eigen::VectorXf> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states) override;

protected:
    double likelihood_gain_;

    int histogram_size_;
};

#endif /* HISTOGRAMNORMTWOCHI_H */
