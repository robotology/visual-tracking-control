#ifndef HISTOGRAMNORMONE_H
#define HISTOGRAMNORMONE_H

#include <BayesFilters/LikelihoodModel.h>


class HistogramNormOne : public bfl::LikelihoodModel
{
public:
    HistogramNormOne(const double likelihood_gain, const int histogram_size) noexcept;

    ~HistogramNormOne() noexcept;

    std::pair<bool, Eigen::VectorXf> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states) override;

protected:
    double likelihood_gain_;

    int histogram_size_;
};

#endif /* HISTOGRAMNORMONE_H */
