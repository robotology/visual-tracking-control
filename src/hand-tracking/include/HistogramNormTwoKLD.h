#ifndef HISTOGRAMNORMTWOKLD_H
#define HISTOGRAMNORMTWOKLD_H

#include <BayesFilters/LikelihoodModel.h>


class HistogramNormTwoKLD : public bfl::LikelihoodModel
{
public:
    HistogramNormTwoKLD(const double likelihood_gain, const int histogram_size) noexcept;

    ~HistogramNormTwoKLD() noexcept;

    std::pair<bool, Eigen::VectorXf> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states) override;

protected:
    double likelihood_gain_;

    int histogram_size_;
};

#endif /* HISTOGRAMNORMTWOKLD_H */
