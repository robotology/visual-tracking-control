#ifndef HISTOGRAMNORMTWOCHI_H
#define HISTOGRAMNORMTWOCHI_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class HistogramNormTwoChi : public bfl::LikelihoodModel
{
public:
    HistogramNormTwoChi(const double likelihood_gain, const int histogram_size) noexcept;

    ~HistogramNormTwoChi() noexcept;

    std::pair<bool, Eigen::VectorXf> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states) override;

private:
    struct ImplHNTC;

    std::unique_ptr<ImplHNTC> pImpl_;
};

#endif /* HISTOGRAMNORMTWOCHI_H */
