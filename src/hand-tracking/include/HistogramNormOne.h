#ifndef HISTOGRAMNORMONE_H
#define HISTOGRAMNORMONE_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class HistogramNormOne : public bfl::LikelihoodModel
{
public:
    HistogramNormOne(const double likelihood_gain) noexcept;

    ~HistogramNormOne() noexcept;

    std::pair<bool, Eigen::VectorXf> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* HISTOGRAMNORMONE_H */
