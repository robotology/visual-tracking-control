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
    struct ImplHNO;

    std::unique_ptr<ImplHNO> pImpl_;
};

#endif /* HISTOGRAMNORMONE_H */
