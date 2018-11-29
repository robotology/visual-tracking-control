#ifndef NORMTWOCHI_H
#define NORMTWOCHI_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class NormTwoChi : public bfl::LikelihoodModel
{
public:
    NormTwoChi(const double likelihood_gain, const std::size_t vector_size) noexcept;

    ~NormTwoChi() noexcept;

    std::pair<bool, Eigen::VectorXf> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* NORMTWOCHI_H */
