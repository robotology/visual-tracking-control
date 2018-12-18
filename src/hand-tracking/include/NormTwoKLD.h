#ifndef NORMTWOKLD_H
#define NORMTWOKLD_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class NormTwoKLD : public bfl::LikelihoodModel
{
public:
    NormTwoKLD(const double likelihood_gain, const std::size_t vector_size) noexcept;

    ~NormTwoKLD() noexcept;

    std::pair<bool, Eigen::VectorXf> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* NORMTWOKLD_H */
