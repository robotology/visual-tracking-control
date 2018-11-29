#ifndef NORMTWO_H
#define NORMTWO_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class NormTwo : public bfl::LikelihoodModel
{
public:
    NormTwo(const double likelihood_gain, const std::size_t vector_size) noexcept;

    ~NormTwo() noexcept;

    std::pair<bool, Eigen::VectorXf> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXf>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* NORMTWO_H */
