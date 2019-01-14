#ifndef NORMTWOCHISQUARE_H
#define NORMTWOCHISQUARE_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class NormTwoChiSquare : public bfl::LikelihoodModel
{
public:
    NormTwoChiSquare(const double likelihood_gain, const std::size_t vector_size) noexcept;

    ~NormTwoChiSquare() noexcept;

    std::pair<bool, Eigen::VectorXd> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXd>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* NORMTWOCHISQUARE_H */
