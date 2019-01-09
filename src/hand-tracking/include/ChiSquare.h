#ifndef CHISQUARE_H
#define CHISQUARE_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class ChiSquare : public bfl::LikelihoodModel
{
public:
    ChiSquare(const double likelihood_gain, const std::size_t vector_size) noexcept;

    ~ChiSquare() noexcept;

    std::pair<bool, Eigen::VectorXd> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXd>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* CHISQUARE_H */
