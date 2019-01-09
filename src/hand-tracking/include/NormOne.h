#ifndef NORMONE_H
#define NORMONE_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class NormOne : public bfl::LikelihoodModel
{
public:
    NormOne(const double likelihood_gain) noexcept;

    ~NormOne() noexcept;

    std::pair<bool, Eigen::VectorXd> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXd>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* NORMONE_H */
