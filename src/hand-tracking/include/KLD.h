#ifndef KLD_H
#define KLD_H

#include <BayesFilters/LikelihoodModel.h>

#include <memory>


class KLD : public bfl::LikelihoodModel
{
public:
    KLD(const double likelihood_gain, const std::size_t vector_size) noexcept;

    ~KLD() noexcept;

    std::pair<bool, Eigen::VectorXd> likelihood(const bfl::MeasurementModel& measurement_model, const Eigen::Ref<const Eigen::MatrixXd>& pred_states) override;

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* KLD_H */
