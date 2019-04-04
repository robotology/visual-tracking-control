/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef DRAWFWDKINPOSES_H
#define DRAWFWDKINPOSES_H

#include <memory>
#include <random>

#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/StateModel.h>

namespace bfl {
    class DrawFwdKinPoses;
}


class bfl::DrawFwdKinPoses : public bfl::PFPrediction
{
public:
    DrawFwdKinPoses() noexcept;

    DrawFwdKinPoses(DrawFwdKinPoses&& pf_prediction) noexcept;

    ~DrawFwdKinPoses() noexcept;

    DrawFwdKinPoses& operator=(DrawFwdKinPoses&& pf_prediction) noexcept;


    StateModel& getStateModel() override;

    void setStateModel(std::unique_ptr<StateModel> state_model) override;

    ExogenousModel& getExogenousModel() override;

    void setExogenousModel(std::unique_ptr<ExogenousModel> exogenous_model) override;

protected:
    void predictStep(const Eigen::Ref<const Eigen::MatrixXf>& prev_states, const Eigen::Ref<const Eigen::VectorXf>& prev_weights,
                     Eigen::Ref<Eigen::MatrixXf> pred_states, Eigen::Ref<Eigen::VectorXf> pred_weights) override;

    std::unique_ptr<StateModel> state_model_;

    std::unique_ptr<ExogenousModel> exogenous_model_;
};

#endif /* DRAWFWDKINPOSES_H */
