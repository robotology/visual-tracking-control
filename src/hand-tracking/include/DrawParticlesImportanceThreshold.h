/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef DRAWPARTICLESIMPORTANCETHRESHOLD_H
#define DRAWPARTICLESIMPORTANCETHRESHOLD_H

#include <BayesFilters/ParticleSet.h>
#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/StateModel.h>

#include <memory>
#include <random>

namespace bfl {
    class DrawParticlesImportanceThreshold;
}


class bfl::DrawParticlesImportanceThreshold : public bfl::PFPrediction
{
public:
    DrawParticlesImportanceThreshold() noexcept;

    DrawParticlesImportanceThreshold(DrawParticlesImportanceThreshold&& pf_prediction) noexcept;

    ~DrawParticlesImportanceThreshold() noexcept;

    DrawParticlesImportanceThreshold& operator=(DrawParticlesImportanceThreshold&& pf_prediction) noexcept;

    StateModel& getStateModel() override;

    void setStateModel(std::unique_ptr<StateModel> state_model) override;

    ExogenousModel& getExogenousModel() override;

    void setExogenousModel(std::unique_ptr<ExogenousModel> exogenous_model) override;

protected:
    void predictStep(const bfl::ParticleSet& prev_particles, bfl::ParticleSet& pred_particles) override;

    std::unique_ptr<StateModel> state_model_;

    std::unique_ptr<ExogenousModel> exogenous_model_;
};

#endif /* DRAWPARTICLESIMPORTANCETHRESHOLD_H */
