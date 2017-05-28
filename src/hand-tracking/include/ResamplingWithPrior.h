#ifndef RESAMPLINGWITHPRIOR_H
#define RESAMPLINGWITHPRIOR_H

#include "InitiCubArm.h"

#include <random>

#include <BayesFiltersLib/Resampling.h>
#include <Eigen/Dense>


class ResamplingWithPrior : public bfl::Resampling,
                            protected InitiCubArm
{
public:
    /* Resampling complete constructor */
    ResamplingWithPrior(unsigned int seed, yarp::os::ConstString port_prefix, yarp::os::ConstString cam_sel, yarp::os::ConstString laterality) noexcept;

    ResamplingWithPrior(yarp::os::ConstString port_prefix, yarp::os::ConstString cam_sel, yarp::os::ConstString laterality) noexcept;

    ResamplingWithPrior(yarp::os::ConstString cam_sel, yarp::os::ConstString laterality) noexcept;

    /* Destructor */
    ~ResamplingWithPrior() noexcept;

    void resample(const Eigen::Ref<const Eigen::MatrixXf>& pred_particles, const Eigen::Ref<const Eigen::VectorXf>& cor_weights,
                  Eigen::Ref<Eigen::MatrixXf> res_particles, Eigen::Ref<Eigen::VectorXf> res_weights, Eigen::Ref<Eigen::VectorXf> res_parents) override;
};

#endif /* RESAMPLINGWITHPRIOR_H */
