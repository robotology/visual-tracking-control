#ifndef RESAMPLINGWITHPRIOR_H
#define RESAMPLINGWITHPRIOR_H

#include <InitiCubArm.h>

#include <BayesFilters/Resampling.h>
#include <Eigen/Dense>


class ResamplingWithPrior : public bfl::Resampling,
                            protected InitiCubArm
{
public:
    ResamplingWithPrior(const unsigned int seed, const yarp::os::ConstString& port_prefix, const yarp::os::ConstString& cam_sel, const yarp::os::ConstString& laterality) noexcept;

    ResamplingWithPrior(const yarp::os::ConstString& port_prefix, const yarp::os::ConstString& cam_sel, const yarp::os::ConstString& laterality) noexcept;

    ResamplingWithPrior(const yarp::os::ConstString& cam_sel, const yarp::os::ConstString& laterality) noexcept;

    ~ResamplingWithPrior() noexcept;

    void resample(const Eigen::Ref<const Eigen::MatrixXf>& pred_particles, const Eigen::Ref<const Eigen::VectorXf>& cor_weights,
                  Eigen::Ref<Eigen::MatrixXf> res_particles, Eigen::Ref<Eigen::VectorXf> res_weights, Eigen::Ref<Eigen::VectorXf> res_parents) override;
};

#endif /* RESAMPLINGWITHPRIOR_H */
