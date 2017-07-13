#include "ResamplingWithPrior.h"

#include <vector>

using namespace bfl;
using namespace Eigen;
using namespace yarp::os;


ResamplingWithPrior::ResamplingWithPrior(const unsigned int seed, const ConstString& port_prefix, const ConstString& cam_sel, const ConstString& laterality) noexcept :
    Resampling(seed), InitiCubArm(port_prefix, cam_sel, laterality) { }


ResamplingWithPrior::ResamplingWithPrior(const ConstString& port_prefix, const ConstString& cam_sel, const ConstString& laterality) noexcept :
    ResamplingWithPrior(1, port_prefix, cam_sel, laterality) { }


ResamplingWithPrior::ResamplingWithPrior(const ConstString& cam_sel, const ConstString& laterality) noexcept :
    ResamplingWithPrior(1, "ResamplingWithPrior", cam_sel, laterality) { }


ResamplingWithPrior::~ResamplingWithPrior() noexcept { }


std::vector<unsigned int> sort_indexes(const Ref<const VectorXf>& vector)
{
    std::vector<unsigned int> idx(vector.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(),
         [&vector](size_t idx1, size_t idx2) { return vector[idx1] < vector[idx2]; });

    return idx;
}


void ResamplingWithPrior::resample(const Ref<const MatrixXf>& pred_particles, const Ref<const VectorXf>& cor_weights,
                                   Ref<MatrixXf> res_particles, Ref<VectorXf> res_weights, Ref<VectorXf> res_parents)
{
    int num_init_particles     = static_cast<int>(std::floor(pred_particles.cols() / 10.0));
    int num_resample_particles = pred_particles.cols() - num_init_particles;

    MatrixXf tmp_particles(pred_particles.rows(), num_resample_particles);
    VectorXf tmp_weights(num_resample_particles);

    int j = 0;
    for (int i: sort_indexes(cor_weights))
    {
        if (j < num_init_particles)
        {
            res_particles.col(j) = pred_particles.col(i);
            res_weights(j)       = cor_weights(i);
        }
        else
        {
            tmp_particles.col(j - num_init_particles) = pred_particles.col(i);
            tmp_weights(j - num_init_particles)       = cor_weights(i);
        }
        j++;
    }

    initialize(res_particles.leftCols(num_init_particles), res_weights.head(num_init_particles));

    Resampling::resample(tmp_particles, tmp_weights / tmp_weights.sum(),
                         res_particles.rightCols(num_resample_particles), res_weights.tail(num_resample_particles), res_parents);

    res_weights.setConstant(1.0 / pred_particles.cols());
}
