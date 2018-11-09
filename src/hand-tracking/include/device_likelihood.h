#ifndef DEVICE_LIKELIHOOD_H
#define DEVICE_LIKELIHOOD_H

#include <cstddef>

#include <opencv2/core/cuda.hpp>
#include <thrust/host_vector.h>


namespace bfl
{
namespace cuda
{

::thrust::host_vector<float> kld(const cv::cuda::GpuMat& p, const cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number, const bool check_pmf = false);

::thrust::host_vector<float> normtwo(const cv::cuda::GpuMat& p, const cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

::thrust::host_vector<float> normtwo_kld(const cv::cuda::GpuMat& p, const cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number, const bool check_pmf = false);

}
}

#endif /* DEVICE_LIKELIHOOD_H */
