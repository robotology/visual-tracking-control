#ifndef DEVICELIKELIHOOD_H
#define DEVICELIKELIHOOD_H

#include <cstddef>

#include <opencv2/core/cuda.hpp>
#include <thrust/host_vector.h>


namespace bfl
{
namespace cuda
{

thrust::host_vector<float> kld(cv::cuda::GpuMat& p, cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

thrust::host_vector<float> normtwo(cv::cuda::GpuMat& p, cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

thrust::host_vector<float> normtwo_kld(cv::cuda::GpuMat& p, cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

}
}

#endif /* DEVICELIKELIHOOD_H */

