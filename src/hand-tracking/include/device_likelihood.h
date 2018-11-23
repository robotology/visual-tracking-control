#ifndef DEVICE_LIKELIHOOD_H
#define DEVICE_LIKELIHOOD_H

#include <cstddef>

#include <cublas_v2.h>
#include <opencv2/core/cuda.hpp>
#include <thrust/host_vector.h>


namespace bfl
{
namespace cuda
{

::thrust::host_vector<float> kld(cublasHandle_t handle, const cv::cuda::GpuMat& p, const cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number, const bool check_pmf = false);

::thrust::host_vector<float> chisquare(cublasHandle_t handle, const cv::cuda::GpuMat& p, const cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

::thrust::host_vector<float> normone(cublasHandle_t handle, const cv::cuda::GpuMat& p, const cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

::thrust::host_vector<float> normtwo(cublasHandle_t handle, const cv::cuda::GpuMat& p, const cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

::thrust::host_vector<float> normtwo_kld(cublasHandle_t handle, const cv::cuda::GpuMat& p, const cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number, const bool check_pmf = false);

::thrust::host_vector<float> normtwo_kld_chisquare(cublasHandle_t handle, const cv::cuda::GpuMat& p, const cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number, const bool check_pmf = false);

}
}

#endif /* DEVICE_LIKELIHOOD_H */
