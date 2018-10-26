#ifndef DEVICELIKELIHOOD_H
#define DEVICELIKELIHOOD_H

#include <cstddef>

#include <opencv2/core/cuda.hpp>
#include <thrust/host_vector.h>


thrust::host_vector<float> kld_device(cv::cuda::GpuMat& p, cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

thrust::host_vector<float> normtwo_device(cv::cuda::GpuMat& p, cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

thrust::host_vector<float> euclidean_kld_device(cv::cuda::GpuMat& p, cv::cuda::GpuMat& q, const std::size_t histogram_size, const std::size_t histogram_number);

#endif /* DEVICELIKELIHOOD_H */
