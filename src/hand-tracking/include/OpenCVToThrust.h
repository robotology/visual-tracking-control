#ifndef OPENCVTOTHRUST_H
#define OPENCVTOTHRUST_H

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/traits.hpp>


template<typename T>
struct step_functor : public thrust::unary_function<int, int>
{
    int columns;
    int step;
    int channels;


    __host__ __device__
    step_functor(int columns_, int step_, int channels_ = 1) :
        columns(columns_),
        step(step_),
        channels(channels_) { };


    __host__
    step_functor(cv::cuda::GpuMat& mat)
    {
        CV_Assert(mat.depth() == cv::DataType<T>::depth);
        columns = mat.cols;
        step = mat.step / sizeof(T);
        channels = mat.channels();
    }


    __host__ __device__
    int operator()(int x) const
    {
        int row = x / columns;
        int idx = (row * step) + (x % columns)*channels;
        return idx;
    }
};


template<typename T>
using OCVIterator = thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>;


/**
 * GpuMatBeginItr generates a thrust-compatible iterator representing the begin of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the begin of a GPU mat's memory.
 */
template<typename T>
OCVIterator<T> GpuMatBeginItr(cv::cuda::GpuMat& mat, int channel = 0)
{
    if (channel == -1)
    {
        mat = mat.reshape(1);
        channel = 0;
    }

    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());

    return thrust::make_permutation_iterator(thrust::device_pointer_cast(mat.ptr<T>(0) + channel), thrust::make_transform_iterator(thrust::make_counting_iterator(0), step_functor<T>(mat.cols, static_cast<int>(mat.step / sizeof(T)), mat.channels())));
}


/**
 * GpuMatEndItr generates a thrust-compatible iterator representing the end of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
OCVIterator<T> GpuMatEndItr(cv::cuda::GpuMat& mat, int channel = 0)
{
    if (channel == -1)
    {
        mat = mat.reshape(1);
        channel = 0;
    }

    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());

    return thrust::make_permutation_iterator(thrust::device_pointer_cast(mat.ptr<T>(0) + channel), thrust::make_transform_iterator(thrust::make_counting_iterator(mat.rows * mat.cols), step_functor<T>(mat.cols, static_cast<int>(mat.step / sizeof(T)), mat.channels())));
}


/**
 * GpuMatEndItr generates a thrust-compatible iterator representing the end of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
OCVIterator<T> GpuMatMidItr(cv::cuda::GpuMat& mat, const int offset, int channel = 0)
{
    if (channel == -1)
    {
        mat = mat.reshape(1);
        channel = 0;
    }

    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());

    return thrust::make_permutation_iterator(thrust::device_pointer_cast(mat.ptr<T>(0) + channel), thrust::make_transform_iterator(thrust::make_counting_iterator(offset), step_functor<T>(mat.cols, static_cast<int>(mat.step / sizeof(T)), mat.channels())));
}

#endif /* OPENCVTOTHRUST_H */
