#ifndef THRUST_OPENCV_ITERATOR_H
#define THRUST_OPENCV_ITERATOR_H

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/traits.hpp>


namespace bfl
{
namespace thrust
{


template<typename T>
class step_functor : public ::thrust::unary_function<int, int>
{
public:
    __host__ __device__
    step_functor(int columns_, int step_, int channels_ = 1) :
        columns(columns_),
        step(step_),
        channels(channels_)
    { };


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
        int idx = (row * step) + (x % columns) * channels;
        return idx;
    }

private:
    int columns;
    int step;
    int channels;
};


template<typename T>
using ocv_iterator = ::thrust::permutation_iterator<::thrust::device_ptr<T>, ::thrust::transform_iterator<step_functor<T>, ::thrust::counting_iterator<int>>>;


template<typename T>
using const_ocv_iterator = ::thrust::permutation_iterator<::thrust::device_ptr<const T>, ::thrust::transform_iterator<step_functor<T>, ::thrust::counting_iterator<int>>>;


/**
 * GpuMat_begin_iterator generates a thrust-compatible iterator representing the begin of a given channel of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the begin of a GPU mat's memory.
 */
template<typename T>
ocv_iterator<T> GpuMat_begin_channel_iterator(cv::cuda::GpuMat& mat, const int channel)
{
    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat.ptr<T>(0) + channel), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(0), step_functor<T>(mat.cols, static_cast<int>(mat.step / sizeof(T)), mat.channels())));
}


/**
 * GpuMat_begin_iterator generates a thrust-compatible const iterator representing the begin of a given channel of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the begin of a GPU mat's memory.
 */
template<typename T>
const_ocv_iterator<T> GpuMat_begin_channel_iterator(const cv::cuda::GpuMat& mat, const int channel)
{
    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat.ptr<T>(0) + channel), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(0), step_functor<T>(mat.cols, static_cast<int>(mat.step / sizeof(T)), mat.channels())));
}


/**
 * GpuMat_end_iterator generates a thrust-compatible iterator representing the end of a given channel of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
ocv_iterator<T> GpuMat_end_channel_iterator(cv::cuda::GpuMat& mat, const int channel)
{
    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat.ptr<T>(0) + channel), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(mat.rows * mat.cols), step_functor<T>(mat.cols, static_cast<int>(mat.step / sizeof(T)), mat.channels())));
}


/**
 * GpuMat_end_iterator generates a thrust-compatible const iterator representing the end of a given channel of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
const_ocv_iterator<T> GpuMat_end_channel_iterator(const cv::cuda::GpuMat& mat, const int channel)
{
    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat.ptr<T>(0) + channel), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(mat.rows * mat.cols), step_functor<T>(mat.cols, static_cast<int>(mat.step / sizeof(T)), mat.channels())));
}


/**
 * GpuMat_end_iterator generates a thrust-compatible iterator representing a given position of a given channel of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
ocv_iterator<T> GpuMat_position_channel_iterator(cv::cuda::GpuMat& mat, const int position, const int channel)
{
    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat.ptr<T>(0) + channel), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(position), step_functor<T>(mat.cols, static_cast<int>(mat.step / sizeof(T)), mat.channels())));
}


/**
 * GpuMat_end_iterator generates a thrust-compatible const iterator representing a given position of a given channel of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
const_ocv_iterator<T> GpuMat_position_channel_iterator(const cv::cuda::GpuMat& mat, const int position, const int channel)
{
    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat.ptr<T>(0) + channel), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(position), step_functor<T>(mat.cols, static_cast<int>(mat.step / sizeof(T)), mat.channels())));
}


/**
 * GpuMat_begin_iterator generates a thrust-compatible iterator representing the begin of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the begin of a GPU mat's memory.
 */
template<typename T>
ocv_iterator<T> GpuMat_begin_iterator(cv::cuda::GpuMat& mat)
{
    cv::cuda::GpuMat mat_reshaped = mat.reshape(1);

    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat_reshaped.ptr<T>(0)), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(0), step_functor<T>(mat_reshaped.cols, static_cast<int>(mat_reshaped.step / sizeof(T)), mat_reshaped.channels())));
}


/**
 * GpuMat_begin_iterator generates a thrust-compatible const iterator representing the begin of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the begin of a GPU mat's memory.
 */
template<typename T>
const_ocv_iterator<T> GpuMat_begin_iterator(const cv::cuda::GpuMat& mat)
{
    const cv::cuda::GpuMat mat_reshaped = mat.reshape(1);

    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat_reshaped.ptr<const T>(0)), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(0), step_functor<T>(mat_reshaped.cols, static_cast<int>(mat_reshaped.step / sizeof(T)), mat_reshaped.channels())));
}


/**
 * GpuMat_end_iterator generates a thrust-compatible iterator representing the end of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
ocv_iterator<T> GpuMat_end_iterator(cv::cuda::GpuMat& mat)
{
    cv::cuda::GpuMat mat_reshaped = mat.reshape(1);

    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat_reshaped.ptr<T>(0)), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(mat_reshaped.rows * mat_reshaped.cols), step_functor<T>(mat_reshaped.cols, static_cast<int>(mat_reshaped.step / sizeof(T)), mat_reshaped.channels())));
}


/**
 * GpuMat_end_iterator generates a thrust-compatible const iterator representing the end of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
const_ocv_iterator<T> GpuMat_end_iterator(const cv::cuda::GpuMat& mat)
{
    const cv::cuda::GpuMat mat_reshaped = mat.reshape(1);

    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat_reshaped.ptr<const T>(0)), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(mat_reshaped.rows * mat_reshaped.cols), step_functor<T>(mat_reshaped.cols, static_cast<int>(mat_reshaped.step / sizeof(T)), mat_reshaped.channels())));
}


/**
 * GpuMat_end_iterator generates a thrust-compatible iterator representing a given position of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
ocv_iterator<T> GpuMat_position_iterator(cv::cuda::GpuMat& mat, const int position)
{
    cv::cuda::GpuMat mat_reshaped = mat.reshape(1);

    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat_reshaped.ptr<T>(0)), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(position), step_functor<T>(mat_reshaped.cols, static_cast<int>(mat_reshaped.step / sizeof(T)), mat_reshaped.channels())));
}


/**
 * GpuMat_end_iterator generates a thrust-compatible const iterator representing a given position of a GPU-allocated matrix memory.
 * @param mat is the input matrix.
 * @param channel is the channel of the matrix that the iterator is accessing. If set to -1, the iterator will access every element in sequential order.
 * @return a thrust-compatible iterator to the end of a GPU mat's memory.
 */
template<typename T>
const_ocv_iterator<T> GpuMat_position_iterator(const cv::cuda::GpuMat& mat, const int position, const int channel = 0)
{
    const cv::cuda::GpuMat mat_reshaped = mat.reshape(1);

    return ::thrust::make_permutation_iterator(::thrust::device_pointer_cast(mat_reshaped.ptr<const T>(0)), ::thrust::make_transform_iterator(::thrust::make_counting_iterator(position), step_functor<T>(mat_reshaped.cols, static_cast<int>(mat_reshaped.step / sizeof(T)), mat_reshaped.channels())));
}


}
}

#endif /* THRUST_OPENCV_ITERATOR_H */
