#include <devicelikelihood.h>
#include <OpenCVToThrust.h>

#include <typeinfo>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <nppdefs.h>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>


/* >- ************************************************** -< */
using FloatIterator = thrust::device_vector<float>::iterator;


/* >- ************************************************** -< */
template<typename Iterator>
class StridedIterator
{
public:
    using DifferenceType = typename thrust::iterator_difference<Iterator>::type;

    StridedIterator(Iterator first, Iterator last, DifferenceType stride) :
    first_(first),
    last_(last),
    stride(stride) { }

    struct stride_functor : public thrust::unary_function<DifferenceType, DifferenceType>
    {
        DifferenceType stride;

        stride_functor(DifferenceType stride) :
        stride(stride) { }

        __host__ __device__
        DifferenceType operator()(const DifferenceType& i) const
        {
            return stride * i;
        }
    };

    using CountingIterator = typename thrust::counting_iterator<DifferenceType>;

    using TransformIterator = typename thrust::transform_iterator<stride_functor, CountingIterator>;

    using PermutationIterator = typename thrust::permutation_iterator<Iterator, TransformIterator>;

    PermutationIterator begin() const
    {
        return PermutationIterator(first_, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    PermutationIterator end() const
    {
        return begin() + ((last_ - first_) + (stride - 1)) / stride;
    }

protected:
    Iterator first_;

    Iterator last_;

    DifferenceType stride;
};


/* >- ************************************************** -< */
template<typename Iterator>
class CircularIterator : public thrust::iterator_adaptor<CircularIterator<Iterator>, Iterator>
{
public:
    using base_t = typename thrust::iterator_adaptor<CircularIterator<Iterator>, Iterator>;

    __host__ __device__
    CircularIterator(const Iterator& begin, const Iterator& end) :
    base_t(begin),
    begin_(begin),
    end_(end),
    range_(end - begin),
    upper_bound_(end - begin - 1) { }

    friend class thrust::iterator_core_access;

protected:
    const Iterator begin_;

    const Iterator end_;

    const typename thrust::iterator_adaptor<CircularIterator<Iterator>, Iterator>::difference_type range_;

    const typename thrust::iterator_adaptor<CircularIterator<Iterator>, Iterator>::difference_type upper_bound_;

    typename thrust::iterator_adaptor<CircularIterator<Iterator>, Iterator>::difference_type current_ = 0;

private:
    __host__ __device__
    void advance(typename thrust::iterator_adaptor<CircularIterator<Iterator>, Iterator>::difference_type n)
    {
        n %= range_;
        if ((n + current_) > upper_bound_)
            n -= range_;

        this->base_reference() += n;
        current_ += n;
    }

    __host__ __device__
    void increment()
    {
        ++(this->base_reference());
        ++current_;

        if (this->base_reference() == end_)
        {
            this->base_reference() = begin_;
            current_ = 0;
        }
    }

    __host__ __device__
    void decrement()
    {
        if (this->base_reference() == begin_)
        {
            this->base_reference() = this->m_itEnd;
            current_ = range_;
        }

        --(this->base_reference());
        --current_;
    }
};


/* >- ************************************************** -< */
thrust::device_vector<float> sum_subvectors_in_device(thrust::device_vector<float>& vec, const std::size_t subvector_size)
{
    thrust::inclusive_scan(vec.begin(), vec.end(), vec.begin());

    std::size_t num_element = vec.size() / subvector_size;

    thrust::device_vector<float> out_gpu(num_element);
    out_gpu[0] = vec[subvector_size - 1];

    if (num_element > 1)
    {
        StridedIterator<FloatIterator> base(vec.begin() + (2 * subvector_size) - 1, vec.end(), subvector_size);
        StridedIterator<FloatIterator> to_remove(vec.begin() + subvector_size - 1,  vec.end(), subvector_size);

        thrust::transform(base.begin(), base.end(),
                          to_remove.begin(),
                          out_gpu.begin() + 1,
                          thrust::minus<float>());
    }

    return out_gpu;
}


/* >- ************************************************** -< */
thrust::device_vector<float> sum_subvectors_in_device(cv::cuda::GpuMat& mat, const std::size_t subvector_size)
{
    thrust::device_vector<float> out_inclusive_scan(mat.cols);

    thrust::inclusive_scan(GpuMatBeginItr<float>(mat, -1), GpuMatEndItr<float>(mat, -1),
                           out_inclusive_scan.begin());

    std::size_t num_element = mat.cols / subvector_size;

    thrust::device_vector<float> out_gpu(num_element);
    out_gpu[0] = out_inclusive_scan[subvector_size - 1];

    if (num_element > 1)
    {
        StridedIterator<FloatIterator> base(out_inclusive_scan.begin() + (2 * subvector_size) - 1, out_inclusive_scan.end(), subvector_size);
        StridedIterator<FloatIterator> to_remove(out_inclusive_scan.begin() + subvector_size - 1,  out_inclusive_scan.end(), subvector_size);

        thrust::transform(base.begin(), base.end(),
                          to_remove.begin(),
                          out_gpu.begin() + 1,
                          thrust::minus<float>());
    }

    return out_gpu;
}


/* >- ************************************************** -< */
struct kld_functor_foreach
{
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple& t)
    {
        thrust::get<2>(t) = thrust::get<0>(t) * (log10f(thrust::get<0>(t) + NPP_MINABS_32F) - log10f(thrust::get<1>(t) + NPP_MINABS_32F));
    }
};


/* >- ************************************************** -< */
struct normalize_kld_functor_foreach
{
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple& t)
    {
        thrust::get<2>(t) = (thrust::get<2>(t) / thrust::get<0>(t)) + log10f(thrust::get<1>(t) + NPP_MINABS_32F) - log10f(thrust::get<0>(t) + NPP_MINABS_32F);

        /* The following line is to force 0 on very low numbers, but may result in thread divergence in warps. */
        //        thrust::get<2>(t) = (thrust::get<2>(t) < 0.00001f) ? 0 : thrust::get<2>(t);
    }
};


/* >- ************************************************** -< */
struct sqared_difference_functor_foreach
{
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple& t)
    {
        thrust::get<2>(t) = powf(thrust::get<0>(t) - thrust::get<1>(t), 2.0f);
    }
};


/* >- ************************************************** -< */
struct sqrt_functor : public thrust::unary_function<float, float>
{
    __host__ __device__
    float operator()(const float x) const
    {
        return sqrtf(x);
    }
};


namespace bfl
{
namespace cuda
{

/* >- ************************************************** -< */
struct euclidean_kld_functor_foreach
{
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple& t)
    {
        thrust::get<3>(t) = sqrtf(thrust::get<0>(t)) * ((thrust::get<3>(t) / thrust::get<1>(t)) + log10f(thrust::get<2>(t) + NPP_MINABS_32F) - log10f(thrust::get<1>(t) + NPP_MINABS_32F));

        /* The following line is to force 0 on very low numbers, but may result in thread divergence in warps. */
        //        thrust::get<3>(t) = (thrust::get<3>(t) < 0.00001f) ? 0 : thrust::get<3>(t);
    }
};


/* >- ************************************************** -< */
thrust::host_vector<float> kld
(
    cv::cuda::GpuMat& p,
    cv::cuda::GpuMat& q,
    const std::size_t histogram_size,
    const std::size_t histogram_number
)
{
    thrust::device_vector<float> unnormalized_kld_gpu(q.cols);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<OCVIterator<float>>(GpuMatBeginItr<float>(p, -1), GpuMatEndItr<float>(p, -1)), GpuMatBeginItr<float>(q, -1), unnormalized_kld_gpu.begin())),
                       unnormalized_kld_gpu.size(),
                       kld_functor_foreach());

    thrust::device_vector<float> kld_gpu      = sum_subvectors_in_device(unnormalized_kld_gpu, histogram_size);
    thrust::device_vector<float> subsum_p_gpu = sum_subvectors_in_device(p,                    histogram_size);
    thrust::device_vector<float> subsum_q_gpu = sum_subvectors_in_device(q,                    histogram_size);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<FloatIterator>(subsum_p_gpu.begin(), subsum_p_gpu.end()), subsum_q_gpu.begin(), kld_gpu.begin())),
                       kld_gpu.size(),
                       normalize_kld_functor_foreach());

    thrust::host_vector<float> kld_cpu(sum_subvectors_in_device(kld_gpu, histogram_number));

    return kld_cpu;
}


/* >- ************************************************** -< */
thrust::host_vector<float> normtwo
(
    cv::cuda::GpuMat& p,
    cv::cuda::GpuMat& q,
    const std::size_t histogram_size,
    const std::size_t histogram_number
)
{
    thrust::device_vector<float> sqared_difference_gpu(q.cols);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<OCVIterator<float>>(GpuMatBeginItr<float>(p, -1), GpuMatEndItr<float>(p, -1)), GpuMatBeginItr<float>(q, -1), sqared_difference_gpu.begin())),
                       q.cols,
                       sqared_difference_functor_foreach());

    thrust::device_vector<float> euclidean_dist_gpu = sum_subvectors_in_device(sqared_difference_gpu, histogram_size);

    thrust::transform(euclidean_dist_gpu.begin(), euclidean_dist_gpu.end(),
                      euclidean_dist_gpu.begin(),
                      sqrt_functor());

    thrust::host_vector<float> euclidean_dist_cpu(sum_subvectors_in_device(euclidean_dist_gpu, histogram_number));

    return euclidean_dist_cpu;
}


/* >- ************************************************** -< */
thrust::host_vector<float> normtwo_kld
(
    cv::cuda::GpuMat& p,
    cv::cuda::GpuMat& q,
    const std::size_t histogram_size,
    const std::size_t histogram_number
)
{
    /* Euclidean */
    thrust::device_vector<float> sqared_difference_gpu(q.cols);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(GpuMatBeginItr<float>(q, -1), CircularIterator<OCVIterator<float>>(GpuMatBeginItr<float>(p, -1), GpuMatEndItr<float>(p, -1)), sqared_difference_gpu.begin())),
                       q.cols,
                       sqared_difference_functor_foreach());

    thrust::device_vector<float> subsum_sqared_difference_gpu = sum_subvectors_in_device(sqared_difference_gpu, histogram_size);

    /* KLD */
    thrust::device_vector<float> unnormalized_kld_gpu(q.cols);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<OCVIterator<float>>(GpuMatBeginItr<float>(p, -1), GpuMatEndItr<float>(p, -1)), GpuMatBeginItr<float>(q, -1), unnormalized_kld_gpu.begin())),
                       unnormalized_kld_gpu.size(),
                       kld_functor_foreach());

    thrust::device_vector<float> euclidean_kld_gpu = sum_subvectors_in_device(unnormalized_kld_gpu, histogram_size);
    thrust::device_vector<float> subsum_p_gpu      = sum_subvectors_in_device(p,                    histogram_size);
    thrust::device_vector<float> subsum_q_gpu      = sum_subvectors_in_device(q,                    histogram_size);

    /* Euclidean * KLD */
    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(subsum_sqared_difference_gpu.begin(), CircularIterator<FloatIterator>(subsum_p_gpu.begin(), subsum_p_gpu.end()), subsum_q_gpu.begin(), euclidean_kld_gpu.begin())),
                       euclidean_kld_gpu.size(),
                       euclidean_kld_functor_foreach());

    /* Sum the contributions of Euclidean x KLD for each image */
    thrust::host_vector<float> image_euclidean_kld_cpu(sum_subvectors_in_device(euclidean_kld_gpu, histogram_number));

    return image_euclidean_kld_cpu;
}

}
}

