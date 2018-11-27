#include <device_likelihood.h>
#include <thrust_opencv_iterator.h>

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
#include <thrust/replace.h>
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

using ConstFloatIterator = thrust::device_vector<float>::const_iterator;


/* >- ************************************************** -< */
template<typename Iterator>
class StridedIterator
{
public:
    using DifferenceType = typename thrust::iterator_difference<Iterator>::type;

    StridedIterator(Iterator first, Iterator last, DifferenceType stride) :
        first_(first),
        last_(last),
        stride_(stride)
    { }

    struct stride_functor : public thrust::unary_function<DifferenceType, DifferenceType>
    {
        DifferenceType stride_;

        stride_functor(DifferenceType stride) :
            stride_(stride)
        { }

        __host__ __device__
        DifferenceType operator()(const DifferenceType& i) const
        {
            return stride_ * i;
        }
    };

    using CountingIterator = typename thrust::counting_iterator<DifferenceType>;

    using TransformIterator = typename thrust::transform_iterator<stride_functor, CountingIterator>;

    using PermutationIterator = typename thrust::permutation_iterator<Iterator, TransformIterator>;

    PermutationIterator begin() const
    {
        return PermutationIterator(first_, TransformIterator(CountingIterator(0), stride_functor(stride_)));
    }

    PermutationIterator end() const
    {
        return begin() + ((last_ - first_) + (stride_ - 1)) / stride_;
    }

protected:
    Iterator first_;

    Iterator last_;

    DifferenceType stride_;
};


/* >- ************************************************** -< */
template<typename Iterator>
class CircularIterator : public thrust::iterator_adaptor<CircularIterator<Iterator>, Iterator>
{
public:
    using base_t = typename thrust::iterator_adaptor<CircularIterator<Iterator>, Iterator>;

    using difference_t = typename thrust::iterator_adaptor<CircularIterator<Iterator>, Iterator>::difference_type;

    __host__ __device__
    CircularIterator(const Iterator& begin, const Iterator& end) :
        base_t(begin),
        begin_(begin),
        end_(end),
        range_(end - begin),
        upper_bound_(end - begin - 1)
    { }

    friend class thrust::iterator_core_access;

protected:
    const Iterator begin_;

    const Iterator end_;

    const difference_t range_;

    const difference_t upper_bound_;

    difference_t current_ = 0;

private:
    __host__ __device__
    void advance(difference_t n)
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
template<typename Iterator>
class SlowPaceIterator : public thrust::iterator_adaptor<SlowPaceIterator<Iterator>, Iterator>
{
public:
    using base_t = typename thrust::iterator_adaptor<SlowPaceIterator<Iterator>, Iterator>;

    using difference_t = typename thrust::iterator_adaptor<SlowPaceIterator<Iterator>, Iterator>::difference_type;

    __host__ __device__
    SlowPaceIterator(const Iterator& begin, const Iterator& end, const std::size_t pace) :
        base_t(begin),
        begin_(begin),
        end_(end),
        pace_(pace)
    { }

    friend class thrust::iterator_core_access;

protected:
    const Iterator begin_;

    const Iterator end_;

    const std::size_t pace_;

    std::size_t pace_accumulator = 0;

private:
    __host__ __device__
    void advance(difference_t n)
    {
        pace_accumulator += n;

        n = pace_accumulator / pace_;

        pace_accumulator %= pace_;

        this->base_reference() += n;
    }

    __host__ __device__
    void increment()
    {
        ++pace_accumulator;

        if (pace_accumulator == pace_)
        {
            ++(this->base_reference());

            pace_accumulator = 0;
        }
    }

    __host__ __device__
    void decrement()
    {
        if (pace_accumulator == 0)
        {
            --(this->base_reference());

            pace_accumulator = pace_ - 1;
        }
        else
            --pace_accumulator;
    }
};


/* >- ************************************************** -< */
thrust::device_vector<float> sum_subvectors_in_device(cublasHandle_t handle, const thrust::device_vector<float>& vec, const std::size_t subvector_size)
{
    std::size_t subvector_num = vec.size() / subvector_size;

    const float *A = ::thrust::raw_pointer_cast(&vec[0]);

    ::thrust::device_vector<float> ones(subvector_size, 1.0f);
    const float *B = ::thrust::raw_pointer_cast(&ones[0]);

    ::thrust::device_vector<float> sum_subvectors_gpu(subvector_num);
    float *C = ::thrust::raw_pointer_cast(&sum_subvectors_gpu[0]);

    int m = static_cast<int>(subvector_size);
    int n = static_cast<int>(subvector_num);
    int lda = m;
    int ldb = 1;
    int ldc = 1;
    const float alf = 1.0f;
    const float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasSgemv(handle, cublasOperation_t::CUBLAS_OP_T,
                m, n,
                alpha, A, lda,
                B, ldb, beta,
                C, ldc);

    return sum_subvectors_gpu;
}


/* >- ************************************************** -< */
thrust::device_vector<float> sum_subvectors_in_device(cublasHandle_t handle, const cv::cuda::GpuMat& mat, const std::size_t subvector_size)
{
    thrust::device_vector<float> p_thrust(mat.size().area());

    thrust::copy(bfl::thrust::GpuMat_begin_iterator<float>(mat), bfl::thrust::GpuMat_end_iterator<float>(mat),
                 p_thrust.begin());

    return sum_subvectors_in_device(handle, p_thrust, subvector_size);
}


/* >- ************************************************** -< */
struct functor_is_less_than_zero
{
    __host__ __device__
        bool operator()(const float& x)
    {
        return x <= 0;
    }
};


/* >- ************************************************** -< */
struct functor_kld
{
    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple& t)
    {
        thrust::get<2>(t) = thrust::get<0>(t) * (log10f(thrust::get<0>(t) + NPP_MINABS_32F) - log10f(thrust::get<1>(t) + NPP_MINABS_32F));
    }
};


/* >- ************************************************** -< */
struct functor_kld_with_consistency
{
    functor_kld_with_consistency(const std::size_t histogram_size) : uniform_pmf_value_(1.0f / histogram_size)
    { }

    const float uniform_pmf_value_;

    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple& t)
    {
        if (thrust::get<2>(t) <= 0.0f)
            thrust::get<3>(t) = thrust::get<0>(t) * (log10f(thrust::get<0>(t) + NPP_MINABS_32F) - log10f(uniform_pmf_value_ + NPP_MINABS_32F));
        else
            thrust::get<3>(t) = thrust::get<0>(t) * (log10f(thrust::get<0>(t) + NPP_MINABS_32F) - log10f(thrust::get<1>(t) + NPP_MINABS_32F));
    }
};


/* >- ************************************************** -< */
struct functor_normalize_kld
{
    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple& t)
    {
        thrust::get<2>(t) = (thrust::get<2>(t) / thrust::get<0>(t)) + log10f(thrust::get<1>(t) + NPP_MINABS_32F) - log10f(thrust::get<0>(t) + NPP_MINABS_32F);

        /* The following line is to force 0 on very low numbers, but may result in thread divergence in warps. */
        //thrust::get<2>(t) = (thrust::get<2>(t) < 0.00001f) ? 0.0f : thrust::get<2>(t);
    }
};


/* >- ************************************************** -< */
struct functor_chi_square
{
    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple& t)
    {
        thrust::get<2>(t) = powf(thrust::get<0>(t) - thrust::get<1>(t), 2.0f) / (2.0f * (thrust::get<0>(t) + thrust::get<1>(t)));
    }
};


/* >- ************************************************** -< */
struct functor_absolute_difference
{
    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple& t)
    {
        thrust::get<2>(t) = fabsf(thrust::get<0>(t) - thrust::get<1>(t));
    }
};


/* >- ************************************************** -< */
struct functor_sqared_difference
{
    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple& t)
    {
        thrust::get<2>(t) = powf(thrust::get<0>(t) - thrust::get<1>(t), 2.0f);
    }
};


/* >- ************************************************** -< */
struct functor_sqrt : public thrust::unary_function<float, float>
{
    __host__ __device__
        float operator()(const float x) const
    {
        return sqrtf(x);
    }
};


/* >- ************************************************** -< */
struct functor_euclidean_kld
{
    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple& t)
    {
        thrust::get<3>(t) = sqrtf(thrust::get<0>(t)) * ((thrust::get<3>(t) / thrust::get<1>(t)) + log10f(thrust::get<2>(t) + NPP_MINABS_32F) - log10f(thrust::get<1>(t) + NPP_MINABS_32F));

        /* The following line is to force 0 on very low numbers, but may result in thread divergence in warps. */
        //thrust::get<3>(t) = (thrust::get<3>(t) < 0.00001f) ? 0.0f : thrust::get<3>(t);
    }
};


/* >- ************************************************** -< */
struct functor_euclidean_kld_chisquare
{
    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple& t)
    {
        thrust::get<4>(t) = sqrtf(thrust::get<0>(t)) * ((thrust::get<4>(t) / thrust::get<1>(t)) + log10f(thrust::get<2>(t) + NPP_MINABS_32F) - log10f(thrust::get<1>(t) + NPP_MINABS_32F)) * thrust::get<3>(t);

        /* The following line is to force 0 on very low numbers, but may result in thread divergence in warps. */
        //thrust::get<4>(t) = (thrust::get<4>(t) < 0.00001f) ? 0.0f : thrust::get<4>(t);
    }
};


/* TODO
* When switching to MSVC 15.7 plus, upgrade this implementation to use tuples and move constructors to return device vectors.
*/
/* >- ************************************************** -< */
void kld_of_pmf_subvectors
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t histogram_size,
    thrust::device_vector<float>& normalizing_term_p_gpu,
    thrust::device_vector<float>& normalizing_term_q_gpu,
    thrust::device_vector<float>& histogram_unnormalized_kld_gpu
)
{
    std::size_t number_element_q = q.size().area();

    normalizing_term_p_gpu = sum_subvectors_in_device(handle, p, histogram_size);
    normalizing_term_q_gpu = sum_subvectors_in_device(handle, q, histogram_size);

    thrust::device_vector<float> unnormalized_kld_gpu(number_element_q);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<bfl::thrust::const_ocv_iterator<float>>(bfl::thrust::GpuMat_begin_iterator<float>(p), bfl::thrust::GpuMat_end_iterator<float>(p)), bfl::thrust::GpuMat_begin_iterator<float>(q), unnormalized_kld_gpu.begin())),
                       unnormalized_kld_gpu.size(),
                       functor_kld());

    histogram_unnormalized_kld_gpu = sum_subvectors_in_device(handle, unnormalized_kld_gpu, histogram_size);
}


/* TODO
* When switching to MSVC 15.7 plus, upgrade this implementation to use tuples and move constructors to return device vectors.
*/
/* >- ************************************************** -< */
void kld_of_versor_subvectors
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t histogram_size,
    thrust::device_vector<float>& normalizing_term_p_gpu,
    thrust::device_vector<float>& normalizing_term_q_gpu,
    thrust::device_vector<float>& histogram_unnormalized_kld_gpu
)
{
    std::size_t number_element_q = q.size().area();

    normalizing_term_p_gpu = sum_subvectors_in_device(handle, p, histogram_size);
    normalizing_term_q_gpu = sum_subvectors_in_device(handle, q, histogram_size);

    thrust::device_vector<float> unnormalized_kld_gpu(number_element_q);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<bfl::thrust::const_ocv_iterator<float>>(bfl::thrust::GpuMat_begin_iterator<float>(p), bfl::thrust::GpuMat_end_iterator<float>(p)), bfl::thrust::GpuMat_begin_iterator<float>(q), SlowPaceIterator<ConstFloatIterator>(normalizing_term_q_gpu.cbegin(), normalizing_term_q_gpu.cend(), histogram_size), unnormalized_kld_gpu.begin())),
                       unnormalized_kld_gpu.size(),
                       functor_kld_with_consistency(histogram_size));

    thrust::replace_if(normalizing_term_q_gpu.begin(), normalizing_term_q_gpu.end(),
                       functor_is_less_than_zero(), 1.0f);

    histogram_unnormalized_kld_gpu = sum_subvectors_in_device(handle, unnormalized_kld_gpu, histogram_size);
}


/* >- ************************************************** -< */
void chisquare_of_subvectors
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t histogram_size,
    thrust::device_vector<float>& histogram_chi_square_gpu
)
{
    std::size_t number_element_q = q.size().area();

    thrust::device_vector<float> component_chi_square_gpu(number_element_q);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<bfl::thrust::const_ocv_iterator<float>>(bfl::thrust::GpuMat_begin_iterator<float>(p), bfl::thrust::GpuMat_end_iterator<float>(p)), bfl::thrust::GpuMat_begin_iterator<float>(q), component_chi_square_gpu.begin())),
                       number_element_q,
                       functor_chi_square());

    histogram_chi_square_gpu = sum_subvectors_in_device(handle, component_chi_square_gpu, histogram_size);
}


/* >- ************************************************** -< */
void normone_of_subvectors
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t histogram_size,
    thrust::device_vector<float>& histogram_absolute_difference_gpu
)
{
    std::size_t number_element_q = q.size().area();

    thrust::device_vector<float> component_absolute_difference_gpu(number_element_q);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<bfl::thrust::const_ocv_iterator<float>>(bfl::thrust::GpuMat_begin_iterator<float>(p), bfl::thrust::GpuMat_end_iterator<float>(p)), bfl::thrust::GpuMat_begin_iterator<float>(q), component_absolute_difference_gpu.begin())),
                       number_element_q,
                       functor_absolute_difference());

    histogram_absolute_difference_gpu = sum_subvectors_in_device(handle, component_absolute_difference_gpu, histogram_size);
}


/* >- ************************************************** -< */
void normone_whole_vector
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    thrust::device_vector<float>& histogram_absolute_difference_gpu
)
{
    std::size_t number_element_q = q.size().area();

    histogram_absolute_difference_gpu.resize(number_element_q);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<bfl::thrust::const_ocv_iterator<float>>(bfl::thrust::GpuMat_begin_iterator<float>(p), bfl::thrust::GpuMat_end_iterator<float>(p)), bfl::thrust::GpuMat_begin_iterator<float>(q), histogram_absolute_difference_gpu.begin())),
                       number_element_q,
                       functor_absolute_difference());
}


/* >- ************************************************** -< */
void normtwo_of_subvectors
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t histogram_size,
    thrust::device_vector<float>& histogram_sqared_difference_gpu
)
{
    std::size_t number_element_q = q.size().area();

    thrust::device_vector<float> component_squared_difference_gpu(number_element_q);

    thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(CircularIterator<bfl::thrust::const_ocv_iterator<float>>(bfl::thrust::GpuMat_begin_iterator<float>(p), bfl::thrust::GpuMat_end_iterator<float>(p)), bfl::thrust::GpuMat_begin_iterator<float>(q), component_squared_difference_gpu.begin())),
                       number_element_q,
                       functor_sqared_difference());

    histogram_sqared_difference_gpu = sum_subvectors_in_device(handle, component_squared_difference_gpu, histogram_size);
}


namespace bfl
{
namespace cuda
{

/* >- ************************************************** -< */
::thrust::host_vector<float> normone
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q
)
{
    /* Absolute difference components */
    ::thrust::device_vector<float> histogram_abs_difference_gpu;

    normone_whole_vector(handle,
                         p, q,
                         histogram_abs_difference_gpu);


    /* Sum over histogram-wise absolute difference */
    std::size_t subvector_size = static_cast<std::size_t>(p.size().area());

    ::thrust::host_vector<float> summed_histogram_normone_cpu(sum_subvectors_in_device(handle, histogram_abs_difference_gpu, subvector_size));


    return summed_histogram_normone_cpu;
}


/* >- ************************************************** -< */
::thrust::host_vector<float> normtwo
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t vector_size
)
{
    /* Euclidean distance components */
    ::thrust::device_vector<float> histogram_sqared_difference_gpu;

    normtwo_of_subvectors(handle,
                          p, q, vector_size,
                          histogram_sqared_difference_gpu);


    /* Histogram-wise Euclidean */
    ::thrust::transform(histogram_sqared_difference_gpu.cbegin(), histogram_sqared_difference_gpu.cend(),
                        histogram_sqared_difference_gpu.begin(),
                        functor_sqrt());


    /* Sum over histogram-wise Euclidean */
    std::size_t vector_number = static_cast<std::size_t>(p.size().area() / vector_size);

    ::thrust::host_vector<float> summed_histogram_normtwo_cpu(sum_subvectors_in_device(handle, histogram_sqared_difference_gpu, vector_number));


    return summed_histogram_normtwo_cpu;
}


/* >- ************************************************** -< */
::thrust::host_vector<float> kld
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t vector_size,
    const bool check_pmf
)
{
    /* KLD components */
    ::thrust::device_vector<float> normalizing_term_p_gpu;
    ::thrust::device_vector<float> normalizing_term_q_gpu;
    ::thrust::device_vector<float> histogram_unnormalized_kld_gpu;

    if (check_pmf)
        kld_of_versor_subvectors(handle,
                                 p, q, vector_size,
                                 normalizing_term_p_gpu, normalizing_term_q_gpu,
                                 histogram_unnormalized_kld_gpu);
    else
        kld_of_pmf_subvectors(handle,
                              p, q, vector_size,
                              normalizing_term_p_gpu, normalizing_term_q_gpu,
                              histogram_unnormalized_kld_gpu);


    /* Histogram-wise KLD */
    ::thrust::for_each_n(::thrust::make_zip_iterator(::thrust::make_tuple(CircularIterator<ConstFloatIterator>(normalizing_term_p_gpu.cbegin(), normalizing_term_p_gpu.cend()), normalizing_term_q_gpu.cbegin(), histogram_unnormalized_kld_gpu.begin())),
                         histogram_unnormalized_kld_gpu.size(),
                         functor_normalize_kld());


    /* Sum over histogram-wise KLD */
    std::size_t vector_number = static_cast<std::size_t>(p.size().area() / vector_size);

    ::thrust::host_vector<float> summed_histogram_kld_cpu(sum_subvectors_in_device(handle, histogram_unnormalized_kld_gpu, vector_number));


    return summed_histogram_kld_cpu;
}


/* >- ************************************************** -< */
::thrust::host_vector<float> chisquare
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t vector_size
)
{
    /* Chi-square components */
    ::thrust::device_vector<float> histogram_chi_square_gpu;

    chisquare_of_subvectors(handle,
                            p, q, vector_size,
                            histogram_chi_square_gpu);


    /* Sum over histogram-wise chi-square */
    std::size_t vector_number = static_cast<std::size_t>(p.size().area() / vector_size);

    ::thrust::host_vector<float> summed_histogram_normone_cpu(sum_subvectors_in_device(handle, histogram_chi_square_gpu, vector_number));


    return summed_histogram_normone_cpu;
}


/* >- ************************************************** -< */
::thrust::host_vector<float> normtwo_kld
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t vector_size,
    const bool check_pmf
)
{
    std::size_t number_element_q = q.size().area();


    /* Euclidean distance components */
    ::thrust::device_vector<float> histogram_sqared_difference_gpu;

    normtwo_of_subvectors(handle,
                          p, q, vector_size,
                          histogram_sqared_difference_gpu);


    /* KLD components */
    ::thrust::device_vector<float> normalizing_term_p_gpu;
    ::thrust::device_vector<float> normalizing_term_q_gpu;
    ::thrust::device_vector<float> histogram_unnormalized_kld_gpu;

    if (check_pmf)
        kld_of_versor_subvectors(handle,
                                 p, q, vector_size,
                                 normalizing_term_p_gpu, normalizing_term_q_gpu,
                                 histogram_unnormalized_kld_gpu);
    else
        kld_of_pmf_subvectors(handle,
                              p, q, vector_size,
                              normalizing_term_p_gpu, normalizing_term_q_gpu,
                              histogram_unnormalized_kld_gpu);


    /* Histogram-wise Euclidean * KLD */
    ::thrust::for_each_n(::thrust::make_zip_iterator(::thrust::make_tuple(histogram_sqared_difference_gpu.cbegin(), CircularIterator<ConstFloatIterator>(normalizing_term_p_gpu.cbegin(), normalizing_term_p_gpu.cend()), normalizing_term_q_gpu.cbegin(), histogram_unnormalized_kld_gpu.begin())),
                         histogram_unnormalized_kld_gpu.size(),
                         functor_euclidean_kld());


    /* Sum over histogram-wise Euclidean * KLD */
    std::size_t vector_number = static_cast<std::size_t>(p.size().area() / vector_size);

    ::thrust::host_vector<float> summed_histogram_euclidean_kld_cpu(sum_subvectors_in_device(handle, histogram_unnormalized_kld_gpu, vector_number));


    return summed_histogram_euclidean_kld_cpu;
}


/* >- ************************************************** -< */
::thrust::host_vector<float> normtwo_kld_chisquare
(
    cublasHandle_t handle,
    const cv::cuda::GpuMat& p,
    const cv::cuda::GpuMat& q,
    const std::size_t vector_size,
    const bool check_pmf
)
{
    std::size_t number_element_q = q.size().area();


    /* Euclidean distance components */
    ::thrust::device_vector<float> histogram_sqared_difference_gpu;

    normtwo_of_subvectors(handle,
                          p, q, vector_size,
                          histogram_sqared_difference_gpu);


    /* KLD components */
    ::thrust::device_vector<float> normalizing_term_p_gpu;
    ::thrust::device_vector<float> normalizing_term_q_gpu;
    ::thrust::device_vector<float> histogram_unnormalized_kld_gpu;

    if (check_pmf)
        kld_of_versor_subvectors(handle,
                                 p, q, vector_size,
                                 normalizing_term_p_gpu, normalizing_term_q_gpu,
                                 histogram_unnormalized_kld_gpu);
    else
        kld_of_pmf_subvectors(handle,
                              p, q, vector_size,
                              normalizing_term_p_gpu, normalizing_term_q_gpu,
                              histogram_unnormalized_kld_gpu);


    /* Chi-square components */
    ::thrust::device_vector<float> histogram_chi_square_gpu;

    chisquare_of_subvectors(handle,
                            p, q, vector_size,
                            histogram_chi_square_gpu);


    /* Histogram-wise Euclidean * KLD * Chi-square */
    ::thrust::for_each_n(::thrust::make_zip_iterator(::thrust::make_tuple(histogram_sqared_difference_gpu.cbegin(), CircularIterator<ConstFloatIterator>(normalizing_term_p_gpu.cbegin(), normalizing_term_p_gpu.cend()), normalizing_term_q_gpu.cbegin(), histogram_chi_square_gpu.cbegin(), histogram_unnormalized_kld_gpu.begin())),
                         histogram_unnormalized_kld_gpu.size(),
                         functor_euclidean_kld_chisquare());


    /* Sum over histogram-wise Euclidean * KLD */
    std::size_t vector_number = static_cast<std::size_t>(p.size().area() / vector_size);

    ::thrust::host_vector<float> summed_histogram_euclidean_kld_cpu(sum_subvectors_in_device(handle, histogram_unnormalized_kld_gpu, vector_number));


    return summed_histogram_euclidean_kld_cpu;
}

}
}
