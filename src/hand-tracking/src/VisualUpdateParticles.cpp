#include <VisualUpdateParticles.h>

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include <opencv2/core/eigen.hpp>
#if HANDTRACKING_USE_OPENCV_CUDA
#include <opencv2/cudaimgproc.hpp>
#endif // HANDTRACKING_USE_OPENCV_CUDA
#include <opencv2/imgproc/imgproc.hpp>

using namespace bfl;
using namespace cv;
using namespace Eigen;


VisualUpdateParticles::VisualUpdateParticles(std::unique_ptr<VisualProprioception> observation_model) noexcept :
    VisualUpdateParticles(std::move(observation_model), 0.001, 1) { }


VisualUpdateParticles::VisualUpdateParticles(std::unique_ptr<VisualProprioception> observation_model, const double likelihood_gain) noexcept :
    VisualUpdateParticles(std::move(observation_model), likelihood_gain, 1) { }


VisualUpdateParticles::VisualUpdateParticles(std::unique_ptr<VisualProprioception> observation_model, const double likelihood_gain, const int num_parallel_processor) noexcept :
    observation_model_(std::move(observation_model)),
    likelihood_gain_(likelihood_gain)
{
    img_width_      = observation_model_->getCamWidth();
    img_height_     = observation_model_->getCamHeight();
    ogl_tiles_cols_ = observation_model_->getOGLTilesCols();
    ogl_tiles_rows_ = observation_model_->getOGLTilesRows();
    feature_dim_    = (img_width_ / block_size_ * 2 - 1) * (img_height_ / block_size_ * 2 - 1) * bin_number_ * 4;

    num_rendered_img_ = observation_model_->getOGLTilesNumber();

#if HANDTRACKING_USE_OPENCV_CUDA
    num_parallel_processor_ = num_parallel_processor;
    cuda_stream_ = std::vector<cuda::Stream>(num_parallel_processor);

    hog_cuda_ = cuda::HOG::create(Size(img_width_, img_height_), Size(block_size_, block_size_), Size(block_size_/2, block_size_/2), Size(block_size_/2, block_size_/2), bin_number_);
    hog_cuda_->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_ROW_BY_ROW);
    hog_cuda_->setGammaCorrection(true);
    hog_cuda_->setWinStride(Size(img_width_, img_height_));

    for (int s = 0; s < num_parallel_processor_; ++s)
    {
        cuda_img_.emplace_back        (cuda::GpuMat(Size(img_width_ * ogl_tiles_cols_, img_height_* ogl_tiles_rows_), CV_8UC3));
        cuda_img_alpha_.emplace_back  (cuda::GpuMat(Size(img_width_ * ogl_tiles_cols_, img_height_* ogl_tiles_rows_), CV_8UC4));
        cuda_descriptors_.emplace_back(cuda::GpuMat(Size(num_rendered_img_, feature_dim_),                            CV_32F ));
        cpu_descriptors_.emplace_back (Mat         (Size(num_rendered_img_, feature_dim_),                            CV_32F));
    }
#else
    hog_cpu_ = std::unique_ptr<HOGDescriptor>(new HOGDescriptor(Size(img_width_, img_height_),
                                                                Size(block_size_, block_size_),
                                                                Size(block_size_ / 2, block_size_ / 2),
                                                                Size(block_size_ / 2, block_size_ / 2),
                                                                bin_number_,
                                                                1, -1.0, HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS, false));

    static_cast<void>(num_parallel_processor);
    num_parallel_processor_ = 1;

    cpu_descriptors_.emplace_back(std::vector<float>(num_rendered_img_ * feature_dim_));
#endif // HANDTRACKING_USE_OPENCV_CUDA

    for (int s = 0; s < num_parallel_processor_; ++s)
        hand_rendered_.emplace_back(Mat(Size(img_width_ * ogl_tiles_cols_, img_height_* ogl_tiles_rows_), CV_8UC3));
}


VisualUpdateParticles::~VisualUpdateParticles() noexcept { }


void VisualUpdateParticles::innovation(const Ref<const MatrixXf>& pred_states, cv::InputArray measurements, Ref<MatrixXf> innovations)
{
    const std::vector<float>* measurements_ptr = static_cast<const std::vector<float>*>(measurements.getObj());

    for (int s = 0; s < num_parallel_processor_; ++s)
        observation_model_->observe(pred_states.block(0, s * num_rendered_img_, 7, num_rendered_img_), hand_rendered_[s]);

#if HANDTRACKING_USE_OPENCV_CUDA
    for (int s = 0; s < num_parallel_processor_; ++s)
    {
        cuda_img_[s].upload(hand_rendered_[s], cuda_stream_[s]);
        cuda::cvtColor(cuda_img_[s], cuda_img_alpha_[s], COLOR_BGR2BGRA, 4, cuda_stream_[s]);
        hog_cuda_->compute(cuda_img_alpha_[s], cuda_descriptors_[s], cuda_stream_[s]);
    }

    for (int s = 0; s < num_parallel_processor_; ++s)
    {
        cuda_stream_[s].waitForCompletion();

        cuda_descriptors_[s].download(cpu_descriptors_[s], cuda_stream_[s]);
#else
    for (int s = 0; s < num_parallel_processor_; ++s)
    {
        hog_cpu_->compute(hand_rendered_[s], cpu_descriptors_[s], Size(img_width_, img_height_));
#endif // HANDTRACKING_USE_OPENCV_CUDA

        for (int i = 0; i < num_rendered_img_; ++i)
        {
            double norm = 0;
            double chi = 0;

            double sum_normchi = 0;

            auto it_cam     = measurements_ptr->begin();
            auto it_cam_end = measurements_ptr->end();
            int  j          = 0;
            while (it_cam < it_cam_end)
            {
#if HANDTRACKING_USE_OPENCV_CUDA
                norm += std::pow((*it_cam) - cpu_descriptors_[s].at<float>(i, j), 2.0);

                chi += (std::pow((*it_cam) - cpu_descriptors_[s].at<float>(i, j), 2.0)) / ((*it_cam) + cpu_descriptors_[s].at<float>(i, j) + std::numeric_limits<float>::min());
#else
                norm += std::pow((*it_cam) - cpu_descriptors_[s][i * feature_dim_ + j], 2.0);

                chi += (std::pow((*it_cam) - cpu_descriptors_[s][i * feature_dim_ + j], 2.0)) / ((*it_cam) + cpu_descriptors_[s][i * feature_dim_ + j] + std::numeric_limits<float>::min());
#endif // HANDTRACKING_USE_OPENCV_CUDA

                ++it_cam;
                ++j;

                if (j % (bin_number_ * 4 - 1))
                {
                    sum_normchi += std::sqrt(norm) * chi;

                    norm = 0;
                    chi = 0;
                }
            }

            innovations(s * num_rendered_img_ + i, 0) = sum_normchi;
        }
    }
}


double VisualUpdateParticles::likelihood(const Ref<const MatrixXf>& innovations)
{
    return exp(-likelihood_gain_ * innovations.cast<double>().coeff(0, 0));
}


bfl::VisualObservationModel& VisualUpdateParticles::getVisualObservationModel()
{
    return *observation_model_;
}


void VisualUpdateParticles::correctStep(const Ref<const MatrixXf>& pred_states, const Ref<const VectorXf>& pred_weights, cv::InputArray measurements,
                                        Ref<MatrixXf> cor_states, Ref<VectorXf> cor_weights)
{
    VectorXf innovations(pred_states.cols());
    innovation(pred_states, measurements, innovations);

    cor_states = pred_states;

    for (int i = 0; i < innovations.rows(); ++i)
    {
        cor_weights(i) = pred_weights(i) * likelihood(innovations.row(i));

        if (cor_weights(i) <= 0)
            cor_weights(i) = std::numeric_limits<float>::min();
    }
}
