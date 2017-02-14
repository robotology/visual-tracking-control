#include "VisualParticleFilterCorrection.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace bfl;
using namespace cv;
using namespace Eigen;


VisualParticleFilterCorrection::VisualParticleFilterCorrection(std::shared_ptr<VisualObservationModel> measurement_model, const int num_particle) noexcept :
    VisualParticleFilterCorrection(measurement_model, num_particle, 2) { };



VisualParticleFilterCorrection::VisualParticleFilterCorrection(std::shared_ptr<VisualObservationModel> measurement_model, const int num_particle, const int num_cuda_stream) noexcept :
    measurement_model_(measurement_model), num_particle_(num_particle), num_cuda_stream_(num_cuda_stream), num_img_stream_(num_particle / num_cuda_stream), cuda_stream_(num_cuda_stream)
{
    cuda_hog_ = cuda::HOG::create(Size(img_width_, img_height_), Size(block_size_, block_size_), Size(block_size_/2, block_size_/2), Size(block_size_/2, block_size_/2), bin_number_);
    cuda_hog_->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_COL_BY_COL);
    cuda_hog_->setGammaCorrection(true);
    cuda_hog_->setWinStride(Size(img_width_, img_height_));

    for (int block = 0; block < num_particle_ / num_img_stream_; ++block) {
        hand_rendered_.insert   (hand_rendered_.begin(),    Mat(         Size(img_width_ * num_img_stream_, img_height_), CV_8UC3));
        cuda_img_.insert        (cuda_img_.begin(),         cuda::GpuMat(Size(img_width_ * num_img_stream_, img_height_), CV_8UC3));
        cuda_img_alpha_.insert  (cuda_img_alpha_.begin(),   cuda::GpuMat(Size(img_width_ * num_img_stream_, img_height_), CV_8UC4));
        cuda_descriptors_.insert(cuda_descriptors_.begin(), cuda::GpuMat(Size(num_img_stream_, ((img_width_/block_size_*2-1) * (img_height_/block_size_*2-1) * bin_number_ * 4)), CV_32F));
        cpu_descriptors_.insert (cpu_descriptors_.begin(),  Mat(         Size(num_img_stream_, ((img_width_/block_size_*2-1) * (img_height_/block_size_*2-1) * bin_number_ * 4)), CV_32F));
    }
}


VisualParticleFilterCorrection::~VisualParticleFilterCorrection() noexcept { }


//VisualParticleFilterCorrection::VisualParticleFilterCorrection(const VisualParticleFilterCorrection& vpf_correction) :
//    num_particle_(vpf_correction.num_particle_), num_image_stream_(vpf_correction.num_image_stream_)
//{
//    // FIXME: riscrivere i costruttori/opertori move e copy con tutti i parametri
//    measurement_model_ = vpf_correction.measurement_model_;
//}
//
//
//VisualParticleFilterCorrection::VisualParticleFilterCorrection(VisualParticleFilterCorrection&& vpf_correction) noexcept :
//measurement_model_(std::move(vpf_correction.measurement_model_)), num_particle_(std::move(vpf_correction.num_particle_)), num_image_stream_(std::move(vpf_correction.num_image_stream_)) { };
//
//
//VisualParticleFilterCorrection& VisualParticleFilterCorrection::operator=(const VisualParticleFilterCorrection& vpf_correction)
//{
//    VisualParticleFilterCorrection tmp(vpf_correction);
//    *this = std::move(tmp);
//
//    return *this;
//}
//
//
//VisualParticleFilterCorrection& VisualParticleFilterCorrection::operator=(VisualParticleFilterCorrection&& vpf_correction) noexcept
//{
//    measurement_model_ = std::move(vpf_correction.measurement_model_);
//
//    return *this;
//}


void VisualParticleFilterCorrection::correct(const Ref<const MatrixXf>& pred_state, InputArray measurements, Ref<MatrixXf> cor_state)
{
    VectorXf innovate(pred_state.cols());

    innovation(pred_state, measurements, innovate);

    likelihood(innovate, cor_state);
}


void VisualParticleFilterCorrection::innovation(const Ref<const MatrixXf>& pred_state, InputArray measurements, Ref<MatrixXf> innovation)
{
    for (int block = 0; block < num_particle_ / num_img_stream_; ++block)
    {
        measurement_model_->observe(pred_state.block(0, block * num_img_stream_, 6, num_img_stream_), hand_rendered_[block]);
    }

    for (int block = 0; block < num_particle_ / num_img_stream_; ++block)
    {
        const unsigned int stream_index = block % num_cuda_stream_;

        cuda_img_[block].upload(hand_rendered_[block], cuda_stream_[stream_index]);
        cuda::cvtColor(cuda_img_[block], cuda_img_alpha_[block], COLOR_BGR2BGRA, 4, cuda_stream_[stream_index]);
        cuda_hog_->compute(cuda_img_alpha_[block], cuda_descriptors_[block], cuda_stream_[stream_index]);
    }

    for (int s = 0; s < num_cuda_stream_; ++s) cuda_stream_[s].waitForCompletion();

    for (int block = 0; block < num_particle_ / num_img_stream_; ++block)
    {
        const unsigned int stream_index = block % num_cuda_stream_;
        cuda_descriptors_[block].download(cpu_descriptors_[block], cuda_stream_[stream_index]);

        for (int i = 0; i < num_img_stream_; ++i)
        {
            float sum_diff = 0;
            {
            auto it_cam     = (static_cast<const std::vector<float>*>(measurements.getObj()))->begin();
            auto it_cam_end = (static_cast<const std::vector<float>*>(measurements.getObj()))->end();
            int  j          = 0;
            while (it_cam < it_cam_end)
            {
                sum_diff += abs((*it_cam) - cpu_descriptors_[block].at<float>(i, j));
//                if (cpu_descriptors_[block].at<float>(i, j) != 0 && *it_cam != 0)
//                    sum_diff += (*it_cam) * std::log((*it_cam) / cpu_descriptors_[block].at<float>(i, j));
//                    sum_diff += cpu_descriptors_[block].at<float>(i, j) * std::log(cpu_descriptors_[block].at<float>(i, j) / (*it_cam));
                ++it_cam;
                ++j;
            }
            }
            innovation(block * num_img_stream_ + i, 0) = sum_diff;
        }
    }
}


void VisualParticleFilterCorrection::likelihood(const Ref<const MatrixXf>& innovation, Ref<MatrixXf> cor_state)
{
    // FIXME: Kernel likelihood need to be tuned!
    for (int i = 0; i < innovation.rows(); ++i)
    {
        cor_state(i, 0) *= ( exp( -0.001 * innovation(i, 0) /* / pow(1, 2.0) */ ) );
        if (cor_state(i, 0) <= 0) cor_state(i, 0) = std::numeric_limits<float>::min();
    }
}
