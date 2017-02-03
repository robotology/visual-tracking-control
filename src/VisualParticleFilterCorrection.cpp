#include "VisualParticleFilterCorrection.h"

#include <cmath>
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
    VisualParticleFilterCorrection(measurement_model, num_particle, 10) { };



VisualParticleFilterCorrection::VisualParticleFilterCorrection(std::shared_ptr<VisualObservationModel> measurement_model, const int num_particle, const int num_cuda_stream) noexcept :
    measurement_model_(measurement_model),
    hog_(HOGDescriptor(Size(img_width_, img_height_), Size(block_size_, block_size_), Size(block_size_/2, block_size_/2), Size(block_size_/2, block_size_/2), bin_number_, 1, -1, HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS, false)),
    num_particle_(num_particle), num_image_stream_(num_particle/num_cuda_stream), stream_(num_cuda_stream)
{
    cuda_hog_ = cuda::HOG::create(Size(img_width_, img_height_), Size(block_size_, block_size_), Size(block_size_/2, block_size_/2), Size(block_size_/2, block_size_/2), bin_number_);
    cuda_hog_->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_COL_BY_COL);
    cuda_hog_->setGammaCorrection(true);
    cuda_hog_->setWinStride(Size(img_width_, img_height_));

    const int descriptor_length = (img_width_/block_size_*2-1) * (img_height_/block_size_*2-1) * bin_number_ * 4;
    std::cout << descriptor_length << std::endl;
    for (int i = 0; i < num_cuda_stream; ++i)
    {
        cuda_img_.insert        (cuda_img_.begin(),         cuda::GpuMat(Size(img_width_ * num_image_stream_, img_height_), CV_8UC3));
        cuda_img_alpha_.insert  (cuda_img_alpha_.begin(),   cuda::GpuMat(Size(img_width_ * num_image_stream_, img_height_), CV_8UC4));
        cuda_descriptors_.insert(cuda_descriptors_.begin(), cuda::GpuMat(Size(num_image_stream_, descriptor_length),        CV_32F ));
    }
}


VisualParticleFilterCorrection::~VisualParticleFilterCorrection() noexcept { }


VisualParticleFilterCorrection::VisualParticleFilterCorrection(const VisualParticleFilterCorrection& vpf_correction) :
    num_particle_(vpf_correction.num_particle_), num_image_stream_(vpf_correction.num_image_stream_)
{
    // FIXME: riscrivere i costruttori/opertori move e copy con tutti i parametri
    measurement_model_ = vpf_correction.measurement_model_;
}


VisualParticleFilterCorrection::VisualParticleFilterCorrection(VisualParticleFilterCorrection&& vpf_correction) noexcept :
measurement_model_(std::move(vpf_correction.measurement_model_)), num_particle_(std::move(vpf_correction.num_particle_)), num_image_stream_(std::move(vpf_correction.num_image_stream_)) { };


VisualParticleFilterCorrection& VisualParticleFilterCorrection::operator=(const VisualParticleFilterCorrection& vpf_correction)
{
    VisualParticleFilterCorrection tmp(vpf_correction);
    *this = std::move(tmp);

    return *this;
}


VisualParticleFilterCorrection& VisualParticleFilterCorrection::operator=(VisualParticleFilterCorrection&& vpf_correction) noexcept
{
    measurement_model_ = std::move(vpf_correction.measurement_model_);

    return *this;
}


void VisualParticleFilterCorrection::correct(const Ref<const MatrixXf>& pred_state, InputArray measurements, Ref<MatrixXf> cor_state)
{
    VectorXf innovate(pred_state.cols());
    innovation(pred_state, measurements, innovate);
    likelihood(innovate, cor_state);
}


void VisualParticleFilterCorrection::innovation(const Ref<const MatrixXf>& pred_state, InputArray measurements, Ref<MatrixXf> innovation)
{
    Mat       hand_ogl_cv(Size(img_width_ * num_image_stream_, img_height_), CV_8UC3);
    Rect      current_ROI(0, 0, img_width_, img_height_);
    Mat       descriptors_cad;

    for (int i = 0; i < stream_.size(); ++i)
    {
        for (int j = 0; j < num_image_stream_; ++j)
        {
            current_ROI.x = j * img_width_;
            measurement_model_->observe(pred_state.col((i * num_image_stream_) + j), hand_ogl_cv(current_ROI));
        }

        cuda_img_[i].upload(hand_ogl_cv, stream_[i]);
        cuda::cvtColor(cuda_img_[i], cuda_img_alpha_[i], COLOR_BGR2BGRA, 4, stream_[i]);
        cuda_hog_->compute(cuda_img_alpha_[i], cuda_descriptors_[i], stream_[i]);
    }

    for (int i = 0; i < stream_.size(); ++i)
    {
        stream_[i].waitForCompletion();
        cuda_descriptors_[i].download(descriptors_cad, stream_[i]);

        for (int j = 0; j < num_image_stream_; ++j)
        {
            float sum_diff = 0;
            {
            auto it_cam = (static_cast<const std::vector<float>*>(measurements.getObj()))->begin();
            auto it_cam_end = (static_cast<const std::vector<float>*>(measurements.getObj()))->end();
            int i = 0;
            for (; it_cam < it_cam_end; ++i, ++it_cam) sum_diff += abs((*it_cam) - descriptors_cad.at<float>(j, i));
            }

            innovation((i * num_image_stream_) + j, 0) = sum_diff;
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
