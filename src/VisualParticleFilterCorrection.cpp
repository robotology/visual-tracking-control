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


VisualParticleFilterCorrection::VisualParticleFilterCorrection(std::shared_ptr<VisualObservationModel> measurement_model) noexcept :
    measurement_model_(measurement_model), hog_(HOGDescriptor(Size(img_width_, img_height_), Size(block_size_, block_size_), Size(block_size_/2, block_size_/2), Size(block_size_/2, block_size_/2), 9, 1, -1, HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS, false))
{
    cuda_hog_ = cuda::HOG::create(Size(img_width_, img_height_), Size(block_size_, block_size_), Size(block_size_/2, block_size_/2), Size(block_size_/2, block_size_/2), 9);
    cuda_hog_->setGammaCorrection(true);
    cuda_hog_->setWinStride(Size(img_width_, img_height_));
}


VisualParticleFilterCorrection::~VisualParticleFilterCorrection() noexcept { }


VisualParticleFilterCorrection::VisualParticleFilterCorrection(const VisualParticleFilterCorrection& vpf_correction)
{
    measurement_model_ = vpf_correction.measurement_model_;
}


VisualParticleFilterCorrection::VisualParticleFilterCorrection(VisualParticleFilterCorrection&& vpf_correction) noexcept :
measurement_model_(std::move(vpf_correction.measurement_model_)) { };


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


void VisualParticleFilterCorrection::correct(const Ref<const VectorXf>& pred_state, cv::InputArray measurements, Ref<VectorXf> cor_state)
{
    VectorXf innovate(1);
    innovation(pred_state, measurements, innovate);
    likelihood(innovate, cor_state);
}


void VisualParticleFilterCorrection::innovation(const Ref<const VectorXf>& pred_state, cv::InputArray measurements, Ref<MatrixXf> innovation)
{
    Mat          hand_ogl_cv;
    cuda::GpuMat cuda_img;
    cuda::GpuMat cuda_img_alpha;
    cuda::GpuMat cuda_descriptors;
    Mat          descriptors_cad;

    measurement_model_->observe(pred_state, hand_ogl_cv);

//    hog_.compute(hand_ogl_cv, descriptors_cad);

    cuda_img.upload(hand_ogl_cv);
    cuda::cvtColor(cuda_img, cuda_img_alpha, COLOR_BGR2BGRA, 4);
    cuda_hog_->compute(cuda_img_alpha, cuda_descriptors);
    cuda_descriptors.download(descriptors_cad);

    float sum_diff = 0;
    {
    auto it_cam = (static_cast<const std::vector<float>*>(measurements.getObj()))->begin();
    auto it_cam_end = (static_cast<const std::vector<float>*>(measurements.getObj()))->end();
    int i = 0;
    for (; it_cam < it_cam_end; ++i, ++it_cam) sum_diff += abs((*it_cam) - descriptors_cad.at<float>(0, i));
    }

    innovation(0, 0) = sum_diff;
}


void VisualParticleFilterCorrection::likelihood(const Ref<const MatrixXf>& innovation, Ref<VectorXf> cor_state)
{
    // FIXME: Kernel likelihood need to be tuned!
    cor_state(0, 0) *= ( exp( -0.001 * innovation(0, 0) /* / pow(1, 2.0) */ ) );
    if (cor_state(0, 0) <= 0) cor_state(0, 0) = std::numeric_limits<float>::min();
}
