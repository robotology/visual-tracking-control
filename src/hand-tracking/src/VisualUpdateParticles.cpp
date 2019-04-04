/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <VisualUpdateParticles.h>

#include <cmath>
#include <exception>
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


VisualUpdateParticles::VisualUpdateParticles(std::unique_ptr<VisualProprioception> observation_model) noexcept :
    VisualUpdateParticles(std::move(observation_model), 0.001, 1) { }


VisualUpdateParticles::VisualUpdateParticles(std::unique_ptr<VisualProprioception> observation_model, const double likelihood_gain) noexcept :
    VisualUpdateParticles(std::move(observation_model), likelihood_gain, 1) { }


VisualUpdateParticles::VisualUpdateParticles(std::unique_ptr<VisualProprioception> observation_model, const double likelihood_gain, const int num_cuda_stream) noexcept :
    observation_model_(std::move(observation_model)),
    likelihood_gain_(likelihood_gain),
    num_cuda_stream_(num_cuda_stream),
    num_img_stream_(observation_model_->getOGLTilesNumber()),
    cuda_stream_(std::vector<cuda::Stream>(num_cuda_stream))
{
    int          block_size     = 16;
    int          bin_number     = 9;
    unsigned int img_width      = observation_model_->getCamWidth();
    unsigned int img_height     = observation_model_->getCamHeight();
    unsigned int ogl_tiles_cols = observation_model_->getOGLTilesCols();
    unsigned int ogl_tiles_rows = observation_model_->getOGLTilesRows();
    unsigned int feature_dim    = (img_width/block_size*2-1) * (img_height/block_size*2-1) * bin_number * 4;

    cuda_hog_ = cuda::HOG::create(Size(img_width, img_height), Size(block_size, block_size), Size(block_size/2, block_size/2), Size(block_size/2, block_size/2), bin_number);
    cuda_hog_->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_ROW_BY_ROW);
    cuda_hog_->setGammaCorrection(true);
    cuda_hog_->setWinStride(Size(img_width, img_height));

    for (int s = 0; s < num_cuda_stream_; ++s)
    {
        hand_rendered_.emplace_back   (Mat(         Size(img_width * ogl_tiles_cols, img_height* ogl_tiles_rows), CV_8UC3));
        cuda_img_.emplace_back        (cuda::GpuMat(Size(img_width * ogl_tiles_cols, img_height* ogl_tiles_rows), CV_8UC3));
        cuda_img_alpha_.emplace_back  (cuda::GpuMat(Size(img_width * ogl_tiles_cols, img_height* ogl_tiles_rows), CV_8UC4));
        cuda_descriptors_.emplace_back(cuda::GpuMat(Size(num_img_stream_, feature_dim),                           CV_32F ));
        cpu_descriptors_.emplace_back (Mat(         Size(num_img_stream_, feature_dim),                           CV_32F ));
    }
}


VisualUpdateParticles::~VisualUpdateParticles() noexcept { }


void VisualUpdateParticles::innovation(const Ref<const MatrixXf>& pred_states, cv::InputArray measurements, Ref<MatrixXf> innovations)
{
    for (int s = 0; s < num_cuda_stream_; ++s)
        observation_model_->observe(pred_states.block(0, s * num_img_stream_, 7, num_img_stream_), hand_rendered_[s]);

    for (int s = 0; s < num_cuda_stream_; ++s)
    {
        cuda_img_[s].upload(hand_rendered_[s], cuda_stream_[s]);
        cuda::cvtColor(cuda_img_[s], cuda_img_alpha_[s], COLOR_BGR2BGRA, 4, cuda_stream_[s]);
        cuda_hog_->compute(cuda_img_alpha_[s], cuda_descriptors_[s], cuda_stream_[s]);
    }

    for (int s = 0; s < num_cuda_stream_; ++s)
    {
        cuda_stream_[s].waitForCompletion();

        cuda_descriptors_[s].download(cpu_descriptors_[s], cuda_stream_[s]);

        for (int i = 0; i < num_img_stream_; ++i)
        {
            float sum_diff = 0;
            {
            auto it_cam     = (static_cast<const std::vector<float>*>(measurements.getObj()))->begin();
            auto it_cam_end = (static_cast<const std::vector<float>*>(measurements.getObj()))->end();
            int  j          = 0;
            while (it_cam < it_cam_end)
            {
                sum_diff += abs((*it_cam) - cpu_descriptors_[s].at<float>(i, j));

                ++it_cam;
                ++j;
            }
            }

            innovations(s * num_img_stream_ + i, 0) = sum_diff;
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


void VisualUpdateParticles::setVisualObservationModel(std::unique_ptr<VisualObservationModel> visual_observation_model)
{
    throw std::runtime_error("ERROR::VISUALUPDATEPARTICLES::SETVISUALOBSERVATIONMODEL\nERROR:\n\tClass under development. Do not invoke this method.");

    VisualObservationModel* tmp_observation_model = visual_observation_model.release();

    observation_model_ = std::unique_ptr<VisualProprioception>(dynamic_cast<VisualProprioception*>(tmp_observation_model));
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
