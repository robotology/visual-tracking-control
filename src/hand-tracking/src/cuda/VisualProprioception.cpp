/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <VisualProprioception.h>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <utility>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace Eigen;



struct VisualProprioception::ImplData
{
    const std::string log_ID_ = "[VisualProprioception]";

    std::unique_ptr<bfl::Camera> camera_ = nullptr;

    std::unique_ptr<bfl::MeshModel> mesh_model_;

    bfl::Camera::CameraIntrinsics cam_params_;

    SICAD::ModelPathContainer mesh_paths_;

    std::string shader_folder_;

    std::unique_ptr<SICAD> si_cad_;

    int num_images_;

    const int block_size_ = 16;

    const int bin_number_ = 9;

    unsigned int feature_dim_;

    cv::Ptr<cv::cuda::HOG> hog_cuda_;

    const GLuint* pbo_ = nullptr;

    size_t pbo_size_ = 0;

    struct cudaGraphicsResource** pbo_cuda_;

    cv::cuda::GpuMat cuda_descriptors_;
};


VisualProprioception::VisualProprioception
(
    std::unique_ptr<bfl::Camera> camera,
    const int num_requested_images,
    std::unique_ptr<bfl::MeshModel> mesh_model
) :
    pImpl_(std::unique_ptr<ImplData>(new ImplData))
{
    ImplData& rImpl = *pImpl_;


    rImpl.camera_ = std::move(camera);

    rImpl.mesh_model_ = std::move(mesh_model);

    rImpl.cam_params_ = rImpl.camera_->getCameraParameters();


    bool valid_parameter = false;

    std::tie(valid_parameter, rImpl.mesh_paths_) = rImpl.mesh_model_->getMeshPaths();
    if (!valid_parameter)
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR\nERROR: Could not find meshe files.");

    std::tie(valid_parameter, rImpl.shader_folder_) = rImpl.mesh_model_->getShaderPaths();
    if (!valid_parameter)
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR\nERROR: Could not find shader folder.");


    rImpl.hog_cuda_ = cv::cuda::HOG::create(cv::Size(rImpl.cam_params_.width, rImpl.cam_params_.height), cv::Size(rImpl.block_size_, rImpl.block_size_), cv::Size(rImpl.block_size_ / 2, rImpl.block_size_ / 2), cv::Size(rImpl.block_size_ / 2, rImpl.block_size_ / 2), rImpl.bin_number_);
    rImpl.hog_cuda_->setDescriptorFormat(cv::cuda::HOG::DESCR_FORMAT_ROW_BY_ROW);
    rImpl.hog_cuda_->setGammaCorrection(true);
    rImpl.hog_cuda_->setWinStride(cv::Size(rImpl.cam_params_.width, rImpl.cam_params_.height));


    try
    {
        rImpl.si_cad_ = std::unique_ptr<SICAD>(new SICAD(rImpl.mesh_paths_,
                                                         rImpl.cam_params_.width, rImpl.cam_params_.height,
                                                         rImpl.cam_params_.fx, rImpl.cam_params_.fy, rImpl.cam_params_.cx, rImpl.cam_params_.cy,
                                                         num_requested_images,
                                                         rImpl.shader_folder_,
                                                         { 1.0, 0.0, 0.0, static_cast<float>(M_PI) }));
    }
    catch (const std::runtime_error& e)
    {
        throw std::runtime_error(e.what());
    }

    rImpl.num_images_ = rImpl.si_cad_->getTilesNumber();

    rImpl.feature_dim_ = (rImpl.cam_params_.width / rImpl.block_size_ * 2 - 1) * (rImpl.cam_params_.height / rImpl.block_size_ * 2 - 1) * rImpl.bin_number_ * 4;

    std::tie(rImpl.pbo_, rImpl.pbo_size_) = rImpl.si_cad_->getPBOs();

    rImpl.pbo_cuda_ = new cudaGraphicsResource*[rImpl.pbo_size_]();

    for (size_t i = 0; i < rImpl.pbo_size_; ++i)
        cudaGraphicsGLRegisterBuffer(rImpl.pbo_cuda_ + i, rImpl.pbo_[i], cudaGraphicsRegisterFlagsNone);

    rImpl.si_cad_->releaseContext();
}


VisualProprioception::~VisualProprioception() noexcept
{
    ImplData& rImpl = *pImpl_;

    delete[] rImpl.pbo_cuda_;
}


std::pair<bool, bfl::Data> VisualProprioception::measure() const
{
    //return std::make_pair(true, std::move(rImpl.cuda_descriptors_));
    return std::make_pair(true, pImpl_->cuda_descriptors_);
}


std::pair<bool, bfl::Data> VisualProprioception::predictedMeasure(const Ref<const MatrixXd>& cur_states) const
{
    ImplData& rImpl = *pImpl_;

    std::array<double, 3> camera_position;
    std::array<double, 4> camera_orientation;

    std::tie(std::ignore, camera_position, camera_orientation) = bfl::any::any_cast<bfl::Camera::CameraData>(rImpl.camera_->getData());

    std::vector<Superimpose::ModelPoseContainer> mesh_poses;
    bool success = false;

    std::tie(success, mesh_poses) = rImpl.mesh_model_->getModelPose(cur_states);
    if (!success)
        return std::make_pair(false, MatrixXf::Zero(1, 1));

    success &= rImpl.si_cad_->superimpose(mesh_poses, camera_position.data(), camera_orientation.data(), 0);
    if (!success)
        return std::make_pair(false, MatrixXf::Zero(1, 1));

    char* pbo_cuda_data;
    cudaGraphicsMapResources(static_cast<int>(rImpl.pbo_size_), rImpl.pbo_cuda_, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_cuda_data), &num_bytes, rImpl.pbo_cuda_[0]);

    cv::cuda::GpuMat cuda_mat_render(rImpl.si_cad_->getTilesRows() * rImpl.cam_params_.height,
                                     rImpl.si_cad_->getTilesCols() * rImpl.cam_params_.width,
                                     CV_8UC3, static_cast<void*>(pbo_cuda_data));


    /* IMPROVEME
     * The following two steps shold be performed by OpenGL.
     */
    cv::cuda::GpuMat cuda_mat_render_flipped;
    cv::cuda::flip(cuda_mat_render, cuda_mat_render_flipped, 0);

    cv::cuda::GpuMat cuda_mat_render_flipped_alpha;
    cv::cuda::cvtColor(cuda_mat_render_flipped, cuda_mat_render_flipped_alpha, cv::COLOR_BGR2BGRA, 4);


    cv::cuda::GpuMat cuda_descriptor;
    rImpl.hog_cuda_->compute(cuda_mat_render_flipped_alpha, cuda_descriptor);


    cudaGraphicsUnmapResources(static_cast<int>(rImpl.pbo_size_), rImpl.pbo_cuda_, 0);
    rImpl.si_cad_->releaseContext();


    //return std::make_pair(true, std::move(cuda_descriptor));
    return std::make_pair(true, cuda_descriptor);
}


std::pair<bool, bfl::Data> VisualProprioception::innovation(const bfl::Data& predicted_measurements, const bfl::Data& measurements) const
{
    MatrixXf innovation = -(bfl::any::any_cast<MatrixXf>(predicted_measurements).rowwise() - bfl::any::any_cast<MatrixXf>(measurements).row(0));

    return std::make_pair(true, std::move(innovation));
}


bool VisualProprioception::freezeMeasurements()
{
    ImplData& rImpl = *pImpl_;

    if(!rImpl.camera_->bufferData())
        return false;

    cv::Mat camera_image;
    std::tie(camera_image, std::ignore, std::ignore) = bfl::any::any_cast<bfl::Camera::CameraData>(rImpl.camera_->getData());

    cv::cuda::GpuMat cuda_img;
    cv::cuda::GpuMat cuda_img_alpha;

    cuda_img.upload(camera_image);

    cv::cuda::cvtColor(cuda_img, cuda_img_alpha, cv::COLOR_BGR2BGRA, 4);

    rImpl.hog_cuda_->compute(cuda_img_alpha, rImpl.cuda_descriptors_);

    return true;
}


std::pair<std::size_t, std::size_t> VisualProprioception::getOutputSize() const
{
    /* The output size is set to (0, 0) since the measurements are stored in a row major format,
       hence not compatible with the required output format. */
    return std::make_pair(0, 0);
}


int VisualProprioception::getNumberOfUsedParticles() const
{
    ImplData& rImpl = *pImpl_;

    return rImpl.num_images_;
}


void VisualProprioception::superimpose(const Superimpose::ModelPoseContainer& obj2pos_map, cv::Mat& img)
{
    ImplData& rImpl = *pImpl_;

    std::array<double, 3> camera_position;
    std::array<double, 4> camera_orientation;

    try
    {
        std::tie(std::ignore, camera_position, camera_orientation) = bfl::any::any_cast<bfl::Camera::CameraData>(rImpl.camera_->getData());
    }
    catch(const bfl::any::bad_any_cast& e)
    {
        std::cout << rImpl.log_ID_ << "[superimpose] Error: " << e.what() << std::endl;

        return;
    }

    rImpl.si_cad_->setBackgroundOpt(true);
    rImpl.si_cad_->setWireframeOpt(true);

    rImpl.si_cad_->superimpose(obj2pos_map, camera_position.data(), camera_orientation.data(), img);

    rImpl.si_cad_->setBackgroundOpt(false);
    rImpl.si_cad_->setWireframeOpt(false);
}
