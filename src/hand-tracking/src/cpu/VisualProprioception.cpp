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

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>

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

    std::unique_ptr<cv::HOGDescriptor> hog_cpu_;

    MatrixXf descriptor_out_;
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

    rImpl.hog_cpu_ = std::unique_ptr<cv::HOGDescriptor>(new cv::HOGDescriptor(cv::Size(rImpl.cam_params_.width, rImpl.cam_params_.height),
                                                                              cv::Size(rImpl.block_size_, rImpl.block_size_),
                                                                              cv::Size(rImpl.block_size_ / 2, rImpl.block_size_ / 2),
                                                                              cv::Size(rImpl.block_size_ / 2, rImpl.block_size_ / 2),
                                                                              rImpl.bin_number_,
                                                                              1, -1.0, cv::HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS, false));

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

    rImpl.descriptor_out_.resize(1, rImpl.feature_dim_);
}


VisualProprioception::~VisualProprioception() noexcept = default;


std::pair<bool, bfl::Data> VisualProprioception::measure() const
{
    return std::make_pair(true, std::move(pImpl_->descriptor_out_));
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

    MatrixXf descriptor(rImpl.num_images_, rImpl.feature_dim_);

    cv::Mat rendered_image;
    success &= rImpl.si_cad_->superimpose(mesh_poses, camera_position.data(), camera_orientation.data(), rendered_image);
    if (!success)
        return std::make_pair(false, MatrixXf::Zero(1, 1));

    /*
     * cv::HOGDescriptor::compute requires a std::vector<float>.
     */
    std::vector<float> descriptors_cpu;
    rImpl.hog_cpu_->compute(rendered_image, descriptors_cpu, cv::Size(rImpl.cam_params_.width, rImpl.cam_params_.height));

    /* FIXME
     * The following copy operation is super slow (approx 4 ms for 2M numbers).
     * Must find out a new, direct, way of doing this.
     */
    descriptor = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(descriptors_cpu.data(), rImpl.num_images_, rImpl.feature_dim_);

    return std::make_pair(true, std::move(descriptor));
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

    /*
     * cv::HOGDescriptor::compute requires a std::vector<float>.
     */
    std::vector<float> descriptors_cpu;
    rImpl.hog_cpu_->compute(camera_image, descriptors_cpu, cv::Size(rImpl.cam_params_.width, rImpl.cam_params_.height));

    /* FIXME
     * The following copy operation is super slow (approx 4 ms for 2M numbers).
     * Must find out a new, direct, way of doing this.
     */
    rImpl.descriptor_out_ = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(descriptors_cpu.data(), 1, rImpl.feature_dim_);

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
