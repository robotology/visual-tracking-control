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


/* FIXME:
   Implement PIMPL pattern to remove global variables.
 */
cv::Ptr<cv::cuda::HOG> hog_cuda_;

const GLuint* pbo_ = nullptr;

size_t pbo_size_ = 0;

struct cudaGraphicsResource** pbo_cuda_;


VisualProprioception::VisualProprioception
(
    std::unique_ptr<bfl::Camera> camera,
    const int num_requested_images,
    std::unique_ptr<bfl::MeshModel> mesh_model
) :
    camera_(std::move(camera)),
    mesh_model_(std::move(mesh_model)),
    cam_params_(camera_->getCameraParameters())
{
    bool valid_parameter = false;

    std::tie(valid_parameter, mesh_paths_) = mesh_model_->getMeshPaths();
    if (!valid_parameter)
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR\nERROR: Could not find meshe files.");

    std::tie(valid_parameter, shader_folder_) = mesh_model_->getShaderPaths();
    if (!valid_parameter)
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR\nERROR: Could not find shader folder.");


    hog_cuda_ = cv::cuda::HOG::create(cv::Size(cam_params_.width, cam_params_.height), cv::Size(block_size_, block_size_), cv::Size(block_size_ / 2, block_size_ / 2), cv::Size(block_size_ / 2, block_size_ / 2), bin_number_);
    hog_cuda_->setDescriptorFormat(cv::cuda::HOG::DESCR_FORMAT_ROW_BY_ROW);
    hog_cuda_->setGammaCorrection(true);
    hog_cuda_->setWinStride(cv::Size(cam_params_.width, cam_params_.height));


    try
    {
        si_cad_ = std::unique_ptr<SICAD>(new SICAD(mesh_paths_,
                                                   cam_params_.width, cam_params_.height,
                                                   cam_params_.fx, cam_params_.fy, cam_params_.cx, cam_params_.cy,
                                                   num_requested_images,
                                                   shader_folder_,
                                                   { 1.0, 0.0, 0.0, static_cast<float>(M_PI) }));
    }
    catch (const std::runtime_error& e)
    {
        throw std::runtime_error(e.what());
    }

    num_images_ = si_cad_->getTilesNumber();

    feature_dim_ = (cam_params_.width / block_size_ * 2 - 1) * (cam_params_.height / block_size_ * 2 - 1) * bin_number_ * 4;

    std::tie(pbo_, pbo_size_) = si_cad_->getPBOs();

    pbo_cuda_ = new cudaGraphicsResource*[pbo_size_]();

    for (size_t i = 0; i < pbo_size_; ++i)
        cudaGraphicsGLRegisterBuffer(pbo_cuda_ + i, pbo_[i], cudaGraphicsRegisterFlagsNone);

    si_cad_->releaseContext();
}


VisualProprioception::~VisualProprioception() noexcept
{
    delete[] pbo_cuda_;
}


/** FIXME
 Sicuri che measure, predicted measure e innovation debbano essere const?!
 */
std::pair<bool, bfl::Data> VisualProprioception::measure(const Ref<const MatrixXf>& cur_states) const
{
    std::tie(std::ignore, camera_position_, camera_orientation_) = bfl::any::any_cast<bfl::Camera::CameraData>(camera_->getData());

    std::vector<Superimpose::ModelPoseContainer> mesh_poses;
    bool success = false;

    std::tie(success, mesh_poses) = mesh_model_->getModelPose(cur_states);
    if (!success)
        return std::make_pair(false, MatrixXf::Zero(1, 1));

    success &= si_cad_->superimpose(mesh_poses, camera_position_.data(), camera_orientation_.data(), 0);
    if (!success)
        return std::make_pair(false, MatrixXf::Zero(1, 1));

    char* pbo_cuda_data;
    cudaGraphicsMapResources(static_cast<int>(pbo_size_), pbo_cuda_, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_cuda_data), &num_bytes, pbo_cuda_[0]);

    cv::cuda::GpuMat cuda_mat_render(si_cad_->getTilesRows() * cam_params_.height, si_cad_->getTilesCols() * cam_params_.width, CV_8UC3, static_cast<void*>(pbo_cuda_data));


    /* FIXME
       The following two steps shold be performed by OpenGL. */
    cv::cuda::GpuMat cuda_mat_render_flipped;
    cv::cuda::flip(cuda_mat_render, cuda_mat_render_flipped, 0);

    cv::cuda::GpuMat cuda_mat_render_flipped_alpha;
    cv::cuda::cvtColor(cuda_mat_render_flipped, cuda_mat_render_flipped_alpha, cv::COLOR_BGR2BGRA, 4);


    cv::cuda::GpuMat cuda_descriptor;
    hog_cuda_->compute(cuda_mat_render_flipped_alpha, cuda_descriptor);


    cudaGraphicsUnmapResources(static_cast<int>(pbo_size_), pbo_cuda_, 0);
    si_cad_->releaseContext();


    //return std::make_pair(true, std::move(cuda_descriptor));
    return std::make_pair(true, cuda_descriptor);
}


std::pair<bool, bfl::Data> VisualProprioception::predictedMeasure(const Ref<const MatrixXf>& cur_states) const
{
    return measure(cur_states);
}


std::pair<bool, bfl::Data> VisualProprioception::innovation(const bfl::Data& predicted_measurements, const bfl::Data& measurements) const
{
    MatrixXf innovation = -(bfl::any::any_cast<MatrixXf>(predicted_measurements).rowwise() - bfl::any::any_cast<MatrixXf>(measurements).row(0));

    return std::make_pair(true, std::move(innovation));
}


bool VisualProprioception::bufferAgentData() const
{
    return camera_->bufferData();
}


std::pair<bool, bfl::Data> VisualProprioception::getAgentMeasurements() const
{
    cv::Mat camera_image;
    std::tie(camera_image, std::ignore, std::ignore) = bfl::any::any_cast<bfl::Camera::CameraData>(camera_->getData());

    cv::cuda::GpuMat cuda_img;
    cv::cuda::GpuMat cuda_img_alpha;
    cv::cuda::GpuMat cuda_descriptors;

    cuda_img.upload(camera_image);

    cv::cuda::cvtColor(cuda_img, cuda_img_alpha, cv::COLOR_BGR2BGRA, 4);

    hog_cuda_->compute(cuda_img_alpha, cuda_descriptors);


    //return std::make_pair(true, std::move(cuda_descriptors));
    return std::make_pair(true, cuda_descriptors);
}


int VisualProprioception::getOGLTilesNumber() const
{
    return num_images_;
}


void VisualProprioception::superimpose(const Superimpose::ModelPoseContainer& obj2pos_map, cv::Mat& img)
{
    try
    {
        std::tie(std::ignore, camera_position_, camera_orientation_) = bfl::any::any_cast<bfl::Camera::CameraData>(camera_->getData());
    }
    catch(const bfl::any::bad_any_cast& e)
    {
        std::cout << log_ID_ << "[superimpose] Error: " << e.what() << std::endl;

        return;
    }

    si_cad_->setBackgroundOpt(true);
    si_cad_->setWireframeOpt(true);

    si_cad_->superimpose(obj2pos_map, camera_position_.data(), camera_orientation_.data(), img);

    si_cad_->setBackgroundOpt(false);
    si_cad_->setWireframeOpt(false);
}
