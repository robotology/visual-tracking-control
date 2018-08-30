#include <VisualProprioception.h>
#include <CameraData.h>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <utility>

#if HANDTRACKING_USE_OPENCV_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif // HANDTRACKING_USE_OPENCV_CUDA

using namespace Eigen;


VisualProprioception::VisualProprioception
(
    const int num_images,
    const bfl::Camera::CameraParameters& cam_params,
    std::unique_ptr<bfl::MeshModel> mesh_model
) :
    mesh_model_(std::move(mesh_model)),
    cam_params_(cam_params),
    num_images_(num_images)
{
    bool valid_parameter = false;

    std::tie(valid_parameter, mesh_paths_) = mesh_model_->getMeshPaths();
    if (!valid_parameter)
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR\nERROR: Could not find meshe files.");

    std::tie(valid_parameter, shader_folder_) = mesh_model_->getShaderPaths();
    if (!valid_parameter)
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR\nERROR: Could not find shader folder.");


#if HANDTRACKING_USE_OPENCV_CUDA
    hog_cuda_ = cv::cuda::HOG::create(cv::Size(cam_params_.width, cam_params_.height), cv::Size(block_size_, block_size_), cv::Size(block_size_ / 2, block_size_ / 2), cv::Size(block_size_ / 2, block_size_ / 2), bin_number_);
    hog_cuda_->setDescriptorFormat(cv::cuda::HOG::DESCR_FORMAT_ROW_BY_ROW);
    hog_cuda_->setGammaCorrection(true);
    hog_cuda_->setWinStride(cv::Size(cam_params_.width, cam_params_.height));
#else
    hog_cpu_ = std::unique_ptr<cv::HOGDescriptor>(new cv::HOGDescriptor(cv::Size(cam_params_.width, cam_params_.height),
                                                                        cv::Size(block_size_, block_size_),
                                                                        cv::Size(block_size_ / 2, block_size_ / 2),
                                                                        cv::Size(block_size_ / 2, block_size_ / 2),
                                                                        bin_number_,
                                                                        1, -1.0, cv::HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS, false));
#endif // HANDTRACKING_USE_OPENCV_CUDA

    try
    {
        si_cad_ = std::unique_ptr<SICAD>(new SICAD(mesh_paths_,
                                                   cam_params_.width, cam_params_.height,
                                                   cam_params_.fx, cam_params_.fy, cam_params_.cx, cam_params_.cy,
                                                   num_images_,
                                                   shader_folder_,
                                                   { 1.0, 0.0, 0.0, static_cast<float>(M_PI) }));
    }
    catch (const std::runtime_error& e)
    {
        throw std::runtime_error(e.what());
    }

    feature_dim_ = (cam_params_.width / block_size_ * 2 - 1) * (cam_params_.height / block_size_ * 2 - 1) * bin_number_ * 4;

#if HANDTRACKING_USE_OPENCV_CUDA
    std::tie(pbo_, pbo_size_) = si_cad_->getPBOs();

    pbo_cuda_ = new cudaGraphicsResource*[pbo_size_]();

    for (size_t i = 0; i < pbo_size_; ++i)
        cudaGraphicsGLRegisterBuffer(pbo_cuda_ + i, pbo_[i], cudaGraphicsRegisterFlagsNone);

    si_cad_->releaseContext();
#endif // HANDTRACKING_USE_OPENCV_CUDA
}


VisualProprioception::~VisualProprioception() noexcept
{
#if HANDTRACKING_USE_OPENCV_CUDA
    cudaGraphicsUnmapResources(static_cast<int>(pbo_size_), pbo_cuda_, 0);

    delete[] pbo_cuda_;
#endif // HANDTRACKING_USE_OPENCV_CUDA
}


std::pair<bool, MatrixXf> VisualProprioception::measure(const Ref<const MatrixXf>& cur_states) const
{
    MatrixXf descriptor_out(num_images_, feature_dim_);

#if HANDTRACKING_USE_OPENCV_CUDA
    std::vector<Superimpose::ModelPoseContainer> mesh_poses;
    bool success = false;

    std::tie(success, mesh_poses) = mesh_model_->getModelPose(cur_states);
    if (!success)
        return std::make_pair(false, MatrixXf::Zero(1, 1));

    success &= si_cad_->superimpose(mesh_poses, camera_position_->data(), camera_orientation_->data(), 0);
    if (!success)
        return std::make_pair(false, MatrixXf::Zero(1, 1));

    cv::cuda::GpuMat cuda_mat_render(si_cad_->getTilesRows() * cam_params_.height, si_cad_->getTilesCols() * cam_params_.width, CV_8UC3, static_cast<void*>(pbo_cuda_));

    cv::cuda::GpuMat cuda_mat_render_flipped;
    cv::cuda::flip(cuda_mat_render, cuda_mat_render_flipped, 0);

    /* FIXME
     This step shold be performed by OpenGL. */
    cv::cuda::GpuMat cuda_mat_render_flipped_alpha;
    cv::cuda::cvtColor(cuda_mat_render_flipped, cuda_mat_render_flipped_alpha, cv::COLOR_BGR2BGRA, 4);

    cv::cuda::GpuMat cuda_descriptor;
    hog_cuda_->compute(cuda_mat_render_flipped_alpha, cuda_descriptor);

    cv::Mat cpu_descriptor;
    cuda_descriptor.download(cpu_descriptor);

    /* FIXME
     Is the following command slow? */
    ocv2eigen(cpu_descriptor, descriptor_out);
#else
    std::vector<Superimpose::ModelPoseContainer> mesh_poses;
    bool success = false;

    std::tie(success, mesh_poses) = mesh_model_->getModelPose(cur_states);
    if (!success)
        return std::make_pair(false, MatrixXf::Zero(1, 1));

    cv::Mat rendered_image;
    success &= si_cad_->superimpose(mesh_poses, camera_position_->data(), camera_orientation_->data(), rendered_image);
    if (!success)
        return std::make_pair(false, MatrixXf::Zero(1, 1));

    std::vector<float> descriptors_cpu;
    hog_cpu_->compute(rendered_image, descriptors_cpu, cv::Size(cam_params_.width, cam_params_.height));

    /* FIXME
     The following copy operation is super slow (approx 4 ms for 2M numbers).
     Must find out a new, direct, way of doing this. */
    descriptor_out.block(s * num_percore_rendered_img_, 0, num_percore_rendered_img_, feature_dim_) = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(descriptors_cpu.data(), num_percore_rendered_img_, feature_dim_);
#endif // HANDTRACKING_USE_OPENCV_CUDA

    return std::make_pair(true, descriptor_out);
}


std::pair<bool, MatrixXf> VisualProprioception::predictedMeasure(const Ref<const MatrixXf>& cur_states) const
{
    return measure(cur_states);
}


std::pair<bool, MatrixXf> VisualProprioception::innovation(const Ref<const MatrixXf>& predicted_measurements, const Ref<const MatrixXf>& measurements) const
{
    return std::make_pair(true, -(predicted_measurements.rowwise() - measurements.row(0)));
}


bool VisualProprioception::registerProcessData(std::shared_ptr<bfl::GenericData> process_data)
{
    std::shared_ptr<bfl::CameraData> process_data_casted = std::dynamic_pointer_cast<bfl::CameraData>(process_data);

    if (process_data)
    {
        camera_image_ = process_data_casted->image_;

        camera_position_ = process_data_casted->position_;

        camera_orientation_ = process_data_casted->orientation_;

        return true;
    }
    else
        return false;
}


std::pair<bool, MatrixXf> VisualProprioception::getProcessMeasurements() const
{
    MatrixXf descriptor_out(1, feature_dim_);

#if HANDTRACKING_USE_OPENCV_CUDA
    cv::cuda::GpuMat img_cuda;
    cv::cuda::GpuMat img_alpha_cuda;
    cv::cuda::GpuMat descriptors_cuda;
    cv::Mat          descriptors_cpu;

    img_cuda.upload(*camera_image_);

    cv::cuda::cvtColor(img_cuda, img_alpha_cuda, cv::COLOR_BGR2BGRA, 4);

    hog_cuda_->compute(img_alpha_cuda, descriptors_cuda);

    descriptors_cuda.download(descriptors_cpu);

    /* FIXME
     Is the following command slow? */
    ocv2eigen(descriptors_cpu, descriptor_out);
#else
    std::vector<float> descriptors_cpu;

    hog_cpu_->compute(*camera_image_, descriptors_cpu, cv::Size(cam_params_.width, cam_params_.height));

    /* FIXME
     The following copy operation is super slow (approx 4 ms for 2M numbers).
     Must find out a new, direct, way of doing this. */
    descriptor_out = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(descriptors_cpu.data(), 1, feature_dim_);
#endif // HANDTRACKING_USE_OPENCV_CUDA

    return std::make_pair(true, descriptor_out);
}


int VisualProprioception::getOGLTilesNumber() const
{
    return num_images_;
}


void VisualProprioception::superimpose(const Superimpose::ModelPoseContainer& obj2pos_map, cv::Mat& img)
{
    si_cad_->setBackgroundOpt(true);
    si_cad_->setWireframeOpt(true);

    si_cad_->superimpose(obj2pos_map, camera_position_->data(), camera_orientation_->data(), img);

    si_cad_->setBackgroundOpt(false);
    si_cad_->setWireframeOpt(false);
}


void VisualProprioception::ocv2eigen(const cv::Mat& src, Ref<MatrixXf> dst) const
{
    CV_DbgAssert(src.rows == dst.rows() && src.cols == dst.cols());

    if (!(dst.Flags & Eigen::RowMajorBit))
    {
        const cv::Mat _dst(src.cols, src.rows, cv::traits::Type<float>::value, dst.data(), (size_t)(dst.outerStride() * sizeof(float)));
        if (src.type() == _dst.type())
            transpose(src, _dst);
        else if (src.cols == src.rows)
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            cv::Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const cv::Mat _dst(src.rows, src.cols, cv::traits::Type<float>::value, dst.data(), (size_t)(dst.outerStride() * sizeof(float)));
        src.convertTo(_dst, _dst.type());
    }
}
