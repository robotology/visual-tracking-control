#include <VisualProprioception.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <utility>

using namespace cv;
using namespace Eigen;


VisualProprioception::VisualProprioception(const int num_images)
{
    readCameraParameters(width_, height_, fx_, fy_, cx_, cy_);

    readMeshPaths(mesh_paths_);

    readShaderPaths(shader_folder_);

    cam_x_[0] = 0;
    cam_x_[1] = 0;
    cam_x_[2] = 0;

    cam_o_[0] = 0;
    cam_o_[1] = 0;
    cam_o_[2] = 0;
    cam_o_[3] = 0;

    try
    {
        si_cad_ = new SICAD(mesh_paths_,
                            width_, height_, fx_, fy_, cx_, cy_,
                            num_images,
                            {1.0, 0.0, 0.0, static_cast<float>(M_PI)},
                            shader_folder_,
                            false);
    }
    catch (const std::runtime_error& e)
    {
        throw std::runtime_error(e.what());
    }
}


VisualProprioception::~VisualProprioception() noexcept
{
    delete si_cad_;
}


VisualProprioception::VisualProprioception(const VisualProprioception& proprio) :
    mesh_paths_(proprio.mesh_paths_),
    shader_folder_(proprio.shader_folder_),
    si_cad_(proprio.si_cad_),
    width_(proprio.width_),
    height_(proprio.height_),
    fx_(proprio.fx_),
    cx_(proprio.cx_),
    fy_(proprio.fy_),
    cy_(proprio.cy_)
{
    cam_x_[0] = proprio.cam_x_[0];
    cam_x_[1] = proprio.cam_x_[1];
    cam_x_[2] = proprio.cam_x_[2];

    cam_o_[0] = proprio.cam_o_[0];
    cam_o_[1] = proprio.cam_o_[1];
    cam_o_[2] = proprio.cam_o_[2];
    cam_o_[3] = proprio.cam_o_[3];
}


VisualProprioception::VisualProprioception(VisualProprioception&& proprio) noexcept :
    mesh_paths_(std::move(proprio.mesh_paths_)),
    shader_folder_(std::move(proprio.shader_folder_)),
    si_cad_(std::move(proprio.si_cad_)),
    width_(std::move(proprio.width_)),
    height_(std::move(proprio.height_)),
    fx_(std::move(proprio.fx_)),
    cx_(std::move(proprio.cx_)),
    fy_(std::move(proprio.fy_)),
    cy_(std::move(proprio.cy_))
{
    cam_x_[0] = proprio.cam_x_[0];
    cam_x_[1] = proprio.cam_x_[1];
    cam_x_[2] = proprio.cam_x_[2];

    cam_o_[0] = proprio.cam_o_[0];
    cam_o_[1] = proprio.cam_o_[1];
    cam_o_[2] = proprio.cam_o_[2];
    cam_o_[3] = proprio.cam_o_[3];

    proprio.cam_x_[0] = 0.0;
    proprio.cam_x_[1] = 0.0;
    proprio.cam_x_[2] = 0.0;

    proprio.cam_o_[0] = 0.0;
    proprio.cam_o_[1] = 0.0;
    proprio.cam_o_[2] = 0.0;
    proprio.cam_o_[3] = 0.0;
}


VisualProprioception& VisualProprioception::operator=(const VisualProprioception& proprio)
{
    width_ = proprio.width_;
    height_ = proprio.height_;

    fx_ = proprio.fx_;
    cx_ = proprio.cx_;
    fy_ = proprio.fy_;
    cy_ = proprio.cy_;

    mesh_paths_ = proprio.mesh_paths_;
    si_cad_ = proprio.si_cad_;
    cam_x_[0] = proprio.cam_x_[0];
    cam_x_[1] = proprio.cam_x_[1];
    cam_x_[2] = proprio.cam_x_[2];

    cam_o_[0] = proprio.cam_o_[0];
    cam_o_[1] = proprio.cam_o_[1];
    cam_o_[2] = proprio.cam_o_[2];
    cam_o_[3] = proprio.cam_o_[3];

    return *this;
}


VisualProprioception& VisualProprioception::operator=(VisualProprioception&& proprio) noexcept
{
    cam_x_[0] = proprio.cam_x_[0];
    cam_x_[1] = proprio.cam_x_[1];
    cam_x_[2] = proprio.cam_x_[2];

    cam_o_[0] = proprio.cam_o_[0];
    cam_o_[1] = proprio.cam_o_[1];
    cam_o_[2] = proprio.cam_o_[2];
    cam_o_[3] = proprio.cam_o_[3];

    width_  = std::move(proprio.width_);
    height_ = std::move(proprio.height_);
    fx_     = std::move(proprio.fx_);
    cx_     = std::move(proprio.cx_);
    fy_     = std::move(proprio.fy_);
    cy_     = std::move(proprio.cy_);

    mesh_paths_ = std::move(proprio.mesh_paths_);
    si_cad_  = std::move(proprio.si_cad_);

    proprio.cam_x_[0] = 0.0;
    proprio.cam_x_[1] = 0.0;
    proprio.cam_x_[2] = 0.0;

    proprio.cam_o_[0] = 0.0;
    proprio.cam_o_[1] = 0.0;
    proprio.cam_o_[2] = 0.0;
    proprio.cam_o_[3] = 0.0;

    return *this;
}


void VisualProprioception::observe(const Ref<const MatrixXf>& cur_states, OutputArray observations)
{
    std::vector<Superimpose::ModelPoseContainer> hand_poses;
    getModelPose(cur_states, hand_poses);

    observations.create(height_ * si_cad_->getTilesRows(), width_ * si_cad_->getTilesCols(), CV_8UC3);
    Mat hand_ogl = observations.getMat();

    si_cad_->superimpose(hand_poses, cam_x_, cam_o_, hand_ogl);
}


bool VisualProprioception::setProperty(const std::string property)
{
    return false;
}


int VisualProprioception::getOGLTilesNumber()
{
    return si_cad_->getTilesNumber();
}


int VisualProprioception::getOGLTilesRows()
{
    return si_cad_->getTilesRows();
}


int VisualProprioception::getOGLTilesCols()
{
    return si_cad_->getTilesCols();
}


unsigned int VisualProprioception::getCamWidth()
{
    return width_;
}


unsigned int VisualProprioception::getCamHeight()
{
    return height_;
}


float VisualProprioception::getCamFx()
{
    return fx_;
}


float VisualProprioception::getCamFy()
{
    return fy_;
}


float VisualProprioception::getCamCx()
{
    return cx_;
}


float VisualProprioception::getCamCy()
{
    return cy_;
}
