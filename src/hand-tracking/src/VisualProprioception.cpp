#include <VisualProprioception.h>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <utility>

using namespace cv;
using namespace Eigen;


VisualProprioception::VisualProprioception(const int num_images, std::unique_ptr<bfl::Camera> camera, std::unique_ptr<bfl::MeshModel> mesh_model) :
    num_images_(num_images),
    camera_(std::move(camera)),
    mesh_model_(std::move(mesh_model))
{
    bool success = false;

    std::tie(success, cam_params_) = camera_->getCameraParameters();
    if (!success)
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR\nERROR: Could not read camera parameters.");

    std::tie(success, mesh_paths_) = mesh_model_->getMeshPaths();
    if (!success)
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR\nERROR: Could not find meshe files.");

    std::tie(success, shader_folder_) = mesh_model_->getShaderPaths();
    if (!success)
        throw std::runtime_error("ERROR::VISUALPROPRIOCEPTION::CTOR\nERROR: Could not find shader folder.");

    try
    {
        si_cad_ = std::unique_ptr<SICAD>(new SICAD(mesh_paths_,
                                                   cam_params_.width, cam_params_.height,
                                                   cam_params_.fx, cam_params_.fy, cam_params_.cx, cam_params_.cy,
                                                   num_images,
                                                   shader_folder_,
                                                   { 1.0, 0.0, 0.0, static_cast<float>(M_PI) }));
    }
    catch (const std::runtime_error& e)
    {
        throw std::runtime_error(e.what());
    }
}


void VisualProprioception::observe(const Ref<const MatrixXf>& cur_states, OutputArray observations)
{
    bool success = false;
    std::vector<Superimpose::ModelPoseContainer> hand_poses;

    std::tie(success, hand_poses)     = mesh_model_->getModelPose(cur_states);
    std::tie(success, cam_x_, cam_o_) = camera_->getCameraPose();

    observations.create(cam_params_.height * si_cad_->getTilesRows(), cam_params_.width * si_cad_->getTilesCols(), CV_8UC3);
    Mat hand_ogl = observations.getMat();

    si_cad_->superimpose(hand_poses, cam_x_.data(), cam_o_.data(), hand_ogl);
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
    return cam_params_.width;
}


unsigned int VisualProprioception::getCamHeight()
{
    return cam_params_.height;
}


float VisualProprioception::getCamFx()
{
    return cam_params_.fx;
}


float VisualProprioception::getCamFy()
{
    return cam_params_.fy;
}


float VisualProprioception::getCamCx()
{
    return cam_params_.cx;
}


float VisualProprioception::getCamCy()
{
    return cam_params_.cy;
}


void VisualProprioception::superimpose(const Superimpose::ModelPoseContainer& obj2pos_map, Mat& img)
{
    si_cad_->setBackgroundOpt(true);
    si_cad_->setWireframeOpt(true);

    si_cad_->superimpose(obj2pos_map, cam_x_.data(), cam_o_.data(), img);

    si_cad_->setBackgroundOpt(false);
    si_cad_->setWireframeOpt(false);
}