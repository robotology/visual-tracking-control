#ifndef VISUALPROPRIOCEPTION_H
#define VISUALPROPRIOCEPTION_H

#include <string>

#include <BayesFilters/VisualObservationModel.h>
#include <opencv2/core/core.hpp>
#include <SuperimposeMesh/SICAD.h>


class VisualProprioception : public bfl::VisualObservationModel
{
public:
    VisualProprioception(const int num_images);

    VisualProprioception(const VisualProprioception& proprio);

    VisualProprioception(VisualProprioception&& proprio) noexcept;

    virtual ~VisualProprioception() noexcept;

    VisualProprioception& operator=(const VisualProprioception& proprio);

    VisualProprioception& operator=(VisualProprioception&& proprio) noexcept;

    void observe(const Eigen::Ref<const Eigen::MatrixXf>& cur_states, cv::OutputArray observations) override;

    bool setProperty(const std::string property) override;

    int getOGLTilesNumber();
    int getOGLTilesRows();
    int getOGLTilesCols();

    unsigned int getCamWidth();
    unsigned int getCamHeight();

    float getCamFx();
    float getCamFy();
    float getCamCx();
    float getCamCy();

protected:
    std::string log_ID_ = "[VisualProprioception]";

    virtual void readCameraParameters(unsigned int& width, unsigned int& height, float& fx, float& fy, float& cx, float& cy) = 0;

    virtual void readMeshPaths(SICAD::ModelPathContainer& mesh_path) = 0;

    virtual void readShaderPaths(std::string& mesh_path) = 0;

    virtual void getModelPose(const Eigen::Ref<const Eigen::MatrixXf>& cur_states, std::vector<Superimpose::ModelPoseContainer>& hand_poses) = 0;


    SICAD::ModelPathContainer mesh_paths_;
    std::string               shader_folder_;

    SICAD*                    si_cad_;
    
    unsigned int width_;
    unsigned int height_;

    float fx_;
    float cx_;
    float fy_;
    float cy_;

    double cam_x_[3];
    double cam_o_[4];
};

#endif /* VISUALPROPRIOCEPTION_H */
