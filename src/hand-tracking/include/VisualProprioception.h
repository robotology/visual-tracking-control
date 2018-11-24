#ifndef VISUALPROPRIOCEPTION_H
#define VISUALPROPRIOCEPTION_H

#include <BayesFilters/MeasurementModel.h>

#include <Camera.h>
#include <MeshModel.h>

#include <array>
#include <string>
#include <memory>

#include <opencv2/core/core.hpp>

#include <SuperimposeMesh/SICAD.h>


class VisualProprioception : public bfl::MeasurementModel
{
public:
    VisualProprioception(std::unique_ptr<bfl::Camera> camera, const int num_requested_images, std::unique_ptr<bfl::MeshModel> mesh_model);

    virtual ~VisualProprioception() noexcept;

    std::pair<bool, bfl::Data> measure(const Eigen::Ref<const Eigen::MatrixXf>& cur_states) const override;

    std::pair<bool, bfl::Data> predictedMeasure(const Eigen::Ref<const Eigen::MatrixXf>& cur_states) const override;

    std::pair<bool, bfl::Data> innovation(const bfl::Data& predicted_measurements, const bfl::Data& measurements) const override;

    bool bufferAgentData() const override;

    std::pair<bool, bfl::Data> getAgentMeasurements() const override;

    /* IMPROVEME
     * Find a way to better communicate with the callee. Maybe a struct.
     */
    int getNumberOfUsedParticles() const;

    /* TODELETE
     * For debugging walkman
     */
    void superimpose(const Superimpose::ModelPoseContainer& obj2pos_map, cv::Mat& img);

private:
    struct ImplData;

    std::unique_ptr<ImplData> pImpl_;
};

#endif /* VISUALPROPRIOCEPTION_H */
