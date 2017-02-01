#include "VisualParticleFilterCorrection.h"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace bfl;
using namespace cv;
using namespace Eigen;


VisualParticleFilterCorrection::VisualParticleFilterCorrection(std::shared_ptr<VisualObservationModel> measurement_model) noexcept :
    measurement_model_(measurement_model) { }


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


void VisualParticleFilterCorrection::correct(const Eigen::Ref<const Eigen::VectorXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::VectorXf> cor_state)
{
    VectorXf innovate(1);
    innovation(pred_state, measurements, innovate);
    likelihood(innovate, cor_state);
}


void VisualParticleFilterCorrection::innovation(const Eigen::Ref<const Eigen::VectorXf>& pred_state, cv::InputArray measurements, Eigen::Ref<Eigen::MatrixXf> innovation)
{
    int block_size = 16;
    Mat hand_edge_ogl_cv;
    std::vector<Point> points;
    std::vector<float> descriptors_cam;
    std::vector<float> descriptors_cad;

    measurement_model_->observe(pred_state, hand_edge_ogl_cv);

    /* In-crop HOG between camera and render edges */
    HOGDescriptor hog(hand_edge_ogl_cv.size(), Size(block_size, block_size), Size(block_size/2, block_size/2), Size(block_size/2, block_size/2), 9, 1, -1, HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS, false);

    hog.compute(hand_edge_ogl_cv, descriptors_cam);
    hog.compute(measurements,     descriptors_cad);

    float sum_diff = 0;
    {
    auto it_cad = descriptors_cad.begin();
    auto it_cam = descriptors_cam.begin();
        for (; it_cad < descriptors_cad.end(); ++it_cad, ++it_cam) sum_diff += abs((*it_cam) - (*it_cad));
    }

    innovation(0, 0) = sum_diff;
}


void VisualParticleFilterCorrection::likelihood(const Eigen::Ref<const Eigen::MatrixXf>& innovation, Eigen::Ref<Eigen::VectorXf> cor_state)
{
    // FIXME: Kernel likelihood need to be tuned!
    cor_state(0) *= ( exp( -0.001 * innovation(0, 0) /* / pow(1, 2.0) */ ) );
    if (cor_state(0) <= 0) cor_state(0) = std::numeric_limits<float>::min();
}
