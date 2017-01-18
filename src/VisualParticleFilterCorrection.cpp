#include "VisualParticleFilterCorrection.h"

#include <cmath>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace bfl;
using namespace cv;
using namespace Eigen;


VisualParticleFilterCorrection::VisualParticleFilterCorrection(std::shared_ptr<ObservationModel> measurement_model) noexcept :
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


void VisualParticleFilterCorrection::correct(const Eigen::Ref<const Eigen::VectorXf>& pred_state, const Eigen::Ref<const Eigen::MatrixXf>& measurements, Eigen::Ref<Eigen::VectorXf> cor_state)
{
    VectorXf innovate(1);
    innovation(pred_state, measurements, innovate);
    likelihood(innovate, cor_state);
}


void VisualParticleFilterCorrection::innovation(const Eigen::Ref<const Eigen::VectorXf>& pred_state, const Eigen::Ref<const Eigen::MatrixXf>& measurements, Eigen::Ref<Eigen::MatrixXf> innovation)
{
    int block_size = 16;
    Mat hand_edge_ogl_cv;
    std::vector<Point> points;

    MatrixXf hand_edge_ogl;
    measurement_model_->observe(pred_state, hand_edge_ogl);

    /* OGL image crop */
    eigen2cv(hand_edge_ogl, hand_edge_ogl_cv);
    for (auto it = hand_edge_ogl_cv.begin<float>(); it != hand_edge_ogl_cv.end<float>(); ++it) if (*it) points.push_back(it.pos());

    if (points.size() > 0)
    {
//        Mat                hand_edge_cam_cv = measurements;
        Mat                hand_edge_cam_cv;
        Mat                cad_edge_crop;
        Mat                cam_edge_crop;
        std::vector<float> descriptors_cam;
        std::vector<float> descriptors_cad;
        std::vector<Point> locations;
        int                rem_not_mult;

        eigen2cv(MatrixXf(measurements), hand_edge_cam_cv);

        Rect cad_crop_roi = boundingRect(points);

        rem_not_mult = div(cad_crop_roi.width,  block_size).rem;
        if (rem_not_mult > 0) cad_crop_roi.width  = cad_crop_roi.width  + (block_size - rem_not_mult);

        rem_not_mult = div(cad_crop_roi.height, block_size).rem;
        if (rem_not_mult > 0) cad_crop_roi.height = cad_crop_roi.height + (block_size - rem_not_mult);

        if (cad_crop_roi.x + cad_crop_roi.width  > hand_edge_ogl_cv.cols) cad_crop_roi.x -= (cad_crop_roi.x + cad_crop_roi.width ) - hand_edge_ogl_cv.cols;

        if (cad_crop_roi.y + cad_crop_roi.height > hand_edge_ogl_cv.rows) cad_crop_roi.y -= (cad_crop_roi.y + cad_crop_roi.height) - hand_edge_ogl_cv.rows;

        hand_edge_ogl_cv(cad_crop_roi).convertTo(cad_edge_crop, CV_8U);
        hand_edge_cam_cv(cad_crop_roi).convertTo(cam_edge_crop, CV_8U);

        /* In-crop HOG between camera and render edges */
        HOGDescriptor hog(Size(cad_crop_roi.width, cad_crop_roi.height), Size(block_size, block_size), Size(block_size/2, block_size/2), Size(block_size/2, block_size/2), 12, 1, -1, HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS, false);

        // FIXME: may not be needed. See default value.
        locations.push_back(Point(0, 0));

        hog.compute(cam_edge_crop, descriptors_cam, Size(), Size(), locations);
        hog.compute(cad_edge_crop, descriptors_cad, Size(), Size(), locations);

        auto it_cad = descriptors_cad.begin();
        auto it_cam = descriptors_cam.begin();
        float sum_diff = 0;
        for (; it_cad < descriptors_cad.end(); ++it_cad, ++it_cam) sum_diff += abs((*it_cad) - (*it_cam));

        innovation(0, 0) = sum_diff;
    }
    else
    {
        innovation(0, 0) = NAN;
    }
}


void VisualParticleFilterCorrection::likelihood(const Eigen::Ref<const Eigen::MatrixXf>& innovation, Eigen::Ref<Eigen::VectorXf> cor_state)
{
    // FIXME: Kernel likelihood need to be tuned!
    cor_state(0) *= ( exp( -0.001 * innovation(0, 0) /* / pow(1, 2.0) */ ) );
    if (cor_state(0) <= 0) cor_state(0) = std::numeric_limits<float>::min();
}
