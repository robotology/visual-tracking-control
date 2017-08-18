#include "VisualSIRParticleFilter.h"

#include <exception>
#include <iostream>
#include <utility>

#include <Eigen/Dense>
#include <iCub/ctrl/math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <yarp/math/Math.h>
#include <yarp/eigen/Eigen.h>

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::eigen;
using namespace yarp::math;
using namespace yarp::os;
using yarp::sig::Vector;
using yarp::sig::ImageOf;
using yarp::sig::PixelRgb;


VisualSIRParticleFilter::VisualSIRParticleFilter(std::unique_ptr<Initialization> initialization,
                                                 std::unique_ptr<ParticleFilterPrediction> prediction, std::unique_ptr<VisualCorrection> correction,
                                                 std::unique_ptr<Resampling> resampling,
                                                 const ConstString& cam_sel, const ConstString& laterality, const int num_particles) :
    initialization_(std::move(initialization)), prediction_(std::move(prediction)), correction_(std::move(correction)), resampling_(std::move(resampling)),
    cam_sel_(cam_sel), laterality_(laterality), num_particles_(num_particles)
{
    cuda_hog_ = cuda::HOG::create(Size(img_width_, img_height_), Size(block_size_, block_size_), Size(block_size_/2, block_size_/2), Size(block_size_/2, block_size_/2), bin_number_);
    cuda_hog_->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_COL_BY_COL);
    cuda_hog_->setGammaCorrection(true);
    cuda_hog_->setWinStride(Size(img_width_, img_height_));


    port_image_in_.open     ("/hand-tracking/" + cam_sel_ + "/img:i");
    port_estimates_out_.open("/hand-tracking/" + cam_sel_ + "/result/estimates:o");


    setCommandPort();
}


VisualSIRParticleFilter::~VisualSIRParticleFilter() noexcept
{
    port_image_in_.close();
    port_estimates_out_.close();
}


void VisualSIRParticleFilter::initialization()
{
    particle_ = MatrixXf(7, num_particles_);
    weight_   = VectorXf(num_particles_, 1);

    initialization_->initialize(particle_, weight_);

    prediction_->setStateModelProperty("ICFW_PLAY_INIT");
}


void VisualSIRParticleFilter::filteringStep()
{
    std::vector<float> descriptors_cam_left (descriptor_length_);
    cuda::GpuMat       cuda_img             (Size(img_width_, img_height_), CV_8UC3);
    cuda::GpuMat       cuda_img_alpha       (Size(img_width_, img_height_), CV_8UC4);
    cuda::GpuMat       descriptors_cam_cuda (Size(descriptor_length_, 1),   CV_32F );

    ImageOf<PixelRgb>* img_in = port_image_in_.read(true);
    if (img_in != YARP_NULLPTR)
    {
        MatrixXf temp_particle(7, num_particles_);
        VectorXf temp_weight(num_particles_, 1);
        VectorXf temp_parent(num_particles_, 1);

        /* PROCESS CURRENT MEASUREMENT */
        Mat measurement;

        measurement = cvarrToMat(img_in->getIplImage());
        cuda_img.upload(measurement);
        cuda::cvtColor(cuda_img, cuda_img_alpha, COLOR_BGR2BGRA, 4);
        cuda_hog_->compute(cuda_img_alpha, descriptors_cam_cuda);
        descriptors_cam_cuda.download(descriptors_cam_left);

        /* PREDICTION */
        VectorXf sorted_pred = weight_;
        std::sort(sorted_pred.data(), sorted_pred.data() + sorted_pred.size());
        float threshold = sorted_pred.tail(6)(0);

        prediction_->setStateModelProperty("ICFW_DELTA");
        for (int j = 0; j < num_particles_; ++j)
        {
            if (weight_(j) <= threshold)
                prediction_->predict(particle_.col(j), particle_.col(j));
            else
                prediction_->motion(particle_.col(j), particle_.col(j));
        }

        /* CORRECTION */
        correction_->setObservationModelProperty("VP_PARAMS");
        correction_->correct(particle_, descriptors_cam_left, weight_);
        weight_ /= weight_.sum();


        /* STATE ESTIMATE EXTRACTION FROM PARTICLE SET */
        VectorXf out_particle(7);
        switch (ext_mode)
        {
            case EstimatesExtraction::mean :
                out_particle = mean(particle_, weight_);
                break;

            case EstimatesExtraction::mode :
                out_particle = mode(particle_, weight_);
                break;

            case EstimatesExtraction::sm_average :
                out_particle = smAverage(particle_, weight_);
                break;

            case EstimatesExtraction::wm_average :
                out_particle = wmAverage(particle_, weight_);
                break;

            case EstimatesExtraction::em_average :
                out_particle = emAverage(particle_, weight_);
                break;

            case EstimatesExtraction::am_average :
                out_particle = amAverage(particle_, weight_);
                break;

            default:
                out_particle.fill(0.0);
                break;
        }


        /* RESAMPLING */
        std::cout << "Step: " << getFilteringStep() << std::endl;
        std::cout << "Neff: " << resampling_->neff(weight_) << std::endl;
        if (resampling_->neff(weight_) < std::round(num_particles_ / 5.f))
        {
            std::cout << "Resampling!" << std::endl;

            resampling_->resample(particle_, weight_,
                                  temp_particle, temp_weight,
                                  temp_parent);

            particle_ = temp_particle;
            weight_   = temp_weight;
        }

        /* STATE ESTIMATE OUTPUT */
        /* INDEX FINGERTIP */
//        Vector q = readRootToEE();
//        icub_kin_arm_.setAng(q.subVector(0, 9) * (M_PI/180.0));
//        Vector chainjoints;
//        if (analogs_) icub_kin_finger_[1].getChainJoints(q.subVector(3, 18), analogs, chainjoints, right_hand_analogs_bounds_);
//        else          icub_kin_finger_[1].getChainJoints(q.subVector(3, 18), chainjoints);
//        icub_kin_finger_[1].setAng(chainjoints * (M_PI/180.0));
//
//        Vector l_ee_t(3);
//        toEigen(l_ee_t) = out_particle.col(0).head(3).cast<double>();
//        l_ee_t.push_back(1.0);
//
//        Vector l_ee_o(3);
//        toEigen(l_ee_o) = out_particle.col(0).tail(3).normalized().cast<double>();
//        l_ee_o.push_back(static_cast<double>(out_particle.col(0).tail(3).norm()));
//
//        yarp::sig::Matrix l_Ha = axis2dcm(l_ee_o);
//        l_Ha.setCol(3, l_ee_t);
//        Vector l_i_x = (l_Ha * (icub_kin_finger_[1].getH(3, true).getCol(3))).subVector(0, 2);
//        Vector l_i_o = dcm2axis(l_Ha * icub_kin_finger_[1].getH(3, true));
//        l_i_o.setSubvector(0, l_i_o.subVector(0, 2) * l_i_o[3]);
//
//
//        Vector r_ee_t(3);
//        toEigen(r_ee_t) = out_particle.col(1).head(3).cast<double>();
//        r_ee_t.push_back(1.0);
//
//        Vector r_ee_o(3);
//        toEigen(r_ee_o) = out_particle.col(1).tail(3).normalized().cast<double>();
//        r_ee_o.push_back(static_cast<double>(out_particle.col(1).tail(3).norm()));
//
//        yarp::sig::Matrix r_Ha = axis2dcm(r_ee_o);
//        r_Ha.setCol(3, r_ee_t);
//        Vector r_i_x = (r_Ha * (icub_kin_finger_[1].getH(3, true).getCol(3))).subVector(0, 2);
//        Vector r_i_o = dcm2axis(r_Ha * icub_kin_finger_[1].getH(3, true));
//        r_i_o.setSubvector(0, r_i_o.subVector(0, 2) * r_i_o[3]);
//
//
//        Vector& estimates_out = port_estimates_out_.prepare();
//        estimates_out.resize(12);
//        estimates_out.setSubvector(0, l_i_x);
//        estimates_out.setSubvector(3, l_i_o.subVector(0, 2));
//        estimates_out.setSubvector(6, r_i_x);
//        estimates_out.setSubvector(9, r_i_o.subVector(0, 2));
//        port_estimates_out_.write();

        /* PALM */
        Vector& estimates_out = port_estimates_out_.prepare();
        estimates_out.resize(7);
        toEigen(estimates_out) = out_particle.cast<double>();
        port_estimates_out_.write();
        /* ********** */
    }
}


void VisualSIRParticleFilter::getResult() { }


bool VisualSIRParticleFilter::attach(yarp::os::Port &source)
{
    return this->yarp().attachAsServer(source);
}


bool VisualSIRParticleFilter::setCommandPort()
{
    std::cout << "Opening RPC command port." << std::endl;
    if (!port_rpc_command_.open("/hand-tracking/" + cam_sel_ + "/cmd:i"))
    {
        std::cerr << "Cannot open the RPC command port." << std::endl;
        return false;
    }
    if (!attach(port_rpc_command_))
    {
        std::cerr << "Cannot attach the RPC command port." << std::endl;
        return false;
    }
    std::cout << "RPC command port opened and attached. Ready to recieve commands!" << std::endl;

    return true;
}


bool VisualSIRParticleFilter::run_filter()
{
    run();

    return true;
}


bool VisualSIRParticleFilter::reset_filter()
{
    reset();

    return false;
}


bool VisualSIRParticleFilter::stop_filter()
{
    reboot();

    return false;
}


bool VisualSIRParticleFilter::use_analogs(const bool status)
{
    if (status)
        return correction_->setObservationModelProperty("VP_ANALOGS_ON");
    else
        return correction_->setObservationModelProperty("VP_ANALOGS_OFF");
}


std::vector<std::string> VisualSIRParticleFilter::get_info()
{
    std::vector<std::string> info;

    info.push_back("<| Information about Visual SIR Particle Filter |>");
    info.push_back("<| The Particle Filter is " + std::string(isRunning() ? "not " : "") + "running |>");
    info.push_back("<| Filtering step: " + std::to_string(getFilteringStep()) + " |>");
    info.push_back("<| Using " + cam_sel_ + " camera images |>");
    info.push_back("<| Using encoders from " + laterality_ + " iCub arm |>");
    info.push_back("<| Using " + std::to_string(num_particles_) + " particles |>");
    info.push_back("<| Available estimate extraction methods:" +
                   std::string(ext_mode == EstimatesExtraction::mean       ? "1) mean <-- In use; "       : "1) mean; ") +
                   std::string(ext_mode == EstimatesExtraction::mode       ? "2) mode <-- In use; "       : "2) mode") +
                   std::string(ext_mode == EstimatesExtraction::sm_average ? "3) sm_average <-- In use; " : "3) sm_average") +
                   std::string(ext_mode == EstimatesExtraction::wm_average ? "4) wm_average <-- In use; " : "4) wm_average") +
                   std::string(ext_mode == EstimatesExtraction::em_average ? "5) em_average <-- In use; " : "5) em_average") +
                   std::string(ext_mode == EstimatesExtraction::am_average ? "6) am_average <-- In use; " : "6) am_average") + " |>");

    return info;
}


bool VisualSIRParticleFilter::set_estimates_extraction_method(const std::string& method)
{
    if (method == "mean")
    {
        ext_mode = EstimatesExtraction::mean;

        return true;
    }
    else if (method == "mode")
    {
        ext_mode = EstimatesExtraction::mode;

        return true;
    }
    else if (method == "sm_average")
    {
        ext_mode = EstimatesExtraction::sm_average;

        return true;
    }
    else if (method == "wm_average")
    {
        ext_mode = EstimatesExtraction::wm_average;

        return true;
    }
    else if (method == "em_average")
    {
        ext_mode = EstimatesExtraction::em_average;

        return true;
    }
    else if (method == "am_average")
    {
        init_filter = true;

        ext_mode = EstimatesExtraction::am_average;

        return true;
    }

    return false;
}


bool VisualSIRParticleFilter::set_mobile_average_window(const int16_t window)
{
    if (window > 0)
        return hist_buffer_.setHistorySize(window);
    else
        return false;
}


bool VisualSIRParticleFilter::quit()
{
    teardown();

    return true;
}


VectorXf VisualSIRParticleFilter::mean(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights) const
{
    VectorXf out_particle = VectorXf::Zero(7);
    float    s_ang        = 0;
    float    c_ang        = 0;

    for (int i = 0; i < particles.cols(); ++i)
    {
        out_particle.head<3>()        += weights(i) * particles.col(i).head<3>();
        out_particle.middleRows<3>(3) += weights(i) * particles.col(i).middleRows<3>(3);

        s_ang += weights(i) * std::sin(particles(6, i));
        c_ang += weights(i) * std::cos(particles(6, i));
    }

    out_particle(6) = std::atan2(s_ang, c_ang) + M_PI;

    return out_particle;
}


VectorXf VisualSIRParticleFilter::mode(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights) const
{
    MatrixXf::Index maxRow;
    MatrixXf::Index maxCol;
    weights.maxCoeff(&maxRow, &maxCol);
    return particles.col(maxRow);
}


VectorXf VisualSIRParticleFilter::smAverage(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights)
{
    VectorXf cur_estimates = mean(particles, weights);

    hist_buffer_.addElement(cur_estimates);

    MatrixXf history = hist_buffer_.getHistoryBuffer();
    if (sm_weights_.size() != history.cols())
        sm_weights_ = VectorXf::Ones(history.cols()) / history.cols();

    return mean(history, sm_weights_);
}


VectorXf VisualSIRParticleFilter::wmAverage(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights)
{
    VectorXf cur_estimates = mean(particles, weights);

    hist_buffer_.addElement(cur_estimates);

    MatrixXf history = hist_buffer_.getHistoryBuffer();
    if (wm_weights_.size() != history.cols())
    {
        wm_weights_.resize(history.cols());
        for (unsigned int i = 0; i < history.cols(); ++i)
            wm_weights_(i) = history.cols() - i;

        wm_weights_ /= wm_weights_.sum();
    }

    return mean(history, wm_weights_);
}


VectorXf VisualSIRParticleFilter::emAverage(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights)
{
    VectorXf cur_estimates = mean(particles, weights);

    hist_buffer_.addElement(cur_estimates);

    MatrixXf history = hist_buffer_.getHistoryBuffer();
    if (em_weights_.size() != history.cols())
    {
        em_weights_.resize(history.cols());
        for (unsigned int i = 0; i < history.cols(); ++i)
            em_weights_(i) = std::exp(-(static_cast<double>(i) / history.cols()));

        em_weights_ /= em_weights_.sum();
    }

    return mean(history, em_weights_);
}


VectorXf VisualSIRParticleFilter::amAverage(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights)
{
    VectorXf cur_estimates = mean(particles, weights);

    if (!init_filter)
    {
        time_2_ =  std::chrono::steady_clock::now();
        t_      += std::chrono::duration_cast<std::chrono::milliseconds>(time_2_ - time_1_);
        time_1_ =  time_2_;
    }
    else
    {
        t_      = std::chrono::milliseconds(0);
        time_1_ = std::chrono::steady_clock::now();
        lin_est_x_.reset();
        lin_est_o_.reset();
        lin_est_theta_.reset();

        init_filter = false;
    }

    Vector x(3);
    toEigen(x) = cur_estimates.head<3>().cast<double>();
    AWPolyElement element_x;
    element_x.data = x;
    element_x.time = t_.count();

    Vector o(3);
    toEigen(o) = cur_estimates.middleRows<3>(3).cast<double>();
    AWPolyElement element_o;
    element_o.data = o;
    element_o.time = t_.count();

    Vector tetha(1);
    tetha(0) = cur_estimates(6);
    AWPolyElement element_theta;
    element_theta.data = tetha;
    element_theta.time = t_.count();

    Vector est_out(7);
    est_out.setSubvector(0, lin_est_x_.estimate(element_x));
    est_out.setSubvector(3, lin_est_o_.estimate(element_o));
    est_out(6) = lin_est_theta_.estimate(element_theta)(0);

    return toEigen(est_out).cast<float>();
}


void HistoryBuffer::addElement(const Ref<const VectorXf>& element)
{
    hist_buffer_.push_front(element);

    if (hist_buffer_.size() > window_)
        hist_buffer_.pop_back();
}


MatrixXf HistoryBuffer::getHistoryBuffer()
{
    MatrixXf hist_out(7, hist_buffer_.size());

    unsigned int i = 0;
    for (const Ref<const VectorXf>& element : hist_buffer_)
        hist_out.col(i++) = element;

    return hist_out;
}


bool HistoryBuffer::setHistorySize(const unsigned int window)
{
    unsigned int tmp;
    if      (window == window_)     return true;
    else if (window < 2)            tmp = 2;
    else if (window >= max_window_) tmp = max_window_;
    else                            tmp = window;

    if (tmp < window_ && tmp < hist_buffer_.size())
    {
        for (unsigned int i = 0; i < (window_ - tmp); ++i)
            hist_buffer_.pop_back();
    }

    window_ = tmp;

    return true;
}
