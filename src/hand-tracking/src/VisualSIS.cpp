#include <VisualSIS.h>

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
#include <yarp/eigen/Eigen.h>
#include <yarp/math/Math.h>
#include <yarp/os/Time.h>

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


VisualSIS::VisualSIS(const ConstString& cam_sel, const int num_particles) :
    cam_sel_(cam_sel),
    num_particles_(num_particles)
{
    cuda_hog_ = cuda::HOG::create(Size(img_width_, img_height_), Size(block_size_, block_size_), Size(block_size_/2, block_size_/2), Size(block_size_/2, block_size_/2), bin_number_);
    cuda_hog_->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_COL_BY_COL);
    cuda_hog_->setGammaCorrection(true);
    cuda_hog_->setWinStride(Size(img_width_, img_height_));


    port_image_in_.open     ("/hand-tracking/" + cam_sel_ + "/img:i");
    port_estimates_out_.open("/hand-tracking/" + cam_sel_ + "/result/estimates:o");

    img_in_.resize(320, 240);
    img_in_.zero();

    setCommandPort();
}


bool VisualSIS::skip(const std::string& what_step, const bool status)
{
    if (what_step == "prediction" ||
        what_step == "state"      ||
        what_step == "exogenous"    )
        return prediction_->skip(what_step, status);

    if (what_step == "correction")
        return correction_->skip(status);

    return false;
}


void VisualSIS::initialization()
{
    pred_particle_ = MatrixXf(7, num_particles_);
    pred_weight_   = VectorXf(num_particles_, 1);

    cor_particle_ = MatrixXf(7, num_particles_);
    cor_weight_   = VectorXf(num_particles_, 1);

    hist_buffer_.initializeHistory();

    initialization_->initialize(pred_particle_, pred_weight_);

    prediction_->getExogenousModel().setProperty("ICFW_INIT");
}


VisualSIS::~VisualSIS() noexcept
{
    port_image_in_.close();
    port_estimates_out_.close();
}


void VisualSIS::filteringStep()
{
    std::vector<float> descriptors_cam_left (descriptor_length_);
    cuda::GpuMat       cuda_img             (Size(img_width_, img_height_), CV_8UC3);
    cuda::GpuMat       cuda_img_alpha       (Size(img_width_, img_height_), CV_8UC4);
    cuda::GpuMat       descriptors_cam_cuda (Size(descriptor_length_, 1),   CV_32F );


    ImageOf<PixelRgb>* tmp_imgin = YARP_NULLPTR;
    tmp_imgin = port_image_in_.read(false);
    if (tmp_imgin != YARP_NULLPTR)
    {
        init_img_in_ = true;
        img_in_ = *tmp_imgin;
    }

    if (init_img_in_)
    {
        /* PROCESS CURRENT MEASUREMENT */
        Mat measurement;

        measurement = cvarrToMat(img_in_.getIplImage());
        cuda_img.upload(measurement);
        cuda::cvtColor(cuda_img, cuda_img_alpha, COLOR_BGR2BGRA, 4);
        cuda_hog_->compute(cuda_img_alpha, descriptors_cam_cuda);
        descriptors_cam_cuda.download(descriptors_cam_left);

        /* PREDICTION */
        if (getFilteringStep() != 0)
            prediction_->predict(cor_particle_, cor_weight_,
                                 pred_particle_, pred_weight_);

        /* CORRECTION */
        correction_->getVisualObservationModel().setProperty("VP_PARAMS");
        correction_->correct(pred_particle_, pred_weight_, descriptors_cam_left,
                             cor_particle_, cor_weight_);
        cor_weight_ /= cor_weight_.sum();


        /* STATE ESTIMATE EXTRACTION FROM PARTICLE SET */
        VectorXf out_particle(7);
        switch (ext_mode)
        {
            case EstimatesExtraction::mean :
                out_particle = mean(cor_particle_, cor_weight_);
                break;

            case EstimatesExtraction::mode :
                out_particle = mode(cor_particle_, cor_weight_);
                break;

            case EstimatesExtraction::sm_average :
                out_particle = smAverage(cor_particle_, cor_weight_);
                break;

            case EstimatesExtraction::wm_average :
                out_particle = wmAverage(cor_particle_, cor_weight_);
                break;

            case EstimatesExtraction::em_average :
                out_particle = emAverage(cor_particle_, cor_weight_);
                break;

            default:
                out_particle.fill(0.0);
                break;
        }


        /* RESAMPLING */
        std::cout << "Step: " << getFilteringStep() << std::endl;
        std::cout << "Neff: " << resampling_->neff(cor_weight_) << std::endl;
        if (resampling_->neff(cor_weight_) < std::round(num_particles_ / 5.f))
        {
            std::cout << "Resampling!" << std::endl;

            MatrixXf res_particle(7, num_particles_);
            VectorXf res_weight(num_particles_, 1);
            VectorXf res_parent(num_particles_, 1);

            resampling_->resample(cor_particle_, cor_weight_,
                                  res_particle, res_weight,
                                  res_parent);

            cor_particle_ = res_particle;
            cor_weight_   = res_weight;
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


void VisualSIS::getResult() { }


bool VisualSIS::attach(yarp::os::Port &source)
{
    return this->yarp().attachAsServer(source);
}


bool VisualSIS::setCommandPort()
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


bool VisualSIS::run_filter()
{
    run();

    return true;
}


bool VisualSIS::reset_filter()
{
    reset();

    return true;
}


bool VisualSIS::stop_filter()
{
    reboot();

    return true;
}


bool VisualSIS::skip_step(const std::string& what_step, const bool status)
{
    return skip(what_step, status);
}


bool VisualSIS::use_analogs(const bool status)
{
    if (status)
        return correction_->getVisualObservationModel().setProperty("VP_ANALOGS_ON");
    else
        return correction_->getVisualObservationModel().setProperty("VP_ANALOGS_OFF");
}


std::vector<std::string> VisualSIS::get_info()
{
    std::vector<std::string> info;

    info.push_back("<| Information about Visual SIR Particle Filter |>");
    info.push_back("<| The Particle Filter is " + std::string(isRunning() ? "not " : "") + "running |>");
    info.push_back("<| Filtering step: " + std::to_string(getFilteringStep()) + " |>");
    info.push_back("<| Using " + cam_sel_ + " camera images |>");
    info.push_back("<| Using " + std::to_string(num_particles_) + " particles |>");
    info.push_back("<| Adaptive window: " +
                   std::string(hist_buffer_.getAdaptiveWindowStatus() ? "enabled" : "disabled") + " |>");
    info.push_back("<| Current window size: " + std::to_string(hist_buffer_.getHistorySize()) + " |>");
    info.push_back("<| Available estimate extraction methods:" +
                   std::string(ext_mode == EstimatesExtraction::mean       ? "1) mean <-- In use; "       : "1) mean; ") +
                   std::string(ext_mode == EstimatesExtraction::mode       ? "2) mode <-- In use; "       : "2) mode; ") +
                   std::string(ext_mode == EstimatesExtraction::sm_average ? "3) sm_average <-- In use; " : "3) sm_average; ") +
                   std::string(ext_mode == EstimatesExtraction::wm_average ? "4) wm_average <-- In use; " : "4) wm_average; ") +
                   std::string(ext_mode == EstimatesExtraction::em_average ? "5) em_average <-- In use; " : "5) em_average") + " |>");

    return info;
}


bool VisualSIS::set_estimates_extraction_method(const std::string& method)
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

    return false;
}


bool VisualSIS::set_mobile_average_window(const int16_t window)
{
    if (window > 0)
        return hist_buffer_.setHistorySize(window);
    else
        return false;
}


bool VisualSIS::enable_adaptive_window(const bool status)
{
    return hist_buffer_.enableAdaptiveWindow(status);
}


bool VisualSIS::quit()
{
    return teardown();
}


VectorXf VisualSIS::mean(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights) const
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

    float versor_norm = out_particle.middleRows<3>(3).norm();
    if ( versor_norm >= 0.99)
        out_particle.middleRows<3>(3) /= versor_norm;
    else
        out_particle.middleRows<3>(3) = mode(particles, weights).middleRows<3>(3);

    out_particle(6) = std::atan2(s_ang, c_ang);

    return out_particle;
}


VectorXf VisualSIS::mode(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights) const
{
    MatrixXf::Index maxRow;
    MatrixXf::Index maxCol;
    weights.maxCoeff(&maxRow, &maxCol);

    return particles.col(maxRow);
}


VectorXf VisualSIS::smAverage(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights)
{
    VectorXf cur_estimates = mean(particles, weights);

    hist_buffer_.addElement(cur_estimates);

    MatrixXf history = hist_buffer_.getHistoryBuffer();
    if (sm_weights_.size() != history.cols())
        sm_weights_ = VectorXf::Ones(history.cols()) / history.cols();

    return mean(history, sm_weights_);
}


VectorXf VisualSIS::wmAverage(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights)
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


VectorXf VisualSIS::emAverage(const Ref<const MatrixXf>& particles, const Ref<const VectorXf>& weights)
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


HistoryBuffer::HistoryBuffer() noexcept
{
    lin_est_x_.reset();
    lin_est_o_.reset();
    lin_est_theta_.reset();
}


void HistoryBuffer::addElement(const Ref<const VectorXf>& element)
{
    if (adaptive_window_)
        adaptWindow(element);

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


bool HistoryBuffer::decreaseHistorySize()
{
    return setHistorySize(window_ - 1);
}


bool HistoryBuffer::increaseHistorySize()
{
    return setHistorySize(window_ + 1);
}


bool HistoryBuffer::initializeHistory()
{
    hist_buffer_.clear();
    return true;
}


bool HistoryBuffer::enableAdaptiveWindow(const bool status)
{
    if (status)
        mem_window_ = window_;
    else
    {
        window_ = mem_window_;
        setHistorySize(window_);
    }

    adaptive_window_ = status;

    return true;
}


void HistoryBuffer::adaptWindow(const Ref<const VectorXf>& element)
{
    double time_now = Time::now();

    Vector x(3);
    toEigen(x) = element.head<3>().cast<double>();
    AWPolyElement element_x(x, time_now);

    Vector dot_x = lin_est_x_.estimate(element_x);


    Vector o(3);
    toEigen(o) = element.middleRows<3>(3).cast<double>();
    AWPolyElement element_o(o, time_now);

    Vector dot_o = lin_est_o_.estimate(element_o);


    Vector theta(1);
    theta(0) = element(6);
    AWPolyElement element_theta(theta, time_now);

    Vector dot_theta = lin_est_theta_.estimate(element_theta);


    if (norm(dot_x) <= lin_est_x_thr_ || norm(dot_o) <= lin_est_o_thr_ || std::abs(dot_theta(0)) <= lin_est_theta_thr_)
        increaseHistorySize();
    else if (norm(dot_x) > lin_est_x_thr_ || norm(dot_o) > lin_est_o_thr_ || std::abs(dot_theta(0)) > lin_est_theta_thr_)
        decreaseHistorySize();
}
