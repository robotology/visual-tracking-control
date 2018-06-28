#include <VisualSIS.h>
#include <VisualUpdateParticles.h>

#include <exception>
#include <utility>

#include <Eigen/Dense>
#include <iCub/ctrl/math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#if HANDTRACKING_USE_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif // HANDTRACKING_USE_OPENCV_CUDA
#include <yarp/eigen/Eigen.h>
#include <yarp/math/Math.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Time.h>

#include <SuperimposeMesh/SICAD.h>
#include <VisualProprioception.h>

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


VisualSIS::VisualSIS(const std::string& cam_sel,
                     const int img_width, const int img_height,
                     const int num_particles,
                     const double resample_ratio,
                     const std::string& port_prefix) :
    port_prefix_(port_prefix),
    cam_sel_(cam_sel),
    img_width_(img_width),
    img_height_(img_height),
    num_particles_(num_particles),
    resample_ratio_(resample_ratio)
{
#if HANDTRACKING_USE_OPENCV_CUDA
    /* Page locked allocator should be faster using CUDA. Apparently it seems not the case. */
    //Mat::setDefaultAllocator(cuda::HostMem::getAllocator(cuda::HostMem::PAGE_LOCKED));

    cuda::DeviceInfo gpu_dev;
    yInfo() << log_ID_ << "CUDA engine capability:" << gpu_engine_count_to_string(gpu_dev.asyncEngineCount());
    yInfo() << log_ID_ << "CUDA concurrent kernels:" << gpu_dev.concurrentKernels();
    yInfo() << log_ID_ << "CUDA streaming multiprocessor count:" << gpu_dev.multiProcessorCount();
    yInfo() << log_ID_ << "CUDA can map host memory:" << gpu_dev.canMapHostMemory();
    yInfo() << log_ID_ << "CUDA clock:" << gpu_dev.clockRate() << "KHz";


    hog_cuda_ = cuda::HOG::create(Size(img_width_, img_height_), Size(block_size_, block_size_), Size(block_size_/2, block_size_/2), Size(block_size_/2, block_size_/2), bin_number_);
    hog_cuda_->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_ROW_BY_ROW);
    hog_cuda_->setGammaCorrection(true);
    hog_cuda_->setWinStride(Size(img_width_, img_height_));
#else
    hog_cpu_ = std::unique_ptr<HOGDescriptor>(new HOGDescriptor(Size(img_width_, img_height_),
                                                                Size(block_size_, block_size_),
                                                                Size(block_size_/2, block_size_/2),
                                                                Size(block_size_/2, block_size_/2),
                                                                bin_number_,
                                                                1, -1.0, HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS, false));
#endif // HANDTRACKING_USE_OPENCV_CUDA

    descriptor_length_ = (img_width_ / block_size_ * 2 - 1) * (img_height_ / block_size_ * 2 - 1) * bin_number_ * 4;

    port_image_in_.open     ("/" + port_prefix_ + "/img:i");
    port_estimates_out_.open("/" + port_prefix_ + "/estimates:o");

    img_in_.resize(img_width_, img_height_);
    img_in_.zero();

    port_image_out_.open("/" + port_prefix_ + "/img:o");

    setCommandPort();
}


void VisualSIS::initialization()
{
    pred_particle_ = MatrixXf(7, num_particles_);
    pred_weight_   = VectorXf(num_particles_, 1);

    cor_particle_ = MatrixXf(7, num_particles_);
    cor_weight_   = VectorXf(num_particles_, 1);

    estimate_extraction_.clear();

    initialization_->initialize(pred_particle_, pred_weight_);

    prediction_->getExogenousModel().setProperty("init");

    skip("all", false);
}


VisualSIS::~VisualSIS() noexcept
{
    port_image_in_.close();
    port_estimates_out_.close();
}


void VisualSIS::filteringStep()
{
    std::vector<float> descriptors_cam_left(descriptor_length_);

#if HANDTRACKING_USE_OPENCV_CUDA
    cuda::GpuMat cuda_img            (Size(img_width_, img_height_), CV_8UC3);
    cuda::GpuMat cuda_img_alpha      (Size(img_width_, img_height_), CV_8UC4);
    cuda::GpuMat descriptors_cam_cuda(Size(descriptor_length_, 1),   CV_32F );
#endif // HANDTRACKING_USE_OPENCV_CUDA

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
        Mat measurement = cvarrToMat(img_in_.getIplImage());

#if HANDTRACKING_USE_OPENCV_CUDA
        cuda_img.upload(measurement);
        cuda::resize(cuda_img, cuda_img, Size(img_width_, img_height_));
        cuda::cvtColor(cuda_img, cuda_img_alpha, COLOR_BGR2BGRA, 4);
        hog_cuda_->compute(cuda_img_alpha, descriptors_cam_cuda);
        descriptors_cam_cuda.download(descriptors_cam_left);
#else
        cv::resize(measurement, measurement, Size(img_width_, img_height_));
        hog_cpu_->compute(measurement, descriptors_cam_left);
#endif // HANDTRACKING_USE_OPENCV_CUDA

        /* PREDICTION */
        if (getFilteringStep() != 0)
            prediction_->predict(cor_particle_, cor_weight_,
                                 pred_particle_, pred_weight_);

        /* CORRECTION */
        dynamic_cast<VisualUpdateParticles*>(correction_.get())->getVisualObservationModel().setProperty("VP_PARAMS");
        correction_->correct(pred_particle_, pred_weight_, descriptors_cam_left,
                             cor_particle_, cor_weight_);
        cor_weight_ /= cor_weight_.sum();


        /* STATE ESTIMATE EXTRACTION FROM PARTICLE SET */
        VectorXf out_particle = estimate_extraction_.extract(cor_particle_, cor_weight_);


        /* RESAMPLING */
        yInfo() << log_ID_ << "Step:" << getFilteringStep();
        yInfo() << log_ID_ << "Neff:" << resampling_->neff(cor_weight_);
        if (resampling_->neff(cor_weight_) < std::round(num_particles_ * resample_ratio_))
        {
            yInfo() << log_ID_ << "Resampling!";

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

        /* STATE ESTIMATE OUTPUT */
        Superimpose::ModelPoseContainer hand_pose;
        Superimpose::ModelPose          pose;
        ImageOf<PixelRgb>& img_out = port_image_out_.prepare();

        pose.assign(out_particle.data(), out_particle.data() + 3);
        pose.insert(pose.end(), out_particle.data() + 3, out_particle.data() + 7);
        hand_pose.emplace("palm", pose);

        dynamic_cast<VisualProprioception*>(&dynamic_cast<VisualUpdateParticles*>(correction_.get())->getVisualObservationModel())->superimpose(hand_pose, measurement);

        img_out.setExternal(measurement.ptr(), img_width_, img_height_);

        port_image_out_.write();
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
    yInfo() << log_ID_ << "Opening RPC command port.";
    if (!port_rpc_command_.open("/" + port_prefix_ + "/cmd:i"))
    {
        yError() << log_ID_ << "Cannot open the RPC command port.";
        return false;
    }
    if (!attach(port_rpc_command_))
    {
        yError() << log_ID_ << "Cannot attach the RPC command port.";
        return false;
    }
    yInfo() << log_ID_ << "RPC command port opened and attached. Ready to recieve commands!";

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
        return dynamic_cast<VisualUpdateParticles*>(correction_.get())->getVisualObservationModel().setProperty("VP_ANALOGS_ON");
    else
        return dynamic_cast<VisualUpdateParticles*>(correction_.get())->getVisualObservationModel().setProperty("VP_ANALOGS_OFF");
}


std::vector<std::string> VisualSIS::get_info()
{
    std::vector<std::string> info;

    info.push_back("<| Information about Visual SIR Particle Filter |>");
    info.push_back("<| The Particle Filter is " + std::string(isRunning() ? "not " : "") + "running |>");
    info.push_back("<| Filtering step: " + std::to_string(getFilteringStep()) + " |>");
    info.push_back("<| Using " + cam_sel_ + " camera images |>");
    info.push_back("<| Using " + std::to_string(num_particles_) + " particles |>");

    std::vector<std::string> est_ext_info = estimate_extraction_.getInfo();

    info.insert(info.end(), est_ext_info.begin(), est_ext_info.end());

    return info;
}


bool VisualSIS::set_estimates_extraction_method(const std::string& method)
{
    if (method == "mean")
    {
        estimate_extraction_.setMethod(EstimatesExtraction::ExtractionMethod::mean);

        return true;
    }
    else if (method == "smean")
    {
        estimate_extraction_.setMethod(EstimatesExtraction::ExtractionMethod::smean);

        return true;
    }
    else if (method == "wmean")
    {
        estimate_extraction_.setMethod(EstimatesExtraction::ExtractionMethod::wmean);

        return true;
    }
    else if (method == "emean")
    {
        estimate_extraction_.setMethod(EstimatesExtraction::ExtractionMethod::emean);

        return true;
    }
    else if (method == "mode")
    {
        estimate_extraction_.setMethod(EstimatesExtraction::ExtractionMethod::mode);

        return true;
    }
    else if (method == "smode")
    {
        estimate_extraction_.setMethod(EstimatesExtraction::ExtractionMethod::smode);

        return true;
    }
    else if (method == "wmode")
    {
        estimate_extraction_.setMethod(EstimatesExtraction::ExtractionMethod::wmode);

        return true;
    }
    else if (method == "emode")
    {
        estimate_extraction_.setMethod(EstimatesExtraction::ExtractionMethod::emode);

        return true;
    }

    return false;
}


bool VisualSIS::set_mobile_average_window(const int16_t window)
{
    if (window > 0)
        return estimate_extraction_.setMobileAverageWindowSize(window);
    else
        return false;
}


bool VisualSIS::quit()
{
    return teardown();
}


std::string VisualSIS::gpu_engine_count_to_string(const int engine_count) const
{
    if (engine_count == 0) return "concurrency is unsupported on this device";
    if (engine_count == 1) return "the device can concurrently copy memory between host and device while executing a kernel";
    if (engine_count == 2) return "the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time";
    return "wrong argument...!";
}