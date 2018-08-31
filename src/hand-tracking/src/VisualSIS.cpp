#include <VisualSIS.h>

#include <exception>
#include <utility>

#include <Eigen/Dense>
//#include <iCub/ctrl/math.h>
#include <yarp/eigen/Eigen.h>
#include <yarp/math/Math.h>
#include <yarp/os/LogStream.h>

//#include <SuperimposeMesh/SICAD.h>
//#include <VisualProprioception.h>

using namespace bfl;
using namespace Eigen;
//using namespace iCub::ctrl;
//using namespace iCub::iKin;
using namespace yarp::eigen;
using namespace yarp::math;
using namespace yarp::os;
using yarp::sig::Vector;
//using yarp::sig::ImageOf;
//using yarp::sig::PixelRgb;


VisualSIS::VisualSIS(const std::string& cam_sel,
                     const int num_particles,
                     const double resample_ratio,
                     const std::string& port_prefix) :
    num_particles_(num_particles),
    resample_ratio_(resample_ratio),
    port_prefix_(port_prefix)
{
    port_estimates_out_.open("/" + port_prefix_ + "/estimates:o");

    //port_image_out_.open("/" + port_prefix_ + "/img:o");

    setCommandPort();
}


bool VisualSIS::initialization()
{
    pred_particle_ = MatrixXf(7, num_particles_);
    pred_weight_   = VectorXf(num_particles_, 1);

    cor_particle_ = MatrixXf(7, num_particles_);
    cor_weight_   = VectorXf(num_particles_, 1);

    estimate_extraction_.clear();

    initialization_->initialize(pred_particle_, pred_weight_);

    prediction_->getExogenousModel().setProperty("init");

    skip("all", false);

    return true;
}


VisualSIS::~VisualSIS() noexcept
{
    port_estimates_out_.close();
}


void VisualSIS::filteringStep()
{
    /* PREDICTION */
    if (getFilteringStep() != 0)
        prediction_->predict(cor_particle_, cor_weight_,
                             pred_particle_, pred_weight_);


    /* CORRECTION */
    correction_->correct(pred_particle_, pred_weight_,
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
//    Vector q = readRootToEE();
//    icub_kin_arm_.setAng(q.subVector(0, 9) * (M_PI/180.0));
//    Vector chainjoints;
//    if (analogs_) icub_kin_finger_[1].getChainJoints(q.subVector(3, 18), analogs, chainjoints, right_hand_analogs_bounds_);
//    else          icub_kin_finger_[1].getChainJoints(q.subVector(3, 18), chainjoints);
//    icub_kin_finger_[1].setAng(chainjoints * (M_PI/180.0));
//
//    Vector l_ee_t(3);
//    toEigen(l_ee_t) = out_particle.col(0).head(3).cast<double>();
//    l_ee_t.push_back(1.0);
//
//    Vector l_ee_o(3);
//    toEigen(l_ee_o) = out_particle.col(0).tail(3).normalized().cast<double>();
//    l_ee_o.push_back(static_cast<double>(out_particle.col(0).tail(3).norm()));
//
//    yarp::sig::Matrix l_Ha = axis2dcm(l_ee_o);
//    l_Ha.setCol(3, l_ee_t);
//    Vector l_i_x = (l_Ha * (icub_kin_finger_[1].getH(3, true).getCol(3))).subVector(0, 2);
//    Vector l_i_o = dcm2axis(l_Ha * icub_kin_finger_[1].getH(3, true));
//    l_i_o.setSubvector(0, l_i_o.subVector(0, 2) * l_i_o[3]);
//
//
//    Vector r_ee_t(3);
//    toEigen(r_ee_t) = out_particle.col(1).head(3).cast<double>();
//    r_ee_t.push_back(1.0);
//
//    Vector r_ee_o(3);
//    toEigen(r_ee_o) = out_particle.col(1).tail(3).normalized().cast<double>();
//    r_ee_o.push_back(static_cast<double>(out_particle.col(1).tail(3).norm()));
//
//    yarp::sig::Matrix r_Ha = axis2dcm(r_ee_o);
//    r_Ha.setCol(3, r_ee_t);
//    Vector r_i_x = (r_Ha * (icub_kin_finger_[1].getH(3, true).getCol(3))).subVector(0, 2);
//    Vector r_i_o = dcm2axis(r_Ha * icub_kin_finger_[1].getH(3, true));
//    r_i_o.setSubvector(0, r_i_o.subVector(0, 2) * r_i_o[3]);
//
//
//    Vector& estimates_out = port_estimates_out_.prepare();
//    estimates_out.resize(12);
//    estimates_out.setSubvector(0, l_i_x);
//    estimates_out.setSubvector(3, l_i_o.subVector(0, 2));
//    estimates_out.setSubvector(6, r_i_x);
//    estimates_out.setSubvector(9, r_i_o.subVector(0, 2));
//    port_estimates_out_.write();


    /* PALM */
    Vector& estimates_out = port_estimates_out_.prepare();
    estimates_out.resize(7);
    toEigen(estimates_out) = out_particle.cast<double>();
    port_estimates_out_.write();


    /* STATE ESTIMATE OUTPUT */
//    Superimpose::ModelPoseContainer hand_pose;
//    Superimpose::ModelPose          pose;
//    ImageOf<PixelRgb>& img_out = port_image_out_.prepare();
//
//    pose.assign(out_particle.data(), out_particle.data() + 3);
//    pose.insert(pose.end(), out_particle.data() + 3, out_particle.data() + 7);
//    hand_pose.emplace("palm", pose);
//
//    dynamic_cast<VisualProprioception*>(&dynamic_cast<VisualUpdateParticles*>(correction_.get())->getVisualObservationModel())->superimpose(hand_pose, measurement);
//
//    img_out.setExternal(measurement.ptr(), img_width_, img_height_);
//
//    port_image_out_.write();
    /* ********** */
}


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


std::vector<std::string> VisualSIS::get_info()
{
    std::vector<std::string> info;

    info.push_back("<| Information about Visual SIR Particle Filter |>");
    info.push_back("<| The Particle Filter is " + std::string(isRunning() ? "not " : "") + "running |>");
    info.push_back("<| Filtering step: " + std::to_string(getFilteringStep()) + " |>");
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
