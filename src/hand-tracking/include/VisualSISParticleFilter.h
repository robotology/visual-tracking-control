#ifndef VISUALSISPARTICLEFILTER_H
#define VISUALSISPARTICLEFILTER_H

#include <chrono>
#include <deque>
#include <memory>
#include <vector>

#include <BayesFilters/FilteringAlgorithm.h>
#include <BayesFilters/Initialization.h>
#include <BayesFilters/StateModel.h>
#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/VisualObservationModel.h>
#include <BayesFilters/PFVisualCorrection.h>
#include <BayesFilters/Resampling.h>
#include <Eigen/Dense>
#include <iCub/ctrl/adaptWinPolyEstimator.h>
#include <iCub/ctrl/filters.h>
#include <iCub/iKin/iKinFwd.h>
#include <opencv2/cudaobjdetect.hpp>
#include <thrift/VisualSISParticleFilterIDL.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IAnalogSensor.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/Port.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>


class HistoryBuffer
{
public:
    HistoryBuffer() noexcept;

    ~HistoryBuffer() noexcept { };

    void            addElement(const Eigen::Ref<const Eigen::VectorXf>& element);

    Eigen::MatrixXf getHistoryBuffer();

    bool            setHistorySize(const unsigned int window);

    unsigned int    getHistorySize() { return window_; };

    bool            decreaseHistorySize();

    bool            increaseHistorySize();

    bool            enableAdaptiveWindow(const bool status);

    bool            getAdaptiveWindowStatus() { return adaptive_window_; };

private:
    unsigned int                window_     = 5;

    const unsigned int          max_window_ = 30;

    unsigned int                mem_window_ = 5;

    std::deque<Eigen::VectorXf> hist_buffer_;

    bool                        adaptive_window_ = false;

    double                      lin_est_x_thr_     = 0.03;
    double                      lin_est_o_thr_     = 0.5;
    double                      lin_est_theta_thr_ = 3.0 * iCub::ctrl::CTRL_DEG2RAD;

    iCub::ctrl::AWLinEstimator  lin_est_x_         {4, 0.1};
    iCub::ctrl::AWLinEstimator  lin_est_o_         {4, 0.6};
    iCub::ctrl::AWLinEstimator  lin_est_theta_     {4, 6.0 * iCub::ctrl::CTRL_DEG2RAD};

    void                        adaptWindow(const Eigen::Ref<const Eigen::VectorXf>& element);
};


class VisualSISParticleFilter: public bfl::FilteringAlgorithm,
                               public VisualSISParticleFilterIDL
{
public:
    VisualSISParticleFilter(std::unique_ptr<bfl::Initialization> initialization,
                            std::unique_ptr<bfl::PFPrediction> prediction, std::unique_ptr<bfl::PFVisualCorrection> correction,
                            std::unique_ptr<bfl::Resampling> resampling,
                            const yarp::os::ConstString& cam_sel, const yarp::os::ConstString& laterality, const int num_particles);

    VisualSISParticleFilter(const VisualSISParticleFilter& vsir_pf) = delete;

    VisualSISParticleFilter(VisualSISParticleFilter&& vsir_pf) noexcept = delete;

    ~VisualSISParticleFilter() noexcept;

    VisualSISParticleFilter& operator=(const VisualSISParticleFilter& vsir_pf) = delete;

    VisualSISParticleFilter& operator=(VisualSISParticleFilter&& vsir_pf) noexcept = delete;

    void initialization() override;

    void filteringStep() override;

    void getResult() override;

    bool runCondition() override { return true; };

protected:
    std::unique_ptr<bfl::Initialization>     initialization_;
    std::unique_ptr<bfl::PFPrediction>       prediction_;
    std::unique_ptr<bfl::PFVisualCorrection> correction_;
    std::unique_ptr<bfl::Resampling>         resampling_;

    yarp::os::ConstString cam_sel_;
    yarp::os::ConstString laterality_;
    const int             num_particles_;

    cv::Ptr<cv::cuda::HOG> cuda_hog_;


    yarp::os::BufferedPort<yarp::sig::Vector>                       port_estimates_out_;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_in_;


    yarp::os::Port port_rpc_command_;
    bool           attach(yarp::os::Port &source);
    bool           setCommandPort();


    bool                     run_filter() override;

    bool                     reset_filter() override;

    bool                     stop_filter() override;

    bool                     use_analogs(const bool status) override;

    std::vector<std::string> get_info() override;

    bool                     quit() override;


    /* ESTIMATE EXTRACTION METHODS */
    bool set_estimates_extraction_method(const std::string& method) override;

    bool set_mobile_average_window(const int16_t window) override;

    bool enable_adaptive_window(const bool status) override;

    HistoryBuffer hist_buffer_;

    enum class EstimatesExtraction
    {
        mean,
        mode,
        sm_average,
        wm_average,
        em_average
    };

    EstimatesExtraction ext_mode = EstimatesExtraction::em_average;

    Eigen::VectorXf mean(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights) const;

    Eigen::VectorXf mode(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights) const;

    Eigen::VectorXf smAverage(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights);
    Eigen::VectorXf sm_weights_;

    Eigen::VectorXf wmAverage(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights);
    Eigen::VectorXf wm_weights_;

    Eigen::VectorXf emAverage(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights);
    Eigen::VectorXf em_weights_;
    /* *************************** */

private:
    Eigen::MatrixXf pred_particle_;
    Eigen::VectorXf pred_weight_;

    Eigen::MatrixXf cor_particle_;
    Eigen::VectorXf cor_weight_;


    const int          block_size_        = 16;
    const int          img_width_         = 320;
    const int          img_height_        = 240;
    const int          bin_number_        = 9;
    const unsigned int descriptor_length_ = (img_width_/block_size_*2-1) * (img_height_/block_size_*2-1) * bin_number_ * 4;
};

#endif /* VISUALSISPARTICLEFILTER_H */