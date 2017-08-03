#ifndef VISUALSIRPARTICLEFILTER_H
#define VISUALSIRPARTICLEFILTER_H

#include <chrono>
#include <deque>
#include <memory>
#include <vector>

#include <BayesFilters/FilteringAlgorithm.h>
#include <BayesFilters/Initialization.h>
#include <BayesFilters/StateModel.h>
#include <BayesFilters/ParticleFilterPrediction.h>
#include <BayesFilters/VisualObservationModel.h>
#include <BayesFilters/VisualCorrection.h>
#include <BayesFilters/Resampling.h>
#include <Eigen/Dense>
#include <iCub/ctrl/adaptWinPolyEstimator.h>
#include <iCub/iKin/iKinFwd.h>
#include <opencv2/cudaobjdetect.hpp>
#include <thrift/VisualSIRParticleFilterIDL.h>
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
    HistoryBuffer() noexcept { };

    ~HistoryBuffer() noexcept { };

    void            addElement(const Eigen::Ref<const Eigen::VectorXf>& element);

    Eigen::MatrixXf getHistoryBuffer();

    bool            setHistorySize(const unsigned int window);

private:
    unsigned int                window_     = 5;

    const unsigned int          max_window_ = 90;

    std::deque<Eigen::VectorXf> hist_buffer_;
};


class VisualSIRParticleFilter: public bfl::FilteringAlgorithm,
                               public VisualSIRParticleFilterIDL
{
public:
    /* Default constructor, disabled */
    VisualSIRParticleFilter() = delete;

    /* VisualSIR complete constructor */
    VisualSIRParticleFilter(std::unique_ptr<bfl::Initialization> initialization,
                            std::unique_ptr<bfl::ParticleFilterPrediction> prediction, std::unique_ptr<bfl::VisualCorrection> correction,
                            std::unique_ptr<bfl::Resampling> resampling,
                            const yarp::os::ConstString& cam_sel, const yarp::os::ConstString& laterality, const int num_particles);

    /* Destructor */
    ~VisualSIRParticleFilter() noexcept override;

    /* Copy constructor */
    VisualSIRParticleFilter(const VisualSIRParticleFilter& vsir_pf) = delete;

    /* Move constructor */
    VisualSIRParticleFilter(VisualSIRParticleFilter&& vsir_pf) noexcept = delete;

    /* Copy assignment operator */
    VisualSIRParticleFilter& operator=(const VisualSIRParticleFilter& vsir_pf) = delete;

    /* Move assignment operator */
    VisualSIRParticleFilter& operator=(VisualSIRParticleFilter&& vsir_pf) noexcept = delete;

    void initialization() override;

    void filteringStep() override;

    void getResult() override;

    bool runCondition() override { return true; };

protected:
    std::unique_ptr<bfl::Initialization>           initialization_;
    std::unique_ptr<bfl::ParticleFilterPrediction> prediction_;
    std::unique_ptr<bfl::VisualCorrection>         correction_;
    std::unique_ptr<bfl::Resampling>               resampling_;

    yarp::os::ConstString                          cam_sel_;
    yarp::os::ConstString                          laterality_;
    const int                                      num_particles_;

    cv::Ptr<cv::cuda::HOG>                         cuda_hog_;


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

    HistoryBuffer hist_buffer_;

    enum class EstimatesExtraction
    {
        mean,
        mode,
        sm_average,
        wm_average,
        em_average,
        am_average
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

    bool                                  init_filter = true;
    iCub::ctrl::AWLinEstimator            lin_est_x_    {10, 0.02};
    iCub::ctrl::AWLinEstimator            lin_est_o_    {10, 0.5};
    iCub::ctrl::AWLinEstimator            lin_est_theta_{10, 3.0 * iCub::ctrl::CTRL_DEG2RAD};
    std::chrono::milliseconds             t_{0};
    std::chrono::steady_clock::time_point time_1_;
    std::chrono::steady_clock::time_point time_2_;
    Eigen::VectorXf amAverage(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights);
    /* *************************** */

private:
    Eigen::MatrixXf particle_;
    Eigen::VectorXf weight_;


    const int          block_size_        = 16;
    const int          img_width_         = 320;
    const int          img_height_        = 240;
    const int          bin_number_        = 9;
    const unsigned int descriptor_length_ = (img_width_/block_size_*2-1) * (img_height_/block_size_*2-1) * bin_number_ * 4;
};

#endif /* VISUALSIRPARTICLEFILTER_H */
