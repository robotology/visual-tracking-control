#ifndef VISUALSIS_H
#define VISUALSIS_H

#include <BayesFilters/EstimatesExtraction.h>
#include <BayesFilters/ParticleFilter.h>
#include <BayesFilters/ParticleSet.h>
#include <BayesFilters/ParticleSetInitialization.h>
#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/PFCorrection.h>
#include <BayesFilters/Resampling.h>

#include <chrono>
#include <deque>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <iCub/iKin/iKinFwd.h>
#include <thrift/VisualSISParticleFilterIDL.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IAnalogSensor.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Port.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>


class VisualSIS: public bfl::ParticleFilter,
                 public VisualSISParticleFilterIDL
{
public:
    VisualSIS(std::unique_ptr<bfl::ParticleSetInitialization> initialization,
              std::unique_ptr<bfl::PFPrediction> prediction,
              std::unique_ptr<bfl::PFCorrection> correction,
              std::unique_ptr<bfl::Resampling> resampling,
              const std::string& cam_sel,
              const int num_particles,
              const double resample_ratio,
              const std::string& port_prefix);

    ~VisualSIS() noexcept;

protected:
    bool initialization() override;

    void filteringStep() override;

    bool runCondition() override { return true; };

    int num_particles_;

    double resample_ratio_;

    unsigned int descriptor_length_;

    yarp::os::BufferedPort<yarp::sig::Vector> port_estimates_out_;

    yarp::os::Port port_rpc_command_;

    bool attach(yarp::os::Port &source);

    bool setCommandPort();

    bool run_filter() override;

    bool reset_filter() override;

    bool stop_filter() override;

    bool skip_step(const std::string& what_step, const bool status) override;

    std::vector<std::string> get_info() override;

    bool quit() override;

    bool set_estimates_extraction_method(const std::string& method) override;

    bool set_mobile_average_window(const int16_t window) override;

    std::string gpu_engine_count_to_string(const int engine_count) const;

private:
    const std::string log_ID_ = "[VisualSIS]";

    std::string port_prefix_;

    bfl::ParticleSet pred_particle_;

    bfl::ParticleSet cor_particle_;

    bfl::EstimatesExtraction estimate_extraction_;

    /* Add to debug Walkman */
    //yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_out_;
};

#endif /* VISUALSIS_H */
