#ifndef VISUALSIRPARTICLEFILTER_H
#define VISUALSIRPARTICLEFILTER_H

#include <memory>
#include <vector>

#include <BayesFiltersLib/FilteringAlgorithm.h>
#include <BayesFiltersLib/StateModel.h>
#include <BayesFiltersLib/ParticleFilterPrediction.h>
#include <BayesFiltersLib/VisualObservationModel.h>
#include <BayesFiltersLib/VisualCorrection.h>
#include <BayesFiltersLib/Resampling.h>
#include <Eigen/Dense>
#include <iCub/iKin/iKinFwd.h>
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


class VisualSIRParticleFilter: public bfl::FilteringAlgorithm,
                               public VisualSIRParticleFilterIDL
{
public:
    /* Default constructor, disabled */
    VisualSIRParticleFilter() = delete;

    /* VisualSIR complete constructor */
    VisualSIRParticleFilter(std::unique_ptr<bfl::ParticleFilterPrediction> prediction, std::unique_ptr<bfl::VisualCorrection> correction,
                            std::unique_ptr<bfl::Resampling> resampling,
                            yarp::os::ConstString cam_sel, yarp::os::ConstString laterality, const int num_particles);

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

    void runFilter() override;

    void getResult() override;

protected:
    std::unique_ptr<bfl::ParticleFilterPrediction> prediction_;
    std::unique_ptr<bfl::VisualCorrection>         correction_;
    std::unique_ptr<bfl::Resampling>               resampling_;

    yarp::os::ConstString                          cam_sel_;
    yarp::os::ConstString                          laterality_;
    const int                                      num_particles_;

    yarp::os::BufferedPort<yarp::sig::Vector>      port_estimates_out_;

    unsigned long int                              filtering_step_ = 0;
    bool                                           is_filter_init_ = false;
    bool                                           is_running_     = false;
    bool                                           use_mean_       = false;
    bool                                           use_mode_       = true;

    yarp::os::Port                                 port_rpc_command_;
    bool                                           setCommandPort();


    bool                     use_analogs(const bool status) override;

    std::vector<std::string> get_info() override;

    bool                     set_estimates_extraction_method(const std::string& method) override;

    bool                     quit() override;

    bool                     visual_correction(const bool status) override;
    bool                     do_visual_correction_ = true;


    Eigen::VectorXf mean(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights) const;

    Eigen::VectorXf mode(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights) const;

private:
    /*         THIS DATA MEMBER AND METHODS SHOULD BE IN INITIALIZATION CLASS          */
    iCub::iKin::iCubArm                                              icub_kin_arm_;
    iCub::iKin::iCubFinger                                           icub_kin_finger_[3];
    yarp::os::BufferedPort<yarp::os::Bottle>                         port_torso_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle>                         port_arm_enc_;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>>  port_image_in_left_;

    yarp::sig::Vector                                                readTorso();

    yarp::sig::Vector                                                readRootToEE();
    /* ******************************************************************************* */
};

#endif /* VISUALSIRPARTICLEFILTER_H */
