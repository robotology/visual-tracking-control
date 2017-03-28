#ifndef VISUALSIRPARTICLEFILTER_H
#define VISUALSIRPARTICLEFILTER_H

#include <future>
#include <memory>
#include <random>
#include <vector>

#include <BayesFiltersLib/FilteringAlgorithm.h>
#include <BayesFiltersLib/StateModel.h>
#include <BayesFiltersLib/Prediction.h>
#include <BayesFiltersLib/VisualObservationModel.h>
#include <BayesFiltersLib/VisualCorrection.h>
#include <BayesFiltersLib/Resampling.h>
#include <Eigen/Dense>
#include <iCub/iKin/iKinFwd.h>
#include <thrift/visualSIRParticleFilterIDL.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IAnalogSensor.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/Port.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>

#include "VisualParticleFilterCorrection.h"


class VisualSIRParticleFilter: public bfl::FilteringAlgorithm,
                               public visualSIRParticleFilterIDL
{
public:
    /* Default constructor, disabled */
    VisualSIRParticleFilter() = delete;

    /* VisualSIR complete constructor */
    VisualSIRParticleFilter(std::unique_ptr<bfl::Prediction> prediction, std::unique_ptr<VisualParticleFilterCorrection> correction,
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

    std::future<void> spawn();

    bool isRunning();

    bool shouldStop();

    void stopThread();

protected:
    std::unique_ptr<bfl::Prediction>                                 prediction_;
    std::unique_ptr<VisualParticleFilterCorrection>                  correction_;
    std::unique_ptr<bfl::Resampling>                                 resampling_;
    yarp::os::ConstString                                            cam_sel_;
    yarp::os::ConstString                                            laterality_;
    const int                                                        num_particles_;

    /* INIT ONLY */
    iCub::iKin::iCubArm                                              icub_kin_arm_;
    iCub::iKin::iCubFinger                                           icub_kin_finger_[3];
    yarp::os::BufferedPort<yarp::os::Bottle>                         port_torso_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle>                         port_arm_enc_;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>>  port_image_in_left_;
    /* OUTPUT */
    yarp::os::BufferedPort<yarp::sig::Vector>                        port_estimates_out_;
    /* DEBUG ONLY */
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>>  port_image_out_left_;
    /* ******************* */

    bool                                                             is_filter_init_;
    bool                                                             is_running_;

    yarp::os::Port port_rpc_command_;
    bool setCommandPort();

    bool stream_result(const bool status) override;
    bool stream_ = true;

    bool use_analogs(const bool status) override;

    void quit() override;

    enum laterality
    {
       LEFT  = 0,
       RIGHT = 1
    };

private:
    Eigen::VectorXf mean(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights) const;

    Eigen::VectorXf mode(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights) const;

    /* THIS CALL SHOULD BE IN ANOTHER CLASS */
    yarp::sig::Vector readTorso();

    yarp::sig::Vector readRootToEE();

    yarp::sig::Matrix getInvertedH(double a, double d, double alpha, double offset, double q);
    /* ************************************ */
};

#endif /* VISUALSIRPARTICLEFILTER_H */
