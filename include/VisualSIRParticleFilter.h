#ifndef VISUALSIRPARTICLEFILTER_H
#define VISUALSIRPARTICLEFILTER_H

#include <future>
#include <memory>
#include <random>

#include <Eigen/Dense>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>

#include <BayesFiltersLib/FilteringAlgorithm.h>
#include <BayesFiltersLib/StateModel.h>
#include <BayesFiltersLib/Prediction.h>
#include <BayesFiltersLib/VisualObservationModel.h>
#include <BayesFiltersLib/VisualCorrection.h>
#include <BayesFiltersLib/Resampling.h>


class VisualSIRParticleFilter: public bfl::FilteringAlgorithm {
public:

    /* Default constructor, disabled */
    VisualSIRParticleFilter() = delete;

    /* VisualSIR complete constructor */
    VisualSIRParticleFilter(std::shared_ptr<bfl::StateModel> state_model, std::shared_ptr<bfl::Prediction> prediction, std::shared_ptr<bfl::VisualObservationModel> observation_model, std::shared_ptr<bfl::VisualCorrection> correction, std::shared_ptr<bfl::Resampling> resampling, const int num_particles) noexcept;

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

    void stopThread();

protected:
    std::shared_ptr<bfl::StateModel>                                state_model_;
    std::shared_ptr<bfl::Prediction>                                prediction_;
    std::shared_ptr<bfl::VisualObservationModel>                    observation_model_;
    std::shared_ptr<bfl::VisualCorrection>                          correction_;
    std::shared_ptr<bfl::Resampling>                                resampling_;
    const int                                                       num_particles_;

    Eigen::MatrixXf                                                 object_;
    Eigen::MatrixXf                                                 measurement_;

    Eigen::MatrixXf                                                 init_particle_;
    Eigen::VectorXf                                                 init_weight_;
    
    std::vector<Eigen::MatrixXf>                                    result_particle_;
    std::vector<Eigen::VectorXf>                                    result_weight_;

    iCub::iKin::iCubEye                                             icub_kin_eye_;
    iCub::iKin::iCubArm                                             icub_kin_arm_;
    iCub::iKin::iCubFinger                                          icub_kin_finger_[3];
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_in_;
    yarp::os::BufferedPort<yarp::os::Bottle>                        port_head_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle>                        port_torso_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle>                        port_arm_enc_;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> port_image_out_;

    bool                                                            is_running_;

private:
    Eigen::MatrixXf mean(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights) const;

    Eigen::MatrixXf mode(const Eigen::Ref<const Eigen::MatrixXf>& particles, const Eigen::Ref<const Eigen::VectorXf>& weights) const;
    
    yarp::sig::Vector readTorso();

    yarp::sig::Vector readRootToFingers();

    yarp::sig::Vector readRootToEye(const yarp::os::ConstString eye);

    yarp::sig::Vector readRootToEE();

    yarp::sig::Matrix getInvertedH(double a, double d, double alpha, double offset, double q);
};

#endif /* VISUALSIRPARTICLEFILTER_H */
