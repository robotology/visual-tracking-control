#ifndef VISUALPROPRIOCEPTION_H
#define VISUALPROPRIOCEPTION_H

#include <string>

#include <BayesFilters/VisualObservationModel.h>
#include <iCub/iKin/iKinFwd.h>
#include <opencv2/core/core.hpp>
#include <SuperimposeMesh/SICAD.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/IAnalogSensor.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>


// FIXME: riscrivere rispettando l'interfaccia, altrimenti salta decorator.
class VisualProprioception : public bfl::VisualObservationModel
{
public:
    /* VisualProprioception constructor */
    VisualProprioception(const int num_images, const yarp::os::ConstString& cam_sel, const yarp::os::ConstString& laterality, const yarp::os::ConstString& context);

    /* Destructor */
    ~VisualProprioception() noexcept override;

    /* Copy constructor */
    VisualProprioception(const VisualProprioception& proprio);

    /* Move constructor */
    VisualProprioception(VisualProprioception&& proprio) noexcept;

    /* Copy assignment operator */
    VisualProprioception& operator=(const VisualProprioception& proprio);

    /* Move assignment operator */
    VisualProprioception& operator=(VisualProprioception&& proprio) noexcept;

    void observe(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, cv::OutputArray observation) override;

    bool setProperty(const std::string property) override;


    int          getOGLTilesNumber();
    int          getOGLTilesRows();
    int          getOGLTilesCols();

    unsigned int getCamWidth();
    unsigned int getCamHeight();
    float        getCamFx();
    float        getCamFy();
    float        getCamCx();
    float        getCamCy();

protected:
    yarp::os::ConstString    log_ID_ = "[VisualProprioception]";

    /* ICUB */
    yarp::os::ConstString                     laterality_;
    yarp::dev::PolyDriver                     drv_gaze_;
    yarp::dev::IGazeControl*                  itf_gaze_;
    iCub::iKin::iCubArm                       icub_arm_;
    iCub::iKin::iCubFinger                    icub_kin_finger_[3];
    iCub::iKin::iCubEye                       icub_kin_eye_;
    yarp::os::BufferedPort<yarp::os::Bottle>  port_head_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle>  port_torso_enc_;
    yarp::os::BufferedPort<yarp::os::Bottle>  port_arm_enc_;
    yarp::dev::PolyDriver                     drv_right_hand_analog_;
    yarp::dev::IAnalogSensor                * itf_right_hand_analog_;
    yarp::sig::Matrix                         right_hand_analogs_bounds_;

    yarp::sig::Matrix getInvertedH(const double a, const double d, const double alpha, const double offset, const double q);

    bool              openGazeController();

    bool              openAnalogs();
    bool              closeAnalogs();
    bool              analogs_ = false;

    bool              setiCubParams();

    void              setArmJoints(const yarp::sig::Vector& q);

    void              setArmJoints(const yarp::sig::Vector& q, const yarp::sig::Vector& analogs, const yarp::sig::Matrix& analog_bounds);

    void              getPoses(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, std::vector<Superimpose::ModelPoseContainer>& hand_poses);

    yarp::sig::Vector readArmEncoders();
    yarp::sig::Vector readTorso();
    yarp::sig::Vector readRootToFingers();
    yarp::sig::Vector readRootToEye(const yarp::os::ConstString cam_sel);
    /* **** */

    yarp::os::ConstString     cam_sel_;
    double                    cam_x_[3];
    double                    cam_o_[4];
    unsigned int              cam_width_;
    unsigned int              cam_height_;
    float                     cam_fx_;
    float                     cam_cx_;
    float                     cam_fy_;
    float                     cam_cy_;

    SICAD::ModelPathContainer cad_obj_;
    SICAD*                    si_cad_;
    int                       ogl_tiles_rows_;
    int                       ogl_tiles_cols_;

    bool file_found(const yarp::os::ConstString& file);

};

#endif /* VISUALPROPRIOCEPTION_H */
