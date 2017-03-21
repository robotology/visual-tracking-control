#ifndef VISUALPROPRIOCEPTION_H
#define VISUALPROPRIOCEPTION_H

#include <string>

// FIXME: perchè sono inclusi GL/GLFW?
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iCub/iKin/iKinFwd.h>
#include <opencv2/core/core.hpp>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>

#include <BayesFiltersLib/VisualObservationModel.h>
#include <SuperImpose/SICAD.h>


class VisualProprioception : public bfl::VisualObservationModel {
public:
    /* VisualProprioception constructor */
    VisualProprioception(const int num_images, const yarp::os::ConstString cam_sel, const yarp::os::ConstString laterality);

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

    int  oglWindowShouldClose();

    void observe(const Eigen::Ref<const Eigen::MatrixXf>& cur_state, cv::OutputArray observation) override;

    void setCamXO(double* cam_x, double* cam_o);

    void setCamIntrinsic(const unsigned int cam_width, const unsigned int cam_height,
                         const float eye_fx, const float eye_cx, const float eye_fy, const float eye_cy);

    void setArmJoints(const yarp::sig::Vector& q);

    void setArmJoints(const yarp::sig::Vector& q, const yarp::sig::Vector& analogs, const yarp::sig::Matrix& analog_bounds);

    void superimpose(const SuperImpose::ObjPoseMap& obj2pos_map, cv::Mat& img);

    int  getOGLTilesRows();
    int  getOGLTilesCols();

protected:
    yarp::os::ConstString    log_ID_;

    /* ICUB */
    yarp::os::ConstString    laterality_;
    yarp::dev::PolyDriver    drv_gaze;
    yarp::dev::IGazeControl* itf_gaze;
    yarp::os::Bottle         cam_info;
    iCub::iKin::iCubArm      icub_arm_;
    iCub::iKin::iCubFinger   icub_kin_finger_[3];
    /* **** */

    yarp::os::ConstString    cam_sel_;
    double                   cam_x_[3];
    double                   cam_o_[4];
    SuperImpose::ObjFileMap  cad_hand_;
    unsigned int             cam_width_;
    unsigned int             cam_height_;
    float                    cam_fx_;
    float                    cam_cx_;
    float                    cam_fy_;
    float                    cam_cy_;

    SICAD                  * si_cad_;
    int                      ogl_tiles_rows_;
    int                      ogl_tiles_cols_;

    bool file_found(const yarp::os::ConstString& file);

private:
    yarp::sig::Matrix getInvertedH(const double a, const double d, const double alpha, const double offset, const double q);

    bool              setGazeController(yarp::dev::PolyDriver& drv_gaze, yarp::dev::IGazeControl*& itf_gaze, yarp::os::ConstString program_name);
};

#endif /* VISUALPROPRIOCEPTION_H */
