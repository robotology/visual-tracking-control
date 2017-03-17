#ifndef VISUALPROPRIOCEPTION_H
#define VISUALPROPRIOCEPTION_H

// FIXME: perchè sono inclusi GL/GLFW?
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iCub/iKin/iKinFwd.h>
#include <opencv2/core/core.hpp>
#include <yarp/os/ConstString.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>

#include <BayesFiltersLib/VisualObservationModel.h>
#include <SuperImpose/SICAD.h>


class VisualProprioception : public bfl::VisualObservationModel {
public:
    /* VisualProprioception constructor */
    VisualProprioception(const int width, const int height, const int num_images, const yarp::os::ConstString lateralirty);

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

protected:
    iCub::iKin::iCubArm      icub_arm_;
    iCub::iKin::iCubFinger   icub_kin_finger_[3];
    double                   cam_x_[3];
    double                   cam_o_[4];
    SuperImpose::ObjFileMap  cad_hand_;
    int                      cam_width_;
    int                      cam_height_;
    float                    eye_fx_;
    float                    eye_cx_;
    float                    eye_fy_;
    float                    eye_cy_;
    SICAD                  * si_cad_;

    bool file_found(const yarp::os::ConstString& file);

    yarp::sig::Matrix getInvertedH(double a, double d, double alpha, double offset, double q);
};

#endif /* VISUALPROPRIOCEPTION_H */
