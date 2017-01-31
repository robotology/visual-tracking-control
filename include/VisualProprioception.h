#ifndef VISUALPROPRIOCEPTION_H
#define VISUALPROPRIOCEPTION_H

#include <exception>
#include <functional>
#include <random>

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
    /* VisualProprioception complete constructor */
    VisualProprioception(GLFWwindow*& window);

    /* Default constructor */
    VisualProprioception() = delete;

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

    void observe(const Eigen::Ref<const Eigen::VectorXf>& cur_state, cv::OutputArray observation) override;

    void setCamXO(double* cam_x, double* cam_o);

    void setArmJoints(const yarp::sig::Vector & q);

    void superimpose(const SuperImpose::ObjPoseMap& obj2pos_map, cv::Mat& img);

protected:
    // FIXME: non ha senso che siano dei puntatori
    double                    cam_x_[3];
    double                    cam_o_[4];
    GLFWwindow             *& window_;
    iCub::iKin::iCubFinger    icub_kin_finger_[3];
    iCub::iKin::iCubArm       icub_arm_;
    SICAD                  *  si_cad_;
    SuperImpose::ObjFileMap   cad_hand_;

    bool file_found(const yarp::os::ConstString& file);

    yarp::sig::Matrix getInvertedH(double a, double d, double alpha, double offset, double q);
};

#endif /* VISUALPROPRIOCEPTION_H */
