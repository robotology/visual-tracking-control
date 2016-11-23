#include <algorithm>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <numeric>
#include <sys/types.h>
#include <limits>
#include <thread>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/sig/Image.h>
#include <yarp/math/Math.h>
#include <iCub/iKin/iKinFwd.h>
#include <iCub/ctrl/math.h>

#include "FilteringContext.h"
#include "SIRParticleFilter.h"
#include "ParticleFilteringFunction.h"
#include "AutoCanny.h"
#include "SICAD.h"

#define WINDOW_WIDTH  320
#define WINDOW_HEIGHT 240

using namespace cv;
using namespace Eigen;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;
using namespace iCub::iKin;
using namespace iCub::ctrl;


cv::String cvwin = "Superimposed Edges";

bool FileFound (const ConstString & file)
{
    if (file.empty()) {
        yError() << "File not found!";
        return false;
    }
    return true;
}


class HTParticleFilteringFunction : public ParticleFilteringFunction
{
private:
    std::normal_distribution<float>        * distribution_theta = nullptr;
    std::uniform_real_distribution<float>  * distribution_phi_z = nullptr;
    std::function<float (float)>             gaussian_random_theta;
    std::function<float (float)>             gaussian_random_phi_z;

    GLFWwindow                             * window_ = nullptr;

    SICAD                                  * si_cad_;
    SuperImpose::ObjFileMap                  cad_hand_;
    double                                   cam_x_[3];
    double                                   cam_o_[4];
    Mat                                      img_back_edge_;

public:

    HTParticleFilteringFunction() {}


    ~HTParticleFilteringFunction()
    {
        delete distribution_theta;
        delete distribution_phi_z;
        delete si_cad_;
    }


    void setOGLWindow(GLFWwindow *& window)
    {
        window_ = window;
    }


    void setCamXO(double * cam_x, double * cam_o)
    {
        memcpy(cam_x_, cam_x, 3 * sizeof(double));
        memcpy(cam_o_, cam_o, 4 * sizeof(double));
    }


    void setImgBackEdge(const Mat & img_back_edge)
    {
        img_back_edge_ = img_back_edge;
    }


    virtual bool Configure()
    {
        _state_cov.resize(3, 1);
        _state_cov <<                0.01,
                      0.5 * (M_PI /180.0),
                      3.0 * (M_PI /180.0);

        generator             = new std::mt19937_64(1);
        distribution_pos      = new std::normal_distribution<float>(0.0, _state_cov(0));
        distribution_theta    = new std::normal_distribution<float>(0.0, _state_cov(1));
        distribution_phi_z    = new std::uniform_real_distribution<float>(0.0, 1.0);
        gaussian_random_pos   = [&] (int) { return (*distribution_pos)  (*generator); };
        gaussian_random_theta = [&] (int) { return (*distribution_theta)(*generator); };
        gaussian_random_phi_z = [&] (int) { return (*distribution_phi_z)(*generator); };

        ResourceFinder rf;
        cad_hand_["palm"] = rf.findFileByName("r_palm.obj");
        if (!FileFound(cad_hand_["palm"])) return false;
//        cad_hand_["thumb1"] = rf.findFileByName("r_tl0.obj");
//        if (!FileFound(cad_hand_["thumb1"])) return false;
//        cad_hand_["thumb2"] = rf.findFileByName("r_tl1.obj");
//        if (!FileFound(cad_hand_["thumb2"])) return false;
//        cad_hand_["thumb3"] = rf.findFileByName("r_tl2.obj");
//        if (!FileFound(cad_hand_["thumb3"])) return false;
//        cad_hand_["thumb4"] = rf.findFileByName("r_tl3.obj");
//        if (!FileFound(cad_hand_["thumb4"])) return false;
//        cad_hand_["thumb5"] = rf.findFileByName("r_tl4.obj");
//        if (!FileFound(cad_hand_["thumb5"])) return false;
//        cad_hand_["index0"] = rf.findFileByName("r_indexbase.obj");
//        if (!FileFound(cad_hand_["index0"])) return false;
//        cad_hand_["index1"] = rf.findFileByName("r_ail0.obj");
//        if (!FileFound(cad_hand_["index1"])) return false;
//        cad_hand_["index2"] = rf.findFileByName("r_ail1.obj");
//        if (!FileFound(cad_hand_["index2"])) return false;
//        cad_hand_["index3"] = rf.findFileByName("r_ail2.obj");
//        if (!FileFound(cad_hand_["index3"])) return false;
//        cad_hand_["index4"] = rf.findFileByName("r_ail3.obj");
//        if (!FileFound(cad_hand_["index4"])) return false;
//        cad_hand_["medium0"] = rf.findFileByName("r_ml0.obj");
//        if (!FileFound(cad_hand_["medium0"])) return false;
//        cad_hand_["medium1"] = rf.findFileByName("r_ml1.obj");
//        if (!FileFound(cad_hand_["medium1"])) return false;
//        cad_hand_["medium2"] = rf.findFileByName("r_ml2.obj");
//        if (!FileFound(cad_hand_["medium2"])) return false;
//        cad_hand_["medium3"] = rf.findFileByName("r_ml3.obj");
//        if (!FileFound(cad_hand_["medium3"])) return false;

        si_cad_ = new SICAD();
        si_cad_->Configure(window_, cad_hand_, 232.921, 232.43, 162.202, 125.738);

        return true;
    }

    virtual void StateModel(const Ref<const VectorXf> & prev_state, Ref<VectorXf> prop_state)
    {
        MatrixXf A = MatrixXf::Identity(6, 6);
        prop_state = A * prev_state;
    }


    virtual void Prediction(const Ref<const VectorXf> & prev_state, Ref<VectorXf> pred_state)
    {
        StateModel(prev_state, pred_state);

        pred_state.head(3) += VectorXf::NullaryExpr(3, gaussian_random_pos);

        /* FROM MATLAB */
        float coneAngle = _state_cov(2);

        /* Generate points on the spherical cap around the north pole [1]. */
        /* [1] http://math.stackexchange.com/a/205589/81266 */
        float z   = gaussian_random_phi_z(0) * (1 - cos(coneAngle)) + cos(coneAngle);
        float phi = gaussian_random_phi_z(0) * (2 * M_PI);
        float x   = sqrt(1 - (z * z)) * cos(phi);
        float y   = sqrt(1 - (z * z)) * sin(phi);

        /* Find the rotation axis 'u' and rotation angle 'rot' [1] */
        Vector3f def_dir(0, 0, 1);
        Vector3f cone_dir = pred_state.tail(3).normalized();
        Vector3f u = def_dir.cross(cone_dir).normalized();
        float rot = static_cast<float>(acos(cone_dir.dot(def_dir)));

        /* Convert rotation axis and angle to 3x3 rotation matrix [2] */
        /* [2] https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle */
        Matrix3f cross_matrix;
        cross_matrix <<     0,  -u(2),   u(1),
                         u(2),      0,  -u(0),
                        -u(1),   u(0),      0;
        Matrix3f R = cos(rot) * Matrix3f::Identity() + sin(rot) * cross_matrix + (1 - cos(rot)) * (u * u.transpose());
                                                                                         
        /* Rotate [x y z]' from north pole to 'cone_dir' */
        Vector3f r_to_rotate(x, y, z);
        Vector3f r = R * r_to_rotate;

        /* *********** */

        VectorXf ang(1);
        ang << pred_state.tail(3).norm() + gaussian_random_theta(0);

        pred_state.tail(3) = r;
        pred_state.tail(3) *= ang;
    }


    virtual Ref<MatrixXf> ObservationModel(const Ref<const VectorXf> & pred_state)
    {
        Mat hand_ogl = Mat::zeros(img_back_edge_.rows, img_back_edge_.cols, img_back_edge_.type());
        Mat hand_edge;
        Mat edge;

        SuperImpose::ObjPoseMap hand_pose;
        SuperImpose::ObjPose    pose;
        pose.assign(pred_state.data(), pred_state.data()+3);

        Vector a_a(4);
        float ang = pred_state.tail(3).norm();
        a_a(0) = pred_state(3) / ang;
        a_a(1) = pred_state(4) / ang;
        a_a(2) = pred_state(5) / ang;
        a_a(3) = ang;
        pose.insert(pose.end(), a_a.data(), a_a.data()+4);

        hand_pose.emplace("palm", pose);

        si_cad_->Superimpose(hand_pose, cam_x_, cam_o_, hand_ogl);
        AutoCanny(hand_ogl, hand_edge);

        MatrixXf m(hand_edge.rows, hand_edge.cols);
        cv2eigen(hand_edge, m);

        /* Debug Only */
        hand_edge = max(hand_edge, img_back_edge_);
        imshow(cvwin, hand_edge);
        /* ********** */

        return m;
    }


    virtual void Correction(const Ref<const VectorXf> & pred_particles, const Ref<const MatrixXf> & measurements, Ref<VectorXf> cor_state)
    {
        MatrixXf hand_edge = ObservationModel(pred_particles);

        Mat hand_edge_cv;
        Mat meas_cv;
        eigen2cv(hand_edge, hand_edge_cv);
        MatrixXf meas = measurements;
        eigen2cv(meas, meas_cv);

        Mat result;
        normalize(meas_cv, meas_cv, 0.0, 1.0, NORM_MINMAX);
        normalize(hand_edge_cv, hand_edge_cv, 0.0, 1.0, NORM_MINMAX);
        matchTemplate(meas_cv, hand_edge_cv, result, TM_CCORR_NORMED);

//        cor_state << (result.at<float>(0, 0) < 0? 0 : exp(-(1 - result.at<float>(0, 0)))) + std::numeric_limits<float>::min();
//        cor_state << (result.at<float>(0, 0) < 0? 0 : exp(-((result.at<float>(0, 0) - 1) * (result.at<float>(0, 0) - 1)))) + std::numeric_limits<float>::min();
        cor_state << (result.at<float>(0, 0) < 0? 0 : result.at<float>(0, 0)) + std::numeric_limits<float>::min();
    }


    virtual void WeightedSum(const Ref<const MatrixXf> & particles, const Ref<const VectorXf> & weights, Ref<MatrixXf> particle)
    {
        particle = (particles.array().rowwise() * weights.array().transpose()).rowwise().sum();
    }


    virtual void Mode(const Ref<const MatrixXf> & particles, const Ref<const VectorXf> & weights, Ref<MatrixXf> particle)
    {
        MatrixXf::Index maxRow;
        MatrixXf::Index maxCol;

        weights.maxCoeff(&maxRow, &maxCol);

        particle = particles.col(maxRow);
    }


    void Superimpose(const SuperImpose::ObjPoseMap & obj2pos_map, cv::Mat & img)
    {
        si_cad_->setBackgroundOpt(true);
        si_cad_->Superimpose(obj2pos_map, cam_x_, cam_o_, img);
        si_cad_->setBackgroundOpt(false);
    }
};


class HTSIRParticleFilter : public FilteringAlgorithm
{
protected:
    HTParticleFilteringFunction      * ht_pf_f_;

    Network                            yarp;
    iCubEye                          * icub_kin_eye_;
    iCubArm                          * icub_kin_arm_;
//    iCubFinger                       * icub_kin_finger_;
    BufferedPort<ImageOf<PixelRgb>>    port_image_in_;
    BufferedPort<Bottle>               port_head_enc;
    BufferedPort<Bottle>               port_torso_enc;
    BufferedPort<Bottle>               port_arm_enc;
    BufferedPort<ImageOf<PixelRgb>>    port_image_out_;

    GLFWwindow                       * window_;

    bool                               is_running_;

private:
    Vector readTorso()
    {
        Bottle * b = port_torso_enc.read();
        Vector torso_enc(3);

        torso_enc(0) = b->get(2).asDouble();
        torso_enc(1) = b->get(1).asDouble();
        torso_enc(2) = b->get(0).asDouble();

        return torso_enc;
    }


    Vector readHead(const ConstString eye)
    {
        Bottle * b = port_head_enc.read();
        Vector head_enc(8);

        head_enc.setSubvector(0, readTorso());
        for (unsigned int i = 0; i < 4; ++i)
        {
            head_enc(i+3) = b->get(i).asDouble();
        }
        if (eye == "left")  head_enc(7) = b->get(4).asDouble() + b->get(5).asDouble()/2.0;
        if (eye == "right") head_enc(7) = b->get(4).asDouble() - b->get(5).asDouble()/2.0;

        return head_enc;
    }


    Vector readArm()
    {
        Bottle * b = port_arm_enc.read();
        Vector arm_enc(10);

        arm_enc.setSubvector(0, readTorso());
        for (unsigned int i = 0; i < 7; ++i)
        {
            arm_enc(i+3) = b->get(i).asDouble();
        }

        return arm_enc;
    }


public:
    HTSIRParticleFilter()
    {
        is_running_ = false;

        icub_kin_eye_    = nullptr;
        icub_kin_arm_    = nullptr;
//        icub_kin_finger_ = nullptr;
    }


    virtual ~HTSIRParticleFilter()
    {
        delete ht_pf_f_;
        delete icub_kin_eye_;
        delete icub_kin_arm_;
//        delete icub_kin_finger_;
    }


    void setOGLWindow(GLFWwindow *& window)
    {
        window_ = window;
    }


    bool Configure()
    {
        ht_pf_f_ = new HTParticleFilteringFunction;
        ht_pf_f_->setOGLWindow(window_);
        ht_pf_f_->Configure();

        if (!yarp.checkNetwork(3.0))
        {
            yError() << "YARP seems unavailable.";
            return false;
        }

        icub_kin_eye_ = new iCubEye("left_v2");
        icub_kin_eye_->setAllConstraints(false);
        icub_kin_eye_->releaseLink(0);
        icub_kin_eye_->releaseLink(1);
        icub_kin_eye_->releaseLink(2);

        icub_kin_arm_ = new iCubArm("right_v2");
        icub_kin_arm_->setAllConstraints(false);
        icub_kin_arm_->releaseLink(0);
        icub_kin_arm_->releaseLink(1);
        icub_kin_arm_->releaseLink(2);
//        icub_kin_finger_ = new iCubFinger();

        /* Images:         /icub/camcalib/left/out
           Head encoders:  /icub/head/state:o
           Arm encoders:   /icub/right_arm/state:o
           Torso encoders: /icub/torso/state:o     */
        port_image_in_.open ("/left_img:i");
        port_head_enc.open  ("/head");
        port_arm_enc.open   ("/right_arm");
        port_torso_enc.open ("/torso");
        port_image_out_.open("/left_img:o");

        if (!yarp.connect("/icub/camcalib/left/out", "/left_img:i")) return false;
        if (!yarp.connect("/icub/head/state:o",      "/head"))       return false;
        if (!yarp.connect("/icub/right_arm/state:o", "/right_arm"))  return false;
        if (!yarp.connect("/icub/torso/state:o",     "/torso"))      return false;

        return true;
    }


    void runFilter()
    {
        /* INITIALIZATION */
        MatrixXf init_particle;
        VectorXf init_weight;
        double cam_x[3];
        double cam_o[4];
        int num_particle = 50;

        init_weight.resize(num_particle, 1);
        init_weight.setConstant(1.0/num_particle);

        init_particle.resize(6, num_particle);

        Vector q = readArm();
        Vector ee_pose = icub_kin_arm_->EndEffPose(CTRL_DEG2RAD * q);

        Map<VectorXd> q_arm(ee_pose.data(), 6, 1);
        q_arm.tail(3) *= ee_pose(6);
        for (int i = 0; i < num_particle; ++i)
        {
            init_particle.col(i) = q_arm.cast<float>();
        }

        /* FILTERING */
        ImageOf<PixelRgb> * imgin = NULL;
        while(is_running_)
        {
            if (imgin == NULL) imgin = port_image_in_.read(true);
//            imgin = port_image_in_.read(true);
            if (imgin != NULL)
            {
                ImageOf<PixelRgb> & imgout = port_image_out_.prepare();
                imgout = *imgin;

                MatrixXf temp_particle(6, num_particle);
                VectorXf temp_weight(num_particle, 1);
                VectorXf temp_parent(num_particle, 1);

                MatrixXf measurement;
                Mat img_back = cvarrToMat(imgout.getIplImage());
                Mat img_back_edge;

                Vector eye_pose = icub_kin_eye_->EndEffPose(CTRL_DEG2RAD * readHead("left"));
                cam_x[0] = eye_pose(0); cam_x[1] = eye_pose(1); cam_x[2] = eye_pose(2);
                cam_o[0] = eye_pose(3); cam_o[1] = eye_pose(4); cam_o[2] = eye_pose(5); cam_o[3] = eye_pose(6);

                //        Snapshot();

                for (int i = 0; i < num_particle; ++i)
                {
                    ht_pf_f_->Prediction(init_particle.col(i), init_particle.col(i));
                }

                //        Snapshot();

                AutoCanny(img_back, img_back_edge);

                MatrixXf img_back_edge_eigen(img_back_edge.rows, img_back_edge.cols);
                cv2eigen(img_back_edge, img_back_edge_eigen);

                ht_pf_f_->setImgBackEdge(img_back_edge);
                ht_pf_f_->setCamXO(cam_x, cam_o);
                for (int i = 0; i < num_particle; ++i)
                {
                    ht_pf_f_->Correction(init_particle.col(i), img_back_edge_eigen, init_weight.row(i));
                }

                init_weight = init_weight / init_weight.sum();

                //        Snapshot();

                MatrixXf out_particle(6, 1);
                ht_pf_f_->WeightedSum(init_particle, init_weight, out_particle);
//                ht_pf_f_->Mode(init_particle, init_weight, out_particle);

//                if (ht_pf_f_->Neff(init_weight) < num_particle/3)
//                {
                    ht_pf_f_->Resampling(init_particle, init_weight, temp_particle, temp_weight, temp_parent);

                    init_particle = temp_particle;
                    init_weight   = temp_weight;
//                }

                // FIXME: out_particle is treatead as a Vector, but it's a Matrix.
                SuperImpose::ObjPoseMap hand_pose;
                SuperImpose::ObjPose    pose;
                pose.assign(out_particle.data(), out_particle.data()+3);

                Vector a_a(4);
                float ang = out_particle.col(0).tail(3).norm();
                a_a(0) = out_particle(3) / ang;
                a_a(1) = out_particle(4) / ang;
                a_a(2) = out_particle(5) / ang;
                a_a(3) = ang;
                pose.insert(pose.end(), a_a.data(), a_a.data()+4);

                hand_pose.emplace("palm", pose);
                
                ht_pf_f_->Superimpose(hand_pose, img_back);

                port_image_out_.write();
            }
        }
    }


    void getResult() {}


    std::thread spawn()
    {
        is_running_ = true;
        return std::thread(&HTSIRParticleFilter::runFilter, this);
    }


    bool isRunning()
    {
        return is_running_;
    }


    void stopThread()
    {
        is_running_ = false;
    }
};


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    /* When a user presses the escape key, we set the WindowShouldClose property to true, closing the application. */
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}


bool openglSetUp(GLFWwindow *& window, const int width, const int height)
{
    ConstString log_ID = "[OpenGL]";
    yInfo() << log_ID << "Start setting up...";

    /* Initialize GLFW. */
    if (glfwInit() == GL_FALSE)
    {
        yError() << log_ID << "Failed to initialize GLFW.";
        return false;
    }

    /* Set context properties by "hinting" specific (property, value) pairs. */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE,             GL_FALSE);
    glfwWindowHint(GLFW_VISIBLE,               GL_TRUE);
#ifdef GLFW_MAC
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    /* Create a window. */
    window = glfwCreateWindow(width, height, "OpenGL Window", nullptr, nullptr);
    if (window == nullptr)
    {
        yError() << log_ID << "Failed to create GLFW window.";
        glfwTerminate();
        return false;
    }
    /* Make the OpenGL context of window the current one handled by this thread. */
    glfwMakeContextCurrent(window);

    /* Set window callback functions. */
    glfwSetKeyCallback(window, key_callback);

    /* Initialize GLEW to use the OpenGL implementation provided by the videocard manufacturer. */
    /* Note: remember that the OpenGL are only specifications, the implementation is provided by the manufacturers. */
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        yError() << log_ID << "Failed to initialize GLEW.";
        return false;
    }

    /* Set OpenGL rendering frame for the current window. */
    /* Note that the real monitor width and height may differ w.r.t. the choosen one in hdpi monitors. */
    int hdpi_width;
    int hdpi_height;
    glfwGetFramebufferSize(window, &hdpi_width, &hdpi_height);
    glViewport(0, 0, hdpi_width, hdpi_height);
    yInfo() << log_ID << "Viewport set to "+std::to_string(hdpi_width)+"x"+std::to_string(hdpi_height)+".";

    /* Set GL property. */
    glEnable(GL_DEPTH_TEST);

    glfwPollEvents();

    yInfo() << log_ID << "Succesfully set up!";
    
    return true;
}


int main(int argc, char const *argv[])
{
    ConstString log_ID = "[Main]";
    yInfo() << log_ID << "Configuring and starting module...";

    namedWindow(cvwin, WINDOW_NORMAL | WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

//    ResourceFinder rf;
//    rf.setVerbose(true);
//    rf.setDefaultConfigFile("superimpose-hand_config.ini");
//    rf.setDefaultContext("superimpose-hand");
//    rf.configure(argc, argv);

    /* Initialize OpenGL context */
    GLFWwindow * window = nullptr;
    if (!openglSetUp(window, WINDOW_WIDTH, WINDOW_HEIGHT)) return EXIT_FAILURE;

    /* SuperimposeHand, derived from RFModule, must be declared by the main thread (thread_0). */
//    SuperimposerFactory sh;
//
//    sh.setWindow(window);
//    if (sh.runModuleThreaded(rf) == 0)
//    {
//        while (!sh.isStopping())
//        {
//            glfwPollEvents();
//        }
//    }
//
//    sh.joinModule();

    HTSIRParticleFilter ht_sir_pf;
    ht_sir_pf.setOGLWindow(window);
    if (!ht_sir_pf.Configure()) return EXIT_FAILURE;

    std::thread t = ht_sir_pf.spawn();
    while (ht_sir_pf.isRunning())
    {
        if (glfwWindowShouldClose(window)) ht_sir_pf.stopThread();
        glfwPollEvents();
    }
//    t.join();

    glfwMakeContextCurrent(NULL);
    glfwTerminate();

    yInfo() << log_ID << "Main returning.";
    yInfo() << log_ID << "Application closed.";

    return EXIT_SUCCESS;
}

//const char* window_autocanny = "AutoCanny Edge Map";
//const char* window_autodist  = "Distance Transform for AutoCanny";
//
//int main()
//{
//    namedWindow(window_autocanny, WINDOW_NORMAL | WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
//    namedWindow(window_autodist,  WINDOW_NORMAL | WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
//
//    while (waitKey(0) != 103);
//
//    Mat src;
//    Mat edge;
//    Mat dist;
//
//    const std::string img_dir("../../../resource/log/camera/left/");
//    DIR * dir = opendir(img_dir.c_str());
//    if (dir!= nullptr)
//    {
//        while (struct dirent * ent = readdir(dir))
//        {
//            size_t len = strlen(ent->d_name);
//
//            if (strcmp(ent->d_name, ".")  != 0 &&
//                strcmp(ent->d_name, "..") != 0 &&
//                len > 4 && strcmp(ent->d_name + len - 4, ".ppm") == 0)
//            {
//
//                src = imread(img_dir + std::string(ent->d_name), IMREAD_COLOR);
//
//                AutoCanny(src, edge);
//
////                distanceTransform(Scalar(255, 255, 255)-edge, dist, DIST_L2, DIST_MASK_5);
//                distanceTransform(Scalar(255, 255, 255)-edge, dist, DIST_L2, DIST_MASK_PRECISE);
//
//
//                /* ------------ PLOT ONLY ------------ */
//                Mat edge_colored(edge.size(), edge.type(), Scalar(0));
//                src.copyTo(edge_colored, edge);
//                normalize(dist, dist, 0.0, 1.0, NORM_MINMAX);
//
//                imshow(window_autocanny, edge_colored);
//                imshow(window_autodist,  dist);
//
//                waitKey(1);
//            }
//        }
//        closedir(dir);
//    }
//    else
//    {
//        perror("Could not open directory.");
//        return EXIT_FAILURE;
//    }
//
//    return EXIT_SUCCESS;
//}

//int main()
//{
//    FilteringContext fc(new SIRParticleFilter);
//
//    fc.run();
//
//    fc.saveResult();
//
//    return EXIT_SUCCESS;
//}
