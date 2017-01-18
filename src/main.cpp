#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <numeric>
#include <limits>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/sig/Image.h>
#include <yarp/math/Math.h>
#include <iCub/iKin/iKinFwd.h>
#include <iCub/ctrl/math.h>

#include <BayesFiltersLib/FilteringContext.h>
#include <BayesFiltersLib/FilteringFunction.h>
#include <BayesFiltersLib/SIRParticleFilter.h>

#include "BrownianMotion.h"
#include "Proprioception.h"
#include "VisualParticleFilterCorrection.h"
#include "VisualSIRParticleFilter.h"

#include "SICAD.h"

#define WINDOW_WIDTH  320
#define WINDOW_HEIGHT 240

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;
using namespace iCub::iKin;
using namespace iCub::ctrl;
//typedef typename yarp::sig::Matrix YMatrix;

//cv::String cvwin = "Superimposed Edges";


#ifdef HT_DEPRECATED

bool FileFound (const ConstString & file)
{
    if (file.empty()) {
        yError() << "File not found!";
        return false;
    }
    return true;
}


class HTParticleFilteringFunction : public ParticleFilteringFunction {
private:
    std::normal_distribution<float>        * distribution_theta;
    std::uniform_real_distribution<float>  * distribution_phi_z;
    std::function<float (float)>             gaussian_random_theta;
    std::function<float (float)>             gaussian_random_phi_z;

    GLFWwindow                             * window_;

    SICAD                                  * si_cad_;
    iCubFinger                             * icub_kin_finger_[3];
    SuperImpose::ObjFileMap                  cad_hand_;
    double                                   cam_x_[3];
    double                                   cam_o_[4];
    Mat                                      img_back_edge_;

public:
    /* C/D-tors */
    HTParticleFilteringFunction()
    {
        distribution_theta = nullptr;
        distribution_phi_z = nullptr;
        window_            = nullptr;
        si_cad_            = nullptr;
        for (size_t i = 0; i < 3; ++i) icub_kin_finger_[i] = nullptr;
    }


    ~HTParticleFilteringFunction()
    {
        delete distribution_theta;
        delete distribution_phi_z;
        delete si_cad_;
        for (size_t i = 0; i < 3; ++i) delete icub_kin_finger_[i];
    }


    /* Extras */
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


    void setArmJoints(const Vector & q)
    {
        Vector chainjoints;
        for (size_t i = 0; i < 3; ++i)
        {
            icub_kin_finger_[i]->getChainJoints(q, chainjoints);
            icub_kin_finger_[i]->setAng(CTRL_DEG2RAD * chainjoints);
        }
    }


    virtual bool Configure()
    {
        _state_cov.resize(3, 1);
        _state_cov <<               0.005,
                      1.0 * (M_PI /180.0),
                      2.0 * (M_PI /180.0);

        generator             = new std::mt19937_64(1);
        distribution_pos      = new std::normal_distribution<float>(0.0, _state_cov(0));
        distribution_theta    = new std::normal_distribution<float>(0.0, _state_cov(1));
        distribution_phi_z    = new std::uniform_real_distribution<float>(0.0, 1.0);
        gaussian_random_pos   = [&] (int) { return (*distribution_pos)  (*generator); };
        gaussian_random_theta = [&] (int) { return (*distribution_theta)(*generator); };
        gaussian_random_phi_z = [&] (int) { return (*distribution_phi_z)(*generator); };

        // FIXME: middle finger only!
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
        cad_hand_["medium0"] = rf.findFileByName("r_ml0.obj");
        if (!FileFound(cad_hand_["medium0"])) return false;
        cad_hand_["medium1"] = rf.findFileByName("r_ml1.obj");
        if (!FileFound(cad_hand_["medium1"])) return false;
        cad_hand_["medium2"] = rf.findFileByName("r_ml2.obj");
        if (!FileFound(cad_hand_["medium2"])) return false;
        cad_hand_["medium3"] = rf.findFileByName("r_ml3.obj");
        if (!FileFound(cad_hand_["medium3"])) return false;

        si_cad_ = new SICAD();
        si_cad_->Configure(window_, cad_hand_, 232.921, 232.43, 162.202, 125.738);

        icub_kin_finger_[0] = new iCubFinger("right_thumb");
        icub_kin_finger_[1] = new iCubFinger("right_index");
        icub_kin_finger_[2] = new iCubFinger("right_middle");
        icub_kin_finger_[0]->setAllConstraints(false);
        icub_kin_finger_[1]->setAllConstraints(false);
        icub_kin_finger_[2]->setAllConstraints(false);

        return true;
    }


    void Superimpose(const SuperImpose::ObjPoseMap & obj2pos_map, cv::Mat & img)
    {
        si_cad_->setBackgroundOpt(true);
        si_cad_->Superimpose(obj2pos_map, cam_x_, cam_o_, img);
        si_cad_->setBackgroundOpt(false);
    }


    /* Filtering */
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
        Mat                     hand_ogl = Mat::zeros(img_back_edge_.rows, img_back_edge_.cols, img_back_edge_.type());
        Mat                     hand_ogl_gray;
        SuperImpose::ObjPoseMap hand_pose;
        SuperImpose::ObjPose    pose;
        Vector                  ee_o(4);
        float                   ang;


        ang     = pred_state.tail(3).norm();
        ee_o(0) = pred_state(3) / ang;
        ee_o(1) = pred_state(4) / ang;
        ee_o(2) = pred_state(5) / ang;
        ee_o(3) = ang;

        pose.assign(pred_state.data(), pred_state.data()+3);
        pose.insert(pose.end(), ee_o.data(), ee_o.data()+4);

        hand_pose.emplace("palm", pose);

        Vector ee_t(3, pose.data());
        ee_t.push_back(1.0);
        YMatrix Ha = axis2dcm(ee_o);
        Ha.setCol(3, ee_t);
        // FIXME: middle finger only!
        for (size_t fng = 2; fng < 3; ++fng)
        {
            std::string finger_s;
            pose.clear();
            if (fng != 0)
            {
                Vector j_x = (Ha * (icub_kin_finger_[fng]->getH0().getCol(3))).subVector(0, 2);
                Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng]->getH0());

                if      (fng == 1) { finger_s = "index0"; }
                else if (fng == 2) { finger_s = "medium0"; }

                pose.assign(j_x.data(), j_x.data()+3);
                pose.insert(pose.end(), j_o.data(), j_o.data()+4);
                hand_pose.emplace(finger_s, pose);
            }

            for (size_t i = 0; i < icub_kin_finger_[fng]->getN(); ++i)
            {
                Vector j_x = (Ha * (icub_kin_finger_[fng]->getH(i, true).getCol(3))).subVector(0, 2);
                Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng]->getH(i, true));

                if      (fng == 0) { finger_s = "thumb"+std::to_string(i+1); }
                else if (fng == 1) { finger_s = "index"+std::to_string(i+1); }
                else if (fng == 2) { finger_s = "medium"+std::to_string(i+1); }

                pose.assign(j_x.data(), j_x.data()+3);
                pose.insert(pose.end(), j_o.data(), j_o.data()+4);
                hand_pose.emplace(finger_s, pose);
            }
        }

        si_cad_->Superimpose(hand_pose, cam_x_, cam_o_, hand_ogl);
        cvtColor(hand_ogl, hand_ogl_gray, CV_RGB2GRAY);

        MatrixXf m(hand_ogl_gray.rows, hand_ogl_gray.cols);
        cv2eigen(hand_ogl_gray, m);

        /* Debug Only */
        imshow(cvwin, max(hand_ogl, img_back_edge_));
        /* ********** */
        
        return m;
    }


    // FIXME: new function using cv::mat measurement instead of eigen::MatrixXf
    virtual void Correction(const Ref<const VectorXf> & pred_particles, const Mat & measurements, Ref<VectorXf> cor_state)
    {
        int block_size = 16;
        Mat hand_edge_ogl_cv;
        std::vector<Point> points;

        MatrixXf hand_edge_ogl = ObservationModel(pred_particles);

        /* OGL image crop */
        eigen2cv(hand_edge_ogl, hand_edge_ogl_cv);
        for (auto it = hand_edge_ogl_cv.begin<float>(); it != hand_edge_ogl_cv.end<float>(); ++it) if (*it) points.push_back(it.pos());

        if (points.size() > 0)
        {
            Mat                hand_edge_cam_cv = measurements;
            Mat                cad_edge_crop;
            Mat                cam_edge_crop;
            std::vector<float> descriptors_cam;
            std::vector<float> descriptors_cad;
            std::vector<Point> locations;
            int                rem_not_mult;

            Rect cad_crop_roi   = boundingRect(points);
            rem_not_mult = div(cad_crop_roi.width,  block_size).rem;
            if (rem_not_mult > 0) cad_crop_roi.width  = cad_crop_roi.width  + (block_size - rem_not_mult);
            rem_not_mult = div(cad_crop_roi.height, block_size).rem;
            if (rem_not_mult > 0) cad_crop_roi.height = cad_crop_roi.height + (block_size - rem_not_mult);
            if (cad_crop_roi.x + cad_crop_roi.width  > hand_edge_ogl_cv.cols) cad_crop_roi.x -= (cad_crop_roi.x + cad_crop_roi.width ) - hand_edge_ogl_cv.cols;
            if (cad_crop_roi.y + cad_crop_roi.height > hand_edge_ogl_cv.rows) cad_crop_roi.y -= (cad_crop_roi.y + cad_crop_roi.height) - hand_edge_ogl_cv.rows;
            hand_edge_ogl_cv(cad_crop_roi).convertTo(cad_edge_crop, CV_8U);
            hand_edge_cam_cv(cad_crop_roi).convertTo(cam_edge_crop, CV_8U);

            /* In-crop HOG between camera and render edges */
            HOGDescriptor hog(Size(cad_crop_roi.width, cad_crop_roi.height), Size(block_size, block_size), Size(block_size/2, block_size/2), Size(block_size/2, block_size/2), 12,
                              1, -1, HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS, false);

            locations.push_back(Point(0, 0));

            hog.compute(cam_edge_crop, descriptors_cam, Size(), Size(), locations);
            hog.compute(cad_edge_crop, descriptors_cad, Size(), Size(), locations);

            auto it_cad = descriptors_cad.begin();
            auto it_cam = descriptors_cam.begin();
            float sum_diff = 0;
            for (; it_cad < descriptors_cad.end(); ++it_cad, ++it_cam) sum_diff += abs((*it_cad) - (*it_cam));

            // FIXME: Kernel likelihood need to be tuned!
            cor_state(0) *= ( exp( -0.001 * sum_diff /* / pow(1, 2.0) */ ) );
            if (cor_state(0) <= 0) cor_state(0) = std::numeric_limits<float>::min();
        }
        else
        {
            cor_state << std::numeric_limits<float>::min();
        }
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
};


class HTSIRParticleFilter : public FilteringAlgorithm {
protected:
    HTParticleFilteringFunction      * ht_pf_f_;

    Network                            yarp;
    iCubEye                          * icub_kin_eye_;
    iCubArm                          * icub_kin_arm_;
    iCubFinger                       * icub_kin_finger_[3];
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

        yAssert(b->size() == 3);

        torso_enc(0) = b->get(2).asDouble();
        torso_enc(1) = b->get(1).asDouble();
        torso_enc(2) = b->get(0).asDouble();

        return torso_enc;
    }


    Vector readArm()
    {
        Bottle * b = port_arm_enc.read();
        Vector arm_enc(16);

        yAssert(b->size() == 16);

        for (size_t i = 0; i < 16; ++i)
        {
            arm_enc(i) = b->get(i).asDouble();
        }

        return arm_enc;
    }


    Vector readRootToEye(const ConstString eye)
    {
        Bottle * b = port_head_enc.read();
        Vector root_eye_enc(8);

        root_eye_enc.setSubvector(0, readTorso());
        for (size_t i = 0; i < 4; ++i)
        {
            root_eye_enc(i+3) = b->get(i).asDouble();
        }
        if (eye == "left")  root_eye_enc(7) = b->get(4).asDouble() + b->get(5).asDouble()/2.0;
        if (eye == "right") root_eye_enc(7) = b->get(4).asDouble() - b->get(5).asDouble()/2.0;

        return root_eye_enc;
    }


    Vector readRootToEE()
    {
        Bottle * b = port_arm_enc.read();
        Vector root_ee_enc(10);

        root_ee_enc.setSubvector(0, readTorso());
        for (size_t i = 0; i < 7; ++i)
        {
            root_ee_enc(i+3) = b->get(i).asDouble();
        }

        return root_ee_enc;
    }

public:
    HTSIRParticleFilter()
    {
        is_running_   = false;
        icub_kin_eye_ = nullptr;
        icub_kin_arm_ = nullptr;
    }


    virtual ~HTSIRParticleFilter()
    {
        delete ht_pf_f_;
        delete icub_kin_eye_;
        delete icub_kin_arm_;
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

        icub_kin_finger_[0] = new iCubFinger("right_thumb");
        icub_kin_finger_[1] = new iCubFinger("right_index");
        icub_kin_finger_[2] = new iCubFinger("right_middle");
        icub_kin_finger_[0]->setAllConstraints(false);
        icub_kin_finger_[1]->setAllConstraints(false);
        icub_kin_finger_[2]->setAllConstraints(false);

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
        unsigned int k = 0;
        MatrixXf init_particle;
        VectorXf init_weight;
        double cam_x[3];
        double cam_o[4];
        int num_particle = 50;

        init_weight.resize(num_particle, 1);
        init_weight.setConstant(1.0/num_particle);

        init_particle.resize(6, num_particle);

        Vector q = readRootToEE();
        Vector ee_pose = icub_kin_arm_->EndEffPose(CTRL_DEG2RAD * q);

        Map<VectorXd> q_arm(ee_pose.data(), 6, 1);
        q_arm.tail(3) *= ee_pose(6);
        for (int i = 0; i < num_particle; ++i)
        {
            init_particle.col(i) = q_arm.cast<float>();
        }

        /* FILTERING */
        ImageOf<PixelRgb> * imgin = YARP_NULLPTR;
        while(is_running_)
        {
            if (imgin == YARP_NULLPTR) imgin = port_image_in_.read(true);
//            imgin = port_image_in_.read(true);
            if (imgin != YARP_NULLPTR)
            {
                ImageOf<PixelRgb> & imgout = port_image_out_.prepare();
                imgout = *imgin;

                MatrixXf temp_particle(6, num_particle);
                VectorXf temp_weight(num_particle, 1);
                VectorXf temp_parent(num_particle, 1);

                MatrixXf measurement;
                Mat img_back = cvarrToMat(imgout.getIplImage());

                Vector eye_pose = icub_kin_eye_->EndEffPose(CTRL_DEG2RAD * readRootToEye("left"));
                cam_x[0] = eye_pose(0); cam_x[1] = eye_pose(1); cam_x[2] = eye_pose(2);
                cam_o[0] = eye_pose(3); cam_o[1] = eye_pose(4); cam_o[2] = eye_pose(5); cam_o[3] = eye_pose(6);

                VectorXf sorted_pred = init_weight;
                std::sort(sorted_pred.data(), sorted_pred.data() + sorted_pred.size());
                float threshold = sorted_pred.tail(6)(0);
                for (int i = 0; i < num_particle; ++i)
                {
                    if(init_weight(i) <= threshold) ht_pf_f_->Prediction(init_particle.col(i), init_particle.col(i));
                }

                /* Set parameters */
                // FIXME: da decidere come sistemare
                Vector q = readArm();
                ht_pf_f_->setArmJoints(q);
                ht_pf_f_->setCamXO(cam_x, cam_o);
                ht_pf_f_->setImgBackEdge(img_back);
                /* ************** */

                for (int i = 0; i < num_particle; ++i)
                {
                    ht_pf_f_->Correction(init_particle.col(i), img_back, init_weight.row(i));
                }

                init_weight = init_weight / init_weight.sum();

                MatrixXf out_particle(6, 1);
                ht_pf_f_->WeightedSum(init_particle, init_weight, out_particle);
//                ht_pf_f_->Mode(init_particle, init_weight, out_particle);

//                VectorXf sorted = init_weight;
//                std::sort(sorted.data(), sorted.data() + sorted.size());
//                std::cout <<  sorted << std::endl;
                std::cout << "Step: " << ++k << std::endl;
                std::cout << "Neff: " << ht_pf_f_->Neff(init_weight) << std::endl;
                if (ht_pf_f_->Neff(init_weight) < 15)
                {
                    std::cout << "Resampling!" << std::endl;

                    ht_pf_f_->Resampling(init_particle, init_weight, temp_particle, temp_weight, temp_parent);

                    init_particle = temp_particle;
                    init_weight   = temp_weight;
                }

                /* DEBUG ONLY */
                // FIXME: out_particle is treatead as a Vector, but it's a Matrix.
                SuperImpose::ObjPoseMap hand_pose;
                SuperImpose::ObjPose    pose;
                Vector ee_o(4);
                float ang;

                ang     = out_particle.col(0).tail(3).norm();
                ee_o(0) = out_particle(3) / ang;
                ee_o(1) = out_particle(4) / ang;
                ee_o(2) = out_particle(5) / ang;
                ee_o(3) = ang;

                pose.assign(out_particle.data(), out_particle.data()+3);
                pose.insert(pose.end(), ee_o.data(), ee_o.data()+4);
                hand_pose.emplace("palm", pose);

                Vector ee_t(3, pose.data());
                ee_t.push_back(1.0);
                YMatrix Ha = axis2dcm(ee_o);
                Ha.setCol(3, ee_t);
                // FIXME: middle finger only!
                for (size_t fng = 2; fng < 3; ++fng)
                {
                    std::string finger_s;
                    pose.clear();
                    if (fng != 0)
                    {
                        Vector j_x = (Ha * (icub_kin_finger_[fng]->getH0().getCol(3))).subVector(0, 2);
                        Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng]->getH0());

                        if      (fng == 1) { finger_s = "index0"; }
                        else if (fng == 2) { finger_s = "medium0"; }

                        pose.assign(j_x.data(), j_x.data()+3);
                        pose.insert(pose.end(), j_o.data(), j_o.data()+4);
                        hand_pose.emplace(finger_s, pose);
                    }

                    for (size_t i = 0; i < icub_kin_finger_[fng]->getN(); ++i)
                    {
                        Vector j_x = (Ha * (icub_kin_finger_[fng]->getH(i, true).getCol(3))).subVector(0, 2);
                        Vector j_o = dcm2axis(Ha * icub_kin_finger_[fng]->getH(i, true));

                        if      (fng == 0) { finger_s = "thumb"+std::to_string(i+1); }
                        else if (fng == 1) { finger_s = "index"+std::to_string(i+1); }
                        else if (fng == 2) { finger_s = "medium"+std::to_string(i+1); }
                        
                        pose.assign(j_x.data(), j_x.data()+3);
                        pose.insert(pose.end(), j_o.data(), j_o.data()+4);
                        hand_pose.emplace(finger_s, pose);
                    }
                }

                ht_pf_f_->Superimpose(hand_pose, img_back);

                port_image_out_.write();
            }
        }
        // FIXME: queste close devono andare da un'altra parte. Simil RFModule.
        port_image_in_.close();
        port_head_enc.close();
        port_arm_enc.close();
        port_torso_enc.close();
        port_image_out_.close();
    }


    void getResult() {}


    std::future<void> spawn()
    {
        is_running_ = true;
        return std::async(std::launch::async, &HTSIRParticleFilter::runFilter, this);
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

#endif

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    /* When a user presses the escape key, we set the WindowShouldClose property to true, closing the application. */
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GL_TRUE);
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

    namedWindow("Superimposed Edges", WINDOW_NORMAL | WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

    /* Initialize OpenGL context */
    GLFWwindow * window = nullptr;
    if (!openglSetUp(window, WINDOW_WIDTH, WINDOW_HEIGHT)) return EXIT_FAILURE;

//    HTSIRParticleFilter ht_sir_pf;
//    ht_sir_pf.setOGLWindow(window);
//    if (!ht_sir_pf.Configure()) return EXIT_FAILURE;
//
//    std::future<void> thr_vpf = ht_sir_pf.spawn();
//    while (ht_sir_pf.isRunning())
//    {
//        if (glfwWindowShouldClose(window))
//        {
//            std::chrono::milliseconds span(1);
//            ht_sir_pf.stopThread();
//            yInfo() << log_ID << "Joining filthering thread...";
//            while (thr_vpf.wait_for(span) == std::future_status::timeout) glfwPollEvents();
//        }
//        else glfwPollEvents();
//    }

    std::shared_ptr<BrownianMotion> brown(new BrownianMotion());
    std::shared_ptr<ParticleFilterPrediction> pf_prediction(new ParticleFilterPrediction(brown));
    std::shared_ptr<Proprioception> proprio(new Proprioception(window));
    std::shared_ptr<VisualParticleFilterCorrection> vpf_correction(new VisualParticleFilterCorrection(proprio));
    std::shared_ptr<Resampling> resampling(new Resampling());
    VisualSIRParticleFilter vsir_pf(brown, pf_prediction, proprio, vpf_correction, resampling);

    std::future<void> thr_vpf = vsir_pf.spawn();
    while (vsir_pf.isRunning())
    {
        if (glfwWindowShouldClose(window))
        {
            std::chrono::milliseconds span(1);
            vsir_pf.stopThread();
            yInfo() << log_ID << "Joining filthering thread...";
            while (thr_vpf.wait_for(span) == std::future_status::timeout) glfwPollEvents();
        }
        else glfwPollEvents();
    }

    glfwMakeContextCurrent(NULL);
    glfwTerminate();

    yInfo() << log_ID << "Main returning.";
    yInfo() << log_ID << "Application closed.";

    return EXIT_SUCCESS;
}
