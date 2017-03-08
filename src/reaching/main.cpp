#include <cmath>
#include <iostream>

#include <iCub/ctrl/minJerkCtrl.h>
#include <iCub/iKin/iKinFwd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <SuperImpose/SISkeleton.h>
#include <yarp/dev/CartesianControl.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/math/Math.h>
#include <yarp/math/SVD.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/Property.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/Time.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Matrix.h>
#include <yarp/sig/Vector.h>

using namespace yarp::dev;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;
using namespace iCub::ctrl;
using namespace iCub::iKin;


class RFMReaching : public RFModule
{
public:
    bool configure(ResourceFinder &rf)
    {
        if (!port_estimates_in_.open("/reaching/estimates:i"))
        {
            yError() << "Could not open /reaching/estimates:i port! Closing.";
            return false;
        }

        if (!port_image_left_in_.open("/reaching/cam_left/img:i"))
        {
            yError() << "Could not open /reaching/cam_left/img:i port! Closing.";
            return false;
        }

        if (!port_image_left_out_.open("/reaching/cam_left/img:o"))
        {
            yError() << "Could not open /reaching/cam_left/img:o port! Closing.";
            return false;
        }

        if (!port_click_left_.open("/reaching/cam_left/click:i"))
        {
            yError() << "Could not open /reaching/cam_left/click:in port! Closing.";
            return false;
        }

        if (!port_image_right_in_.open("/reaching/cam_right/img:i"))
        {
            yError() << "Could not open /reaching/cam_right/img:i port! Closing.";
            return false;
        }

        if (!port_image_right_out_.open("/reaching/cam_right/img:o"))
        {
            yError() << "Could not open /reaching/cam_right/img:o port! Closing.";
            return false;
        }

        if (!port_click_right_.open("/reaching/cam_right/click:i"))
        {
            yError() << "Could not open /reaching/cam_right/click:i port! Closing.";
            return false;
        }

        if (!port_px_left_endeffector.open("/reaching/cam_left/x:o"))
        {
            yError() << "Could not open /reaching/cam_left/x:o port! Closing.";
            return false;
        }

        if (!port_px_right_endeffector.open("/reaching/cam_right/x:o"))
        {
            yError() << "Could not open /reaching/cam_right/x:o port! Closing.";
            return false;
        }

        if (!setGazeController()) return false;

        if (!setTorsoRemoteControlboard()) return false;

        if (!setRightArmRemoteControlboard()) return false;

        if (!setRightArmCartesianController()) return false;

        Bottle btl_cam_info;
        itf_gaze_->getInfo(btl_cam_info);
        yInfo() << "[CAM INFO]" << btl_cam_info.toString();
        Bottle* cam_left_info = btl_cam_info.findGroup("camera_intrinsics_left").get(1).asList();
        Bottle* cam_right_info = btl_cam_info.findGroup("camera_intrinsics_right").get(1).asList();

        float left_fx  = static_cast<float>(cam_left_info->get(0).asDouble());
        float left_cx  = static_cast<float>(cam_left_info->get(2).asDouble());
        float left_fy  = static_cast<float>(cam_left_info->get(5).asDouble());
        float left_cy  = static_cast<float>(cam_left_info->get(6).asDouble());

        left_proj_ = zeros(3, 4);
        left_proj_(0, 0) = left_fx;
        left_proj_(0, 2) = left_cx;
        left_proj_(1, 1) = left_fy;
        left_proj_(1, 2) = left_cy;
        left_proj_(2, 2) = 1.0;

        yInfo() << "left_proj_ =\n" << left_proj_.toString();

        float right_fx = static_cast<float>(cam_right_info->get(0).asDouble());
        float right_cx = static_cast<float>(cam_right_info->get(2).asDouble());
        float right_fy = static_cast<float>(cam_right_info->get(5).asDouble());
        float right_cy = static_cast<float>(cam_right_info->get(6).asDouble());

        right_proj_ = zeros(3, 4);
        right_proj_(0, 0) = right_fx;
        right_proj_(0, 2) = right_cx;
        right_proj_(1, 1) = right_fy;
        right_proj_(1, 2) = right_cy;
        right_proj_(2, 2) = 1.0;

        yInfo() << "right_proj_ =\n" << right_proj_.toString();


        Vector left_eye_x;
        Vector left_eye_o;
        itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

        Vector right_eye_x;
        Vector right_eye_o;
        itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

        yInfo() << "left_eye_o =" << left_eye_o.toString();
        yInfo() << "right_eye_o =" << right_eye_o.toString();


        Matrix l_H_eye = axis2dcm(left_eye_o);
        left_eye_x.push_back(1.0);
        l_H_eye.setCol(3, left_eye_x);
        Matrix l_H_r_to_eye = SE3inv(l_H_eye);

        Matrix r_H_eye = axis2dcm(right_eye_o);
        right_eye_x.push_back(1.0);
        r_H_eye.setCol(3, right_eye_x);
        Matrix r_H_r_to_eye = SE3inv(r_H_eye);

        yInfo() << "l_H_r_to_eye =\n" << l_H_r_to_eye.toString();
        yInfo() << "r_H_r_to_eye =\n" << r_H_r_to_eye.toString();

        l_H_r_to_cam_ = left_proj_  * l_H_r_to_eye;
        r_H_r_to_cam_ = right_proj_ * r_H_r_to_eye;

        yInfo() << "l_H_r_to_cam_ =\n" << l_H_r_to_cam_.toString();
        yInfo() << "r_H_r_to_cam_ =\n" << r_H_r_to_cam_.toString();


        l_si_skel_ = new SISkeleton(left_fx,  left_fy,  left_cx,  left_cy);
        r_si_skel_ = new SISkeleton(right_fx, right_fy, right_cx, right_cy);

        icub_index_ = iCubFinger("right_index");
        std::deque<IControlLimits*> temp_lim;
        temp_lim.push_front(itf_fingers_lim_);
        if (!icub_index_.alignJointsBounds(temp_lim))
        {
            yError() << "Cannot set joint bound for index finger.";
            return false;
        }
        
        handler_port_.open("/reaching/cmd:i");
        attach(handler_port_);
        
        return true;
    }

    
    double getPeriod() { return 0; }


    bool updateModule()
    {
        while (!take_estimates_);

        /* Get the initial end-effector pose from hand-tracking */
        Vector* estimates = port_estimates_in_.read(true);

        /* Get the initial end-effector pose from the cartesian controller */
//        Vector pose(12);
//        Vector pose_x;
//        Vector pose_o;
//        itf_rightarm_cart_->getPose(pose_x, pose_o);
//        pose_x.push_back(1.0);
//        Matrix Ha = axis2dcm(pose_o);
//        Ha.setCol(3, pose_x);
//
//        Vector init_chain_joints;
//        icub_index_.getChainJoints(readRootToFingers().subVector(3, 18), init_chain_joints);
//        Vector init_tip_pose_index = (Ha * icub_index_.getH((M_PI/180.0) * init_chain_joints).getCol(3)).subVector(0, 2);
//
//        pose.setSubvector(0,  init_tip_pose_index);
//        pose.setSubvector(3,  zeros(3));
//        pose.setSubvector(6,  init_tip_pose_index);
//        pose.setSubvector(9,  zeros(3));
//
//        Vector* estimates = &pose;
        /* **************************************************** */


        if (should_stop_) return false;

        yInfo() << "estimates = ["  << estimates->toString() << "]";

        Vector px_img_left;
        px_img_left.push_back(l_px_location_[0]);
        px_img_left.push_back(l_px_location_[1]);

        yInfo() << "px_img_left = [" << px_img_left.toString() << "]";

        Vector px_img_right;
        px_img_right.push_back(r_px_location_[0]);
        px_img_right.push_back(r_px_location_[1]);

        yInfo() << "px_img_right = [" << px_img_right.toString() << "]";

        Vector px_des;
        px_des.push_back(px_img_left[0]);   /* u_l */
        px_des.push_back(px_img_right[0]);  /* u_r */
        px_des.push_back(px_img_left[1]);   /* v_l */

        yInfo() << "px_des = ["  << px_des.toString() << "]";


        Vector px_ee_left;  /* u_ee_l, v_ee_l */
        itf_gaze_->get2DPixel(LEFT,  estimates->subVector(0, 2), px_ee_left);
        yInfo() << "estimates(0, 2) = ["  << estimates->subVector(0, 2).toString() << "]";
        yInfo() << "px_ee_left = ["  << px_ee_left.toString() << "]";


        Vector px_ee_right; /* u_ee_r, v_ee_r */
        itf_gaze_->get2DPixel(RIGHT, estimates->subVector(6, 8), px_ee_right);
        yInfo() << "estimates(6, 8) = ["  << estimates->subVector(6, 8).toString() << "]";
        yInfo() << "px_ee_right = [" << px_ee_right.toString() << "]";


        Vector left_eye_x;
        Vector left_eye_o;
        itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

        Vector l_px = (l_H_r_to_cam_.submatrix(0, 2, 0, 2) * (estimates->subVector(0, 2) - left_eye_x));
        yInfo() << "Proj left ee = [" << (l_px.subVector(0, 1) / l_px[2]).toString() << "]";

        Vector right_eye_x;
        Vector right_eye_o;
        itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

        Vector r_px = (r_H_r_to_cam_.submatrix(0, 2, 0, 2) * (estimates->subVector(6, 8) - right_eye_x));
        yInfo() << "Proj right ee = [" << (r_px.subVector(0, 1) / r_px[2]).toString() << "]";

        Vector px_ee_now;
        px_ee_now.push_back(l_px [0] / l_px[2]);    /* u_ee_l */
        px_ee_now.push_back(r_px[0] / r_px[2]);     /* u_ee_r */
        px_ee_now.push_back(l_px [1] / l_px[2]);    /* v_ee_l */
        yInfo() << "px_ee_now = [" << px_ee_now.toString() << "]";


        /* Jacobian */
        Matrix jacobian(3, 3);

        Vector l_ee_x = estimates->subVector(0, 2);
        l_ee_x.push_back(1.0);

        double l_num_u     = dot(l_H_r_to_cam_.subrow(0, 0, 4), l_ee_x);
        double l_num_v     = dot(l_H_r_to_cam_.subrow(1, 0, 4), l_ee_x);
        double l_lambda    = dot(l_H_r_to_cam_.subrow(2, 0, 4), l_ee_x);
        double l_lambda_sq = pow(l_lambda, 2.0);

        jacobian(0, 0) = (l_H_r_to_cam_(0, 0) * l_lambda - l_H_r_to_cam_(2, 0) * l_num_u) / l_lambda_sq;
        jacobian(0, 1) = (l_H_r_to_cam_(0, 1) * l_lambda - l_H_r_to_cam_(2, 1) * l_num_u) / l_lambda_sq;
        jacobian(0, 2) = (l_H_r_to_cam_(0, 2) * l_lambda - l_H_r_to_cam_(2, 2) * l_num_u) / l_lambda_sq;

        jacobian(2, 0) = (l_H_r_to_cam_(1, 0) * l_lambda - l_H_r_to_cam_(2, 0) * l_num_v) / l_lambda_sq;
        jacobian(2, 1) = (l_H_r_to_cam_(1, 1) * l_lambda - l_H_r_to_cam_(2, 1) * l_num_v) / l_lambda_sq;
        jacobian(2, 2) = (l_H_r_to_cam_(1, 2) * l_lambda - l_H_r_to_cam_(2, 2) * l_num_v) / l_lambda_sq;

        Vector r_ee_x = estimates->subVector(6, 8);
        r_ee_x.push_back(1.0);

        double r_num_u     = dot(r_H_r_to_cam_.subrow(0, 0, 4), r_ee_x);
//        double r_num_v     = dot(r_H_r_to_cam_.subrow(1, 0, 4), r_ee_x);
        double r_lambda    = dot(r_H_r_to_cam_.subrow(2, 0, 4), r_ee_x);
        double r_lambda_sq = pow(r_lambda, 2.0);

        jacobian(1, 0) = (r_H_r_to_cam_(0, 0) * r_lambda - r_H_r_to_cam_(2, 0) * r_num_u) / r_lambda_sq;
        jacobian(1, 1) = (r_H_r_to_cam_(0, 1) * r_lambda - r_H_r_to_cam_(2, 1) * r_num_u) / r_lambda_sq;
        jacobian(1, 2) = (r_H_r_to_cam_(0, 2) * r_lambda - r_H_r_to_cam_(2, 2) * r_num_u) / r_lambda_sq;


        double Ts    = 0.05;    // controller's sample time [s]
        double K     = 1;       // how long it takes to move to the target [s]
        double v_max = 0.005;   // max cartesian velocity [m/s]

        bool done = false;
        while (!should_stop_ && !done)
        {
            Vector e = px_des - px_ee_now;

            yInfo() << "e = [" << e.toString() << "]";

            jacobian(0, 0) *= -2.0 * e[0];
            jacobian(0, 1) *= -2.0 * e[0];
            jacobian(0, 2) *= -2.0 * e[0];

            jacobian(1, 0) *= -2.0 * e[1];
            jacobian(1, 1) *= -2.0 * e[1];
            jacobian(1, 2) *= -2.0 * e[1];

            jacobian(2, 0) *= -2.0 * e[2];
            jacobian(2, 1) *= -2.0 * e[2];
            jacobian(2, 2) *= -2.0 * e[2];

            Matrix inv_jacobian = luinv(jacobian);

            e[0] *= e[0];
            e[1] *= e[1];
            e[2] *= e[2];
            Vector vel_x = -1.0 * K * inv_jacobian * e;

            yInfo() << "vel_x = [" << vel_x.toString() << "]";

            /* Enforce velocity bounds */
            for (size_t i = 0; i < vel_x.length(); ++i)
                vel_x[i] = sign(vel_x[i]) * std::min(v_max, std::fabs(vel_x[i]));

            yInfo() << "bounded vel_x = [" << vel_x.toString() << "]";
            yInfo() << "px_des = [" << px_des.toString() << "]";
            yInfo() << "px_ee_now = [" << px_ee_now.toString() << "]";

            itf_rightarm_cart_->setTaskVelocities(vel_x, Vector(4, 0.0));

            yInfo() << "Pixel error norm (0): " << std::abs(px_des(0) - px_ee_now(0));
            yInfo() << "Pixel error norm (1): " << std::abs(px_des(1) - px_ee_now(1));
            yInfo() << "Pixel error norm (2): " << std::abs(px_des(2) - px_ee_now(2));
            yInfo() << "Poistion error norm: " << norm(vel_x);

            Time::delay(Ts);

            done = ((std::abs(px_des(0) - px_ee_now(0)) < 1.0) && (std::abs(px_des(1) - px_ee_now(1)) < 1.0) && (std::abs(px_des(2) - px_ee_now(2)) < 1.0));
            if (done)
            {
                yInfo() << "\npx_des =" << px_des.toString() << "px_ee_now =" << px_ee_now.toString();
                yInfo() << "Terminating!\n";
            }
            else
            {
                /* Get the new end-effector pose from hand-tracking */
                estimates = port_estimates_in_.read(true);

                /* Get the initial end-effector pose from the cartesian controller */
//                itf_rightarm_cart_->getPose(pose_x, pose_o);
//                pose_x.push_back(1.0);
//                Ha = axis2dcm(pose_o);
//                Ha.setCol(3, pose_x);
//
//                Vector chain_joints;
//                icub_index_.getChainJoints(readRootToFingers().subVector(3, 18), chain_joints);
//                Vector tip_pose_index = (Ha * icub_index_.getH((M_PI/180.0) * chain_joints).getCol(3)).subVector(0, 2);
//
//                pose.setSubvector(0,  tip_pose_index);
//                pose.setSubvector(3,  zeros(3));
//                pose.setSubvector(6,  tip_pose_index);
//                pose.setSubvector(9,  zeros(3));

                /* Simulate reaching starting from the initial position */
                /* Comment any previous write on variable 'estimates' */
//                yInfo() << "EE L now: " << estimates->subVector(0, 2).toString();
//                yInfo() << "EE R now: " << estimates->subVector(6, 8).toString() << "\n";
//
//                estimates->setSubvector(0, estimates->subVector(0, 2) + vel_x);
//                estimates->setSubvector(6, estimates->subVector(6, 8) + vel_x);
                /* **************************************************** */

                yInfo() << "EE L cor: " << estimates->subVector(0, 2).toString();
                yInfo() << "EE R cor: " << estimates->subVector(6, 8).toString() << "\n";

                l_px = (l_H_r_to_cam_.submatrix(0, 2, 0, 2) * (estimates->subVector(0, 2) - left_eye_x));
                r_px = (r_H_r_to_cam_.submatrix(0, 2, 0, 2) * (estimates->subVector(6, 8) - right_eye_x));

                l_px[0] /= l_px[2];
                l_px[1] /= l_px[2];

                r_px[0] /= r_px[2];
                r_px[1] /= r_px[2];

                px_ee_now[0] = l_px [0];    /* u_ee_l */
                px_ee_now[1] = r_px[0];     /* u_ee_r */
                px_ee_now[2] = l_px [1];    /* v_ee_l */


                /* Dump pixel coordinates of the end-effector */
                Bottle& l_px_endeffector = port_px_left_endeffector.prepare();
                l_px_endeffector.clear();
                l_px_endeffector.addInt(l_px[0]);
                l_px_endeffector.addInt(l_px[1]);
                port_px_left_endeffector.write();

                Bottle& r_px_endeffector = port_px_right_endeffector.prepare();
                r_px_endeffector.clear();
                r_px_endeffector.addInt(r_px[0]);
                r_px_endeffector.addInt(r_px[1]);
                port_px_right_endeffector.write();


                /* Update Jacobian */
                jacobian(3, 3);
                l_ee_x = estimates->subVector(0, 2);
                l_ee_x.push_back(1.0);

                l_num_u     = dot(l_H_r_to_cam_.subrow(0, 0, 4), l_ee_x);
                l_num_v     = dot(l_H_r_to_cam_.subrow(1, 0, 4), l_ee_x);
                l_lambda    = dot(l_H_r_to_cam_.subrow(2, 0, 4), l_ee_x);
                l_lambda_sq = pow(l_lambda, 2.0);

                jacobian(0, 0) = (l_H_r_to_cam_(0, 0) * l_lambda - l_H_r_to_cam_(2, 0) * l_num_u) / l_lambda_sq;
                jacobian(0, 1) = (l_H_r_to_cam_(0, 1) * l_lambda - l_H_r_to_cam_(2, 1) * l_num_u) / l_lambda_sq;
                jacobian(0, 2) = (l_H_r_to_cam_(0, 2) * l_lambda - l_H_r_to_cam_(2, 2) * l_num_u) / l_lambda_sq;

                jacobian(2, 0) = (l_H_r_to_cam_(1, 0) * l_lambda - l_H_r_to_cam_(2, 0) * l_num_v) / l_lambda_sq;
                jacobian(2, 1) = (l_H_r_to_cam_(1, 1) * l_lambda - l_H_r_to_cam_(2, 1) * l_num_v) / l_lambda_sq;
                jacobian(2, 2) = (l_H_r_to_cam_(1, 2) * l_lambda - l_H_r_to_cam_(2, 2) * l_num_v) / l_lambda_sq;

                r_ee_x = estimates->subVector(6, 8);
                r_ee_x.push_back(1.0);

                r_num_u     = dot(r_H_r_to_cam_.subrow(0, 0, 4), r_ee_x);
//                r_num_v     = dot(r_H_r_to_cam_.subrow(1, 0, 4), r_ee_x);
                r_lambda    = dot(r_H_r_to_cam_.subrow(2, 0, 4), r_ee_x);
                r_lambda_sq = pow(r_lambda, 2.0);

                jacobian(1, 0) = (r_H_r_to_cam_(0, 0) * r_lambda - r_H_r_to_cam_(2, 0) * r_num_u) / r_lambda_sq;
                jacobian(1, 1) = (r_H_r_to_cam_(0, 1) * r_lambda - r_H_r_to_cam_(2, 1) * r_num_u) / r_lambda_sq;
                jacobian(1, 2) = (r_H_r_to_cam_(0, 2) * r_lambda - r_H_r_to_cam_(2, 2) * r_num_u) / r_lambda_sq;


                /* Left eye end-effector superimposition */
                ImageOf<PixelRgb>* l_imgin  = port_image_left_in_.read(true);
                ImageOf<PixelRgb>& l_imgout = port_image_left_out_.prepare();
                l_imgout = *l_imgin;
                cv::Mat l_img = cv::cvarrToMat(l_imgout.getIplImage());

                cv::circle(l_img, cv::Point(l_px[0],      l_px[1]),      4, cv::Scalar(0, 255, 0), 4);
                cv::circle(l_img, cv::Point(l_px_location_[0], l_px_location_[1]), 4, cv::Scalar(0, 255, 0), 4);

                port_image_left_out_.write();

                /* Right eye end-effector superimposition */
                ImageOf<PixelRgb>* r_imgin  = port_image_right_in_.read(true);
                ImageOf<PixelRgb>& r_imgout = port_image_right_out_.prepare();
                r_imgout = *r_imgin;
                cv::Mat r_img = cv::cvarrToMat(r_imgout.getIplImage());

                cv::circle(r_img, cv::Point(r_px[0],     r_px[1]),     4, cv::Scalar(0, 255, 0), 4);
                cv::circle(r_img, cv::Point(r_px_location_[0], r_px_location_[1]), 4, cv::Scalar(0, 255, 0), 4);

                port_image_right_out_.write();
            }
        }

        itf_rightarm_cart_->stopControl();

        Time::delay(0.5);

        return false;
    }


    bool respond(const Bottle& command, Bottle& reply)
    {
        int cmd = command.get(0).asVocab();
        switch (cmd)
        {
            /* Safely close the application */
            case VOCAB4('q','u','i','t'):
            {
                itf_rightarm_cart_->stopControl();

                take_estimates_ = true;
                should_stop_    = true;

                this->interruptModule();

                reply = command;

                break;
            }
            /* Take pixel coordinates from the left and right camera images */
            /* PLUS: Compute again the roto-translation and projection matrices from root to left and right camera planes */
            case VOCAB3('i','m','g'):
            {
                Vector left_eye_x;
                Vector left_eye_o;
                itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

                Vector right_eye_x;
                Vector right_eye_o;
                itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

                yInfo() << "left_eye_o =" << left_eye_o.toString();
                yInfo() << "right_eye_o =" << right_eye_o.toString();


                Matrix l_H_eye = axis2dcm(left_eye_o);
                left_eye_x.push_back(1.0);
                l_H_eye.setCol(3, left_eye_x);
                Matrix l_H_r_to_eye = SE3inv(l_H_eye);

                Matrix r_H_eye = axis2dcm(right_eye_o);
                right_eye_x.push_back(1.0);
                r_H_eye.setCol(3, right_eye_x);
                Matrix r_H_r_to_eye = SE3inv(r_H_eye);

                yInfo() << "l_H_r_to_eye =\n" << l_H_r_to_eye.toString();
                yInfo() << "r_H_r_to_eye =\n" << r_H_r_to_eye.toString();


                l_H_r_to_cam_ = left_proj_  * l_H_r_to_eye;
                r_H_r_to_cam_ = right_proj_ * r_H_r_to_eye;


                Bottle* click_left  = port_click_left_.read  (true);
                Bottle* click_right = port_click_right_.read (true);

                l_px_location_.resize(2);
                l_px_location_[0] = click_left->get(0).asDouble();
                l_px_location_[1] = click_left->get(1).asDouble();

                r_px_location_.resize(2);
                r_px_location_[0] = click_right->get(0).asDouble();
                r_px_location_[1] = click_right->get(1).asDouble();

                reply = command;

                break;
            }
            /* Set a fixed goal in pixel coordinates */
            /* PLUS: Compute again the roto-translation and projection matrices from root to left and right camera planes */
            case VOCAB4('g', 'o', 'a', 'l'):
            {
                Vector left_eye_x;
                Vector left_eye_o;
                itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

                Vector right_eye_x;
                Vector right_eye_o;
                itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

                yInfo() << "left_eye_o =" << left_eye_o.toString();
                yInfo() << "right_eye_o =" << right_eye_o.toString();


                Matrix l_H_eye = axis2dcm(left_eye_o);
                left_eye_x.push_back(1.0);
                l_H_eye.setCol(3, left_eye_x);
                Matrix l_H_r_to_eye = SE3inv(l_H_eye);

                Matrix r_H_eye = axis2dcm(right_eye_o);
                right_eye_x.push_back(1.0);
                r_H_eye.setCol(3, right_eye_x);
                Matrix r_H_r_to_eye = SE3inv(r_H_eye);

                yInfo() << "l_H_r_to_eye =\n" << l_H_r_to_eye.toString();
                yInfo() << "r_H_r_to_eye =\n" << r_H_r_to_eye.toString();

                l_H_r_to_cam_ = left_proj_  * l_H_r_to_eye;
                r_H_r_to_cam_ = right_proj_ * r_H_r_to_eye;


                l_px_location_.resize(2);
                l_px_location_[0] = 125;
                l_px_location_[1] = 135;

                r_px_location_.resize(2);
                r_px_location_[0] = 89;
                r_px_location_[1] = 135;

                reply = command;

                break;
            }
            /* Start reaching phase */
            case VOCAB2('g','o'):
            {
                reply = command;
                take_estimates_ = true;

                break;
            }
            /* Get 3D information from the OPC of IOL */
            case VOCAB3('o', 'p', 'c'):
            {
                Network yarp;
                Bottle cmd;
                Bottle rep;

                RpcClient port_memory;
                port_memory.open("/reaching/tomemory");
                yarp.connect("/reaching/tomemory", "/memory/rpc");

                cmd.addVocab(Vocab::encode("ask"));
                Bottle &content = cmd.addList().addList();
                content.addString("name");
                content.addString("==");
                content.addString("Duck");

                port_memory.write(cmd, rep);

                if (rep.size()>1)
                {
                    if (rep.get(0).asVocab() == Vocab::encode("ack"))
                    {
                        if (Bottle *idField = rep.get(1).asList())
                        {
                            if (Bottle *idValues = idField->get(1).asList())
                            {
                                int id = idValues->get(0).asInt();

                                cmd.clear();

                                cmd.addVocab(Vocab::encode("get"));
                                Bottle& content = cmd.addList();
                                Bottle& list_bid = content.addList();

                                list_bid.addString("id");
                                list_bid.addInt(id);

                                Bottle& list_propSet = content.addList();
                                list_propSet.addString("propSet");

                                Bottle& list_items = list_propSet.addList();
                                list_items.addString("position_2d_left");

                                Bottle reply_prop;
                                port_memory.write(cmd, reply_prop);

                                if (reply_prop.get(0).asVocab() == Vocab::encode("ack"))
                                {
                                    if (Bottle* propField = reply_prop.get(1).asList())
                                    {
                                        if (Bottle* position_2d = propField->find("position_2d_left").asList())
                                        {
                                            if (position_2d->size() == 4)
                                            {
                                                l_px_location_.resize(2);
                                                l_px_location_[0] = position_2d->get(0).asDouble() + (position_2d->get(2).asDouble() - position_2d->get(0).asDouble()) / 2;
                                                l_px_location_[1] = position_2d->get(1).asDouble() + (position_2d->get(3).asDouble() - position_2d->get(1).asDouble()) / 2;

                                                RpcClient port_sfm;
                                                port_sfm.open("/reaching/tosfm");
                                                yarp.connect("/reaching/tosfm", "/SFM/rpc");

                                                cmd.clear();

                                                cmd.addInt(l_px_location_[0]);
                                                cmd.addInt(l_px_location_[1]);

                                                Bottle reply_pos;
                                                port_sfm.write(cmd, reply_pos);

                                                if (reply_pos.size() == 5)
                                                {
                                                    location_.resize(3);
                                                    location_[0] = reply_pos.get(0).asDouble();
                                                    location_[1] = reply_pos.get(1).asDouble();
                                                    location_[2] = reply_pos.get(2).asDouble();

                                                    r_px_location_.resize(2);
                                                    r_px_location_[0] = reply_pos.get(3).asDouble();
                                                    r_px_location_[1] = reply_pos.get(4).asDouble();
                                                }
                                                else
                                                {
                                                    reply.addString("nack_9");
                                                }

                                                yarp.disconnect("/reaching/tosfm", "/SFM/rpc");
                                                port_sfm.close();
                                            }
                                            else
                                            {
                                                reply.addString("nack_8");
                                            }
                                        }
                                        else
                                        {
                                            reply.addString("nack_7");
                                        }
                                    }
                                    else
                                    {
                                        reply.addString("nack_6");
                                    }
                                }
                                else
                                {
                                    reply.addString("nack_5");
                                }
                            }
                            else
                            {
                                reply.addString("nack_4");
                            }
                        }
                        else
                        {
                            reply.addString("nack_3");
                        }
                    }
                    else
                    {
                        reply.addString("nack_2");
                    }
                }
                else
                {
                    reply.addString("nack_1");
                }
                yarp.disconnect("/reaching/tomemory", "/memory/rpc");
                port_memory.close();

                yInfo() << "l_px_location: " << l_px_location_.toString();
                yInfo() << "r_px_location: " << r_px_location_.toString();
                yInfo() << "location: " << location_.toString();

                reply.addString("ack");

                break;
            }
            /* Set a fixed (hard-coded) 3D position for open-loop reaching */
            case VOCAB3('p', 'o', 's'):
            {
                Bottle  cmd;

                location_.resize(3);
                location_[0] = -0.412;
                location_[1] =  0.0435;
                location_[2] =  0.0524;

                yInfo() << "location: " << location_.toString();
                
                reply = command;
                
                break;
            }
            /* Get 3D point from Structure From Motion clicking on the left camera image */
            case VOCAB3('s', 'f', 'm'):
            {
                Network yarp;
                Bottle  cmd;
                Bottle  rep;


                Bottle* click_left  = port_click_left_.read  (true);

                l_px_location_.resize(2);
                l_px_location_[0] = click_left->get(0).asDouble();
                l_px_location_[1] = click_left->get(1).asDouble();


                RpcClient port_sfm;
                port_sfm.open("/reaching/tosfm");
                yarp.connect("/reaching/tosfm", "/SFM/rpc");

                cmd.clear();

                cmd.addInt(l_px_location_[0]);
                cmd.addInt(l_px_location_[1]);

                Bottle reply_pos;
                port_sfm.write(cmd, reply_pos);

                if (reply_pos.size() == 5)
                {
                    location_.resize(3);
                    location_[0] = reply_pos.get(0).asDouble();
                    location_[1] = reply_pos.get(1).asDouble();
                    location_[2] = reply_pos.get(2).asDouble();

                    yInfo() << "location: " << location_.toString();

                    r_px_location_.resize(2);
                    r_px_location_[0] = reply_pos.get(3).asDouble();
                    r_px_location_[1] = reply_pos.get(4).asDouble();
                }
                else
                {
                    reply.addString("nack");
                }

                yarp.disconnect("/reaching/tosfm", "/SFM/rpc");
                port_sfm.close();

                reply = command;

                break;
            }
            /* Go to initial position (open-loop) */
            case VOCAB4('i', 'n', 'i', 't'):
            {
                /* FINGERTIP */
                Matrix Od(3, 3);
                Od(0, 0) = -1.0;
                Od(1, 1) =  1.0;
                Od(2, 2) = -1.0;
                Vector od = dcm2axis(Od);

                /* KARATE */
//                Matrix Od = zeros(3, 3);
//                Od(0, 0) = -1.0;
//                Od(2, 1) = -1.0;
//                Od(1, 2) = -1.0;
//                Vector od = dcm2axis(Od);

                double traj_time = 0.0;
                itf_rightarm_cart_->getTrajTime(&traj_time);

                if (traj_time == traj_time_)
                {
                    /* FINGERTIP */
                    Vector chain_joints;
                    icub_index_.getChainJoints(readRootToFingers().subVector(3, 18), chain_joints);

                    Matrix tip_pose_index = icub_index_.getH((M_PI/180.0) * chain_joints);
                    Vector tip_x = tip_pose_index.getCol(3);
                    Vector tip_o = dcm2axis(tip_pose_index);
                    itf_rightarm_cart_->attachTipFrame(tip_x, tip_o);

                    location_[0] += 0.03;
                    location_[1] += 0.07;
                    location_[2] =  0.12;

                    setTorsoDOF();

                    /* KARATE */
//                    location_[0] += 0.10;
//                    location_[1] += 0.10;
//                    location_[2] =  0.06;

                    Vector gaze_loc(3);
                    gaze_loc(0) = -0.579;
                    gaze_loc(1) =  0.472;
                    gaze_loc(2) = -0.280;


//                    itf_gaze_->lookAtFixationPointSync(gaze_loc);
//                    itf_gaze_->waitMotionDone();

                    int ctxt;
                    itf_rightarm_cart_->storeContext(&ctxt);

                    itf_rightarm_cart_->setLimits(0, 20.0, 20.0);

                    itf_rightarm_cart_->goToPoseSync(location_, od);
                    itf_rightarm_cart_->waitMotionDone();
                    itf_rightarm_cart_->stopControl();

                    itf_rightarm_cart_->restoreContext(ctxt);
                    itf_rightarm_cart_->deleteContext(ctxt);

                    itf_gaze_->lookAtFixationPointSync(gaze_loc);
                    itf_gaze_->waitMotionDone();

                    unsetTorsoDOF();
                    itf_rightarm_cart_->removeTipFrame();

                    reply.addString("ack");
                }
                else
                {
                    reply.addString("nack");
                }
                
                break;
            }
            /* Stop the Cartesian controller */
            case VOCAB4('s', 't', 'o', 'p'):
            {
                if (itf_rightarm_cart_->stopControl())
                {
                    itf_rightarm_cart_->removeTipFrame();

                    reply.addString("ack");
                }
                else
                {
                    reply.addString("nack_2");
                }

                break;
            }
            default:
            {
                reply.addString("nack");
            }
        }

        return true;
    }

    bool interruptModule()
    {
        yInfo() << "Interrupting module.\nPort cleanup...";

        Time::delay(3.0);

        port_estimates_in_.interrupt();
        port_image_left_in_.interrupt();
        port_image_left_out_.interrupt();
        port_click_left_.interrupt();
        port_image_right_in_.interrupt();
        port_image_right_out_.interrupt();
        port_click_right_.interrupt();
        handler_port_.interrupt();

        return true;
    }

    bool close()
    {
        yInfo() << "Calling close functions...";

        port_estimates_in_.close();
        port_image_left_in_.close();
        port_image_left_out_.close();
        port_click_left_.close();
        port_image_right_in_.close();
        port_image_right_out_.close();
        port_click_right_.close();

        itf_rightarm_cart_->removeTipFrame();

        if (rightarm_cartesian_driver_.isValid()) rightarm_cartesian_driver_.close();
        if (gaze_driver_.isValid())               gaze_driver_.close();

        handler_port_.close();

        return true;
    }

private:
    Port                             handler_port_;
    bool                             should_stop_ = false;

    SISkeleton                     * l_si_skel_;
    SISkeleton                     * r_si_skel_;

    BufferedPort<Vector>             port_estimates_in_;

    BufferedPort<ImageOf<PixelRgb>>  port_image_left_in_;
    BufferedPort<ImageOf<PixelRgb>>  port_image_left_out_;
    BufferedPort<Bottle>             port_click_left_;

    BufferedPort<ImageOf<PixelRgb>>  port_image_right_in_;
    BufferedPort<ImageOf<PixelRgb>>  port_image_right_out_;
    BufferedPort<Bottle>             port_click_right_;

    BufferedPort<Bottle>             port_px_left_endeffector;
    BufferedPort<Bottle>             port_px_right_endeffector;

    PolyDriver                       rightarm_cartesian_driver_;
    ICartesianControl              * itf_rightarm_cart_;

    PolyDriver                       gaze_driver_;
    IGazeControl                   * itf_gaze_;

    PolyDriver                       rightarm_remote_driver_;
    IEncoders                      * itf_rightarm_enc_;
    IControlLimits                 * itf_fingers_lim_;

    PolyDriver                       torso_remote_driver_;
    IEncoders                      * itf_torso_enc_;

    iCubFinger                       icub_index_;

    Matrix                           left_proj_;
    Matrix                           right_proj_;
    Matrix                           l_H_r_to_cam_;
    Matrix                           r_H_r_to_cam_;
    Matrix                           px_to_cartesian_;

    double                           traj_time_ = 2.5;
    Vector                           l_px_location_;
    Vector                           r_px_location_;
    Vector                           location_;
    bool                             take_estimates_ = false;

    enum camsel
    {
        LEFT = 0,
        RIGHT = 1
    };

    bool setRightArmCartesianController()
    {
        Property rightarm_cartesian_options;
        rightarm_cartesian_options.put("device", "cartesiancontrollerclient");
        rightarm_cartesian_options.put("local",  "/reaching/cart_right_arm");
        rightarm_cartesian_options.put("remote", "/icub/cartesianController/right_arm");

        rightarm_cartesian_driver_.open(rightarm_cartesian_options);
        if (rightarm_cartesian_driver_.isValid())
        {
            rightarm_cartesian_driver_.view(itf_rightarm_cart_);
            if (!itf_rightarm_cart_)
            {
                yError() << "Error getting ICartesianControl interface.";
                return false;
            }
            yInfo() << "cartesiancontrollerclient succefully opened.";
        }
        else
        {
            yError() << "Error opening cartesiancontrollerclient device.";
            return false;
        }

        if (!itf_rightarm_cart_->setTrajTime(traj_time_))
        {
            yError() << "Error setting ICartesianControl trajectory time.";
            return false;
        }
        yInfo() << "Succesfully set ICartesianControl trajectory time!";

        if(!itf_rightarm_cart_->setInTargetTol(0.01))
        {
            yError() << "Error setting ICartesianControl target tolerance.";
            return false;
        }
        yInfo() << "Succesfully set ICartesianControl target tolerance!";
        
        return true;
    }

    bool setGazeController()
    {
        Property gaze_option;
        gaze_option.put("device", "gazecontrollerclient");
        gaze_option.put("local",  "/reaching/gaze");
        gaze_option.put("remote", "/iKinGazeCtrl");

        gaze_driver_.open(gaze_option);
        if (gaze_driver_.isValid())
        {
            gaze_driver_.view(itf_gaze_);
            if (!itf_gaze_)
            {
                yError() << "Error getting IGazeControl interface.";
                return false;
            }
        }
        else
        {
            yError() << "Gaze control device not available.";
            return false;
        }
        
        return true;
    }

    bool setRightArmRemoteControlboard()
    {
        Property rightarm_remote_options;
        rightarm_remote_options.put("device", "remote_controlboard");
        rightarm_remote_options.put("local",  "/reaching/control_right_arm");
        rightarm_remote_options.put("remote", "/icub/right_arm");

        rightarm_remote_driver_.open(rightarm_remote_options);
        if (rightarm_remote_driver_.isValid())
        {
            yInfo() << "Right arm remote_controlboard succefully opened.";

            rightarm_remote_driver_.view(itf_rightarm_enc_);
            if (!itf_rightarm_enc_)
            {
                yError() << "Error getting right arm IEncoders interface.";
                return false;
            }

            rightarm_remote_driver_.view(itf_fingers_lim_);
            if (!itf_fingers_lim_)
            {
                yError() << "Error getting IControlLimits interface.";
                return false;
            }
        }
        else
        {
            yError() << "Error opening right arm remote_controlboard device.";
            return false;
        }
        
        return true;
    }

    bool setTorsoRemoteControlboard()
    {
        Property torso_remote_options;
        torso_remote_options.put("device", "remote_controlboard");
        torso_remote_options.put("local",  "/reaching/control_torso");
        torso_remote_options.put("remote", "/icub/torso");

        torso_remote_driver_.open(torso_remote_options);
        if (torso_remote_driver_.isValid())
        {
            yInfo() << "Torso remote_controlboard succefully opened.";

            torso_remote_driver_.view(itf_torso_enc_);
            if (!itf_torso_enc_)
            {
                yError() << "Error getting torso IEncoders interface.";
                return false;
            }

            return true;
        }
        else
        {
            yError() << "Error opening Torso remote_controlboard device.";
            return false;
        }
    }

    bool setTorsoDOF()
    {
        Vector curDOF;
        itf_rightarm_cart_->getDOF(curDOF);
        yInfo() << "Old DOF: [" + curDOF.toString(0) + "].";
        yInfo() << "Setting iCub to use the DOF from the torso.";
        Vector newDOF(curDOF);
        newDOF[0] = 1;
        newDOF[1] = 2;
        newDOF[2] = 1;
        if (!itf_rightarm_cart_->setDOF(newDOF, curDOF))
        {
            yError() << "Cannot use torso DOF.";
            return false;
        }
        yInfo() << "Setting the DOF done.";
        yInfo() << "New DOF: [" + curDOF.toString(0) + "]";

        return true;
    }

    bool unsetTorsoDOF()
    {
        Vector curDOF;
        itf_rightarm_cart_->getDOF(curDOF);
        yInfo() << "Old DOF: [" + curDOF.toString(0) + "].";
        yInfo() << "Setting iCub to use the DOF from the torso.";
        Vector newDOF(curDOF);
        newDOF[0] = 0;
        newDOF[1] = 2;
        newDOF[2] = 0;
        if (!itf_rightarm_cart_->setDOF(newDOF, curDOF))
        {
            yError() << "Cannot use torso DOF.";
            return false;
        }
        yInfo() << "Setting the DOF done.";
        yInfo() << "New DOF: [" + curDOF.toString(0) + "]";

        return true;
    }

    Vector readTorso()
    {
        Vector torso_enc(3);
        itf_torso_enc_->getEncoders(torso_enc.data());

        std::swap(torso_enc(0), torso_enc(2));

        return torso_enc;
    }

    Vector readRootToFingers()
    {
        Vector rightarm_encoder(16);
        itf_rightarm_enc_->getEncoders(rightarm_encoder.data());

        Vector root_fingers_enc(19);
        root_fingers_enc.setSubvector(0, readTorso());

        root_fingers_enc.setSubvector(3, rightarm_encoder);
        
        return root_fingers_enc;
    }
};


int main()
{
    Network yarp;
    if (!yarp.checkNetwork(3.0))
    {
        yError() << "YARP seems unavailable!";
        return EXIT_FAILURE;
    }

    ResourceFinder rf;
    RFMReaching reaching;
    reaching.runModule(rf);

    return EXIT_SUCCESS;
}
