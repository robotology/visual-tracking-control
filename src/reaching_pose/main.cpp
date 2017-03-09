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
        if (!port_estimates_in_.open("/reaching_pose/estimates:i"))
        {
            yError() << "Could not open /reaching_pose/estimates:i port! Closing.";
            return false;
        }

        if (!port_image_left_in_.open("/reaching_pose/cam_left/img:i"))
        {
            yError() << "Could not open /reaching_pose/cam_left/img:i port! Closing.";
            return false;
        }

        if (!port_image_left_out_.open("/reaching_pose/cam_left/img:o"))
        {
            yError() << "Could not open /reaching_pose/cam_left/img:o port! Closing.";
            return false;
        }

        if (!port_click_left_.open("/reaching_pose/cam_left/click:i"))
        {
            yError() << "Could not open /reaching_pose/cam_left/click:in port! Closing.";
            return false;
        }

        if (!port_image_right_in_.open("/reaching_pose/cam_right/img:i"))
        {
            yError() << "Could not open /reaching_pose/cam_right/img:i port! Closing.";
            return false;
        }

        if (!port_image_right_out_.open("/reaching_pose/cam_right/img:o"))
        {
            yError() << "Could not open /reaching_pose/cam_right/img:o port! Closing.";
            return false;
        }

        if (!port_click_right_.open("/reaching_pose/cam_right/click:i"))
        {
            yError() << "Could not open /reaching_pose/cam_right/click:i port! Closing.";
            return false;
        }

        if (!port_px_left_endeffector.open("/reaching_pose/cam_left/x:o"))
        {
            yError() << "Could not open /reaching_pose/cam_left/x:o port! Closing.";
            return false;
        }

        if (!port_px_right_endeffector.open("/reaching_pose/cam_right/x:o"))
        {
            yError() << "Could not open /reaching_pose/cam_right/x:o port! Closing.";
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
        
        handler_port_.open("/reaching_pose/cmd:i");
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

        yInfo() << "RUNNING!\n";

        yInfo() << "estimates = ["  << estimates->toString() << "]";
        yInfo() << "l_px_goal_ = [" << l_px_goal_.toString() << "]";
        yInfo() << "r_px_goal_ = [" << r_px_goal_.toString() << "]";

        Vector px_des;
        px_des.push_back(l_px_goal_[0]);    /* u_ee_l */
        px_des.push_back(r_px_goal_[0]);    /* u_ee_r */
        px_des.push_back(l_px_goal_[1]);    /* v_ee_l */
        px_des.push_back(l_px_goal_[2]);    /* u_x1_l */
        px_des.push_back(r_px_goal_[2]);    /* u_x1_r */
        px_des.push_back(l_px_goal_[3]);    /* v_x1_l */
        px_des.push_back(l_px_goal_[4]);    /* u_x2_l */
        px_des.push_back(r_px_goal_[4]);    /* u_x2_r */
        px_des.push_back(l_px_goal_[5]);    /* v_x2_l */
        yInfo() << "px_des = ["  << px_des.toString() << "]";


        // FIXME: solo per controllo con l/r_px?
        Vector px_ee_left;  /* u_ee_l, v_ee_l */
        itf_gaze_->get2DPixel(LEFT,  estimates->subVector(0, 2), px_ee_left);
        yInfo() << "estimates(0, 2) = ["  << estimates->subVector(0, 2).toString() << "]";
        yInfo() << "px_ee_left = ["  << px_ee_left.toString() << "]";


        Vector px_ee_right; /* u_ee_r, v_ee_r */
        itf_gaze_->get2DPixel(RIGHT, estimates->subVector(6, 8), px_ee_right);
        yInfo() << "estimates(6, 8) = ["  << estimates->subVector(6, 8).toString() << "]";
        yInfo() << "px_ee_right = [" << px_ee_right.toString() << "]";
        /* ********************************** */


        Vector l_ee_x  = estimates->subVector(0, 2);
        Vector l_ee_x1 = zeros(4);
        Vector l_ee_x2 = zeros(4);
        getPalmPoints(estimates->subVector(0, 5), l_ee_x1, l_ee_x2);

        Vector left_eye_x;
        Vector left_eye_o;
        itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

        Vector l_px = (l_H_r_to_cam_.submatrix(0, 2, 0, 2) * (l_ee_x - left_eye_x));
        l_px[0] /= l_px[2];
        l_px[1] /= l_px[2];
        Vector l_px1 = l_H_r_to_cam_ * l_ee_x1;
        l_px1[0] /= l_px1[2];
        l_px1[1] /= l_px1[2];
        Vector l_px2 = l_H_r_to_cam_ * l_ee_x2;
        l_px2[0] /= l_px2[2];
        l_px2[1] /= l_px2[2];
        yInfo() << "Proj left ee    = [" << l_px.subVector(0, 1).toString()  << "]";
        yInfo() << "Proj left ee x1 = [" << l_px1.subVector(0, 1).toString() << "]";
        yInfo() << "Proj left ee x2 = [" << l_px2.subVector(0, 1).toString() << "]";

        Vector r_ee_x = estimates->subVector(6, 8);
        Vector r_ee_x1 = zeros(4);
        Vector r_ee_x2 = zeros(4);
        getPalmPoints(estimates->subVector(6, 11), r_ee_x1, r_ee_x2);

        Vector right_eye_x;
        Vector right_eye_o;
        itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

        Vector r_px = (r_H_r_to_cam_.submatrix(0, 2, 0, 2) * (r_ee_x - right_eye_x));
        r_px[0] /= r_px[2];
        r_px[1] /= r_px[2];
        Vector r_px1 = r_H_r_to_cam_ * r_ee_x1;
        r_px1[0] /= r_px1[2];
        r_px1[1] /= r_px1[2];
        Vector r_px2 = r_H_r_to_cam_ * r_ee_x2;
        r_px2[0] /= r_px2[2];
        r_px2[1] /= r_px2[2];
        yInfo() << "Proj right ee    = [" << r_px.subVector(0, 1).toString() << "]";
        yInfo() << "Proj right ee x1 = [" << r_px1.subVector(0, 1).toString() << "]";
        yInfo() << "Proj right ee x2 = [" << r_px2.subVector(0, 1).toString() << "]";


        Vector px_ee_now;
        px_ee_now.push_back(l_px [0]);  /* u_ee_l */
        px_ee_now.push_back(r_px [0]);  /* u_ee_r */
        px_ee_now.push_back(l_px [1]);  /* v_ee_l */
        px_ee_now.push_back(l_px1[0]);  /* u_x1_l */
        px_ee_now.push_back(r_px1[0]);  /* u_x1_r */
        px_ee_now.push_back(l_px1[1]);  /* v_x1_l */
        px_ee_now.push_back(l_px2[0]);  /* u_x2_l */
        px_ee_now.push_back(r_px2[0]);  /* u_x2_r */
        px_ee_now.push_back(l_px2[1]);  /* v_x2_l */
        yInfo() << "px_ee_now = [" << px_ee_now.toString() << "]";


        /* Jacobian */
        Matrix jacobian = zeros(9, 6);

        /* End-effector */
        jacobian.setRow(0, setJacobianU(LEFT,  l_ee_x(2),  l_px ));
        jacobian.setRow(1, setJacobianU(RIGHT, r_ee_x(2),  r_px ));
        jacobian.setRow(2, setJacobianV(LEFT,  l_ee_x(2),  l_px ));

        /* Extra point 1 */
        jacobian.setRow(3, setJacobianU(LEFT,  l_ee_x1(2), l_px1));
        jacobian.setRow(4, setJacobianU(RIGHT, r_ee_x1(2), r_px1));
        jacobian.setRow(5, setJacobianV(LEFT,  l_ee_x1(2), l_px1));

        /* Extra point 2 */
        jacobian.setRow(6, setJacobianU(LEFT,  l_ee_x2(2), l_px2));
        jacobian.setRow(7, setJacobianU(RIGHT, r_ee_x2(2), r_px2));
        jacobian.setRow(8, setJacobianV(LEFT,  l_ee_x2(2), l_px2));
        /* ******** */


        double Ts    = 0.05;   // controller's sample time [s]
        double K_x   = 1.0;  // visual servoing proportional gain
        double K_o   = 0.001;  // visual servoing proportional gain
        double v_max = 0.0005; // max cartesian velocity [m/s]

        bool done = false;
        while (!should_stop_ && !done)
        {
            Vector e            = px_des - px_ee_now;
            Matrix inv_jacobian = pinv(jacobian);
            Vector vel_x        = K_x * inv_jacobian.submatrix(0, 2, 0, 8) * e;
            Vector vel_o        = - K_o * inv_jacobian.submatrix(3, 5, 0, 8) * e;

//            yInfo() << "jacobian = [\n"     << jacobian.toString()     << "]";
//            yInfo() << "inv_jacobian = [\n" << inv_jacobian.toString() << "]";
            yInfo() << "px_des = ["         << px_des.toString()       << "]";
            yInfo() << "px_ee_now = ["      << px_ee_now.toString()    << "]";
            yInfo() << "e = ["              << e.toString()            << "]";
            yInfo() << "vel_x = ["          << vel_x.toString()        << "]";
            yInfo() << "vel_o = ["          << vel_o.toString()        << "]";

            /* Enforce velocity bounds */
            for (size_t i = 0; i < vel_x.length(); ++i)
            {
                vel_x[i] = sign(vel_x[i]) * std::min(v_max, std::fabs(vel_x[i]));
            }

            yInfo() << "bounded vel_x = [" << vel_x.toString() << "]";

            vel_o = dcm2axis(rpy2dcm(vel_o));
            yInfo() << "transformed vel_o = [" << vel_o.toString() << "]";

            itf_rightarm_cart_->setTaskVelocities(vel_x, vel_o);

            yInfo() << "Pixel error: " << std::abs(px_des(0) - px_ee_now(0)) << std::abs(px_des(1) - px_ee_now(1)) << std::abs(px_des(2) - px_ee_now(2))
                                       << std::abs(px_des(3) - px_ee_now(3)) << std::abs(px_des(4) - px_ee_now(4)) << std::abs(px_des(5) - px_ee_now(5))
                                       << std::abs(px_des(6) - px_ee_now(6)) << std::abs(px_des(7) - px_ee_now(7)) << std::abs(px_des(8) - px_ee_now(8));

            Time::delay(Ts);

            done = ((std::abs(px_des(0) - px_ee_now(0)) < 5.0) && (std::abs(px_des(1) - px_ee_now(1)) < 5.0) && (std::abs(px_des(2) - px_ee_now(2)) < 5.0) &&
                    (std::abs(px_des(3) - px_ee_now(3)) < 5.0) && (std::abs(px_des(4) - px_ee_now(4)) < 5.0) && (std::abs(px_des(5) - px_ee_now(5)) < 5.0) &&
                    (std::abs(px_des(6) - px_ee_now(6)) < 5.0) && (std::abs(px_des(7) - px_ee_now(7)) < 5.0) && (std::abs(px_des(8) - px_ee_now(8)) < 5.0));
            if (done)
            {
                yInfo() << "\npx_des ="  << px_des.toString();
                yInfo() << "px_ee_now =" << px_ee_now.toString();
                yInfo() << "\nTERMINATING!\n";
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

                yInfo() << "EE L coord: " << estimates->subVector(0, 2).toString();
                yInfo() << "EE R coord: " << estimates->subVector(6, 8).toString() << "\n";

                l_ee_x  = estimates->subVector(0, 2);
                l_ee_x1 = zeros(4);
                l_ee_x2 = zeros(4);
                getPalmPoints(estimates->subVector(0, 5), l_ee_x1, l_ee_x2);

                l_px = (l_H_r_to_cam_.submatrix(0, 2, 0, 2) * (l_ee_x - left_eye_x));
                l_px[0] /= l_px[2];
                l_px[1] /= l_px[2];
                l_px1 = l_H_r_to_cam_ * l_ee_x1;
                l_px1[0] /= l_px1[2];
                l_px1[1] /= l_px1[2];
                l_px2 = l_H_r_to_cam_ * l_ee_x2;
                l_px2[0] /= l_px2[2];
                l_px2[1] /= l_px2[2];


                r_ee_x = estimates->subVector(6, 8);
                r_ee_x1 = zeros(4);
                r_ee_x2 = zeros(4);
                getPalmPoints(estimates->subVector(6, 11), r_ee_x1, r_ee_x2);

                r_px = (r_H_r_to_cam_.submatrix(0, 2, 0, 2) * (r_ee_x - right_eye_x));
                r_px[0] /= r_px[2];
                r_px[1] /= r_px[2];
                r_px1 = r_H_r_to_cam_ * r_ee_x1;
                r_px1[0] /= r_px1[2];
                r_px1[1] /= r_px1[2];
                r_px2 = r_H_r_to_cam_ * r_ee_x2;
                r_px2[0] /= r_px2[2];
                r_px2[1] /= r_px2[2];


                px_ee_now[0] = l_px[0];     /* u_ee_l */
                px_ee_now[1] = r_px[0];     /* u_ee_r */
                px_ee_now[2] = l_px[1];     /* v_ee_l */
                px_ee_now[3] = l_px1[0];    /* u_x1_l */
                px_ee_now[4] = r_px1[0];    /* u_x1_r */
                px_ee_now[5] = l_px1[1];    /* v_x1_l */
                px_ee_now[6] = l_px2[0];    /* u_x2_l */
                px_ee_now[7] = r_px2[0];    /* u_x2_r */
                px_ee_now[8] = l_px2[1];    /* v_x2_l */


                /* Update Jacobian */
                jacobian = zeros(9, 6);

                /* End-effector */
                jacobian.setRow(0, setJacobianU(LEFT,  l_ee_x(2),  l_px ));
                jacobian.setRow(1, setJacobianU(RIGHT, r_ee_x(2),  r_px ));
                jacobian.setRow(2, setJacobianV(LEFT,  l_ee_x(2),  l_px ));

                /* Extra point 1 */
                jacobian.setRow(3, setJacobianU(LEFT,  l_ee_x1(2), l_px1));
                jacobian.setRow(4, setJacobianU(RIGHT, r_ee_x1(2), r_px1));
                jacobian.setRow(5, setJacobianV(LEFT,  l_ee_x1(2), l_px1));

                /* Extra point 2 */
                jacobian.setRow(6, setJacobianU(LEFT,  l_ee_x2(2), l_px2));
                jacobian.setRow(7, setJacobianU(RIGHT, r_ee_x2(2), r_px2));
                jacobian.setRow(8, setJacobianV(LEFT,  l_ee_x2(2), l_px2));
                /* *************** */


                /* Dump pixel coordinates of the goal */
                Bottle& l_px_endeffector = port_px_left_endeffector.prepare();
                l_px_endeffector.clear();
                l_px_endeffector.addInt(l_px [0]);
                l_px_endeffector.addInt(l_px [1]);
                l_px_endeffector.addInt(l_px1[0]);
                l_px_endeffector.addInt(l_px1[1]);
                l_px_endeffector.addInt(l_px2[0]);
                l_px_endeffector.addInt(l_px2[1]);
                port_px_left_endeffector.write();

                Bottle& r_px_endeffector = port_px_right_endeffector.prepare();
                r_px_endeffector.clear();
                r_px_endeffector.addInt(r_px [0]);
                r_px_endeffector.addInt(r_px [1]);
                r_px_endeffector.addInt(r_px1[0]);
                r_px_endeffector.addInt(r_px1[1]);
                r_px_endeffector.addInt(r_px2[0]);
                r_px_endeffector.addInt(r_px2[1]);
                port_px_right_endeffector.write();


                /* Left eye end-effector superimposition */
                ImageOf<PixelRgb>* l_imgin  = port_image_left_in_.read(true);
                ImageOf<PixelRgb>& l_imgout = port_image_left_out_.prepare();
                l_imgout = *l_imgin;
                cv::Mat l_img = cv::cvarrToMat(l_imgout.getIplImage());

                cv::circle(l_img, cv::Point(l_px[0],       l_px[1] ),      4, cv::Scalar(255,   0,   0), 4);
                cv::circle(l_img, cv::Point(l_px1[0],      l_px1[1]),      4, cv::Scalar(0,   255,   0), 4);
                cv::circle(l_img, cv::Point(l_px2[0],      l_px2[1]),      4, cv::Scalar(0,     0, 255), 4);
                cv::circle(l_img, cv::Point(l_px_goal_[0], l_px_goal_[1]), 4, cv::Scalar(255,   0,   0), 4);
                cv::circle(l_img, cv::Point(l_px_goal_[2], l_px_goal_[3]), 4, cv::Scalar(0,   255,   0), 4);
                cv::circle(l_img, cv::Point(l_px_goal_[4], l_px_goal_[5]), 4, cv::Scalar(0,     0, 255), 4);

                port_image_left_out_.write();

                /* Right eye end-effector superimposition */
                ImageOf<PixelRgb>* r_imgin  = port_image_right_in_.read(true);
                ImageOf<PixelRgb>& r_imgout = port_image_right_out_.prepare();
                r_imgout = *r_imgin;
                cv::Mat r_img = cv::cvarrToMat(r_imgout.getIplImage());

                cv::circle(r_img, cv::Point(r_px[0],       r_px[1] ),      4, cv::Scalar(255,   0,   0), 4);
                cv::circle(r_img, cv::Point(r_px1[0],      r_px1[1]),      4, cv::Scalar(0,   255,   0), 4);
                cv::circle(r_img, cv::Point(r_px2[0],      r_px2[1]),      4, cv::Scalar(0,     0, 255), 4);
                cv::circle(r_img, cv::Point(r_px_goal_[0], r_px_goal_[1]), 4, cv::Scalar(255,   0,   0), 4);
                cv::circle(r_img, cv::Point(r_px_goal_[2], r_px_goal_[3]), 4, cv::Scalar(0,   255,   0), 4);
                cv::circle(r_img, cv::Point(r_px_goal_[4], r_px_goal_[5]), 4, cv::Scalar(0,     0, 255), 4);

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
            /* Go to initial position (open-loop) */
            case VOCAB4('i', 'n', 'i', 't'):
            {
                /* FINGERTIP */
//                Matrix Od(3, 3);
//                Od(0, 0) = -1.0;
//                Od(1, 1) =  1.0;
//                Od(2, 2) = -1.0;
//                Vector od = dcm2axis(Od);

                /* KARATE */
//                Matrix Od = zeros(3, 3);
//                Od(0, 0) = -1.0;
//                Od(2, 1) = -1.0;
//                Od(1, 2) = -1.0;
//                Vector od = dcm2axis(Od);

                /* GRASPING */
                Vector od = zeros(4);
                od(0) = -0.141;
                od(1) =  0.612;
                od(2) = -0.777;
                od(4) =  3.012;


                double traj_time = 0.0;
                itf_rightarm_cart_->getTrajTime(&traj_time);

                if (traj_time == traj_time_)
                {
                    Vector init_pos = zeros(3);

                    /* FINGERTIP */
//                    Vector chain_joints;
//                    icub_index_.getChainJoints(readRootToFingers().subVector(3, 18), chain_joints);
//
//                    Matrix tip_pose_index = icub_index_.getH((M_PI/180.0) * chain_joints);
//                    Vector tip_x = tip_pose_index.getCol(3);
//                    Vector tip_o = dcm2axis(tip_pose_index);
//                    itf_rightarm_cart_->attachTipFrame(tip_x, tip_o);
//
                    //FIXME: to implement
//                    init_pos[0] = -0.345;
//                    init_pos[1] =  0.139;
//                    init_pos[2] =  0.089;

                    /* KARATE */
                    // FIXME: to implement

                    /* GRASPING */
                    init_pos[0] = -0.370;
                    init_pos[1] =  0.103;
                    init_pos[2] =  0.064;


                    setTorsoDOF();

                    Vector gaze_loc(3);
                    gaze_loc(0) = -0.660;
                    gaze_loc(1) =  0.115;
                    gaze_loc(2) = -0.350;


                    int ctxt;
                    itf_rightarm_cart_->storeContext(&ctxt);

                    itf_rightarm_cart_->setLimits(0,  15.0,  15.0);
                    itf_rightarm_cart_->setLimits(2, -23.0, -23.0);
                    itf_rightarm_cart_->setLimits(3, -16.0, -16.0);
                    itf_rightarm_cart_->setLimits(4,  53.0,  53.0);
                    itf_rightarm_cart_->setLimits(5,   0.0,   0.0);
                    itf_rightarm_cart_->setLimits(7, -58.0, -58.0);

                    itf_rightarm_cart_->goToPoseSync(init_pos, od);
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
            /* Get 3D point from Structure From Motion clicking on the left camera image */
            /* PLUS: Compute again the roto-translation and projection matrices from root to left and right camera planes */
            case VOCAB3('s', 'f', 'm'):
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


                Network yarp;
                Bottle  cmd;
                Bottle  rep;

                Bottle* click_left = port_click_left_.read(true);

                l_px_goal_.resize(6);
                l_px_goal_[0] = click_left->get(0).asDouble();
                l_px_goal_[1] = click_left->get(1).asDouble();


                RpcClient port_sfm;
                port_sfm.open("/reaching_pose/tosfm");
                yarp.connect("/reaching_pose/tosfm", "/SFM/rpc");

                cmd.clear();

                cmd.addInt(l_px_goal_[0]);
                cmd.addInt(l_px_goal_[1]);

                Bottle reply_pos;
                port_sfm.write(cmd, reply_pos);
                if (reply_pos.size() == 5)
                {
                    Matrix R_ee = zeros(3, 3);
                    R_ee(0, 0) = -1.0;
                    R_ee(1, 1) =  1.0;
                    R_ee(2, 2) = -1.0;
                    Vector ee_o = dcm2axis(R_ee);

                    Vector goal_pos = zeros(9);
                    goal_pos[0] = reply_pos.get(0).asDouble();
                    goal_pos[1] = reply_pos.get(1).asDouble();
                    goal_pos[2] = reply_pos.get(2).asDouble();

                    Vector p = zeros(7);
                    p.setSubvector(0, goal_pos.subVector(0, 2));
                    p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));
                    Vector p1 = zeros(4);
                    Vector p2 = zeros(4);
                    getPalmPoints(p, p1, p2);
                    goal_pos.setSubvector(3, p1.subVector(0, 2));
                    goal_pos.setSubvector(6, p2.subVector(0, 2));

                    yInfo() << "goal_pos: [" << goal_pos.toString() << "];";


                    Vector l_px1 = l_H_r_to_cam_ * p1;
                    l_px1[0] /= l_px1[2];
                    l_px1[1] /= l_px1[2];
                    Vector l_px2 = l_H_r_to_cam_ * p2;
                    l_px2[0] /= l_px2[2];
                    l_px2[1] /= l_px2[2];

                    l_px_goal_[2] = l_px1[0];
                    l_px_goal_[3] = l_px1[1];
                    l_px_goal_[4] = l_px2[0];
                    l_px_goal_[5] = l_px2[1];


                    Vector r_px1 = r_H_r_to_cam_ * p1;
                    r_px1[0] /= r_px1[2];
                    r_px1[1] /= r_px1[2];
                    Vector r_px2 = r_H_r_to_cam_ * p2;
                    r_px2[0] /= r_px2[2];
                    r_px2[1] /= r_px2[2];

                    r_px_goal_.resize(6);
                    r_px_goal_[0] = reply_pos.get(3).asDouble();
                    r_px_goal_[1] = reply_pos.get(4).asDouble();
                    r_px_goal_[2] = r_px1[0];
                    r_px_goal_[3] = r_px1[1];
                    r_px_goal_[4] = r_px2[0];
                    r_px_goal_[5] = r_px2[1];
                }
                else
                {
                    reply.addString("nack");
                }

                yarp.disconnect("/reaching_pose/tosfm", "/SFM/rpc");
                port_sfm.close();

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


                Matrix R_ee = zeros(3, 3);
                R_ee(0, 0) = -1.0;
                R_ee(1, 1) =  1.0;
                R_ee(2, 2) = -1.0;
                Vector ee_o = dcm2axis(R_ee);

                // -0.416311	-0.026632	 0.055334	-0.381311	-0.036632	 0.055334	-0.381311	-0.016632	 0.055334
                Vector goal_pos = zeros(9);
                goal_pos[0] = -0.416311;
                goal_pos[1] = -0.026632;
                goal_pos[2] =  0.055334;

                Vector p = zeros(7);
                p.setSubvector(0, goal_pos.subVector(0, 2));
                p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));
                Vector p1 = zeros(4);
                Vector p2 = zeros(4);
                getPalmPoints(p, p1, p2);
                goal_pos.setSubvector(3, p1.subVector(0, 2));
                goal_pos.setSubvector(6, p2.subVector(0, 2));

                yInfo() << "goal_pos: [" << goal_pos.toString() << "];";


                Vector ee_x  = p.subVector(0, 2);
                ee_x.push_back(1.0);

                Vector l_px  = l_H_r_to_cam_ * ee_x;
                l_px[0] /= l_px[2];
                l_px[1] /= l_px[2];
                Vector l_px1 = l_H_r_to_cam_ * p1;
                l_px1[0] /= l_px1[2];
                l_px1[1] /= l_px1[2];
                Vector l_px2 = l_H_r_to_cam_ * p2;
                l_px2[0] /= l_px2[2];
                l_px2[1] /= l_px2[2];

                l_px_goal_[0] = l_px [0];
                l_px_goal_[1] = l_px [1];
                l_px_goal_[2] = l_px1[0];
                l_px_goal_[3] = l_px1[1];
                l_px_goal_[4] = l_px2[0];
                l_px_goal_[5] = l_px2[1];


                Vector r_px  = r_H_r_to_cam_ * ee_x;
                r_px[0] /= r_px[2];
                r_px[1] /= r_px[2];
                Vector r_px1 = r_H_r_to_cam_ * p1;
                r_px1[0] /= r_px1[2];
                r_px1[1] /= r_px1[2];
                Vector r_px2 = r_H_r_to_cam_ * p2;
                r_px2[0] /= r_px2[2];
                r_px2[1] /= r_px2[2];

                r_px_goal_.resize(6);
                r_px_goal_[0] = r_px [0];
                r_px_goal_[1] = r_px [1];
                r_px_goal_[2] = r_px1[0];
                r_px_goal_[3] = r_px1[1];
                r_px_goal_[4] = r_px2[0];
                r_px_goal_[5] = r_px2[1];

                
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

    double                           traj_time_ = 3.0;
    Vector                           l_px_goal_;
    Vector                           r_px_goal_;
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
        rightarm_cartesian_options.put("local",  "/reaching_pose/cart_right_arm");
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
        gaze_option.put("local",  "/reaching_pose/gaze");
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
        rightarm_remote_options.put("local",  "/reaching_pose/control_right_arm");
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
        torso_remote_options.put("local",  "/reaching_pose/control_torso");
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

    void getPalmPoints(const Vector& endeffector, Vector& p1, Vector& p2)
    {
        Vector ee_x = endeffector.subVector(0, 2);
        ee_x.push_back(1.0);
        double ang  = norm(endeffector.subVector(3, 5));
        Vector ee_o = endeffector.subVector(3, 5) / ang;
        ee_o.push_back(ang);

        Matrix H_ee_to_root = axis2dcm(ee_o);
        H_ee_to_root.setCol(3, ee_x);

        p1 = zeros(4);
        p2 = zeros(4);
        Vector p = zeros(4);

        p(0) = -0.035;
        p(1) = -0.01;
        p(2) = 0;
        p(3) = 1.0;

        p1 = H_ee_to_root * p;

        p(0) = -0.035;
        p(1) = 0.01;
        p(2) = 0;
        p(3) = 1.0;

        p2 = H_ee_to_root * p;
    }

    Vector setJacobianU(const int cam, const double z, const Vector& px)
    {
        Vector jacobian = zeros(6);

        if (cam == LEFT)
        {
            jacobian(0) = left_proj_(0, 0) / z;
            jacobian(2) = -(px(0) - left_proj_(0, 2)) / z;
            jacobian(3) = -((px(0) - left_proj_(0, 2)) * (px(1) - left_proj_(1, 2))) / left_proj_(1, 1);
            jacobian(4) = (pow(left_proj_(0, 0), 2.0) + pow(px(0) - left_proj_(0, 2), 2.0)) / left_proj_(0, 0);
            jacobian(5) = left_proj_(0, 0) / left_proj_(1, 1) * (px(1) - left_proj_(1, 2));
        }
        else if (cam == RIGHT)
        {
            jacobian(0) = right_proj_(0, 0) / z;
            jacobian(2) = -(px(0) - right_proj_(0, 2)) / z;
            jacobian(3) = -((px(0) - right_proj_(0, 2)) * (px(1) - right_proj_(1, 2))) / right_proj_(1, 1);
            jacobian(4) = (pow(right_proj_(0, 0), 2.0) + pow(px(0) - right_proj_(0, 2), 2.0)) / right_proj_(0, 0);
            jacobian(5) = right_proj_(0, 0) / right_proj_(1, 1) * (px(1) - right_proj_(1, 2));
        }

        return jacobian;
    }

    Vector setJacobianV(const int cam, const double z, const Vector& px)
    {
        Vector jacobian = zeros(6);

        if (cam == LEFT)
        {
            jacobian(1) = left_proj_(1, 1) / z;
            jacobian(2) = -(px(1) - left_proj_(1, 2)) / z;
            jacobian(3) = -(pow(left_proj_(1, 1), 2.0) + pow(px(1) - left_proj_(1, 2), 2.0)) / left_proj_(1, 1);
            jacobian(4) = ((px(0) - left_proj_(0, 2)) * (px(1) - left_proj_(1, 2))) / left_proj_(0, 0);
            jacobian(5) = left_proj_(1, 1) / left_proj_(0, 0) * (px(0) - left_proj_(0, 2));
        }
        else if (cam == RIGHT)
        {
            jacobian(1) = right_proj_(1, 1) / z;
            jacobian(2) = -(px(1) - right_proj_(1, 2)) / z;
            jacobian(3) = -(pow(right_proj_(1, 1), 2.0) + pow(px(1) - right_proj_(1, 2), 2.0)) / right_proj_(1, 1);
            jacobian(4) = ((px(0) - right_proj_(0, 2)) * (px(1) - right_proj_(1, 2))) / right_proj_(0, 0);
            jacobian(5) = right_proj_(1, 1) / right_proj_(0, 0) * (px(0) - right_proj_(0, 2));
        }

        return jacobian;
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
