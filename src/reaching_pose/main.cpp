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
#include <yarp/os/ConstString.h>
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
        robot_name_ = rf.find("robot").asString();
        if (robot_name_.empty())
        {
            yError() << "Robot name not provided! Closing.";
            return false;
        }

        if (!port_estimates_left_in_.open("/reaching_pose/estimates/left:i"))
        {
            yError() << "Could not open /reaching_pose/estimates/left:i port! Closing.";
            return false;
        }

        if (!port_estimates_right_in_.open("/reaching_pose/estimates/right:i"))
        {
            yError() << "Could not open /reaching_pose/estimates/right:i port! Closing.";
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

        l_proj_ = zeros(3, 4);
        l_proj_(0, 0)  = left_fx;
        l_proj_(0, 2)  = left_cx;
        l_proj_(1, 1)  = left_fy;
        l_proj_(1, 2)  = left_cy;
        l_proj_(2, 2)  = 1.0;

        yInfo() << "l_proj_ =\n" << l_proj_.toString();

        float right_fx = static_cast<float>(cam_right_info->get(0).asDouble());
        float right_cx = static_cast<float>(cam_right_info->get(2).asDouble());
        float right_fy = static_cast<float>(cam_right_info->get(5).asDouble());
        float right_cy = static_cast<float>(cam_right_info->get(6).asDouble());

        r_proj_ = zeros(3, 4);
        r_proj_(0, 0)  = right_fx;
        r_proj_(0, 2)  = right_cx;
        r_proj_(1, 1)  = right_fy;
        r_proj_(1, 2)  = right_cy;
        r_proj_(2, 2)  = 1.0;

        yInfo() << "r_proj_ =\n" << r_proj_.toString();


        Vector left_eye_x;
        Vector left_eye_o;
        itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

        Vector right_eye_x;
        Vector right_eye_o;
        itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

        yInfo() << "left_eye_o =" << left_eye_o.toString();
        yInfo() << "right_eye_o =" << right_eye_o.toString();


        l_H_eye_to_r_ = axis2dcm(left_eye_o);
        left_eye_x.push_back(1.0);
        l_H_eye_to_r_.setCol(3, left_eye_x);
        l_H_r_to_eye_ = SE3inv(l_H_eye_to_r_);

        r_H_eye_to_r_ = axis2dcm(right_eye_o);
        right_eye_x.push_back(1.0);
        r_H_eye_to_r_.setCol(3, right_eye_x);
        r_H_r_to_eye_ = SE3inv(r_H_eye_to_r_);

        yInfo() << "l_H_r_to_eye_ =\n" << l_H_r_to_eye_.toString();
        yInfo() << "r_H_r_to_eye_ =\n" << r_H_r_to_eye_.toString();

        l_H_r_to_cam_ = l_proj_ * l_H_r_to_eye_;
        r_H_r_to_cam_ = r_proj_ * r_H_r_to_eye_;

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

        if (should_stop_) return false;


        Vector est_copy(12);

        /* Get the initial end-effector pose from left eye particle filter */
        Vector* estimates = port_estimates_left_in_.read(true);
        yInfo() << "Got [" << estimates->toString() << "] from left eye particle filter.";
        est_copy.setSubvector(0, *estimates);

        /* Get the initial end-effector pose from right eye particle filter */
        estimates = port_estimates_right_in_.read(true);
        yInfo() << "Got [" << estimates->toString() << "] from right eye particle filter.";
        est_copy.setSubvector(6, *estimates);


        yInfo() << "RUNNING!\n";

        yInfo() << "estimates = ["  << est_copy.toString() << "]";
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

        px_des.push_back(l_px_goal_[6]);    /* u_x3_l */
        px_des.push_back(r_px_goal_[6]);    /* u_x3_r */
        px_des.push_back(l_px_goal_[7]);    /* v_x3_l */

        yInfo() << "px_des = ["  << px_des.toString() << "]";


        // FIXME: solo per controllo con l/r_px?
        Vector px_ee_left;  /* u_ee_l, v_ee_l */
        itf_gaze_->get2DPixel(LEFT,  est_copy.subVector(0, 2), px_ee_left);
        yInfo() << "estimates(0, 2) = ["  << est_copy.subVector(0, 2).toString() << "]";
        yInfo() << "px_ee_left = ["  << px_ee_left.toString() << "]";


        Vector px_ee_right; /* u_ee_r, v_ee_r */
        itf_gaze_->get2DPixel(RIGHT, est_copy.subVector(6, 8), px_ee_right);
        yInfo() << "estimates(6, 8) = ["  << est_copy.subVector(6, 8).toString() << "]";
        yInfo() << "px_ee_right = [" << px_ee_right.toString() << "]";
        /* ********************************** */


        Vector l_ee_x0 = zeros(4);
        Vector l_ee_x1 = zeros(4);
        Vector l_ee_x2 = zeros(4);
        Vector l_ee_x3 = zeros(4);
        getPalmPoints(est_copy.subVector(0, 5), l_ee_x0, l_ee_x1, l_ee_x2, l_ee_x3);


        Vector l_px0 = l_H_r_to_cam_ * l_ee_x0;
        l_px0[0] /= l_px0[2];
        l_px0[1] /= l_px0[2];
        Vector l_px1 = l_H_r_to_cam_ * l_ee_x1;
        l_px1[0] /= l_px1[2];
        l_px1[1] /= l_px1[2];
        Vector l_px2 = l_H_r_to_cam_ * l_ee_x2;
        l_px2[0] /= l_px2[2];
        l_px2[1] /= l_px2[2];
        Vector l_px3 = l_H_r_to_cam_ * l_ee_x3;
        yInfo() << "Proj left ee    = [" << l_px0.subVector(0, 1).toString() << "]";
        yInfo() << "Proj left ee x1 = [" << l_px1.subVector(0, 1).toString() << "]";
        yInfo() << "Proj left ee x2 = [" << l_px2.subVector(0, 1).toString() << "]";
        yInfo() << "Proj left ee x3 = [" << l_px3.subVector(0, 1).toString() << "]";


        Vector r_ee_x0 = zeros(4);
        Vector r_ee_x1 = zeros(4);
        Vector r_ee_x2 = zeros(4);
        Vector r_ee_x3 = zeros(4);
        getPalmPoints(est_copy.subVector(6, 11), r_ee_x0, r_ee_x1, r_ee_x2, r_ee_x3);


        Vector r_px0 = r_H_r_to_cam_ * r_ee_x0;
        r_px0[0] /= r_px0[2];
        r_px0[1] /= r_px0[2];
        Vector r_px1 = r_H_r_to_cam_ * r_ee_x1;
        r_px1[0] /= r_px1[2];
        r_px1[1] /= r_px1[2];
        Vector r_px2 = r_H_r_to_cam_ * r_ee_x2;
        r_px2[0] /= r_px2[2];
        r_px2[1] /= r_px2[2];
        Vector r_px3 = r_H_r_to_cam_ * r_ee_x3;
        r_px3[0] /= r_px3[2];
        r_px3[1] /= r_px3[2];
        yInfo() << "Proj right ee    = [" << r_px0.subVector(0, 1).toString() << "]";
        yInfo() << "Proj right ee x1 = [" << r_px1.subVector(0, 1).toString() << "]";
        yInfo() << "Proj right ee x2 = [" << r_px2.subVector(0, 1).toString() << "]";
        yInfo() << "Proj right ee x3 = [" << r_px3.subVector(0, 1).toString() << "]";


        Vector px_ee_now;
        px_ee_now.push_back(l_px0[0]);  /* u_ee_l */
        px_ee_now.push_back(r_px0[0]);  /* u_ee_r */
        px_ee_now.push_back(l_px0[1]);  /* v_ee_l */

        px_ee_now.push_back(l_px1[0]);  /* u_x1_l */
        px_ee_now.push_back(r_px1[0]);  /* u_x1_r */
        px_ee_now.push_back(l_px1[1]);  /* v_x1_l */

        px_ee_now.push_back(l_px2[0]);  /* u_x2_l */
        px_ee_now.push_back(r_px2[0]);  /* u_x2_r */
        px_ee_now.push_back(l_px2[1]);  /* v_x2_l */

        px_ee_now.push_back(l_px3[0]);  /* u_x3_l */
        px_ee_now.push_back(r_px3[0]);  /* u_x3_r */
        px_ee_now.push_back(l_px3[1]);  /* v_x3_l */

        yInfo() << "px_ee_now = [" << px_ee_now.toString() << "]";


        /* Jacobian */
        Matrix jacobian = zeros(12, 6);

        /* Point 0 */
        jacobian.setRow(0,  setJacobianU(LEFT,  l_px0));
        jacobian.setRow(1,  setJacobianU(RIGHT, r_px0));
        jacobian.setRow(2,  setJacobianV(LEFT,  l_px0));

        /* Point 1 */
        jacobian.setRow(3,  setJacobianU(LEFT,  l_px1));
        jacobian.setRow(4,  setJacobianU(RIGHT, r_px1));
        jacobian.setRow(5,  setJacobianV(LEFT,  l_px1));

        /* Point 2 */
        jacobian.setRow(6,  setJacobianU(LEFT,  l_px2));
        jacobian.setRow(7,  setJacobianU(RIGHT, r_px2));
        jacobian.setRow(8,  setJacobianV(LEFT,  l_px2));

        /* Point 3 */
        jacobian.setRow(9,  setJacobianU(LEFT,  l_px3));
        jacobian.setRow(10, setJacobianU(RIGHT, r_px3));
        jacobian.setRow(11, setJacobianV(LEFT,  l_px3));
        /* ******** */


        double Ts    = 0.1;   // controller's sample time [s]
        double K_x   = 0.5;  // visual servoing proportional gain
        double K_o   = 0.5;  // visual servoing proportional gain
//        double v_max = 0.0005; // max cartesian velocity [m/s]

        bool done = false;
        while (!should_stop_ && !done)
        {
            Vector e            = px_des - px_ee_now;
            Matrix inv_jacobian = pinv(jacobian);


            Vector vel_x = zeros(3);
            Vector vel_o = zeros(3);
            for (int i = 0; i < inv_jacobian.cols(); ++i)
            {
                Vector delta_vel = inv_jacobian.getCol(i) * e(i);

                if (i == 1 || i == 4 || i == 7 || i == 10)
                {
                    vel_x += r_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel.subVector(0, 2);
                    vel_o += r_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel.subVector(3, 5);
                }
                else
                {
                    vel_x += l_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel.subVector(0, 2);
                    vel_o += l_H_eye_to_r_.submatrix(0, 2, 0, 2) * delta_vel.subVector(3, 5);
                }
            }


            yInfo() << "px_des = ["    << px_des.toString()    << "]";
            yInfo() << "px_ee_now = [" << px_ee_now.toString() << "]";
            yInfo() << "e = ["         << e.toString()         << "]";
            yInfo() << "vel_x = ["     << vel_x.toString()     << "]";
            yInfo() << "vel_o = ["     << vel_o.toString()     << "]";

            /* Enforce velocity bounds */
//            for (size_t i = 0; i < vel_x.length(); ++i)
//            {
//                vel_x[i] = sign(vel_x[i]) * std::min(v_max, std::fabs(vel_x[i]));
//            }

            yInfo() << "bounded vel_x = [" << vel_x.toString() << "]";

            double ang = norm(vel_o);
            vel_o /= ang;
            vel_o.push_back(ang);
            yInfo() << "axis-angle vel_o = [" << vel_o.toString() << "]";

            double K_ctrl_x = 0;
            if (vel_o(3) > (3.0 * CTRL_DEG2RAD)) K_ctrl_x = exp(-(vel_o(3) - (3.0 * CTRL_DEG2RAD)) / 0.1);
            else                                 K_ctrl_x = 1.0;
            yInfo() << "K_ctrl_x: " << K_ctrl_x;

            /* Visual control law */
            /* SIM */
//            vel_x    *= K_x;
            vel_x    *= (K_x * K_ctrl_x);
            vel_o(3) *= K_o;
            /* Real robot - Pose */
            itf_rightarm_cart_->setTaskVelocities(vel_x, vel_o);
            /* Real robot - Orientation */
//            itf_rightarm_cart_->setTaskVelocities(Vector(3, 0.0), vel_o);
            /* Real robot - Translation */
//            itf_rightarm_cart_->setTaskVelocities(vel_x, Vector(4, 0.0));

            yInfo() << "Pixel errors: " << std::abs(px_des(0) - px_ee_now(0)) << std::abs(px_des(1)  - px_ee_now(1))  << std::abs(px_des(2)  - px_ee_now(2))
                                        << std::abs(px_des(3) - px_ee_now(3)) << std::abs(px_des(4)  - px_ee_now(4))  << std::abs(px_des(5)  - px_ee_now(5))
                                        << std::abs(px_des(6) - px_ee_now(6)) << std::abs(px_des(7)  - px_ee_now(7))  << std::abs(px_des(8)  - px_ee_now(8))
                                        << std::abs(px_des(9) - px_ee_now(9)) << std::abs(px_des(10) - px_ee_now(10)) << std::abs(px_des(11) - px_ee_now(11));

            Time::delay(Ts);

            done = ((std::abs(px_des(0) - px_ee_now(0)) < 1.0) && (std::abs(px_des(1)  - px_ee_now(1))  < 1.0) && (std::abs(px_des(2)  - px_ee_now(2))  < 1.0) &&
                    (std::abs(px_des(3) - px_ee_now(3)) < 1.0) && (std::abs(px_des(4)  - px_ee_now(4))  < 1.0) && (std::abs(px_des(5)  - px_ee_now(5))  < 1.0) &&
                    (std::abs(px_des(6) - px_ee_now(6)) < 1.0) && (std::abs(px_des(7)  - px_ee_now(7))  < 1.0) && (std::abs(px_des(8)  - px_ee_now(8))  < 1.0) &&
                    (std::abs(px_des(9) - px_ee_now(9)) < 1.0) && (std::abs(px_des(10) - px_ee_now(10)) < 1.0) && (std::abs(px_des(11) - px_ee_now(11)) < 1.0));
            if (done)
            {
                yInfo() << "\npx_des ="  << px_des.toString();
                yInfo() << "px_ee_now =" << px_ee_now.toString();
                yInfo() << "\nTERMINATING!\n";
            }
            else
            {
                /* Get the new end-effector pose from left eye particle filter */
                estimates = port_estimates_left_in_.read(true);
                yInfo() << "Got [" << estimates->toString() << "] from left eye particle filter.";
                est_copy.setSubvector(0, *estimates);

                /* Get the new end-effector pose from right eye particle filter */
                yInfo() << "Got [" << estimates->toString() << "] from right eye particle filter.";
                estimates = port_estimates_right_in_.read(true);
                est_copy.setSubvector(6, *estimates);

                /* SIM */
//                /* Simulate reaching starting from the initial position */
//                /* Comment any previous write on variable 'estimates' */
//                yInfo() << "EE L now: " << est_copy.subVector(0, 2).toString();
//                yInfo() << "EE R now: " << est_copy.subVector(6, 8).toString() << "\n";
//
//                /* Evaluate the new orientation vector from axis-angle representation */
//                /* The following code is a copy of the setTaskVelocities() code */
//                Vector l_o = getAxisAngle(est_copy.subVector(3, 5));
//                Matrix l_R = axis2dcm(l_o);
//                Vector r_o = getAxisAngle(est_copy.subVector(9, 11));
//                Matrix r_R = axis2dcm(r_o);
//
//                vel_o[3] *= Ts;
//                l_R = axis2dcm(vel_o) * l_R;
//                r_R = axis2dcm(vel_o) * r_R;
//
//                Vector l_new_o = dcm2axis(l_R);
//                double l_ang = l_new_o(3);
//                l_new_o.pop_back();
//                l_new_o *= l_ang;
//
//                Vector r_new_o = dcm2axis(r_R);
//                double r_ang = r_new_o(3);
//                r_new_o.pop_back();
//                r_new_o *= r_ang;
//
//                est_copy.setSubvector(0, est_copy.subVector(0, 2)  + vel_x * Ts);
//                est_copy.setSubvector(3, l_new_o);
//                est_copy.setSubvector(6, est_copy.subVector(6, 8)  + vel_x * Ts);
//                est_copy.setSubvector(9, r_new_o);
                /* **************************************************** */

                yInfo() << "EE L coord: " << est_copy.subVector(0, 2).toString();
                yInfo() << "EE R coord: " << est_copy.subVector(6, 8).toString() << "\n";


                l_ee_x0 = zeros(4);
                l_ee_x1 = zeros(4);
                l_ee_x2 = zeros(4);
                l_ee_x3 = zeros(4);
                getPalmPoints(est_copy.subVector(0, 5), l_ee_x0, l_ee_x1, l_ee_x2, l_ee_x3);

                l_px0 = l_H_r_to_cam_ * l_ee_x0;
                l_px0[0] /= l_px0[2];
                l_px0[1] /= l_px0[2];
                l_px1 = l_H_r_to_cam_ * l_ee_x1;
                l_px1[0] /= l_px1[2];
                l_px1[1] /= l_px1[2];
                l_px2 = l_H_r_to_cam_ * l_ee_x2;
                l_px2[0] /= l_px2[2];
                l_px2[1] /= l_px2[2];
                l_px3 = l_H_r_to_cam_ * l_ee_x3;
                l_px3[0] /= l_px3[2];
                l_px3[1] /= l_px3[2];


                r_ee_x0 = zeros(4);
                r_ee_x1 = zeros(4);
                r_ee_x2 = zeros(4);
                r_ee_x3 = zeros(4);
                getPalmPoints(est_copy.subVector(6, 11), r_ee_x0, r_ee_x1, r_ee_x2, r_ee_x3);

                r_px0 = r_H_r_to_cam_ * r_ee_x0;
                r_px0[0] /= r_px0[2];
                r_px0[1] /= r_px0[2];
                r_px1 = r_H_r_to_cam_ * r_ee_x1;
                r_px1[0] /= r_px1[2];
                r_px1[1] /= r_px1[2];
                r_px2 = r_H_r_to_cam_ * r_ee_x2;
                r_px2[0] /= r_px2[2];
                r_px2[1] /= r_px2[2];
                r_px3 = r_H_r_to_cam_ * r_ee_x3;
                r_px3[0] /= r_px3[2];
                r_px3[1] /= r_px3[2];


                px_ee_now[0]  = l_px0[0];   /* u_ee_l */
                px_ee_now[1]  = r_px0[0];   /* u_ee_r */
                px_ee_now[2]  = l_px0[1];   /* v_ee_l */

                px_ee_now[3]  = l_px1[0];   /* u_x1_l */
                px_ee_now[4]  = r_px1[0];   /* u_x1_r */
                px_ee_now[5]  = l_px1[1];   /* v_x1_l */

                px_ee_now[6]  = l_px2[0];   /* u_x2_l */
                px_ee_now[7]  = r_px2[0];   /* u_x2_r */
                px_ee_now[8]  = l_px2[1];   /* v_x2_l */

                px_ee_now[9]  = l_px3[0];   /* u_x3_l */
                px_ee_now[10] = r_px3[0];   /* u_x3_r */
                px_ee_now[11] = l_px3[1];   /* v_x3_l */


                /* Update Jacobian */
                jacobian = zeros(12, 6);

                /* Point 0 */
                jacobian.setRow(0,  setJacobianU(LEFT,  l_px0));
                jacobian.setRow(1,  setJacobianU(RIGHT, r_px0));
                jacobian.setRow(2,  setJacobianV(LEFT,  l_px0));

                /* Point 1 */
                jacobian.setRow(3,  setJacobianU(LEFT,  l_px1));
                jacobian.setRow(4,  setJacobianU(RIGHT, r_px1));
                jacobian.setRow(5,  setJacobianV(LEFT,  l_px1));

                /* Point 2 */
                jacobian.setRow(6,  setJacobianU(LEFT,  l_px2));
                jacobian.setRow(7,  setJacobianU(RIGHT, r_px2));
                jacobian.setRow(8,  setJacobianV(LEFT,  l_px2));

                /* Point 3 */
                jacobian.setRow(9,  setJacobianU(LEFT,  l_px3));
                jacobian.setRow(10, setJacobianU(RIGHT, r_px3));
                jacobian.setRow(11, setJacobianV(LEFT,  l_px3));
                /* *************** */


                /* Dump pixel coordinates of the goal */
                Bottle& l_px_endeffector = port_px_left_endeffector.prepare();
                l_px_endeffector.clear();
                l_px_endeffector.addInt(l_px0[0]);
                l_px_endeffector.addInt(l_px0[1]);
                l_px_endeffector.addInt(l_px1[0]);
                l_px_endeffector.addInt(l_px1[1]);
                l_px_endeffector.addInt(l_px2[0]);
                l_px_endeffector.addInt(l_px2[1]);
                l_px_endeffector.addInt(l_px3[0]);
                l_px_endeffector.addInt(l_px3[1]);
                port_px_left_endeffector.write();

                Bottle& r_px_endeffector = port_px_right_endeffector.prepare();
                r_px_endeffector.clear();
                r_px_endeffector.addInt(r_px0[0]);
                r_px_endeffector.addInt(r_px0[1]);
                r_px_endeffector.addInt(r_px1[0]);
                r_px_endeffector.addInt(r_px1[1]);
                r_px_endeffector.addInt(r_px2[0]);
                r_px_endeffector.addInt(r_px2[1]);
                r_px_endeffector.addInt(r_px3[0]);
                r_px_endeffector.addInt(r_px3[1]);
                port_px_right_endeffector.write();


                /* Left eye end-effector superimposition */
                ImageOf<PixelRgb>* l_imgin  = port_image_left_in_.read(true);
                ImageOf<PixelRgb>& l_imgout = port_image_left_out_.prepare();
                l_imgout = *l_imgin;
                cv::Mat l_img = cv::cvarrToMat(l_imgout.getIplImage());

                cv::circle(l_img, cv::Point(l_px0[0],      l_px0[1]),      4, cv::Scalar(255,   0,   0), 4);
                cv::circle(l_img, cv::Point(l_px1[0],      l_px1[1]),      4, cv::Scalar(0,   255,   0), 4);
                cv::circle(l_img, cv::Point(l_px2[0],      l_px2[1]),      4, cv::Scalar(0,     0, 255), 4);
                cv::circle(l_img, cv::Point(l_px3[0],      l_px3[1]),      4, cv::Scalar(255, 255,   0), 4);
                cv::circle(l_img, cv::Point(l_px_goal_[0], l_px_goal_[1]), 4, cv::Scalar(255,   0,   0), 4);
                cv::circle(l_img, cv::Point(l_px_goal_[2], l_px_goal_[3]), 4, cv::Scalar(0,   255,   0), 4);
                cv::circle(l_img, cv::Point(l_px_goal_[4], l_px_goal_[5]), 4, cv::Scalar(0,     0, 255), 4);
                cv::circle(l_img, cv::Point(l_px_goal_[6], l_px_goal_[7]), 4, cv::Scalar(255, 255,   0), 4);

                port_image_left_out_.write();

                /* Right eye end-effector superimposition */
                ImageOf<PixelRgb>* r_imgin  = port_image_right_in_.read(true);
                ImageOf<PixelRgb>& r_imgout = port_image_right_out_.prepare();
                r_imgout = *r_imgin;
                cv::Mat r_img = cv::cvarrToMat(r_imgout.getIplImage());

                cv::circle(r_img, cv::Point(r_px0[0],      r_px0[1]),      4, cv::Scalar(255,   0,   0), 4);
                cv::circle(r_img, cv::Point(r_px1[0],      r_px1[1]),      4, cv::Scalar(0,   255,   0), 4);
                cv::circle(r_img, cv::Point(r_px2[0],      r_px2[1]),      4, cv::Scalar(0,     0, 255), 4);
                cv::circle(r_img, cv::Point(r_px3[0],      r_px3[1]),      4, cv::Scalar(255, 255,   0), 4);
                cv::circle(r_img, cv::Point(r_px_goal_[0], r_px_goal_[1]), 4, cv::Scalar(255,   0,   0), 4);
                cv::circle(r_img, cv::Point(r_px_goal_[2], r_px_goal_[3]), 4, cv::Scalar(0,   255,   0), 4);
                cv::circle(r_img, cv::Point(r_px_goal_[4], r_px_goal_[5]), 4, cv::Scalar(0,     0, 255), 4);
                cv::circle(r_img, cv::Point(r_px_goal_[6], r_px_goal_[7]), 4, cv::Scalar(255, 255,   0), 4);

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

                /* Trial 27/04/17 */
                // -0.346 0.133 0.162 0.140 -0.989 0.026 2.693
                Vector od(4);
                od[0] =  0.140;
                od[1] = -0.989;
                od[2] =  0.026;
                od[3] =  2.693;

                /* KARATE */
//                // -0.319711 0.128912 0.075052 0.03846 -0.732046 0.680169 2.979943
//                Matrix Od = zeros(3, 3);
//                Od(0, 0) = -1.0;
//                Od(2, 1) = -1.0;
//                Od(1, 2) = -1.0;
//                Vector od = dcm2axis(Od);

                /* GRASPING */
//                Vector od = zeros(4);
//                od(0) = -0.141;
//                od(1) =  0.612;
//                od(2) = -0.777;
//                od(4) =  3.012;

                /* SIM */
//                Matrix Od(3, 3);
//                Od(0, 0) = -1.0;
//                Od(1, 1) = -1.0;
//                Od(2, 2) =  1.0;
//                Vector od = dcm2axis(Od);


                double traj_time = 0.0;
                itf_rightarm_cart_->getTrajTime(&traj_time);

                if (traj_time == traj_time_)
                {
                    Vector init_pos = zeros(3);

                    /* FINGERTIP init */
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

                    /* Trial 27/04/17 */
                    // -0.346 0.133 0.162 0.140 -0.989 0.026 2.693
                    init_pos[0] = -0.346;
                    init_pos[1] =  0.133;
                    init_pos[2] =  0.162;

//                    /* KARATE init */
//                    // -0.319711 0.128912 0.075052 0.03846 -0.732046 0.680169 2.979943
//                    init_pos[0] = -0.319;
//                    init_pos[1] =  0.128;
//                    init_pos[2] =  0.075;

                    /* GRASPING init */
//                    init_pos[0] = -0.370;
//                    init_pos[1] =  0.103;
//                    init_pos[2] =  0.064;

                    /* SIM init 1 */
//                    init_pos[0] = -0.416;
//                    init_pos[1] =  0.024 + 0.1;
//                    init_pos[2] =  0.055;

                    /* SIM init 2 */
//                    init_pos[0] = -0.35;
//                    init_pos[1] =  0.025 + 0.05;
//                    init_pos[2] =  0.10;

                    yInfo() << "Init: " << init_pos.toString() << " " << od.toString();


//                    setTorsoDOF();

                    /* Normal trials */
//                    Vector gaze_loc(3);
//                    gaze_loc[0] = init_pos[0];
//                    gaze_loc[1] = init_pos[1];
//                    gaze_loc[2] = init_pos[2];

                    /* Trial 27/04/17 */
                    // -6.706 1.394 -3.618
                    Vector gaze_loc(3);
                    gaze_loc[0] = -6.706;
                    gaze_loc[1] =  1.394;
                    gaze_loc[2] = -3.618;

                    yInfo() << "Fixation point: " << gaze_loc.toString();

                    int ctxt;
                    itf_rightarm_cart_->storeContext(&ctxt);

//                    itf_rightarm_cart_->setLimits(0,  15.0,  15.0);
//                    itf_rightarm_cart_->setLimits(2, -23.0, -23.0);
//                    itf_rightarm_cart_->setLimits(3, -16.0, -16.0);
//                    itf_rightarm_cart_->setLimits(4,  53.0,  53.0);
//                    itf_rightarm_cart_->setLimits(5,   0.0,   0.0);
//                    itf_rightarm_cart_->setLimits(7, -58.0, -58.0);

                    itf_rightarm_cart_->goToPoseSync(init_pos, od);
                    itf_rightarm_cart_->waitMotionDone(0.1, 10.0);
                    itf_rightarm_cart_->stopControl();

                    itf_rightarm_cart_->restoreContext(ctxt);
                    itf_rightarm_cart_->deleteContext(ctxt);
                    

                    itf_rightarm_cart_->storeContext(&ctxt);

                    itf_gaze_->lookAtFixationPointSync(gaze_loc);
                    itf_gaze_->waitMotionDone(0.1, 10.0);
                    itf_gaze_->stopControl();

                    itf_rightarm_cart_->restoreContext(ctxt);
                    itf_rightarm_cart_->deleteContext(ctxt);

                    
//                    unsetTorsoDOF();
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


                l_H_eye_to_r_ = axis2dcm(left_eye_o);
                left_eye_x.push_back(1.0);
                l_H_eye_to_r_.setCol(3, left_eye_x);
                l_H_r_to_eye_ = SE3inv(l_H_eye_to_r_);

                r_H_eye_to_r_ = axis2dcm(right_eye_o);
                right_eye_x.push_back(1.0);
                r_H_eye_to_r_.setCol(3, right_eye_x);
                r_H_r_to_eye_ = SE3inv(r_H_eye_to_r_);

                yInfo() << "l_H_r_to_eye_ =\n" << l_H_r_to_eye_.toString();
                yInfo() << "r_H_r_to_eye_ =\n" << r_H_r_to_eye_.toString();

                l_H_r_to_cam_ = l_proj_ * l_H_r_to_eye_;
                r_H_r_to_cam_ = r_proj_ * r_H_r_to_eye_;


                Network yarp;
                Bottle  cmd;
                Bottle  rep;

                Bottle* click_left = port_click_left_.read(true);
                Vector l_click = zeros(2);
                l_click[0] = click_left->get(0).asDouble();
                l_click[1] = click_left->get(1).asDouble();

                RpcClient port_sfm;
                port_sfm.open("/reaching_pose/tosfm");
                yarp.connect("/reaching_pose/tosfm", "/SFM/rpc");

                cmd.clear();

                cmd.addInt(l_click[0]);
                cmd.addInt(l_click[1]);

                Bottle reply_pos;
                port_sfm.write(cmd, reply_pos);
                if (reply_pos.size() == 5)
                {
                    Matrix R_ee = zeros(3, 3);
                    R_ee(0, 0) = -1.0;
                    R_ee(1, 1) =  1.0;
                    R_ee(2, 2) = -1.0;
                    Vector ee_o = dcm2axis(R_ee);

                    Vector sfm_pos = zeros(3);
                    sfm_pos[0] = reply_pos.get(0).asDouble();
                    sfm_pos[1] = reply_pos.get(1).asDouble();
                    sfm_pos[2] = reply_pos.get(2).asDouble();

                    Vector p = zeros(7);
                    p.setSubvector(0, sfm_pos.subVector(0, 2));
                    p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));
                    Vector p0 = zeros(4);
                    Vector p1 = zeros(4);
                    Vector p2 = zeros(4);
                    Vector p3 = zeros(4);
                    getPalmPoints(p, p0, p1, p2, p3);

                    yInfo() << "goal px: [" << p0.toString() << ";" << p1.toString() << ";" << p2.toString() << ";" << p3.toString() << "];";


                    Vector l_px0 = l_H_r_to_cam_ * p0;
                    l_px0[0] /= l_px0[2];
                    l_px0[1] /= l_px0[2];
                    Vector l_px1 = l_H_r_to_cam_ * p1;
                    l_px1[0] /= l_px1[2];
                    l_px1[1] /= l_px1[2];
                    Vector l_px2 = l_H_r_to_cam_ * p2;
                    l_px2[0] /= l_px2[2];
                    l_px2[1] /= l_px2[2];
                    Vector l_px3 = l_H_r_to_cam_ * p3;
                    l_px3[0] /= l_px3[2];
                    l_px3[1] /= l_px3[2];

                    l_px_goal_.resize(8);
                    l_px_goal_[0] = l_px0[0];
                    l_px_goal_[1] = l_px0[1];
                    l_px_goal_[2] = l_px1[0];
                    l_px_goal_[3] = l_px1[1];
                    l_px_goal_[4] = l_px2[0];
                    l_px_goal_[5] = l_px2[1];
                    l_px_goal_[6] = l_px3[0];
                    l_px_goal_[7] = l_px3[1];


                    Vector r_px0 = r_H_r_to_cam_ * p0;
                    r_px0[0] /= r_px0[2];
                    r_px0[1] /= r_px0[2];
                    Vector r_px1 = r_H_r_to_cam_ * p1;
                    r_px1[0] /= r_px1[2];
                    r_px1[1] /= r_px1[2];
                    Vector r_px2 = r_H_r_to_cam_ * p2;
                    r_px2[0] /= r_px2[2];
                    r_px2[1] /= r_px2[2];
                    Vector r_px3 = r_H_r_to_cam_ * p3;
                    r_px3[0] /= r_px3[2];
                    r_px3[1] /= r_px3[2];

                    r_px_goal_.resize(8);
                    r_px_goal_[0] = r_px0[0];
                    r_px_goal_[1] = r_px0[1];
                    r_px_goal_[2] = r_px1[0];
                    r_px_goal_[3] = r_px1[1];
                    r_px_goal_[4] = r_px2[0];
                    r_px_goal_[5] = r_px2[1];
                    r_px_goal_[6] = r_px3[0];
                    r_px_goal_[7] = r_px3[1];
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

                yInfo() << "left_eye_o = ["  << left_eye_o.toString()  << "]";
                yInfo() << "right_eye_o = [" << right_eye_o.toString() << "]";


                l_H_eye_to_r_ = axis2dcm(left_eye_o);
                left_eye_x.push_back(1.0);
                l_H_eye_to_r_.setCol(3, left_eye_x);
                l_H_r_to_eye_ = SE3inv(l_H_eye_to_r_);

                r_H_eye_to_r_ = axis2dcm(right_eye_o);
                right_eye_x.push_back(1.0);
                r_H_eye_to_r_.setCol(3, right_eye_x);
                r_H_r_to_eye_ = SE3inv(r_H_eye_to_r_);

                yInfo() << "l_H_r_to_eye_ = [\n" << l_H_r_to_eye_.toString() << "]";
                yInfo() << "r_H_r_to_eye_ = [\n" << r_H_r_to_eye_.toString() << "]";

                l_H_r_to_cam_ = l_proj_ * l_H_r_to_eye_;
                r_H_r_to_cam_ = r_proj_ * r_H_r_to_eye_;


//                /* Hand pointing forward, palm looking down */
//                Matrix R_ee = zeros(3, 3);
//                R_ee(0, 0) = -1.0;
//                R_ee(1, 1) =  1.0;
//                R_ee(2, 2) = -1.0;
//                Vector ee_o = dcm2axis(R_ee);

                /* Trial 27/04/17 */
                // -0.323 0.018 0.121 0.310 -0.873 0.374 3.008
                Vector p = zeros(6);
                p[0] = -0.323;
                p[1] =  0.018;
                p[2] =  0.121;
                p[3] =  0.310 * 3.008;
                p[4] = -0.873 * 3.008;
                p[5] =  0.374 * 3.008;

                /* KARATE */
//                Vector p = zeros(6);
//                p[0] = -0.319;
//                p[1] =  0.128;
//                p[2] =  0.075;
//                p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));

                /* SIM init 1 */
                // -0.416311	-0.026632	 0.055334	-0.381311	-0.036632	 0.055334	-0.381311	-0.016632	 0.055334
//                Vector p = zeros(6);
//                p[0] = -0.416;
//                p[1] = -0.024;
//                p[2] =  0.055;
//                p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));

                /* SIM init 2 */
//                Vector p = zeros(6);
//                p[0] = -0.35;
//                p[1] =  0.025;
//                p[2] =  0.10;
//                p.setSubvector(3, ee_o.subVector(0, 2) * ee_o(3));

                yInfo() << "Goal: " << p.toString();

                Vector p0 = zeros(4);
                Vector p1 = zeros(4);
                Vector p2 = zeros(4);
                Vector p3 = zeros(4);
                getPalmPoints(p, p0, p1, p2, p3);

                yInfo() << "Goal px: [" << p0.toString() << ";" << p1.toString() << ";" << p2.toString() << ";" << p3.toString() << "];";


                Vector l_px0 = l_H_r_to_cam_ * p0;
                l_px0[0] /= l_px0[2];
                l_px0[1] /= l_px0[2];
                Vector l_px1 = l_H_r_to_cam_ * p1;
                l_px1[0] /= l_px1[2];
                l_px1[1] /= l_px1[2];
                Vector l_px2 = l_H_r_to_cam_ * p2;
                l_px2[0] /= l_px2[2];
                l_px2[1] /= l_px2[2];
                Vector l_px3 = l_H_r_to_cam_ * p3;
                l_px3[0] /= l_px3[2];
                l_px3[1] /= l_px3[2];

                l_px_goal_.resize(8);
                l_px_goal_[0] = l_px0[0];
                l_px_goal_[1] = l_px0[1];
                l_px_goal_[2] = l_px1[0];
                l_px_goal_[3] = l_px1[1];
                l_px_goal_[4] = l_px2[0];
                l_px_goal_[5] = l_px2[1];
                l_px_goal_[6] = l_px3[0];
                l_px_goal_[7] = l_px3[1];


                Vector r_px0 = r_H_r_to_cam_ * p0;
                r_px0[0] /= r_px0[2];
                r_px0[1] /= r_px0[2];
                Vector r_px1 = r_H_r_to_cam_ * p1;
                r_px1[0] /= r_px1[2];
                r_px1[1] /= r_px1[2];
                Vector r_px2 = r_H_r_to_cam_ * p2;
                r_px2[0] /= r_px2[2];
                r_px2[1] /= r_px2[2];
                Vector r_px3 = r_H_r_to_cam_ * p3;
                r_px3[0] /= r_px3[2];
                r_px3[1] /= r_px3[2];

                r_px_goal_.resize(8);
                r_px_goal_[0] = r_px0[0];
                r_px_goal_[1] = r_px0[1];
                r_px_goal_[2] = r_px1[0];
                r_px_goal_[3] = r_px1[1];
                r_px_goal_[4] = r_px2[0];
                r_px_goal_[5] = r_px2[1];
                r_px_goal_[6] = r_px3[0];
                r_px_goal_[7] = r_px3[1];

                
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
            /* Safely close the application */
            case VOCAB4('q','u','i','t'):
            {
                itf_rightarm_cart_->stopControl();
                itf_gaze_->stopControl();

                take_estimates_ = true;
                should_stop_    = true;

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
        yInfo() << "Interrupting module...";

        yInfo() << "...blocking controllers...";
        itf_rightarm_cart_->stopControl();
        itf_gaze_->stopControl();

        Time::delay(3.0);

        yInfo() << "...port cleanup...";
        port_estimates_left_in_.interrupt();
        port_estimates_right_in_.interrupt();
        port_image_left_in_.interrupt();
        port_image_left_out_.interrupt();
        port_click_left_.interrupt();
        port_image_right_in_.interrupt();
        port_image_right_out_.interrupt();
        port_click_right_.interrupt();
        handler_port_.interrupt();

        yInfo() << "...done!";
        return true;
    }

    bool close()
    {
        yInfo() << "Calling close functions...";

        port_estimates_left_in_.close();
        port_estimates_right_in_.close();
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

        yInfo() << "...done!";
        return true;
    }

private:
    ConstString                      robot_name_;

    Port                             handler_port_;
    bool                             should_stop_ = false;

    SISkeleton                     * l_si_skel_;
    SISkeleton                     * r_si_skel_;

    BufferedPort<Vector>             port_estimates_left_in_;
    BufferedPort<Vector>             port_estimates_right_in_;

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

    Matrix                           l_proj_;
    Matrix                           r_proj_;
    Matrix                           l_H_r_to_eye_;
    Matrix                           r_H_r_to_eye_;
    Matrix                           l_H_eye_to_r_;
    Matrix                           r_H_eye_to_r_;
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
        rightarm_cartesian_options.put("remote", "/"+robot_name_+"/cartesianController/right_arm");

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
        rightarm_remote_options.put("remote", "/"+robot_name_+"/right_arm");

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
        torso_remote_options.put("remote", "/"+robot_name_+"/torso");

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

    void getPalmPoints(const Vector& endeffector, Vector& p0, Vector& p1, Vector& p2, Vector& p3)
    {
        Vector ee_x = endeffector.subVector(0, 2);
        ee_x.push_back(1.0);
        double ang  = norm(endeffector.subVector(3, 5));
        Vector ee_o = endeffector.subVector(3, 5) / ang;
        ee_o.push_back(ang);

        Matrix H_ee_to_root = axis2dcm(ee_o);
        H_ee_to_root.setCol(3, ee_x);


        Vector p = zeros(4);

        p(0) =  0;
        p(1) = -0.015;
        p(2) =  0;
        p(3) =  1.0;

        p0 = zeros(4);
        p0 = H_ee_to_root * p;

        p(0) = 0;
        p(1) = 0.015;
        p(2) = 0;
        p(3) = 1.0;

        p1 = zeros(4);
        p1 = H_ee_to_root * p;

        p(0) = -0.035;
        p(1) =  0.015;
        p(2) =  0;
        p(3) =  1.0;

        p2 = zeros(4);
        p2 = H_ee_to_root * p;

        p(0) = -0.035;
        p(1) = -0.015;
        p(2) =  0;
        p(3) =  1.0;

        p3 = zeros(4);
        p3 = H_ee_to_root * p;
    }

    Vector setJacobianU(const int cam, const Vector& px)
    {
        Vector jacobian = zeros(6);

        if (cam == LEFT)
        {
            jacobian(0) = l_proj_(0, 0) / px(2);
            jacobian(2) = - (px(0) - l_proj_(0, 2)) / px(2);
            jacobian(3) = - ((px(0) - l_proj_(0, 2)) * (px(1) - l_proj_(1, 2))) / l_proj_(1, 1);
            jacobian(4) = (pow(l_proj_(0, 0), 2.0) + pow(px(0) - l_proj_(0, 2), 2.0)) / l_proj_(0, 0);
            jacobian(5) = - l_proj_(0, 0) / l_proj_(1, 1) * (px(1) - l_proj_(1, 2));
        }
        else if (cam == RIGHT)
        {
            jacobian(0) = r_proj_(0, 0) / px(2);
            jacobian(2) = - (px(0) - r_proj_(0, 2)) / px(2);
            jacobian(3) = - ((px(0) - r_proj_(0, 2)) * (px(1) - r_proj_(1, 2))) / r_proj_(1, 1);
            jacobian(4) = (pow(r_proj_(0, 0), 2.0) + pow(px(0) - r_proj_(0, 2), 2.0)) / r_proj_(0, 0);
            jacobian(5) = - r_proj_(0, 0) / r_proj_(1, 1) * (px(1) - r_proj_(1, 2));
        }

        return jacobian;
    }

    Vector setJacobianV(const int cam, const Vector& px)
    {
        Vector jacobian = zeros(6);

        if (cam == LEFT)
        {
            jacobian(1) = l_proj_(1, 1) / px(2);
            jacobian(2) = - (px(1) - l_proj_(1, 2)) / px(2);
            jacobian(3) = - (pow(l_proj_(1, 1), 2.0) + pow(px(1) - l_proj_(1, 2), 2.0)) / l_proj_(1, 1);
            jacobian(4) = ((px(0) - l_proj_(0, 2)) * (px(1) - l_proj_(1, 2))) / l_proj_(0, 0);
            jacobian(5) = l_proj_(1, 1) / l_proj_(0, 0) * (px(0) - l_proj_(0, 2));
        }
        else if (cam == RIGHT)
        {
            jacobian(1) = r_proj_(1, 1) / px(2);
            jacobian(2) = - (px(1) - r_proj_(1, 2)) / px(2);
            jacobian(3) = - (pow(r_proj_(1, 1), 2.0) + pow(px(1) - r_proj_(1, 2), 2.0)) / r_proj_(1, 1);
            jacobian(4) = ((px(0) - r_proj_(0, 2)) * (px(1) - r_proj_(1, 2))) / r_proj_(0, 0);
            jacobian(5) = r_proj_(1, 1) / r_proj_(0, 0) * (px(0) - r_proj_(0, 2));
        }

        return jacobian;
    }

    Matrix getSkew(const Vector& v)
    {
        Matrix skew = zeros(3, 3);

        skew(0, 1) = -v(2);
        skew(0, 2) =  v(1);

        skew(1, 0) =  v(2);
        skew(1, 2) = -v(0);

        skew(2, 0) = -v(1);
        skew(2, 1) =  v(0);

        return skew;
    }

    Matrix getGamma(const Vector& p)
    {
        Matrix G = zeros(6, 6);

        G.setSubmatrix(-1.0 * eye(3, 3), 0, 0);
        G.setSubmatrix(getSkew(p)      , 0, 3);
        G.setSubmatrix(-1.0 * eye(3, 3), 3, 3);

        return G;
    }

    Vector getAxisAngle(const Vector& v)
    {
        double ang  = norm(v);
        Vector aa   = v / ang;
        aa.push_back(ang);

        return aa;
    }
};


int main(int argc, char **argv)
{
    Network yarp;
    if (!yarp.checkNetwork(3.0))
    {
        yError() << "YARP seems unavailable!";
        return EXIT_FAILURE;
    }

    ResourceFinder rf;
    rf.configure(argc, argv);
    RFMReaching reaching;
    reaching.runModule(rf);

    return EXIT_SUCCESS;
}
