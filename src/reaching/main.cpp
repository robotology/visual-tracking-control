#include <cmath>
#include <iostream>

#include <iCub/ctrl/minJerkCtrl.h>
#include <iCub/iKin/iKinFwd.h>
#include <opencv2/core/core.hpp>
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
    double getPeriod() { return 0; }

    bool updateModule()
    {
        Bottle* click_left  = port_click_left_.read  (true);
        Bottle* click_right = port_click_right_.read (true);
        Vector* estimates   = port_estimates_in_.read(true);

        yInfo() << "estimates = ["  << estimates->toString() << "]";

        Vector px_img_left;
        px_img_left.push_back(click_left->get(0).asDouble());
        px_img_left.push_back(click_left->get(1).asDouble());

        yInfo() << "px_img_left = [" << px_img_left.toString() << "]";

        Vector px_img_right;
        px_img_right.push_back(click_right->get(0).asDouble());
        px_img_right.push_back(click_right->get(1).asDouble());

        yInfo() << "px_img_right = [" << px_img_right.toString() << "]";

        Vector click_3d_point;
        itf_gaze_->triangulate3DPoint(px_img_left, px_img_right, click_3d_point);
        click_3d_point.push_back(1.0);

        yInfo() << "click_3d_point = ["  << click_3d_point.toString() << "]";

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

        Vector left_proj = (l_H_r_to_cam_.submatrix(0, 2, 0, 2) * (estimates->subVector(0, 2) - left_eye_x));
        yInfo() << "Proj left ee = [" << (left_proj.subVector(0, 1) / left_proj[2]).toString() << "]";

        Vector right_eye_x;
        Vector right_eye_o;
        itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

        Vector right_proj = (r_H_r_to_cam_.submatrix(0, 2, 0, 2) * (estimates->subVector(6, 8) - right_eye_x));
        yInfo() << "Proj right ee = [" << (right_proj.subVector(0, 1) / right_proj[2]).toString() << "]";

        Vector px_ee_now;
        px_ee_now.push_back(left_proj [0] / left_proj[2]); /* u_ee_l */
        px_ee_now.push_back(right_proj[0] / right_proj[2]); /* u_ee_r */
        px_ee_now.push_back(left_proj [1] / left_proj[2]); /* v_ee_l */
        yInfo() << "px_ee_now = [" << px_ee_now.toString() << "]";


        /* Gradient */
        Matrix gradient(3, 3);

        Vector l_ee_x = estimates->subVector(0, 2);
        l_ee_x.push_back(1.0);

        double l_num_u     = dot(l_H_r_to_cam_.subrow(0, 0, 4), l_ee_x);
        double l_num_v     = dot(l_H_r_to_cam_.subrow(1, 0, 4), l_ee_x);
        double l_lambda    = dot(l_H_r_to_cam_.subrow(2, 0, 4), l_ee_x);
        double l_lambda_sq = pow(l_lambda, 2.0);

        gradient(0, 0) = (l_H_r_to_cam_(0, 0) * l_lambda - l_H_r_to_cam_(2, 0) * l_num_u) / l_lambda_sq;
        gradient(0, 1) = (l_H_r_to_cam_(0, 1) * l_lambda - l_H_r_to_cam_(2, 1) * l_num_u) / l_lambda_sq;
        gradient(0, 2) = (l_H_r_to_cam_(0, 2) * l_lambda - l_H_r_to_cam_(2, 2) * l_num_u) / l_lambda_sq;

        gradient(2, 0) = (l_H_r_to_cam_(1, 0) * l_lambda - l_H_r_to_cam_(2, 0) * l_num_v) / l_lambda_sq;
        gradient(2, 1) = (l_H_r_to_cam_(1, 1) * l_lambda - l_H_r_to_cam_(2, 1) * l_num_v) / l_lambda_sq;
        gradient(2, 2) = (l_H_r_to_cam_(1, 2) * l_lambda - l_H_r_to_cam_(2, 2) * l_num_v) / l_lambda_sq;

        Vector r_ee_x = estimates->subVector(6, 8);
        r_ee_x.push_back(1.0);

        double r_num_u     = dot(r_H_r_to_cam_.subrow(0, 0, 4), r_ee_x);
//        double r_num_v     = dot(r_H_r_to_cam_.subrow(1, 0, 4), r_ee_x);
        double r_lambda    = dot(r_H_r_to_cam_.subrow(2, 0, 4), r_ee_x);
        double r_lambda_sq = pow(r_lambda, 2.0);

        gradient(1, 0) = (r_H_r_to_cam_(0, 0) * r_lambda - r_H_r_to_cam_(2, 0) * r_num_u) / r_lambda_sq;
        gradient(1, 1) = (r_H_r_to_cam_(0, 1) * r_lambda - r_H_r_to_cam_(2, 1) * r_num_u) / r_lambda_sq;
        gradient(1, 2) = (r_H_r_to_cam_(0, 2) * r_lambda - r_H_r_to_cam_(2, 2) * r_num_u) / r_lambda_sq;


        double Ts    = 0.25;  // controller's sample time [s]
        double K     = 0.05; // how long it takes to move to the target [s]
        double v_max = 0.005; // max cartesian velocity [m/s]

        bool done = false;
        while (!should_stop_ && !done)
        {
            Vector e = px_des - px_ee_now;
//            Vector vel_x = K * (px_to_cartesian_ * e);

            yInfo() << "e = [" << e.toString() << "]";

            gradient(0, 0) *= -2.0 * e[0];
            gradient(0, 1) *= -2.0 * e[0];
            gradient(0, 2) *= -2.0 * e[0];

            gradient(1, 0) *= -2.0 * e[1];
            gradient(1, 1) *= -2.0 * e[1];
            gradient(1, 2) *= -2.0 * e[1];

            gradient(2, 0) *= -2.0 * e[2];
            gradient(2, 1) *= -2.0 * e[2];
            gradient(2, 2) *= -2.0 * e[2];

            Matrix inv_gradient = luinv(gradient);

            e[0] *= e[0];
            e[1] *= e[1];
            e[2] *= e[2];
            Vector vel_x = -1.0 * K * inv_gradient * e;

            yInfo() << "vel_x = [" << vel_x.toString() << "]";

            yInfo() << "px_des = [" << px_des.toString() << "]";
            yInfo() << "px_ee_now = [" << px_ee_now.toString() << "]";

            itf_rightarm_cart_->setTaskVelocities(vel_x, Vector(4, 0.0));

            yInfo() << "Error norm: " << norm(px_des - px_ee_now) << "\n";

            Time::delay(Ts);

            done = (norm(px_des - px_ee_now) < 5.0);
            if (done)
            {
                yInfo() << "\npx_des =" << px_des.toString() << "px_ee_now =" << px_ee_now.toString();
                yInfo() << "Terminating!\n";
            }
            else
            {
                estimates         = port_estimates_in_.read(true);
//                yInfo() << "EE L now: " << estimates->subVector(0, 2).toString();
//                yInfo() << "EE R now: " << estimates->subVector(6, 8).toString() << "\n";
//
//                estimates->setSubvector(0, estimates->subVector(0, 2) + vel_x);
//                estimates->setSubvector(6, estimates->subVector(6, 8) + vel_x);

                yInfo() << "EE L cor: " << estimates->subVector(0, 2).toString();
                yInfo() << "EE R cor: " << estimates->subVector(6, 8).toString() << "\n";

                Vector left_proj  = (l_H_r_to_cam_.submatrix(0, 2, 0, 2) * (estimates->subVector(0, 2) - left_eye_x));
                Vector right_proj = (r_H_r_to_cam_.submatrix(0, 2, 0, 2) * (estimates->subVector(6, 8) - right_eye_x));

                px_ee_now[0] = left_proj [0] / left_proj[2];    /* u_ee_l */
                px_ee_now[1] = right_proj[0] / right_proj[2];   /* u_ee_r */
                px_ee_now[2] = left_proj [1] / left_proj[2];    /* v_ee_l */


                /* Gradient */
                gradient(3, 3);
                l_ee_x = estimates->subVector(0, 2);
                l_ee_x.push_back(1.0);

                l_num_u     = dot(l_H_r_to_cam_.subrow(0, 0, 4), l_ee_x);
                l_num_v     = dot(l_H_r_to_cam_.subrow(1, 0, 4), l_ee_x);
                l_lambda    = dot(l_H_r_to_cam_.subrow(2, 0, 4), l_ee_x);
                l_lambda_sq = pow(l_lambda, 2.0);

                gradient(0, 0) = (l_H_r_to_cam_(0, 0) * l_lambda - l_H_r_to_cam_(2, 0) * l_num_u) / l_lambda_sq;
                gradient(0, 1) = (l_H_r_to_cam_(0, 1) * l_lambda - l_H_r_to_cam_(2, 1) * l_num_u) / l_lambda_sq;
                gradient(0, 2) = (l_H_r_to_cam_(0, 2) * l_lambda - l_H_r_to_cam_(2, 2) * l_num_u) / l_lambda_sq;

                gradient(2, 0) = (l_H_r_to_cam_(1, 0) * l_lambda - l_H_r_to_cam_(2, 0) * l_num_v) / l_lambda_sq;
                gradient(2, 1) = (l_H_r_to_cam_(1, 1) * l_lambda - l_H_r_to_cam_(2, 1) * l_num_v) / l_lambda_sq;
                gradient(2, 2) = (l_H_r_to_cam_(1, 2) * l_lambda - l_H_r_to_cam_(2, 2) * l_num_v) / l_lambda_sq;

                r_ee_x = estimates->subVector(6, 8);
                r_ee_x.push_back(1.0);

                r_num_u     = dot(r_H_r_to_cam_.subrow(0, 0, 4), r_ee_x);
//                r_num_v     = dot(r_H_r_to_cam_.subrow(1, 0, 4), r_ee_x);
                r_lambda    = dot(r_H_r_to_cam_.subrow(2, 0, 4), r_ee_x);
                r_lambda_sq = pow(r_lambda, 2.0);

                gradient(1, 0) = (r_H_r_to_cam_(0, 0) * r_lambda - r_H_r_to_cam_(2, 0) * r_num_u) / r_lambda_sq;
                gradient(1, 1) = (r_H_r_to_cam_(0, 1) * r_lambda - r_H_r_to_cam_(2, 1) * r_num_u) / r_lambda_sq;
                gradient(1, 2) = (r_H_r_to_cam_(0, 2) * r_lambda - r_H_r_to_cam_(2, 2) * r_num_u) / r_lambda_sq;


                /* Left eye end-effector superimposition */
                SuperImpose::ObjPoseMap l_click_pose;
                SuperImpose::ObjPose    l_click;
                l_click.assign(click_3d_point.data(), click_3d_point.data()+3);
                l_click_pose.emplace("palm", l_click);

                SuperImpose::ObjPoseMap l_ee_pose;
                SuperImpose::ObjPose    l_pose;
                l_pose.assign((*estimates).data(), (*estimates).data()+3);
                l_ee_pose.emplace("palm", l_pose);

                ImageOf<PixelRgb>* l_imgin  = port_image_left_in_.read(true);
                ImageOf<PixelRgb>& l_imgout = port_image_left_out_.prepare();
                l_imgout = *l_imgin;
                cv::Mat l_img = cv::cvarrToMat(l_imgout.getIplImage());

                Vector left_eye_x;
                Vector left_eye_o;
                itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

                l_si_skel_->superimpose(l_ee_pose,    left_eye_x.data(), left_eye_o.data(), l_img);
                l_si_skel_->superimpose(l_click_pose, left_eye_x.data(), left_eye_o.data(), l_img);
                
                port_image_left_out_.write();

                /* Right eye end-effector superimposition */
                SuperImpose::ObjPoseMap r_click_pose;
                SuperImpose::ObjPose    r_click;
                r_click.assign(click_3d_point.data(), click_3d_point.data()+3);
                r_click_pose.emplace("palm", r_click);

                SuperImpose::ObjPoseMap r_ee_pose;
                SuperImpose::ObjPose    r_pose;
                r_pose.assign((*estimates).data()+6, (*estimates).data()+9);
                r_ee_pose.emplace("palm", r_pose);

                ImageOf<PixelRgb>* r_imgin  = port_image_right_in_.read(true);
                ImageOf<PixelRgb>& r_imgout = port_image_right_out_.prepare();
                r_imgout = *r_imgin;
                cv::Mat r_img = cv::cvarrToMat(r_imgout.getIplImage());

                Vector right_eye_x;
                Vector right_eye_o;
                itf_gaze_->getRightEyePose(right_eye_x, right_eye_o);

                r_si_skel_->superimpose(r_ee_pose,    right_eye_x.data(), right_eye_o.data(), r_img);
                r_si_skel_->superimpose(r_click_pose, right_eye_x.data(), right_eye_o.data(), r_img);
                
                port_image_right_out_.write();
            }
        }
        
        itf_rightarm_cart_->stopControl();

        Time::delay(0.5);

        return false;
    }

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

        if (!setRightArmCartesianController()) return false;

        if (!setGazeController()) return false;

        Bottle btl_cam_info;
        itf_gaze_->getInfo(btl_cam_info);
        yInfo() << "[CAM INFO]" << btl_cam_info.toString();
        Bottle* cam_left_info = btl_cam_info.findGroup("camera_intrinsics_left").get(1).asList();
        Bottle* cam_right_info = btl_cam_info.findGroup("camera_intrinsics_right").get(1).asList();

        float left_fx  = static_cast<float>(cam_left_info->get(0).asDouble());
        float left_cx  = static_cast<float>(cam_left_info->get(2).asDouble());
        float left_fy  = static_cast<float>(cam_left_info->get(5).asDouble());
        float left_cy  = static_cast<float>(cam_left_info->get(6).asDouble());

        Matrix left_proj(3, 4);
        left_proj(0, 0) = left_fx;
        left_proj(0, 2) = left_cx;
        left_proj(1, 1) = left_fy;
        left_proj(1, 2) = left_cy;
        left_proj(2, 2) = 1.0;

        yInfo() << "left_proj =\n" << left_proj.toString();

        float right_fx = static_cast<float>(cam_right_info->get(0).asDouble());
        float right_cx = static_cast<float>(cam_right_info->get(2).asDouble());
        float right_fy = static_cast<float>(cam_right_info->get(5).asDouble());
        float right_cy = static_cast<float>(cam_right_info->get(6).asDouble());

        Matrix right_proj(3, 4);
        right_proj(0, 0) = right_fx;
        right_proj(0, 2) = right_cx;
        right_proj(1, 1) = right_fy;
        right_proj(1, 2) = right_cy;
        right_proj(2, 2) = 1.0;

        yInfo() << "right_proj =\n" << right_proj.toString();

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

        l_H_r_to_cam_ = left_proj  * l_H_r_to_eye;
        r_H_r_to_cam_ = right_proj * r_H_r_to_eye;

        yInfo() << "l_H_r_to_cam_ =\n" << l_H_r_to_cam_.toString();
        yInfo() << "r_H_r_to_cam_ =\n" << r_H_r_to_cam_.toString();


        Matrix cartesian_to_px(5, 3);

        cartesian_to_px(0, 0) = l_H_r_to_cam_(0, 0);
        cartesian_to_px(0, 1) = l_H_r_to_cam_(0, 1);
        cartesian_to_px(0, 2) = l_H_r_to_cam_(0, 2);

        cartesian_to_px(1, 0) = r_H_r_to_cam_(0, 0);
        cartesian_to_px(1, 1) = r_H_r_to_cam_(0, 1);
        cartesian_to_px(1, 2) = r_H_r_to_cam_(0, 2);

        cartesian_to_px(2, 0) = l_H_r_to_cam_(1, 0);
        cartesian_to_px(2, 1) = l_H_r_to_cam_(1, 1);
        cartesian_to_px(2, 2) = l_H_r_to_cam_(1, 2);

        cartesian_to_px(3, 0) = l_H_r_to_cam_(2, 0);
        cartesian_to_px(3, 1) = l_H_r_to_cam_(2, 1);
        cartesian_to_px(3, 2) = l_H_r_to_cam_(2, 2);

        cartesian_to_px(4, 0) = r_H_r_to_cam_(2, 0);
        cartesian_to_px(4, 1) = r_H_r_to_cam_(2, 1);
        cartesian_to_px(4, 2) = r_H_r_to_cam_(2, 2);

        px_to_cartesian_ = pinv(cartesian_to_px);

        yInfo() << "px_to_cartesian_ =\n" << px_to_cartesian_.toString();

        l_si_skel_ = new SISkeleton(left_fx,  left_fy,  left_cx,  left_cy);
        r_si_skel_ = new SISkeleton(right_fx, right_fy, right_cx, right_cy);

        handler_port_.open("/reaching/cmd:i");
        attach(handler_port_);

        return true;
    }

    bool respond(const Bottle& command, Bottle& reply)
    {
        int cmd = command.get(0).asVocab();
        switch (cmd)
        {
            case VOCAB4('q','u','i','t'):
            {
                reply = command;
                should_stop_ = true;
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

    PolyDriver                       rightarm_cartesian_driver_;
    ICartesianControl              * itf_rightarm_cart_;

    PolyDriver                       gaze_driver_;
    IGazeControl                   * itf_gaze_;

    Matrix                           l_H_r_to_cam_;
    Matrix                           r_H_r_to_cam_;
    Matrix                           px_to_cartesian_;
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

        if (!itf_rightarm_cart_->setTrajTime(20.0))
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
