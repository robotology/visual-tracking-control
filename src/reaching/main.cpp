#include <cmath>
#include <iostream>

#include <iCub/ctrl/minJerkCtrl.h>
#include <iCub/iKin/iKinFwd.h>
#include <yarp/dev/CartesianControl.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/math/Math.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/Property.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Time.h>
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

        Vector px_img_left;
        px_img_left.push_back(click_left->get(0).asDouble());
        px_img_left.push_back(click_left->get(1).asDouble());

        Vector px_img_right;
        px_img_right.push_back(click_right->get(0).asDouble());
        px_img_right.push_back(click_right->get(1).asDouble());

        Vector px_des;
        px_des.push_back(px_img_left[0]);  /* u_l */
        px_des.push_back(px_img_right[0]); /* u_r */
        px_des.push_back(px_img_left[1]);  /* v_l */


        Vector px_left;  /* u_h_l, v_h_l */
        Vector px_right; /* u_h_r, v_h_r */
        itf_gaze_->get2DPixel(LEFT,  estimates->subVector(0, 2), px_left);
        itf_gaze_->get2DPixel(RIGHT, estimates->subVector(6, 8), px_right);

        Vector px_hand;
        px_hand.push_back(px_left[0]);  /* u_h_l */
        px_hand.push_back(px_right[0]); /* u_h_r */
        px_hand.push_back(px_left[1]);  /* v_h_l */

        double Ts    = 0.1;  // controller's sample time [s]
        double T     = 20.0; // how long it takes to move to the target [s]
        double v_max = 0.01; // max cartesian velocity [m/s]

        bool done = false;

        double start = Time::now();
        double checkTime;
        while (!this->isStopping() && !done)
        {
            Vector e     = px_des - px_hand;
            Vector vel_x = (px_to_cartesian_ * e) / T;

            /* Enforce velocity bounds */
            for (size_t i = 0; i < vel_x.length(); ++i)
                vel_x[i] = sign(e[i]) * std::min(v_max, std::fabs(vel_x[i]));

            itf_rightarm_cart_->setTaskVelocities(vel_x, Vector(4, 0.0));
            Time::delay(Ts);
            checkTime = Time::now();

            done = (norm(e.subVector(0, 2)) < 0.01 || (checkTime-start) >= 60.0);
            if (done)
            {
                yDebug() << "px_des =" << px_des.toString(3, 3).c_str() << "px_hand =" << px_hand.toString(3, 3).c_str();
                yDebug() << "Checktime " << (checkTime - start);
            }
            else
            {
                estimates = port_estimates_in_.read(true);
                itf_gaze_->get2DPixel(LEFT,  estimates->subVector(0, 2), px_left);
                itf_gaze_->get2DPixel(RIGHT, estimates->subVector(6, 8), px_right);

                px_hand.push_back(px_left[0]);  /* u_h_l */
                px_hand.push_back(px_right[0]); /* u_h_r */
                px_hand.push_back(px_left[1]);  /* v_h_l */
            }
        }
        
        itf_rightarm_cart_->stopControl();

        return true;
    }

    bool configure(ResourceFinder &rf)
    {
        if (!port_estimates_in_.open("/reaching/estimates:i"))
        {
            yError() << "Could not open /reaching/estimates:in port! Closing.";
            return false;
        }

        if (!port_click_left_.open("/reaching/cam_left/click:i"))
        {
            yError() << "Could not open /reaching/cam_left/click:in port! Closing.";
            return false;
        }

        if (!port_click_right_.open("/reaching/cam_right/click:i"))
        {
            yError() << "Could not open /reaching/cam_right/click:in port! Closing.";
            return false;
        }

        if (!setRightArmCartesianController()) return false;

        if (!setGazeController()) return false;

        Bottle btl_cam_info;
        itf_gaze_->getInfo(btl_cam_info);
        yInfo() << "[CAM INFO]" << btl_cam_info.toString();
        Bottle* cam_left_info = btl_cam_info.findGroup("camera_intrinsics_left").get(1).asList();
        Bottle* cam_right_info = btl_cam_info.findGroup("camera_intrinsics_right").get(1).asList();
        float left_fx_ = static_cast<float>(cam_left_info->get(0).asDouble());
        float left_cx_ = static_cast<float>(cam_left_info->get(2).asDouble());
        float left_fy_ = static_cast<float>(cam_left_info->get(5).asDouble());
        float left_cy_ = static_cast<float>(cam_left_info->get(6).asDouble());
        float right_fx_ = static_cast<float>(cam_right_info->get(0).asDouble());
        float right_cx_ = static_cast<float>(cam_right_info->get(2).asDouble());

        Vector left_eye_x;
        Vector left_eye_o;
        itf_gaze_->getLeftEyePose(left_eye_x, left_eye_o);

        Vector right_eye_x;
        Vector right_eye_o;
        itf_gaze_->getLeftEyePose(right_eye_x, right_eye_o);

       
        double l_theta  = left_eye_o[3];
        double l_c = cos(l_theta);
        double l_s = sin(l_theta);
        double l_C = 1.0 - l_c;

        double l_xs  = left_eye_o[0] * l_s;
        double l_ys  = left_eye_o[1] * l_s;
        double l_zs  = left_eye_o[2] * l_s;
        double l_xC  = left_eye_o[0] * l_C;
        double l_yC  = left_eye_o[1] * l_C;
        double l_zC  = left_eye_o[2] * l_C;
        double l_xyC = left_eye_o[0] * l_yC;
        double l_yzC = left_eye_o[1] * l_zC;
        double l_zxC = left_eye_o[2] * l_xC;

        double l_h00 = left_eye_o[0] * l_xC + l_c;
        double l_h01 = l_xyC - l_zs;
        double l_h02 = l_zxC + l_ys;
        double l_h10 = l_xyC + l_zs;
        double l_h11 = left_eye_o[1] * l_yC + l_c;
        double l_h12 = l_yzC - l_xs;
        double l_h20 = l_zxC - l_ys;
        double l_h21 = l_yzC + l_xs;
        double l_h22 = left_eye_o[2] * l_zC + l_c;


        double r_theta = right_eye_o[3];
        double r_c = cos(r_theta);
        double r_s = sin(r_theta);
        double r_C = 1.0 - r_c;
        
        double r_xs  = right_eye_o[0] * r_s;
        double r_ys  = right_eye_o[1] * r_s;
        double r_zs  = right_eye_o[2] * r_s;
        double r_xC  = right_eye_o[0] * r_C;
        double r_yC  = right_eye_o[1] * r_C;
        double r_zC  = right_eye_o[2] * r_C;
        double r_xyC = right_eye_o[0] * r_yC;
        double r_yzC = right_eye_o[1] * r_zC;
        double r_zxC = right_eye_o[2] * r_xC;
        
        double r_h00 = right_eye_o[0] * r_xC + r_c;
        double r_h01 = r_xyC - r_zs;
        double r_h02 = r_zxC + r_ys;
//        double r_h10 = r_xyC + r_zs;
//        double r_h11 = right_eye_o[1] * r_yC + r_c;
//        double r_h12 = r_yzC - r_xs;
        double r_h20 = r_zxC - r_ys;
        double r_h21 = r_yzC + r_xs;
        double r_h22 = right_eye_o[2] * r_zC + r_c;

        px_to_cartesian_ = Matrix(3, 3);
        px_to_cartesian_(0, 0) = left_fx_  * l_h00 + left_cx_  * l_h20;
        px_to_cartesian_(0, 1) = left_fx_  * l_h01 + left_cx_  * l_h21;
        px_to_cartesian_(0, 2) = left_fx_  * l_h02 + left_cx_  * l_h22;

        px_to_cartesian_(1, 0) = right_fx_ * r_h00 + right_cx_ * r_h20;
        px_to_cartesian_(1, 1) = right_fx_ * r_h01 + right_cx_ * r_h21;
        px_to_cartesian_(1, 2) = right_fx_ * r_h02 + right_cx_ * r_h22;

        px_to_cartesian_(2, 0) = left_fy_  * l_h10 + left_cy_  * l_h20;
        px_to_cartesian_(2, 1) = left_fy_  * l_h11 + left_cy_  * l_h21;
        px_to_cartesian_(2, 2) = left_fy_  * l_h12 + left_cy_  * l_h22;

        px_to_cartesian_ = luinv(px_to_cartesian_);

        return true;
    }

    bool interruptModule()
    {
        yInfo() << "Interrupting module.\nPort cleanup...";

        port_estimates_in_.interrupt();
        port_click_left_.interrupt();
        port_click_right_.interrupt();

        return true;
    }

    bool close()
    {
        yInfo() << "Calling close functions...";

        port_estimates_in_.close();
        port_click_left_.close();
        port_click_right_.close();

        itf_rightarm_cart_->removeTipFrame();

        if (rightarm_cartesian_driver_.isValid()) rightarm_cartesian_driver_.close();
        if (gaze_driver_.isValid())               gaze_driver_.close();

        return true;
    }

private:
    BufferedPort<Vector>  port_estimates_in_;

    BufferedPort<Bottle>  port_click_left_;

    BufferedPort<Bottle>  port_click_right_;

    PolyDriver            rightarm_cartesian_driver_;
    ICartesianControl   * itf_rightarm_cart_;

    PolyDriver            gaze_driver_;
    IGazeControl        * itf_gaze_;

    Matrix                px_to_cartesian_;
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
