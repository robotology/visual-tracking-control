#include "playGatePose.h"

#include <iCub/ctrl/math.h>
#include <yarp/eigen/Eigen.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Property.h>

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace iCub::ctrl;
using namespace iCub::iKin;
using namespace yarp::eigen;
using namespace yarp::math;
using namespace yarp::os;
using namespace yarp::sig;


playGatePose::playGatePose(std::unique_ptr<VisualCorrection> visual_correction,
                           double gate_x, double gate_y, double gate_z,
                           double gate_aperture, double gate_rotation,
                           const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept :
    GatePose(std::move(visual_correction),
             gate_x, gate_y, gate_z,
             gate_aperture, gate_rotation),
    robot_(robot), laterality_(laterality), port_prefix_(port_prefix)
{
    /* Arm encoders:   /icub/right_arm/state:o
       Torso encoders: /icub/torso/state:o     */
    port_arm_enc_.open  ("/hand-tracking/" + ID_ + "/" + port_prefix_ + "/" + laterality_ + "_arm:i");
    port_torso_enc_.open("/hand-tracking/" + ID_ + "/" + port_prefix_ + "/torso:i");

    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);

    yInfo() << log_ID_ << "Succesfully initialized.";
}


playGatePose::playGatePose(std::unique_ptr<VisualCorrection> visual_correction,
                           const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept :
playGatePose(std::move(visual_correction), 0.1, 0.1, 0.1, 30, 5, robot, laterality, port_prefix) { }


playGatePose::~playGatePose() noexcept { }


Vector playGatePose::readTorso()
{
    Bottle* b = port_torso_enc_.read();
    if (!b) return Vector(3, 0.0);

    yAssert(b->size() == 3);

    Vector torso_enc(3);
    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector playGatePose::readRootToEE()
{
    Bottle* b = port_arm_enc_.read();
    if (!b) return Vector(10, 0.0);

    yAssert(b->size() == 16);

    Vector root_ee_enc(10);
    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
    {
        root_ee_enc(i+3) = b->get(i).asDouble();
    }

    return root_ee_enc;
}


VectorXd playGatePose::readPose()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    
    return toEigen(ee_pose);
}
