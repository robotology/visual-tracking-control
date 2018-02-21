#include <PlayGatePose.h>

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


PlayGatePose::PlayGatePose(std::unique_ptr<PFVisualCorrection> visual_correction,
                           const double gate_x, const double gate_y, const double gate_z,
                           const double gate_rotation,
                           const double gate_aperture,
                           const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept :
    GatePose(std::move(visual_correction),
             gate_x, gate_y, gate_z,
             gate_rotation,
             gate_aperture),
    icub_kin_arm_(iCubArm(laterality + "_v2")), robot_(robot), laterality_(laterality), port_prefix_(port_prefix)
{
    icub_kin_arm_.setAllConstraints(false);
    icub_kin_arm_.releaseLink(0);
    icub_kin_arm_.releaseLink(1);
    icub_kin_arm_.releaseLink(2);


    port_arm_enc_.open  ("/" + port_prefix_ + "/" + laterality_ + "_arm:i");
    port_torso_enc_.open("/" + port_prefix_ + "/torso:i");


    yInfo() << log_ID_ << "Succesfully initialized.";
}


PlayGatePose::PlayGatePose(std::unique_ptr<PFVisualCorrection> visual_correction,
                           const yarp::os::ConstString& robot, const yarp::os::ConstString& laterality, const yarp::os::ConstString& port_prefix) noexcept :
    PlayGatePose(std::move(visual_correction), 0.1, 0.1, 0.1, 5, 30, robot, laterality, port_prefix) { }


PlayGatePose::~PlayGatePose() noexcept { }


Vector PlayGatePose::readTorso()
{
    Bottle* b = port_torso_enc_.read(true);
    if (!b) return Vector(3, 0.0);

    Vector torso_enc(3);
    torso_enc(0) = b->get(2).asDouble();
    torso_enc(1) = b->get(1).asDouble();
    torso_enc(2) = b->get(0).asDouble();

    return torso_enc;
}


Vector PlayGatePose::readRootToEE()
{
    Bottle* b = port_arm_enc_.read(true);
    if (!b) return Vector(10, 0.0);

    Vector root_ee_enc(10);
    root_ee_enc.setSubvector(0, readTorso());
    for (size_t i = 0; i < 7; ++i)
    {
        root_ee_enc(i+3) = b->get(i).asDouble();
    }

    return root_ee_enc;
}


VectorXd PlayGatePose::readPose()
{
    Vector ee_pose = icub_kin_arm_.EndEffPose(CTRL_DEG2RAD * readRootToEE());
    
    return toEigen(ee_pose);
}
