#include <yarp/os/LogStream.h>
#include <yarp/os/Property.h>
#include <yarp/dev/Drivers.h>
#include <yarp/dev/IVisualServoing.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/math/Math.h>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;
using namespace yarp::math;

YARP_DECLARE_PLUGINS(visualservoingplugin);

int main(int argc, char **argv)
{
    Network yarp;
    if (!yarp.checkNetwork(3.0))
    {
        yError("YARP seems unavailable!");
        return EXIT_FAILURE;
    }

    YARP_REGISTER_PLUGINS(visualservoingplugin);

    Property prop_server_vs;
    prop_server_vs.put("device",    "visualservoingclient");
    prop_server_vs.put("verbosity", true);
    prop_server_vs.put("local",     "/VisualServoingClientTest");
    prop_server_vs.put("remote",    "/visual-servoing");

    PolyDriver drv_server_vs(prop_server_vs);
    if (!drv_server_vs.isValid())
    {
        yError("Could not run VisualServoingClient!");
        return EXIT_FAILURE;
    }

    IVisualServoing *visual_servoing;
    drv_server_vs.view(visual_servoing);
    if (visual_servoing == YARP_NULLPTR)
    {
        yError("Could not get interfacate to VisualServoingClient!");
        return EXIT_FAILURE;
    }

//    Vector x(3);
//    Vector o(4);
//    x[0] = -0.356; x[1] = 0.024; x[2] = -0.053;
//    o[0] = 0.057;  o[1] = 0.98;  o[2] = -0.189; o[3] = 2.525;
//    std::vector<Vector> px_l = visual_servoing->getPixelPositionGoalFrom3DPose(x, o, IVisualServoing::CamSel::left);
//    std::vector<Vector> px_r = visual_servoing->getPixelPositionGoalFrom3DPose(x, o, IVisualServoing::CamSel::right);

    visual_servoing->storedInit("t170427");
    visual_servoing->storedGoToGoal("t170427");

    visual_servoing->checkVisualServoingController();
    visual_servoing->waitVisualServoingDone();

    return EXIT_SUCCESS;
}
