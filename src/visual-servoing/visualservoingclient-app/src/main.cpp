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

    Property prop_client_vs;
    prop_client_vs.put("device",    "visualservoingclient");
    prop_client_vs.put("verbosity", true);
    prop_client_vs.put("local",     "/VisualServoingClientTest");
    prop_client_vs.put("remote",    "/visual-servoing");

    PolyDriver drv_client_vs(prop_client_vs);
    if (!drv_client_vs.isValid())
    {
        yError("Could not run VisualServoingClient!");
        return EXIT_FAILURE;
    }

    IVisualServoing *visual_servoing;
    drv_client_vs.view(visual_servoing);
    if (visual_servoing == YARP_NULLPTR)
    {
        yError("Could not get interfacate to VisualServoingClient!");
        return EXIT_FAILURE;
    }

    /* Stored set-up */
    visual_servoing->storedInit("t170427");
    visual_servoing->storedGoToGoal("t170427");

    visual_servoing->checkVisualServoingController();
    visual_servoing->waitVisualServoingDone();

    /* Stored set-up */
    visual_servoing->storedInit("t170713");
    visual_servoing->storedGoToGoal("t170713");

    visual_servoing->checkVisualServoingController();
    visual_servoing->waitVisualServoingDone();

    /* Pixel go to goal */
    visual_servoing->storedInit("t170713");

    Vector x(3);
    Vector o(4);
    x[0] = -0.282; x[1] = 0.061; x[2] = 0.068;
    o[0] = 0.213; o[1] = -0.94; o[2] = 0.265; o[3] = 2.911;

    std::vector<Vector> px_l = visual_servoing->getPixelPositionGoalFrom3DPose(x, o, IVisualServoing::CamSel::left);
    std::vector<Vector> px_r = visual_servoing->getPixelPositionGoalFrom3DPose(x, o, IVisualServoing::CamSel::right);
    visual_servoing->goToGoal(px_l, px_r);

    visual_servoing->checkVisualServoingController();
    visual_servoing->waitVisualServoingDone();

    /* Pose go to goal */
    visual_servoing->storedInit("t170713");
    visual_servoing->goToGoal(x, o);

    visual_servoing->checkVisualServoingController();
    visual_servoing->waitVisualServoingDone();

    return EXIT_SUCCESS;
}
