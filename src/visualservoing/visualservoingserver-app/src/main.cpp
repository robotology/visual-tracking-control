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
    prop_server_vs.put("device",    "visualservoingserver");
    prop_server_vs.put("verbosity", true);
    prop_server_vs.put("simulate",  true);
    prop_server_vs.put("robot",     "icubSim");
    bool use_fwd_kin = false;

    PolyDriver drv_server_vs(prop_server_vs);
    if (!drv_server_vs.isValid())
    {
        yError("Could not run VisualServoingServer!");
        return EXIT_FAILURE;
    }

    IVisualServoing *visual_servoing;
    drv_server_vs.view(visual_servoing);
    if (visual_servoing == YARP_NULLPTR)
    {
        yError("Could not get interfacate to VisualServoingServer!");
        return EXIT_FAILURE;
    }

    /* Stored set-up: t170904 */
    visual_servoing->storedInit("t170904");

    visual_servoing->initFacilities(use_fwd_kin);
    visual_servoing->storedGoToGoal("t170904");

    visual_servoing->checkVisualServoingController();
    visual_servoing->waitVisualServoingDone();

    visual_servoing->stopFacilities();

    /* Stored set-up: t170427 */
    visual_servoing->storedInit("t170427");

    visual_servoing->initFacilities(use_fwd_kin);
    visual_servoing->storedGoToGoal("t170427");

    visual_servoing->checkVisualServoingController();
    visual_servoing->waitVisualServoingDone();

    visual_servoing->stopFacilities();

    return EXIT_SUCCESS;
}
