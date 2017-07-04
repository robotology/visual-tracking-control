#include "ServerVisualServoing.h"

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


int main(int argc, char **argv)
{
    Network yarp;
    if (!yarp.checkNetwork(3.0))
    {
        yError("YARP seems unavailable!");
        return EXIT_FAILURE;
    }

    DriverCreator *server_vs = new DriverCreatorOf<ServerVisualServoing>("server_visualsevoing", "", "ServerVisualServoing");
    Drivers::factory().add(server_vs);

    Property prop_server_vs;
    prop_server_vs.put("device",    "server_visualsevoing");
    prop_server_vs.put("verbosity", true);
    prop_server_vs.put("simulate",  true);
    prop_server_vs.put("robot",     "icubSim");

    PolyDriver drv_server_vs(prop_server_vs);
    if (!drv_server_vs.isValid())
    {
        yError("Could not run ServerVisualServoing!");
        return EXIT_FAILURE;
    }

    IVisualServoing *visual_servoing;
    drv_server_vs.view(visual_servoing);
    if (visual_servoing == YARP_NULLPTR)
    {
        yError("Could not get interfacate to ServerVisualServoing!");
        return EXIT_FAILURE;
    }

    visual_servoing->storedInit("t170427");
    visual_servoing->storedGoToGoal("t170427");
    visual_servoing->checkVisualServoingController();
    visual_servoing->waitVisualServoingDone();

    return EXIT_SUCCESS;
}
