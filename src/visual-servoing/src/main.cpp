#include "ServerVisualServoing.h"

#include <yarp/os/LogStream.h>
#include <yarp/os/Property.h>
#include <yarp/dev/Drivers.h>
#include <yarp/dev/IVisualServoing.h>
#include <yarp/dev/PolyDriver.h>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;


int main(int argc, char **argv)
{
    DriverCreator *server_vs = new DriverCreatorOf<ServerVisualServoing>("server_visualsevoing", "", "ServerVisualServoing");
    Drivers::factory().add(server_vs);

    Property prop_server_vs;
    prop_server_vs.put("device",    "server_visualsevoing");
    prop_server_vs.put("verbosity", true);
    prop_server_vs.put("robot",     "iCub");

    PolyDriver drv_server_vs(prop_server_vs);
    if (!drv_server_vs.isValid())
    {
        yError("drv_server_vs not available.");
        return EXIT_FAILURE;
    }

    IVisualServoing *visual_servoing;
    drv_server_vs.view(visual_servoing);
    if (visual_servoing == YARP_NULLPTR)
    {
        yError("Could not view the visual servoing.");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
