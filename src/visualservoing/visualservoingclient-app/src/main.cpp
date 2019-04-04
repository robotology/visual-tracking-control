/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

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
    prop_client_vs.put("remote",    "/visualservoing");
    bool use_fwd_kin = false;

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

    /* Set visual servoing control*/
    visual_servoing->setVisualServoControl("decoupled");

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
