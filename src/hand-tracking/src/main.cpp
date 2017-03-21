#include <chrono>
#include <future>
#include <iostream>

#include <BayesFiltersLib/FilteringContext.h>
#include <BayesFiltersLib/FilteringFunction.h>
#include <BayesFiltersLib/SIRParticleFilter.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/Value.h>
#include <opencv2/core/cuda.hpp>

#include "BrownianMotion.h"
#include "VisualProprioception.h"
#include "VisualParticleFilterCorrection.h"
#include "VisualSIRParticleFilter.h"

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace yarp::dev;
using namespace yarp::os;


/* UTILITY FUNCTIONS FOREWORD DECLARATIONS */
std::string engine_count_to_string(int engine_count);


/* MAIN */
int main(int argc, char *argv[])
{
    ConstString log_ID = "[Main]";
    yInfo() << log_ID << "Configuring and starting module...";

    Network yarp;
    if (!yarp.checkNetwork(3.0))
    {
        yError() << "YARP seems unavailable!";
        return EXIT_FAILURE;
    }

    cuda::DeviceInfo gpu_dev;
    yInfo() << log_ID << "[CUDA] Engine capability:"              << engine_count_to_string(gpu_dev.asyncEngineCount());
    yInfo() << log_ID << "[CUDA] Can have concurrent kernel:"     << gpu_dev.concurrentKernels();
    yInfo() << log_ID << "[CUDA] Streaming multiprocessor count:" << gpu_dev.multiProcessorCount();
    yInfo() << log_ID << "[CUDA] Device can map host memory:"     << gpu_dev.canMapHostMemory();

    ResourceFinder rf;
    rf.setVerbose();
    rf.setDefaultContext("hand-tracking");
    rf.setDefaultConfigFile("parameters.ini");
    rf.configure(argc, argv);

    ConstString robot_name       = rf.find("robot").asString();
    ConstString robot_cam_sel    = rf.find("cam").asString();
    ConstString robot_laterality = rf.find("laterality").asString();
    const int   num_particles    = rf.findGroup("PF").check("num_particles", Value(50)).asInt();

    yInfo() << log_ID << "Running with:";
    yInfo() << log_ID << " - robot name:"          << robot_name;
    yInfo() << log_ID << " - robot camera:"        << robot_cam_sel;
    yInfo() << log_ID << " - robot laterality:"    << robot_laterality;
    yInfo() << log_ID << " - number of particles:" << num_particles;

    /* Initialize filtering functions */
    std::shared_ptr<BrownianMotion> brown(new BrownianMotion(0.005, 0.005, 3.0, 1.5, 1));

    std::shared_ptr<ParticleFilterPrediction> pf_prediction(new ParticleFilterPrediction(brown));

    std::shared_ptr<VisualProprioception> proprio(new VisualProprioception(num_particles / gpu_dev.multiProcessorCount(), robot_cam_sel, robot_laterality));

    std::shared_ptr<VisualParticleFilterCorrection> vpf_correction(new VisualParticleFilterCorrection(proprio, gpu_dev.multiProcessorCount()));

    std::shared_ptr<Resampling> resampling(new Resampling());

    VisualSIRParticleFilter vsir_pf(pf_prediction,
                                    proprio, vpf_correction,
                                    resampling,
                                    robot_cam_sel, robot_laterality, num_particles);

    std::future<void> thr_vpf = vsir_pf.spawn();
    while (vsir_pf.isRunning())
    {
        if (proprio->oglWindowShouldClose())
        {
            vsir_pf.stopThread();
            yInfo() << log_ID << "Joining filthering thread...";
            while (thr_vpf.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) glfwPollEvents();
        }
        else glfwWaitEvents();
    }

    glfwMakeContextCurrent(NULL);
    glfwTerminate();

    yInfo() << log_ID << "Main returning.";
    yInfo() << log_ID << "Application closed.";

    return EXIT_SUCCESS;
}


/* UTILITY FUNCTIONS */
std::string engine_count_to_string(int engine_count)
{
    if (engine_count == 0) return "concurrency is unsupported on this device";
    if (engine_count == 1) return "the device can concurrently copy memory between host and device while executing a kernel";
    if (engine_count == 2) return "the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time";
    return "wrong argument...!";
}
