#include <chrono>
#include <future>
#include <iostream>

#include <yarp/os/ConstString.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <opencv2/core/cuda.hpp>

#include <BayesFiltersLib/FilteringContext.h>
#include <BayesFiltersLib/FilteringFunction.h>
#include <BayesFiltersLib/SIRParticleFilter.h>

#include "BrownianMotion.h"
#include "VisualProprioception.h"
#include "VisualParticleFilterCorrection.h"
#include "VisualSIRParticleFilter.h"

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace yarp::os;


std::string engine_count_to_string(int engine_count)
{
    if (engine_count == 0) return "concurrency is unsupported on this device";
    if (engine_count == 1) return "the device can concurrently copy memory between host and device while executing a kernel";
    if (engine_count == 2) return "the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time";
    return "wrong argument...!";
}


int main(int argc, char const *argv[])
{
    ConstString log_ID = "[Main]";
    yInfo() << log_ID << "Configuring and starting module...";

    cuda::DeviceInfo gpu_dev;
    yInfo() << log_ID << "[CUDA] Engine count:" << engine_count_to_string(gpu_dev.asyncEngineCount());
    yInfo() << log_ID << "[CUDA] Concurrent kernel:" << gpu_dev.concurrentKernels();
    yInfo() << log_ID << "[CUDA] Multiprocessor count:" << gpu_dev.multiProcessorCount();
    yInfo() << log_ID << "[CUDA] Device can map host memory:" << gpu_dev.canMapHostMemory();

    Network yarp;
    if (!yarp.checkNetwork(3.0))
    {
        yError() << "YARP seems unavailable!";
        return EXIT_FAILURE;
    }

    /* Initialize OpenGL context */
    const int num_particles = 50; /* Must be even. */
    if (!VisualProprioception::initOGL(320, 240, num_particles / 2)) return EXIT_FAILURE;

    std::shared_ptr<BrownianMotion> brown(new BrownianMotion());

    std::shared_ptr<ParticleFilterPrediction> pf_prediction(new ParticleFilterPrediction(brown));

    std::shared_ptr<VisualProprioception> proprio(new VisualProprioception("right"));

    std::shared_ptr<VisualParticleFilterCorrection> vpf_correction(new VisualParticleFilterCorrection(proprio, num_particles, 2));

    std::shared_ptr<Resampling> resampling(new Resampling());

    VisualSIRParticleFilter vsir_pf(brown, pf_prediction,
                                    proprio, vpf_correction,
                                    resampling,
                                    num_particles, 2);

    std::future<void> thr_vpf = vsir_pf.spawn();
    while (vsir_pf.isRunning())
    {
        if (VisualProprioception::oglWindowShouldClose())
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

