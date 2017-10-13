#include <chrono>
#include <future>
#include <iostream>
#include <memory>

#include <BayesFilters/Resampling.h>
#include <BayesFilters/SIRParticleFilter.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/Value.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "BrownianMotionPose.h"
#include "DrawParticlesPose.h"
#include "iCubGatePose.h"
#include "iCubFwdKinModel.h"
#include "InitiCubArm.h"
#include "PlayFwdKinModel.h"
#include "PlayGatePose.h"
#include "ResamplingWithPrior.h"
#include "VisualProprioception.h"
#include "VisualSIRParticleFilter.h"
#include "VisualUpdateParticles.h"

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


    // Page locked dovrebbe essere più veloce da utilizzate con CUDA, non sembra essere il caso.
//    Mat::setDefaultAllocator(cuda::HostMem::getAllocator(cuda::HostMem::PAGE_LOCKED));

    cuda::DeviceInfo gpu_dev;
    yInfo() << log_ID << "[CUDA] Engine capability:"              << engine_count_to_string(gpu_dev.asyncEngineCount());
    yInfo() << log_ID << "[CUDA] Can have concurrent kernel:"     << gpu_dev.concurrentKernels();
    yInfo() << log_ID << "[CUDA] Streaming multiprocessor count:" << gpu_dev.multiProcessorCount();
    yInfo() << log_ID << "[CUDA] Can map host memory:"            << gpu_dev.canMapHostMemory();
    yInfo() << log_ID << "[CUDA] Clock:"                          << gpu_dev.clockRate() << "KHz";


    ResourceFinder rf;
    rf.setVerbose();
    rf.setDefaultContext("hand-tracking");
    rf.setDefaultConfigFile("config.ini");
    rf.configure(argc, argv);

    FilteringParamtersD paramsd;
    paramsd["play"]            = ((rf.findGroup("play").size() == 1 ? 1.0 : (rf.findGroup("play").size() == 2 ? rf.find("play").asBool() : 0.0)));
    paramsd["num_particles"]   = rf.findGroup("PF").check("num_particles", Value(50)).asInt();
    paramsd["gpu_count"]       = 1;
    paramsd["num_images"]      = paramsd["num_particles"] / paramsd["gpu_count"];
    paramsd["likelihood_gain"] = 0.001;

    FilteringParamtersS paramss;
    paramss["robot_name"]       = rf.check("robot",      Value("icub")).asString();
    paramss["robot_cam_sel"]    = rf.check("cam",        Value("left")).asString();
    paramss["robot_laterality"] = rf.check("laterality", Value("right")).asString();


    yInfo() << log_ID << "Running with:";
    yInfo() << log_ID << " - robot name:"          << paramss["robot_name"];
    yInfo() << log_ID << " - robot camera:"        << paramss["robot_cam_sel"];
    yInfo() << log_ID << " - robot laterality:"    << paramss["robot_laterality"];
    yInfo() << log_ID << " - use data from ports:" << (paramsd["play"] == 1.0 ? "true" : "false");
    yInfo() << log_ID << " - number of particles:" << paramsd["num_particles"];


    /* INITIALIZATION */
    std::unique_ptr<InitiCubArm> init_arm(new InitiCubArm("hand-tracking/InitiCubArm", paramss["robot_cam_sel"], paramss["robot_laterality"]));


    /* MOTION MODEL */
    std::unique_ptr<BrownianMotionPose> brown(new BrownianMotionPose(0.005, 0.005, 3.0, 2.5, 1));
    std::unique_ptr<StateModel> icub_motion;
    if (paramsd["play"] != 1.0)
    {
        std::unique_ptr<iCubFwdKinModel> icub_fwdkin(new iCubFwdKinModel(std::move(brown), paramss["robot_name"], paramss["robot_laterality"], paramss["robot_cam_sel"]));
        icub_motion = std::move(icub_fwdkin);
    }
    else
    {
        std::unique_ptr<PlayFwdKinModel> play_fwdkin(new PlayFwdKinModel(std::move(brown), paramss["robot_name"], paramss["robot_laterality"], paramss["robot_cam_sel"]));
        icub_motion = std::move(play_fwdkin);
    }

    /* PREDICTION */
    std::unique_ptr<DrawParticlesPose> pf_prediction(new DrawParticlesPose(std::move(icub_motion)));


    /* SENSOR MODEL */
    std::unique_ptr<VisualProprioception> proprio;
    try
    {
        std::unique_ptr<VisualProprioception> vp(new VisualProprioception(paramsd["num_images"], paramss["robot_cam_sel"], paramss["robot_laterality"], rf.getContext()));

        proprio = std::move(vp);
        paramsd["num_particles"] = proprio->getOGLTilesRows() * proprio->getOGLTilesCols() * paramsd["gpu_count"];
    }
    catch (const std::runtime_error& e)
    {
        yError() << e.what();
        return EXIT_FAILURE;
    }

    /* CORRECTION */
    std::unique_ptr<VisualUpdateParticles> vpf_correction(new VisualUpdateParticles(std::move(proprio), paramsd["likelihood_gain"], paramsd["gpu_count"]));

    std::unique_ptr<GatePose> vpf_correction_gated;
    if (paramsd["play"] != 1.0)
    {
        std::unique_ptr<iCubGatePose> icub_gate_pose(new iCubGatePose(std::move(vpf_correction),
                                                                      0.1, 0.1, 0.1, 15, 30,
                                                                      paramss["robot_name"], paramss["robot_laterality"], paramss["robot_cam_sel"]));
        vpf_correction_gated = std::move(icub_gate_pose);
    }
    else
    {
        std::unique_ptr<PlayGatePose> icub_gate_pose(new PlayGatePose(std::move(vpf_correction),
                                                                      0.1, 0.1, 0.1, 15, 30,
                                                                      paramss["robot_name"], paramss["robot_laterality"], paramss["robot_cam_sel"]));
        vpf_correction_gated = std::move(icub_gate_pose);
    }


    /* RESAMPLING */
//    std::unique_ptr<Resampling> resampling(new Resampling());
    std::unique_ptr<Resampling> resampling(new ResamplingWithPrior("hand-tracking/ResamplingWithPrior", paramss["robot_cam_sel"], paramss["robot_laterality"]));


    /* PARTICLE FILTER */
    VisualSIRParticleFilter vsir_pf(std::move(init_arm),
                                    std::move(pf_prediction), std::move(vpf_correction_gated),
                                    std::move(resampling),
                                    paramss["robot_cam_sel"], paramss["robot_laterality"], paramsd["num_particles"]);


    vsir_pf.prepare();
    vsir_pf.wait();


    yInfo() << log_ID << "Application closed succesfully.";
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
