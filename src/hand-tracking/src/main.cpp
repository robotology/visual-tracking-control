#include <chrono>
#include <future>
#include <iostream>
#include <memory>

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
#include "VisualSIS.h"
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
    paramsd["num_particles"] = rf.findGroup("PF").check("num_particles", Value(50.0)).asDouble();
    paramsd["gpu_count"]     = rf.findGroup("PF").check("gpu_count",     Value(1.0)).asDouble();
    paramsd["num_images"]    = paramsd["num_particles"] / paramsd["gpu_count"];
    paramsd["play"]          = rf.findGroup("PF").check("play",          Value(1.0)).asDouble();

    paramsd["q_xy"]       = rf.findGroup("BROWNIANMOTION").check("q_xy",       Value(0.005)).asDouble();
    paramsd["q_z"]        = rf.findGroup("BROWNIANMOTION").check("q_z",        Value(0.005)).asDouble();
    paramsd["theta"]      = rf.findGroup("BROWNIANMOTION").check("theta",      Value(3.0)).asDouble();
    paramsd["cone_angle"] = rf.findGroup("BROWNIANMOTION").check("cone_angle", Value(2.5)).asDouble();
    paramsd["seed"]       = rf.findGroup("BROWNIANMOTION").check("seed",       Value(1.0)).asDouble();

    paramsd["use_thumb"]   = rf.findGroup("VISUALPROPRIOCEPTION").check("use_thumb", Value(0.0)).asDouble();
    paramsd["use_forearm"] = rf.findGroup("VISUALPROPRIOCEPTION").check("use_forearm", Value(0.0)).asDouble();

    paramsd["likelihood_gain"] = rf.findGroup("VISUALUPDATEPARTICLES").check("likelihood_gain", Value(0.001)).asDouble();

    paramsd["gate_x"]        = rf.findGroup("GATEPOSE").check("gate_x",        Value(0.1)).asDouble();
    paramsd["gate_y"]        = rf.findGroup("GATEPOSE").check("gate_y",        Value(0.1)).asDouble();
    paramsd["gate_z"]        = rf.findGroup("GATEPOSE").check("gate_z",        Value(0.1)).asDouble();
    paramsd["gate_aperture"] = rf.findGroup("GATEPOSE").check("gate_aperture", Value(15.0)).asDouble();
    paramsd["gate_rotation"] = rf.findGroup("GATEPOSE").check("gate_rotation", Value(30.0)).asDouble();

    FilteringParamtersS paramss;
    paramss["robot"]      = rf.findGroup("PF").check("robot",      Value("icub")).asString();
    paramss["cam_sel"]    = rf.findGroup("PF").check("cam_sel",    Value("left")).asString();
    paramss["laterality"] = rf.findGroup("PF").check("laterality", Value("right")).asString();


    yInfo() << log_ID << "Running with:";
    yInfo() << log_ID << " - robot:"         << paramss["robot"];
    yInfo() << log_ID << " - cam_sel:"       << paramss["cam_sel"];
    yInfo() << log_ID << " - laterality:"    << paramss["laterality"];

    yInfo() << log_ID << " - num_particles:" << paramsd["num_particles"];
    yInfo() << log_ID << " - gpu_count:"     << paramsd["gpu_count"];
    yInfo() << log_ID << " - num_images:"    << paramsd["num_images"];
    yInfo() << log_ID << " - play:"          << (paramsd["play"] == 1.0 ? "true" : "false");

    yInfo() << log_ID << " - q_xy:"       << paramsd["q_xy"];
    yInfo() << log_ID << " - q_z:"        << paramsd["q_z"];
    yInfo() << log_ID << " - theta:"      << paramsd["theta"];
    yInfo() << log_ID << " - cone_angle:" << paramsd["cone_angle"];
    yInfo() << log_ID << " - seed:"       << paramsd["seed"];

    yInfo() << log_ID << " - use_thumb:"   << paramsd["use_thumb"];
    yInfo() << log_ID << " - use_forearm:" << paramsd["use_forearm"];

    yInfo() << log_ID << " - likelihood_gain:" << paramsd["likelihood_gain"];

    yInfo() << log_ID << " - gate_x:"        << paramsd["gate_x"];
    yInfo() << log_ID << " - gate_y:"        << paramsd["gate_y"];
    yInfo() << log_ID << " - gate_z:"        << paramsd["gate_z"];
    yInfo() << log_ID << " - gate_aperture:" << paramsd["gate_aperture"];
    yInfo() << log_ID << " - gate_rotation:" << paramsd["gate_rotation"];


    /* INITIALIZATION */
    std::unique_ptr<InitiCubArm> init_arm(new InitiCubArm("hand-tracking/InitiCubArm", paramss["cam_sel"], paramss["laterality"]));


    /* MOTION MODEL */
    std::unique_ptr<BrownianMotionPose> brown(new BrownianMotionPose(paramsd["q_xy"], paramsd["q_z"], paramsd["theta"], paramsd["cone_angle"], paramsd["seed"]));
    std::unique_ptr<StateModel> icub_motion;
    if (paramsd["play"] != 1.0)
    {
        std::unique_ptr<iCubFwdKinModel> icub_fwdkin(new iCubFwdKinModel(std::move(brown), paramss["robot"], paramss["laterality"], paramss["cam_sel"]));
        icub_motion = std::move(icub_fwdkin);
    }
    else
    {
        std::unique_ptr<PlayFwdKinModel> play_fwdkin(new PlayFwdKinModel(std::move(brown), paramss["robot"], paramss["laterality"], paramss["cam_sel"]));
        icub_motion = std::move(play_fwdkin);
    }

    /* PREDICTION */
//    std::unique_ptr<DrawParticlesPose> pf_prediction(new DrawParticlesPose());
    std::unique_ptr<DrawParticlesPoseCondWeight> pf_prediction(new DrawParticlesPoseCondWeight());
    pf_prediction->setStateModel(std::move(icub_motion));


    /* SENSOR MODEL */
    std::unique_ptr<VisualProprioception> proprio;
    try
    {
        std::unique_ptr<VisualProprioception> vp(new VisualProprioception(paramsd["use_thumb"], paramsd["use_forearm"],
                                                                          paramsd["num_images"], paramss["cam_sel"], paramss["laterality"], rf.getContext()));

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
                                                                      paramsd["gate_x"], paramsd["gate_y"], paramsd["gate_z"],
                                                                      paramsd["gate_aperture"], paramsd["gate_rotation"],
                                                                      paramss["robot"], paramss["laterality"], paramss["cam_sel"]));
        vpf_correction_gated = std::move(icub_gate_pose);
    }
    else
    {
        std::unique_ptr<PlayGatePose> icub_gate_pose(new PlayGatePose(std::move(vpf_correction),
                                                                      paramsd["gate_x"], paramsd["gate_y"], paramsd["gate_z"],
                                                                      paramsd["gate_aperture"], paramsd["gate_rotation"],
                                                                      paramss["robot"], paramss["laterality"], paramss["cam_sel"]));
        vpf_correction_gated = std::move(icub_gate_pose);
    }


    /* RESAMPLING */
//    std::unique_ptr<Resampling> resampling(new Resampling());
    std::unique_ptr<Resampling> resampling(new ResamplingWithPrior("hand-tracking/ResamplingWithPrior", paramss["cam_sel"], paramss["laterality"]));


    /* PARTICLE FILTER */
    VisualSIS vsis_pf(paramss["cam_sel"], paramss["laterality"], paramsd["num_particles"]);
    vsis_pf.setInitialization(std::move(init_arm));
    vsis_pf.setPrediction(std::move(pf_prediction));
    vsis_pf.setCorrection(std::move(vpf_correction_gated));
    vsis_pf.setResampling(std::move(resampling));


    vsis_pf.boot();
    vsis_pf.wait();


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
