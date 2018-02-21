#include <chrono>
#include <future>
#include <iostream>
#include <memory>

#include <BayesFilters/ResamplingWithPrior.h>
#include <yarp/os/ConstString.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/Value.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <BrownianMotionPose.h>
#include <DrawParticlesImportanceThreshold.h>
#include <iCubCamera.h>
#include <iCubArmModel.h>
#include <iCubGatePose.h>
#include <iCubFwdKinModel.h>
#include <InitiCubArm.h>
#include <InitWalkmanArm.h>
#include <PlayiCubFwdKinModel.h>
#include <PlayWalkmanPoseModel.h>
#include <PlayGatePose.h>
#include <VisualProprioception.h>
#include <VisualSIS.h>
#include <VisualUpdateParticles.h>
#include <WalkmanArmModel.h>
#include <WalkmanCamera.h>

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
    paramsd["num_particles"]    = rf.findGroup("PF").check("num_particles",    Value(50)).asInt();
    paramsd["gpu_count"]        = rf.findGroup("PF").check("gpu_count",        Value(1.0)).asInt();
    paramsd["resample_prior"]   = rf.findGroup("PF").check("resample_prior",   Value(1.0)).asInt();
    paramsd["gate_pose"]        = rf.findGroup("PF").check("gate_pose",        Value(0.0)).asInt();
    paramsd["resolution_ratio"] = rf.findGroup("PF").check("resolution_ratio", Value(1.0)).asInt();
    paramsd["num_images"]       = paramsd["num_particles"] / paramsd["gpu_count"];
    if (rf.check("play"))
        paramsd["play"] = 1.0;
    else
        paramsd["play"] = rf.findGroup("PF").check("play", Value(1.0)).asDouble();

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

    paramsd["resample_ratio"] = rf.findGroup("RESAMPLING").check("resample_ratio", Value(0.3)).asDouble();
    paramsd["prior_ratio"]    = rf.findGroup("RESAMPLING").check("prior_ratio",    Value(0.5)).asDouble();

    FilteringParamtersS paramss;
    paramss["robot"] = rf.findGroup("PF").check("robot", Value("icub")).asString();

    if (paramss["robot"] != "icub" && paramss["robot"] != "walkman")
    {
        yError() << log_ID << "Wrong robot name. Provided: " << paramss["robot"] << ". Can be iCub, Walkman.";
        return EXIT_FAILURE;
    }

    if (rf.check("cam"))
        paramss["cam_sel"] = rf.find("cam").asString();
    else
        paramss["cam_sel"] = rf.findGroup("PF").check("cam_sel", Value("left")).asString();

    paramss["laterality"]  = rf.findGroup("PF").check("laterality", Value("right")).asString();


    yInfo() << log_ID << "General PF parameters:";
    yInfo() << log_ID << " - robot:"          << paramss["robot"];
    yInfo() << log_ID << " - cam_sel:"        << paramss["cam_sel"];
    yInfo() << log_ID << " - laterality:"     << paramss["laterality"];
    yInfo() << log_ID << " - num_particles:"  << paramsd["num_particles"];
    yInfo() << log_ID << " - gpu_count:"      << paramsd["gpu_count"];
    yInfo() << log_ID << " - num_images:"     << paramsd["num_images"];
    yInfo() << log_ID << " - resample_prior:" << paramsd["resample_prior"];
    yInfo() << log_ID << " - gate_pose:"      << paramsd["gate_pose"];
    yInfo() << log_ID << " - play:"           << (paramsd["play"] == 1.0 ? "true" : "false");

    yInfo() << log_ID << "Motion modle parameters:";
    yInfo() << log_ID << " - q_xy:"       << paramsd["q_xy"];
    yInfo() << log_ID << " - q_z:"        << paramsd["q_z"];
    yInfo() << log_ID << " - theta:"      << paramsd["theta"];
    yInfo() << log_ID << " - cone_angle:" << paramsd["cone_angle"];
    yInfo() << log_ID << " - seed:"       << paramsd["seed"];

    yInfo() << log_ID << "Sensor model parameters:";
    yInfo() << log_ID << " - use_thumb:"   << paramsd["use_thumb"];
    yInfo() << log_ID << " - use_forearm:" << paramsd["use_forearm"];

    yInfo() << log_ID << "Correction parameters:";
    yInfo() << log_ID << " - likelihood_gain:" << paramsd["likelihood_gain"];

    yInfo() << log_ID << "Resampling parameters:";
    yInfo() << log_ID << " - resample_ratio:" << paramsd["resample_ratio"];
    
    if (paramsd["resample_prior"] == 1.0)
    {
        yInfo() << log_ID << "Resampling with prior parameters:";
        yInfo() << log_ID << " - prior_ratio:" << paramsd["prior_ratio"];
    }

    if (paramsd["gate_pose"] == 1.0)
    {
        yInfo() << log_ID << "Pose gating parameters:";
        yInfo() << log_ID << " - gate_x:"        << paramsd["gate_x"];
        yInfo() << log_ID << " - gate_y:"        << paramsd["gate_y"];
        yInfo() << log_ID << " - gate_z:"        << paramsd["gate_z"];
        yInfo() << log_ID << " - gate_aperture:" << paramsd["gate_aperture"];
        yInfo() << log_ID << " - gate_rotation:" << paramsd["gate_rotation"];
    }


    /* INITIALIZATION */
    std::unique_ptr<Initialization> init_arm;
    if (paramss["robot"] == "icub")
        init_arm = std::unique_ptr<InitiCubArm>(new InitiCubArm(paramss["cam_sel"], paramss["laterality"],
                                                "handTracking/InitiCubArm/" + paramss["cam_sel"]));
    else if (paramss["robot"] == "walkman")
        init_arm = std::unique_ptr<InitWalkmanArm>(new InitWalkmanArm(paramss["cam_sel"], paramss["laterality"],
                                                   "handTracking/InitWalkmanArm/" + paramss["cam_sel"]));


    /* MOTION MODEL */
    std::unique_ptr<StateModel> brown(new BrownianMotionPose(paramsd["q_xy"], paramsd["q_z"], paramsd["theta"], paramsd["cone_angle"], paramsd["seed"]));

    std::unique_ptr<ExogenousModel> robot_motion;
    if (paramss["robot"] == "icub")
    {
        if (paramsd["play"] != 1.0)
            robot_motion = std::unique_ptr<iCubFwdKinModel>(new iCubFwdKinModel(paramss["robot"], paramss["laterality"],
                                                                                "handTracking/iCubFwdKinModel/" + paramss["cam_sel"]));
        else
        {
            robot_motion = std::unique_ptr<PlayiCubFwdKinModel>(new PlayiCubFwdKinModel(paramss["robot"], paramss["laterality"],
                                                                                        "handTracking/PlayiCubFwdKinModel/" + paramss["cam_sel"]));
        }
    }
    else if (paramss["robot"] == "walkman")
    {
        if (paramsd["play"] != 1.0)
            robot_motion = std::unique_ptr<PlayWalkmanPoseModel>(new PlayWalkmanPoseModel(paramss["robot"], paramss["laterality"],
                                                                                          "handTracking/PlayWalkmanPoseModel/" + paramss["cam_sel"]));
        else
        {
            yError() << log_ID << "Pose model method for Walkman is unimplemented.";
            return EXIT_FAILURE;
        }
    }
    else
    {
        yError() << log_ID << "Wrong robot name. Provided: " << paramss["robot"] << ". Can be iCub, Walkman.";
        return EXIT_FAILURE;
    }

    /* PREDICTION */
    std::unique_ptr<DrawParticlesImportanceThreshold> pf_prediction(new DrawParticlesImportanceThreshold());
    pf_prediction->setStateModel(std::move(brown));
    pf_prediction->setExogenousModel(std::move(robot_motion));


    /* SENSOR MODEL */
    std::unique_ptr<Camera> camera;
    std::unique_ptr<MeshModel> mesh_model;
    if (paramss["robot"] == "icub")
    {
        camera = std::unique_ptr<iCubCamera>(new iCubCamera(paramss["cam_sel"],
                                                            paramsd["resolution_ratio"],
                                                            rf.getContext(),
                                                            "handTracking/VisualProprioception/iCubCamera/" + paramss["cam_sel"]));

        mesh_model = std::unique_ptr<iCubArmModel>(new iCubArmModel(paramsd["use_thumb"],
                                                                    paramsd["use_forearm"],
                                                                    paramss["laterality"],
                                                                    rf.getContext(),
                                                                    "handTracking/VisualProprioception/iCubArmModel/" + paramss["cam_sel"]));
    }
    else if (paramss["robot"] == "walkman")
    {
        camera = std::unique_ptr<WalkmanCamera>(new WalkmanCamera(paramss["cam_sel"],
                                                                  paramsd["resolution_ratio"],
                                                                  rf.getContext(),
                                                                  "handTracking/VisualProprioception/WalkmanCamera/" + paramss["cam_sel"]));

        mesh_model = std::unique_ptr<WalkmanArmModel>(new WalkmanArmModel(paramss["laterality"],
                                                                          rf.getContext(),
                                                                          "handTracking/VisualProprioception/WalkmanArmModel/" + paramss["cam_sel"]));
    }
    else
    {
        yError() << log_ID << "Wrong robot name. Provided: " << paramss["robot"] << ". Can be iCub, Walkman.";
        return EXIT_FAILURE;
    }


    std::unique_ptr<VisualProprioception> proprio;
    try
    {
        proprio = std::unique_ptr<VisualProprioception>(new VisualProprioception(paramsd["num_images"], std::move(camera), std::move(mesh_model)));

        paramsd["num_particles"] = proprio->getOGLTilesRows() * proprio->getOGLTilesCols() * paramsd["gpu_count"];
        paramsd["cam_width"]     = proprio->getCamWidth();
        paramsd["cam_height"]    = proprio->getCamHeight();
    }
    catch (const std::runtime_error& e)
    {
        yError() << e.what();
        return EXIT_FAILURE;
    }

    /* CORRECTION */
    std::unique_ptr<PFVisualCorrection> vpf_update_particles(new VisualUpdateParticles(std::move(proprio), paramsd["likelihood_gain"], paramsd["gpu_count"]));

    std::unique_ptr<PFVisualCorrection> vpf_correction;

    if (paramsd["gate_pose"] == 1.0)
    {
        std::cerr << "GatePose is disabled due to a change in the interface!" << std::endl;

        return EXIT_FAILURE;

        if (paramsd["play"] != 1.0)
            vpf_correction = std::unique_ptr<iCubGatePose>(new iCubGatePose(std::move(vpf_update_particles),
                                                                            paramsd["gate_x"], paramsd["gate_y"], paramsd["gate_z"],
                                                                            paramsd["gate_aperture"], paramsd["gate_rotation"],
                                                                            paramss["robot"], paramss["laterality"],
                                                                            "handTracking/iCubGatePose/" + paramss["cam_sel"]));
        else
            vpf_correction = std::unique_ptr<PlayGatePose>(new PlayGatePose(std::move(vpf_update_particles),
                                                                            paramsd["gate_x"], paramsd["gate_y"], paramsd["gate_z"],
                                                                            paramsd["gate_aperture"], paramsd["gate_rotation"],
                                                                            paramss["robot"], paramss["laterality"],
                                                                            "handTracking/PlayGatePose/" + paramss["cam_sel"]));
    }
    else
        vpf_correction = std::move(vpf_update_particles);


    /* RESAMPLING */
    std::unique_ptr<Resampling> pf_resampling;
    if (paramsd["resample_prior"] != 1.0)
        pf_resampling = std::unique_ptr<Resampling>(new Resampling());
    else
    {
        std::unique_ptr<Initialization> resample_init_arm;

        if (paramss["robot"] == "icub")
            resample_init_arm = std::unique_ptr<InitiCubArm>(new InitiCubArm(paramss["cam_sel"], paramss["laterality"],
                                                                             "handTracking/ResamplingWithPrior/InitiCubArm/" + paramss["cam_sel"]));
        else if (paramss["robot"] == "walkman")
            resample_init_arm = std::unique_ptr<InitWalkmanArm>(new InitWalkmanArm(paramss["cam_sel"], paramss["laterality"],
                                                                                   "handTracking/ResamplingWithPrior/InitWalkmanArm/" + paramss["cam_sel"]));

        pf_resampling = std::unique_ptr<Resampling>(new ResamplingWithPrior(std::move(resample_init_arm), paramsd["prior_ratio"]));
    }


    /* PARTICLE FILTER */
    VisualSIS vsis_pf(paramss["cam_sel"],
                      paramsd["cam_width"], paramsd["cam_height"],
                      paramsd["num_particles"],
                      paramsd["resample_ratio"],
                      "handTracking/VisualSIS/" + paramss["cam_sel"]);
    vsis_pf.setInitialization(std::move(init_arm));
    vsis_pf.setPrediction(std::move(pf_prediction));
    vsis_pf.setCorrection(std::move(vpf_correction));
    vsis_pf.setResampling(std::move(pf_resampling));


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
