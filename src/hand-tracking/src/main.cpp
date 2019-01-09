#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <string>

#include <BayesFilters/BootstrapCorrection.h>
#include <BayesFilters/ResamplingWithPrior.h>

#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/Value.h>
#include <opencv2/core/core.hpp>

#include <BrownianMotionPose.h>
#include <ChiSquare.h>
#include <DrawParticlesImportanceThreshold.h>
#include <NormOne.h>
#include <NormTwo.h>
#include <NormTwoChiSquare.h>
#include <NormTwoKLD.h>
#include <NormTwoKLDChiSquare.h>
#include <iCubCamera.h>
#include <iCubArmModel.h>
#include <iCubGatePose.h>
#include <iCubFwdKinModel.h>
#include <InitiCubArm.h>
#include <InitWalkmanArm.h>
#include <KLD.h>
#include <PlayiCubFwdKinModel.h>
#include <PlayWalkmanPoseModel.h>
#include <PlayGatePose.h>
#include <VisualProprioception.h>
#include <VisualSIS.h>

#include <WalkmanArmModel.h>
#include <WalkmanCamera.h>

using namespace bfl;
using namespace cv;
using namespace Eigen;
using namespace yarp::os;


/* MAIN */
int main(int argc, char *argv[])
{
    const std::string log_ID = "[Main]";
    yInfo() << log_ID << "Configuring and starting module...";

    Network yarp;
    if (!yarp.checkNetwork(3.0))
    {
        yError() << "YARP seems unavailable!";
        return EXIT_FAILURE;
    }

    ResourceFinder rf;
    rf.setVerbose();
    rf.setDefaultContext("hand-tracking");
    rf.setDefaultConfigFile("config.ini");
    rf.configure(argc, argv);

    FilteringParamtersD paramsd;
    FilteringParamtersS paramss;

    /* Get Particle Filter parameters */
    yarp::os::Bottle bottle_pf_params = rf.findGroup("PF");
    paramsd["num_particles"]    = bottle_pf_params.check("num_particles",    Value(50)).asInt();
    paramsd["gpu_count"]        = bottle_pf_params.check("gpu_count",        Value(1.0)).asInt();
    paramsd["resample_prior"]   = bottle_pf_params.check("resample_prior",   Value(1.0)).asInt();
    paramsd["gate_pose"]        = bottle_pf_params.check("gate_pose",        Value(0.0)).asInt();
    paramsd["resolution_ratio"] = bottle_pf_params.check("resolution_ratio", Value(1.0)).asInt();
    paramss["laterality"]       = bottle_pf_params.check("laterality", Value("right")).asString();

    paramsd["num_images"]       = paramsd["num_particles"] / paramsd["gpu_count"];

    if (rf.check("play"))
        paramsd["play"] = 1.0;
    else
        paramsd["play"] = bottle_pf_params.check("play", Value(1.0)).asDouble();

    if (rf.check("robot"))
        paramss["robot"] = rf.find("robot").asString();
    else
        paramss["robot"] = bottle_pf_params.check("robot", Value("icub")).asString();

    if (rf.check("cam"))
        paramss["cam_sel"] = rf.find("cam").asString();
    else
        paramss["cam_sel"] = bottle_pf_params.check("cam_sel", Value("left")).asString();


    /* Get Brownian Motion parameters */
    yarp::os::Bottle bottle_brownianmotion_params = rf.findGroup("BROWNIANMOTION");
    paramsd["q_xy"]       = bottle_brownianmotion_params.check("q_xy",       Value(0.005)).asDouble();
    paramsd["q_z"]        = bottle_brownianmotion_params.check("q_z",        Value(0.005)).asDouble();
    paramsd["theta"]      = bottle_brownianmotion_params.check("theta",      Value(3.0)).asDouble();
    paramsd["cone_angle"] = bottle_brownianmotion_params.check("cone_angle", Value(2.5)).asDouble();
    paramsd["seed"]       = bottle_brownianmotion_params.check("seed",       Value(1.0)).asDouble();


    /* Get Visual Proprioception parameters */
    yarp::os::Bottle bottle_visualproprioception_params = rf.findGroup("VISUALPROPRIOCEPTION");
    paramsd["use_thumb"]   = bottle_visualproprioception_params.check("use_thumb", Value(0.0)).asDouble();
    paramsd["use_forearm"] = bottle_visualproprioception_params.check("use_forearm", Value(0.0)).asDouble();


    /* Get Likelihood parameters */
    yarp::os::Bottle bottle_likelihood_params = rf.findGroup("LIKELIHOOD");
    paramss["likelihood_type"] = bottle_likelihood_params.check("likelihood_type", Value("norm_one")).asString();
    paramsd["likelihood_gain"] = bottle_likelihood_params.check("likelihood_gain", Value(0.001)).asDouble();


    /* Get Gate Pose parameters */
    yarp::os::Bottle bottle_gatepose_params = rf.findGroup("GATEPOSE");
    paramsd["gate_x"]        = bottle_gatepose_params.check("gate_x",        Value(0.1)).asDouble();
    paramsd["gate_y"]        = bottle_gatepose_params.check("gate_y",        Value(0.1)).asDouble();
    paramsd["gate_z"]        = bottle_gatepose_params.check("gate_z",        Value(0.1)).asDouble();
    paramsd["gate_aperture"] = bottle_gatepose_params.check("gate_aperture", Value(15.0)).asDouble();
    paramsd["gate_rotation"] = bottle_gatepose_params.check("gate_rotation", Value(30.0)).asDouble();


    /* Get Resampling parameters */
    yarp::os::Bottle bottle_resampling_params = rf.findGroup("RESAMPLING");
    paramsd["resample_ratio"] = bottle_resampling_params.check("resample_ratio", Value(0.3)).asDouble();
    paramsd["prior_ratio"]    = bottle_resampling_params.check("prior_ratio",    Value(0.5)).asDouble();


    /* Log parameters */
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
    yInfo() << log_ID << " - likelihood_type:" << paramss["likelihood_type"];
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
    std::unique_ptr<ParticleSetInitialization> init_arm;
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
            robot_motion = std::unique_ptr<KinPoseModel>(new iCubFwdKinModel(paramss["robot"], paramss["laterality"],
                                                                             "handTracking/iCubFwdKinModel/" + paramss["cam_sel"]));
        else
        {
            robot_motion = std::unique_ptr<KinPoseModel>(new PlayiCubFwdKinModel(paramss["robot"], paramss["laterality"],
                                                                                 "handTracking/PlayiCubFwdKinModel/" + paramss["cam_sel"]));
        }
    }
    else if (paramss["robot"] == "walkman")
    {
        if (paramsd["play"] != 1.0)
        {
            yError() << log_ID << "Pose model method for Walkman is unimplemented.";
            return EXIT_FAILURE;
        }
        else
            robot_motion = std::unique_ptr<KinPoseModel>(new PlayWalkmanPoseModel(paramss["robot"], paramss["laterality"],
                                                                                  "handTracking/PlayWalkmanPoseModel/" + paramss["cam_sel"]));
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


    /* PROCESS MODEL */
    std::unique_ptr<Camera> camera;
    std::unique_ptr<MeshModel> mesh_model;
    if (paramss["robot"] == "icub")
    {
        camera = std::unique_ptr<Camera>(new iCubCamera(paramss["cam_sel"],
                                                        paramsd["resolution_ratio"],
                                                        rf.getContext(),
                                                        "handTracking/Process/iCubCamera/" + paramss["cam_sel"]));

        mesh_model = std::unique_ptr<MeshModel>(new iCubArmModel(paramsd["use_thumb"],
                                                                 paramsd["use_forearm"],
                                                                 paramss["laterality"],
                                                                 rf.getContext(),
                                                                 "handTracking/MeshModel/iCubArmModel/" + paramss["cam_sel"]));
    }
    else if (paramss["robot"] == "walkman")
    {
        camera = std::unique_ptr<Camera>(new WalkmanCamera(paramss["cam_sel"],
                                                           paramsd["resolution_ratio"],
                                                           rf.getContext(),
                                                           "handTracking/Process/WalkmanCamera/" + paramss["cam_sel"]));

        mesh_model = std::unique_ptr<MeshModel>(new WalkmanArmModel(paramss["laterality"],
                                                                    rf.getContext(),
                                                                    "handTracking/MeshModel/WalkmanArmModel/" + paramss["cam_sel"]));
    }
    else
    {
        yError() << log_ID << "Wrong robot name. Provided: " << paramss["robot"] << ". Shall be either 'icub' or 'walkman'.";
        return EXIT_FAILURE;
    }

    /* SENSOR MODEL */
    std::unique_ptr<VisualProprioception> proprio;
    try
    {
        proprio = std::unique_ptr<VisualProprioception>(new VisualProprioception(std::move(camera),
                                                                                 paramsd["num_images"],
                                                                                 std::move(mesh_model)));

        paramsd["num_particles"] = proprio->getNumberOfUsedParticles();

        yInfo() << log_ID << "General PF parameters changed after constructing VisualProprioception:";
        yInfo() << log_ID << " - num_particles:" << paramsd["num_particles"];
    }
    catch (const std::runtime_error& e)
    {
        yError() << e.what();
        return EXIT_FAILURE;
    }

    /* LIKELIHOOD */
    std::unique_ptr<LikelihoodModel> likelihood;
    if (paramss["likelihood_type"] == "chi")
        likelihood = std::unique_ptr<ChiSquare>(new ChiSquare(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "kld")
        likelihood = std::unique_ptr<KLD>(new KLD(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "norm_one")
        likelihood = std::unique_ptr<NormOne>(new NormOne(paramsd["likelihood_gain"]));
    else if (paramss["likelihood_type"] == "norm_two")
        likelihood = std::unique_ptr<NormTwo>(new NormTwo(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "norm_two_chi")
        likelihood = std::unique_ptr<NormTwoChiSquare>(new NormTwoChiSquare(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "norm_two_kld")
        likelihood = std::unique_ptr<NormTwoKLD>(new NormTwoKLD(paramsd["likelihood_gain"], 36));
    else if (paramss["likelihood_type"] == "norm_two_kld_chi")
        likelihood = std::unique_ptr<NormTwoKLDChiSquare>(new NormTwoKLDChiSquare(paramsd["likelihood_gain"], 36));
    else
    {
        yError() << log_ID << "Wrong likelihood type. Provided: " << paramss["likelihood_type"] << ". Shalle be either 'norm_one' or 'norm_two_chi'.";
        return EXIT_FAILURE;
    }

    /* CORRECTION */
    std::unique_ptr<PFCorrection> vpf_update_particles(new BootstrapCorrection());
    vpf_update_particles->setLikelihoodModel(std::move(likelihood));
    vpf_update_particles->setMeasurementModel(std::move(proprio));

    std::unique_ptr<PFCorrection> vpf_correction;

    if (paramsd["gate_pose"] == 1.0)
    {
        std::cerr << "GatePose is disabled due to a change in the interface!" << std::endl;

        return EXIT_FAILURE;

//        if (paramsd["play"] != 1.0)
//            vpf_correction = std::unique_ptr<iCubGatePose>(new iCubGatePose(std::move(vpf_update_particles),
//                                                                            paramsd["gate_x"], paramsd["gate_y"], paramsd["gate_z"],
//                                                                            paramsd["gate_aperture"], paramsd["gate_rotation"],
//                                                                            paramss["robot"], paramss["laterality"],
//                                                                            "handTracking/iCubGatePose/" + paramss["cam_sel"]));
//        else
//            vpf_correction = std::unique_ptr<PlayGatePose>(new PlayGatePose(std::move(vpf_update_particles),
//                                                                            paramsd["gate_x"], paramsd["gate_y"], paramsd["gate_z"],
//                                                                            paramsd["gate_aperture"], paramsd["gate_rotation"],
//                                                                            paramss["robot"], paramss["laterality"],
//                                                                            "handTracking/PlayGatePose/" + paramss["cam_sel"]));
    }
    else
        vpf_correction = std::move(vpf_update_particles);

    /* RESAMPLING */
    std::unique_ptr<Resampling> pf_resampling;
    if (paramsd["resample_prior"] != 1.0)
        pf_resampling = std::unique_ptr<Resampling>(new Resampling());
    else
    {
        std::unique_ptr<ParticleSetInitialization> resample_init_arm;

        if (paramss["robot"] == "icub")
            resample_init_arm = std::unique_ptr<InitiCubArm>(new InitiCubArm(paramss["cam_sel"], paramss["laterality"],
                                                                             "handTracking/ResamplingWithPrior/InitiCubArm/" + paramss["cam_sel"]));
        else if (paramss["robot"] == "walkman")
            resample_init_arm = std::unique_ptr<InitWalkmanArm>(new InitWalkmanArm(paramss["cam_sel"], paramss["laterality"],
                                                                                   "handTracking/ResamplingWithPrior/InitWalkmanArm/" + paramss["cam_sel"]));

        pf_resampling = std::unique_ptr<Resampling>(new ResamplingWithPrior(std::move(resample_init_arm), paramsd["prior_ratio"]));
    }

    /* PARTICLE FILTER */
    VisualSIS vsis_pf(std::move(init_arm),
                      std::move(pf_prediction),
                      std::move(vpf_correction),
                      std::move(pf_resampling),
                      paramss["cam_sel"],
                      paramsd["num_particles"],
                      paramsd["resample_ratio"],
                      "handTracking/VisualSIS/" + paramss["cam_sel"]);


    vsis_pf.boot();
    vsis_pf.wait();


    yInfo() << log_ID << "Application closed succesfully.";
    return EXIT_SUCCESS;
}
