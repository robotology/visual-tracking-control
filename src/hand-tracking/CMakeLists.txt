set(EXE_TARGET_NAME hand-tracking)

# Dependencies
find_package(Eigen3 REQUIRED CONFIG)
find_package(ICUB   REQUIRED)
find_package(OpenCV REQUIRED)
find_package(YARP   REQUIRED)
find_package(BayesFilters 0.7.0.0 QUIET)
if(NOT BayesFilters_FOUND)
    message(STATUS "Did not found required master release of BayesFilters. Looking for devel version.")
    find_package(BayesFilters 0.7.100.0 REQUIRED)
endif()
find_package(SuperimposeMesh 0.9.4.0 QUIET)
if(NOT BayesFilters_FOUND)
    message(STATUS "Did not found required master release of SuperimposeMesh. Looking for devel version.")
    find_package(SuperimposeMesh 0.9.104.0 REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH}
                              ${ICUB_MODULE_PATH})

include(YarpInstallationHelpers)

# Application source and header files
set(${EXE_TARGET_NAME}_HDR
        include/BrownianMotionPose.h
		include/Camera.h
        include/DrawParticlesImportanceThreshold.h
        include/KinPoseModel.h
        include/GatePose.h
        include/iCubFwdKinModel.h
		include/iCubArmModel.h
		include/iCubCamera.h
        include/iCubGatePose.h
        include/InitiCubArm.h
        include/InitPoseParticles.h
        include/InitWalkmanArm.h
		include/MeshModel.h
        include/PlayiCubFwdKinModel.h
        include/PlayGatePose.h
        include/PlayWalkmanPoseModel.h
        include/VisualProprioception.h
        include/VisualSIS.h
        include/VisualUpdateParticles.h
		include/WalkmanArmModel.h
		include/WalkmanCamera.h)

set(${EXE_TARGET_NAME}_SRC
        src/main.cpp
        src/BrownianMotionPose.cpp
        src/DrawParticlesImportanceThreshold.cpp
        src/KinPoseModel.cpp
        src/GatePose.cpp
		src/iCubArmModel.cpp
		src/iCubCamera.cpp
        src/iCubFwdKinModel.cpp
        src/iCubGatePose.cpp
        src/InitiCubArm.cpp
        src/InitPoseParticles.cpp
        src/InitWalkmanArm.cpp
        src/PlayiCubFwdKinModel.cpp
        src/PlayGatePose.cpp
        src/PlayWalkmanPoseModel.cpp
        src/VisualProprioception.cpp
        src/VisualSIS.cpp
        src/VisualUpdateParticles.cpp
		src/WalkmanArmModel.cpp
		src/WalkmanCamera.cpp)

set(${EXE_TARGET_NAME}_THRIFT_HDR
        thrift/visualsisparticlefilter.thrift)

# Application target calls
yarp_add_idl(${EXE_TARGET_NAME}_THRIFT_SRC ${${EXE_TARGET_NAME}_THRIFT_HDR})

add_executable(${EXE_TARGET_NAME} ${${EXE_TARGET_NAME}_HDR} ${${EXE_TARGET_NAME}_SRC}
                                  ${${EXE_TARGET_NAME}_THRIFT_SRC})

target_include_directories(${EXE_TARGET_NAME}
                           PRIVATE
                               "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                               ${ICUB_INCLUDE_DIRS}
                               ${YARP_INCLUDE_DIRS})

if(UNIX AND NOT APPLE)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    target_link_libraries(${EXE_TARGET_NAME}
                          PRIVATE
                              BayesFilters::BayesFilters
                              ctrlLib
                              Eigen3::Eigen
                              ${ICUB_LIBRARIES}
                              iKin
                              ${OpenCV_LIBS}
                              SuperimposeMesh::SuperimposeMesh
                              Threads::Threads
                              ${YARP_LIBRARIES})
else()
    target_link_libraries(${EXE_TARGET_NAME}
                          PRIVATE
                              BayesFilters::BayesFilters
                              ctrlLib
                              Eigen3::Eigen
                              ${ICUB_LIBRARIES}
                              iKin
                              ${OpenCV_LIBS}
                              SuperimposeMesh::SuperimposeMesh
                              ${YARP_LIBRARIES})
endif()


# Custom command for RESOURCE/EXTRA files
set(${EXE_TARGET_NAME}_APP
        ${CMAKE_CURRENT_SOURCE_DIR}/app/hand-tracking-iCub.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/app/hand-tracking-iCub-BATCH.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/app/hand-tracking-iCub-SIM.xml
		${CMAKE_CURRENT_SOURCE_DIR}/app/hand-tracking-Walkman-BATCH.xml)

set(${EXE_TARGET_NAME}_CONF
        ${CMAKE_CURRENT_SOURCE_DIR}/conf/config.ini)

set(${EXE_TARGET_NAME}_SHADER_VERT
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_frame.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_model.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_background.vert)

set(${EXE_TARGET_NAME}_SHADER_FRAG
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_frame.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_model.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_background.frag)

set(${EXE_TARGET_NAME}_MESH
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_palm.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ail0.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ail1.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ail2.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ail3.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_forearm.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_indexbase.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ml0.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ml1.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ml2.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ml3.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_tl0.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_tl1.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_tl2.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_tl3.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_tl4.obj
		${CMAKE_CURRENT_SOURCE_DIR}/mesh/walkman_right_arm/soft_hand_open.stl)

add_custom_command(TARGET  ${EXE_TARGET_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${${EXE_TARGET_NAME}_SHADER_VERT} $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>
                   COMMAND ${CMAKE_COMMAND} -E copy ${${EXE_TARGET_NAME}_SHADER_FRAG} $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>
                   COMMAND ${CMAKE_COMMAND} -E copy ${${EXE_TARGET_NAME}_MESH}        $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>)

install(TARGETS ${EXE_TARGET_NAME}                DESTINATION bin)
install(FILES   ${${EXE_TARGET_NAME}_APP}         DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})
install(FILES   ${${EXE_TARGET_NAME}_CONF}        DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${EXE_TARGET_NAME})
install(FILES   ${${EXE_TARGET_NAME}_SHADER_VERT} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${EXE_TARGET_NAME}/shader)
install(FILES   ${${EXE_TARGET_NAME}_SHADER_FRAG} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${EXE_TARGET_NAME}/shader)
install(FILES   ${${EXE_TARGET_NAME}_MESH}        DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${EXE_TARGET_NAME}/mesh)
