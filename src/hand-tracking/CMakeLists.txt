set(EXE_TARGET_NAME hand-tracking)

# Thread preference flags
if(NOT APPLE)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
endif()

# Dependencies
find_package(BayesFilters    REQUIRED)
find_package(EIGEN3          REQUIRED)
find_package(glew            REQUIRED)
find_package(glfw3           REQUIRED)
find_package(ICUB            REQUIRED)
find_package(OpenCV          REQUIRED)
find_package(SuperimposeMesh REQUIRED)
find_package(YARP            REQUIRED)

LIST(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH}
                              ${ICUB_MODULE_PATH})

include(YarpInstallationHelpers)

# Application source and header files
set(${EXE_TARGET_NAME}_HDR
        include/BrownianMotionPose.h
        include/DrawParticlesPose.h
        include/GatePose.h
        include/iCubFwdKinModel.h
        include/iCubGatePose.h
        include/InitiCubArm.h
        include/PlayFwdKinModel.h
        include/PlayGatePose.h
        include/ResamplingWithPrior.h
        include/VisualProprioception.h
        include/VisualSIRParticleFilter.h
        include/VisualUpdateParticles.h)

set(${EXE_TARGET_NAME}_SRC
        src/main.cpp
        src/BrownianMotionPose.cpp
        src/DrawParticlesPose.cpp
        src/GatePose.cpp
        src/iCubFwdKinModel.cpp
        src/iCubGatePose.cpp
        src/InitiCubArm.cpp
        src/PlayFwdKinModel.cpp
        src/PlayGatePose.cpp
        src/ResamplingWithPrior.cpp
        src/VisualProprioception.cpp
        src/VisualSIRParticleFilter.cpp
        src/VisualUpdateParticles.cpp)

set(${EXE_TARGET_NAME}_THRIFT_HDR
        thrift/visualsirpf.thrift)

# Application target calls
yarp_add_idl(${EXE_TARGET_NAME}_THRIFT_SRC ${${EXE_TARGET_NAME}_THRIFT_HDR})

add_executable(${EXE_TARGET_NAME}   ${${EXE_TARGET_NAME}_HDR} ${${EXE_TARGET_NAME}_SRC}
                                    ${${EXE_TARGET_NAME}_THRIFT_SRC})

target_include_directories(${EXE_TARGET_NAME} PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                      ${EIGEN3_INCLUDE_DIRS}
                                                      ${ICUB_INCLUDE_DIRS}
                                                      ${YARP_INCLUDE_DIRS})

target_link_libraries(${EXE_TARGET_NAME} BayesFilters::BayesFilters
                                         ctrlLib
                                         GLEW::GLEW
                                         glfw
                                         ${ICUB_LIBRARIES}
                                         iKin
                                         ${OpenCV_LIBS}
                                         SuperimposeMesh::SuperimposeMesh
                                         ${YARP_LIBRARIES})

if(NOT APPLE)
    target_link_libraries(${EXE_TARGET_NAME} Threads::Threads)
endif()


# Custom command for RESOURCE/EXTRA files
set(${EXE_TARGET_NAME}_APP
        ${PROJECT_SOURCE_DIR}/app/hand-tracking.xml
        ${PROJECT_SOURCE_DIR}/app/hand-tracking-BATCH.xml
        ${PROJECT_SOURCE_DIR}/app/hand-tracking-SIM.xml)

set(${EXE_TARGET_NAME}_CONF
${CMAKE_CURRENT_SOURCE_DIR}/conf/config.ini)

set(${EXE_TARGET_NAME}_SHADER_VERT
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_model.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_background.vert)

set(${EXE_TARGET_NAME}_SHADER_FRAG
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_model.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_background.frag)

set(${EXE_TARGET_NAME}_MESH
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_palm.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ail0.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ail1.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ail2.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ail3.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_forearm.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_indexbase.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ml0.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ml1.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ml2.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ml3.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl0.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl1.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl2.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl3.obj
        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl4.obj)

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
