set(EXE_TARGET_NAME hand-tracking)

include(AddDependenciesPrintUtils)


# Dependencies
find_package(CUDA   QUIET)
find_package(OpenCV REQUIRED)
find_package(YARP   CONFIG REQUIRED
             COMPONENTS OS math sig eigen gsl)
find_package(ICUB   REQUIRED)

find_package(Eigen3 QUIET CONFIG)
if(NOT EIGEN3_FOUND)
    find_package(Eigen3 REQUIRED)
endif()

find_package(BayesFilters 0.7.1.0 QUIET)
if(NOT BayesFilters_FOUND)
    message(STATUS "Did not found required master release of BayesFilters. Looking for devel version.")
    find_package(BayesFilters 0.7.101 REQUIRED)
endif()

find_package(SuperimposeMesh 0.9.4.0 QUIET)
if(NOT SuperimposeMesh_FOUND)
    message(STATUS "Did not found required master release of SuperimposeMesh. Looking for devel version.")
    find_package(SuperimposeMesh 0.9.104.0 REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH ${ICUB_MODULE_PATH})

if(DEFINED OpenCV_CUDA_VERSION)
    message(STATUS "Found OpenCV with CUDA compiled modules.")

    if(CUDA_FOUND)
        option(FORCE_NO_OPENCV_CUDA "Force no CUDA compiled modules?" OFF)

        if(FORCE_NO_OPENCV_CUDA)
            message(STATUS "Forcing no CUDA compiled modules.")

            set(BFL_CUDA OFF)
        else()
            message(STATUS "Compiling with CUDA support and modules.")

            set(BFL_CUDA ON)
        endif()
    endif()
else()
    message(STATUS "Found no OpenCV CUDA compiled modules.")
    message(STATUS "Forcing no CUDA compiled modules.")

    set(BFL_CUDA OFF)
endif()
message(STATUS "Set BFL_CUDA to ${BFL_CUDA}.")


# CUDA settings
if(BFL_CUDA)
    if(WIN32)
        option(CUDA_LIB_DYN "Build CUDA files as dynamic libraries (default OFF - STATIC)" OFF)
    else()
        option(CUDA_LIB_DYN "Build CUDA files as dynamic libraries (default OFF - STATIC)" ON)
    endif()

    if(CUDA_LIB_DYN)
        set(CUDA_LIB_TYPE SHARED)
    else()
        set(CUDA_LIB_TYPE STATIC)
    endif()

    message(STATUS "CUDA_LIB_TYPE: " ${CUDA_LIB_TYPE})


    set(CUDA_SEPARABLE_COMPILATION ON)
    message(STATUS "CUDA_SEPARABLE_COMPILATION: " ${CUDA_SEPARABLE_COMPILATION})


    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
    message(STATUS "CUDA_ARCH_FLAGS: " ${CUDA_ARCH_FLAGS})


    list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

    if (UNIX)
        list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    endif()

    message(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})


    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    message(STATUS "CUDA_HOST_COMPILER: " ${CUDA_HOST_COMPILER})


    set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
    message(STATUS "CUDA_LINK_LIBRARIES_KEYWORD: " ${CUDA_LINK_LIBRARIES_KEYWORD})


    message(STATUS "CUDA_PROPAGATE_HOST_FLAGS: " ${CUDA_PROPAGATE_HOST_FLAGS})


    set(CUDA_LIB_NAME cudadevicelikelihood)
    message(STATUS "Compiling CUDA library: " ${CUDA_LIB_NAME})

    # CUDA source and header files
    set(CUDA_SRC_FILES src/devicelikelihood.cu
    )

    set(CUDA_HDR_FILES include/devicelikelihood.h
                       include/OpenCVToThrust.h
    )
endif()


# Application source and header files
set(${EXE_TARGET_NAME}_HDR
        include/BrownianMotionPose.h
		include/Camera.h
        include/DrawParticlesImportanceThreshold.h
        include/KinPoseModel.h
        include/GatePose.h
        include/HistogramNormOne.h
        include/HistogramNormTwoChi.h
        include/HistogramNormTwoKLD.h
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
		include/WalkmanArmModel.h
		include/WalkmanCamera.h
)

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
		src/WalkmanArmModel.cpp
		src/WalkmanCamera.cpp
)

set(${EXE_TARGET_NAME}_CPU_SRC
        src/cpu/HistogramNormOne.cpp
        src/cpu/HistogramNormTwoChi.cpp
)

set(${EXE_TARGET_NAME}_CUDA_SRC
        src/cuda/HistogramNormOne.cpp
        src/cuda/HistogramNormTwoKLD.cpp
)

set(${EXE_TARGET_NAME}_THRIFT_HDR
        thrift/visualsisparticlefilter.thrift)

yarp_add_idl(${EXE_TARGET_NAME}_THRIFT_SRC ${${EXE_TARGET_NAME}_THRIFT_HDR})


# CUDA library target calls
if(BFL_CUDA)
    cuda_add_library(${CUDA_LIB_NAME} ${CUDA_HDR_FILES} ${CUDA_SRC_FILES} ${CUDA_LIB_TYPE})

    target_include_directories(${CUDA_LIB_NAME} ${CUDA_LINK_LIBRARIES_KEYWORD}
                                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
    )

    target_link_libraries(${CUDA_LIB_NAME} ${CUDA_LINK_LIBRARIES_KEYWORD}
                                ${CUDA_nppc_LIBRARY}
                                ${OpenCV_LIBS})

    cuda_build_clean_target()
endif()


# Application target calls
add_executable(${EXE_TARGET_NAME}
                   ${${EXE_TARGET_NAME}_HDR}
                   ${${EXE_TARGET_NAME}_SRC}
                   ${${EXE_TARGET_NAME}_CUDA_SRC}
                   ${${EXE_TARGET_NAME}_CPU_SRC}
                   ${${EXE_TARGET_NAME}_THRIFT_SRC}
)

if(BFL_CUDA)
    set_source_files_properties(${${EXE_TARGET_NAME}_CPU_SRC}
                                PROPERTIES LANGUAGE "DONOTCOMPILE")
else()
    set_source_files_properties(${${EXE_TARGET_NAME}_CUDA_SRC}
                                PROPERTIES LANGUAGE "DONOTCOMPILE")
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
             PREFIX "Source Files"
             FILES
                 ${${EXE_TARGET_NAME}_CUDA_SRC}
                 ${${EXE_TARGET_NAME}_CPU_SRC}
)

target_include_directories(${EXE_TARGET_NAME}
                           PRIVATE
                               "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                               ${ICUB_INCLUDE_DIRS}
                               ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${EXE_TARGET_NAME}
                      PRIVATE
                          BayesFilters::BayesFilters
                          ctrlLib
                          ${CUDA_LIBRARIES}
                          ${CUDA_LIB_NAME}
                          ${ICUB_LIBRARIES}
                          iKin
                          ${OpenCV_LIBS}
                          SuperimposeMesh::SuperimposeMesh
                          YARP::YARP_OS
                          YARP::YARP_init
                          YARP::YARP_math
                          YARP::YARP_sig
                          YARP::YARP_eigen
                          YARP::YARP_gsl
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    target_link_libraries(${EXE_TARGET_NAME}
                          PRIVATE
                              Threads::Threads
    )
endif()
if(NOT TARGET Eigen3)
    target_include_directories(${EXE_TARGET_NAME}
                               PRIVATE
                                   ${EIGEN3_INCLUDE_DIR}
    )
else()
    target_link_libraries(${EXE_TARGET_NAME}
                          PRIVATE
                              Eigen3::Eigen
    )
endif()

if(BFL_CUDA)
    message(STATUS "Set macro HANDTRACKING_USE_OPENCV_CUDA = 1.")

    target_compile_definitions(${EXE_TARGET_NAME}
                               PRIVATE
                                   HANDTRACKING_USE_OPENCV_CUDA=1
    )
else()
    message(STATUS "Set macro HANDTRACKING_USE_OPENCV_CUDA = 0.")

    target_compile_definitions(${EXE_TARGET_NAME}
                               PRIVATE
                                   HANDTRACKING_USE_OPENCV_CUDA=0
    )
endif()


# Custom command for RESOURCE/EXTRA files
set(${EXE_TARGET_NAME}_APP
        ${CMAKE_CURRENT_SOURCE_DIR}/app/hand-tracking-iCub.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/app/hand-tracking-iCub-Batch.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/app/hand-tracking-iCub-Sim.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/app/hand-tracking-Walkman-Batch.xml
)

set(${EXE_TARGET_NAME}_CONF
        ${CMAKE_CURRENT_SOURCE_DIR}/conf/config.ini
)

set(${EXE_TARGET_NAME}_SHADER_VERT
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_frame.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_model.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_background.vert
)

set(${EXE_TARGET_NAME}_SHADER_FRAG
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_frame.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_model.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_background.frag
)

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
		${CMAKE_CURRENT_SOURCE_DIR}/mesh/walkman_right_arm/soft_hand_open.stl
)

add_custom_command(TARGET  ${EXE_TARGET_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${${EXE_TARGET_NAME}_CONF}        $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>
                   COMMAND ${CMAKE_COMMAND} -E copy ${${EXE_TARGET_NAME}_SHADER_VERT} $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>
                   COMMAND ${CMAKE_COMMAND} -E copy ${${EXE_TARGET_NAME}_SHADER_FRAG} $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>
                   COMMAND ${CMAKE_COMMAND} -E copy ${${EXE_TARGET_NAME}_MESH}        $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>
)

install(TARGETS ${EXE_TARGET_NAME}                DESTINATION bin)
install(FILES   ${${EXE_TARGET_NAME}_APP}         DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})
install(FILES   ${${EXE_TARGET_NAME}_CONF}        DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${EXE_TARGET_NAME})
install(FILES   ${${EXE_TARGET_NAME}_SHADER_VERT} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${EXE_TARGET_NAME}/shader)
install(FILES   ${${EXE_TARGET_NAME}_SHADER_FRAG} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${EXE_TARGET_NAME}/shader)
install(FILES   ${${EXE_TARGET_NAME}_MESH}        DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${EXE_TARGET_NAME}/mesh)
