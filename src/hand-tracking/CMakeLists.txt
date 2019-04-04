#===============================================================================
#
# Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# BSD 3-Clause license. See the accompanying LICENSE file for details.
#
#===============================================================================

set(EXE_TARGET_NAME hand-tracking)

include(AddDependenciesPrintUtils)


# Dependencies
find_package(CUDA QUIET)
print_dependency(CUDA)


find_package(OpenCV REQUIRED)
print_dependency(OpenCV)


find_package(YARP CONFIG REQUIRED
             COMPONENTS OS math sig eigen gsl cv
)
print_dependency(YARP)


find_package(ICUB REQUIRED)
print_dependency(ICUB)


find_package(Eigen3 QUIET CONFIG)
if(NOT EIGEN3_FOUND)
    find_package(Eigen3 REQUIRED)
endif()
print_dependency(Eigen3)


find_package(BayesFilters 0.9.0 QUIET)
if(NOT BayesFilters_FOUND)
  message(STATUS "Did not found required master release of BayesFilters. Looking for devel version.")
  find_package(BayesFilters 0.9.100 REQUIRED)
endif()
print_dependency(BayesFilters)


find_package(SuperimposeMesh 0.9.4.0 QUIET)
if(NOT SuperimposeMesh_FOUND)
  message(STATUS "Did not found required master release of SuperimposeMesh. Looking for devel version.")
  find_package(SuperimposeMesh 0.10.100.0 REQUIRED)
endif()
print_dependency(SuperimposeMesh)


list(APPEND CMAKE_MODULE_PATH ${ICUB_MODULE_PATH})

if(DEFINED OpenCV_CUDA_VERSION)
  message(STATUS "Found OpenCV with CUDA compiled modules.")

  if(CUDA_FOUND)
    option(VTC_FORCE_NO_OPENCV_CUDA "Force no CUDA compiled modules?" OFF)

    if(VTC_FORCE_NO_OPENCV_CUDA)
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
set(CUDA_LIB_NAME cudadevicelikelihood)
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

  message(STATUS "CUDA_LIB_TYPE: ${CUDA_LIB_TYPE}")


  set(CUDA_SEPARABLE_COMPILATION ON)
  message(STATUS "CUDA_SEPARABLE_COMPILATION: ${CUDA_SEPARABLE_COMPILATION}")


  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
  message(STATUS "CUDA_ARCH_FLAGS: ${CUDA_ARCH_FLAGS}")


  list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

  if (UNIX)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  endif()

  message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")


  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  message(STATUS "CUDA_HOST_COMPILER: ${CUDA_HOST_COMPILER}")


  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  message(STATUS "CUDA_LINK_LIBRARIES_KEYWORD: ${CUDA_LINK_LIBRARIES_KEYWORD}")


  message(STATUS "CUDA_PROPAGATE_HOST_FLAGS: ${CUDA_PROPAGATE_HOST_FLAGS}")


  message(STATUS "CUDA library ${CUDA_LIB_NAME} compilation is enabled.")
else()
  message(STATUS "CUDA library ${CUDA_LIB_NAME} compilation is disabled.")
endif()


# Application source and header files
set(${EXE_TARGET_NAME}_HDR
      include/BrownianMotionPose.h
      include/Camera.h
      include/ChiSquare.h
      include/DrawParticlesImportanceThreshold.h
      include/KinPoseModel.h
      include/KinPoseModelAxisAngle.h
      include/GatePose.h
      include/iCubFwdKinModel.h
      include/iCubArmModel.h
      include/iCubCamera.h
      include/iCubGatePose.h
      include/InitiCubArm.h
      include/InitPoseParticles.h
      include/InitPoseParticlesAxisAngle.h
      include/InitWalkmanArm.h
      include/KLD.h
      include/MeshModel.h
      include/NormOne.h
      include/NormTwo.h
      include/NormTwoChiSquare.h
      include/NormTwoKLD.h
      include/NormTwoKLDChiSquare.h
      include/PlayiCubFwdKinModel.h
      include/PlayGatePose.h
      include/PlayWalkmanPoseModel.h
      include/VisualProprioception.h
      include/VisualSIS.h
      include/WalkmanArmModel.h
      include/WalkmanCamera.h
      include/utils.h
)

set(${EXE_TARGET_NAME}_SRC
      src/main.cpp
      src/BrownianMotionPose.cpp
      src/DrawParticlesImportanceThreshold.cpp
      src/KinPoseModel.cpp
      src/KinPoseModelAxisAngle.cpp
      src/GatePose.cpp
      src/iCubArmModel.cpp
      src/iCubCamera.cpp
      src/iCubFwdKinModel.cpp
      src/iCubGatePose.cpp
      src/InitiCubArm.cpp
      src/InitPoseParticles.cpp
      src/InitPoseParticlesAxisAngle.cpp
      src/InitWalkmanArm.cpp
      src/PlayiCubFwdKinModel.cpp
      src/PlayGatePose.cpp
      src/PlayWalkmanPoseModel.cpp
      src/VisualSIS.cpp
      src/WalkmanArmModel.cpp
      src/WalkmanCamera.cpp
      src/utils.cpp
)

set(${EXE_TARGET_NAME}_CPU_SRC
      src/cpu/ChiSquare.cpp
      src/cpu/KLD.cpp
      src/cpu/NormOne.cpp
      src/cpu/NormTwo.cpp
      src/cpu/NormTwoChiSquare.cpp
      src/cpu/NormTwoKLD.cpp
      src/cpu/NormTwoKLDChiSquare.cpp
      src/cpu/VisualProprioception.cpp
)

set(${EXE_TARGET_NAME}_CUDA_SRC
      src/cuda/ChiSquare.cpp
      src/cuda/KLD.cpp
      src/cuda/NormOne.cpp
      src/cuda/NormTwo.cpp
      src/cuda/NormTwoChiSquare.cpp
      src/cuda/NormTwoKLD.cpp
      src/cuda/NormTwoKLDChiSquare.cpp
      src/cuda/VisualProprioception.cpp
)

set(${EXE_TARGET_NAME}_THRIFT_HDR
      thrift/visualsisparticlefilter.thrift)

yarp_add_idl(${EXE_TARGET_NAME}_THRIFT_SRC ${${EXE_TARGET_NAME}_THRIFT_HDR})

set(${CUDA_LIB_NAME}_SRC_FILES
      src/cuda/device_likelihood.cu
)

set(${CUDA_LIB_NAME}_HDR_FILES
      include/device_likelihood.h
      include/thrust_opencv_iterator.h
)


# CUDA library target calls
if(BFL_CUDA)
  cuda_add_library(${CUDA_LIB_NAME}
                     ${${CUDA_LIB_NAME}_HDR_FILES}
                     ${${CUDA_LIB_NAME}_SRC_FILES}
                     ${CUDA_LIB_TYPE})

  target_include_directories(${CUDA_LIB_NAME} ${CUDA_LINK_LIBRARIES_KEYWORD}
                              "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                              "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
  )

  target_link_libraries(${CUDA_LIB_NAME} ${CUDA_LINK_LIBRARIES_KEYWORD}
                          ${CUDA_nppc_LIBRARY}
                          ${OpenCV_LIBS})

  cuda_add_cublas_to_target(${CUDA_LIB_NAME})

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

target_include_directories(${EXE_TARGET_NAME}
                           PRIVATE
                            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                            ${ICUB_INCLUDE_DIRS}
)

target_link_libraries(${EXE_TARGET_NAME}
                      PRIVATE
                        BayesFilters::BayesFilters
                        ctrlLib
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
                        YARP::YARP_cv
)

if(BFL_CUDA)
    target_include_directories(${EXE_TARGET_NAME}
                               PRIVATE
                                 ${CUDA_INCLUDE_DIRS}
    )

    target_link_libraries(${EXE_TARGET_NAME}
                          PRIVATE
                            ${CUDA_LIBRARIES}
                            ${CUDA_LIB_NAME}
    )

    cuda_add_cublas_to_target(${EXE_TARGET_NAME})

    set_source_files_properties(${${EXE_TARGET_NAME}_CPU_SRC}
                                PROPERTIES HEADER_FILE_ONLY ON)
else()
    set_source_files_properties(${${CUDA_LIB_NAME}_HDR_FILES}
                                PROPERTIES HEADER_FILE_ONLY ON)

    set_source_files_properties(${${CUDA_LIB_NAME}_SRC_FILES}
                                PROPERTIES HEADER_FILE_ONLY ON)

    set_source_files_properties(${${EXE_TARGET_NAME}_CUDA_SRC}
                                PROPERTIES HEADER_FILE_ONLY ON)
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
             PREFIX "Source Files"
             FILES
               ${${EXE_TARGET_NAME}_CUDA_SRC}
               ${${EXE_TARGET_NAME}_CPU_SRC}
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
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ail0.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ail1.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ail2.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ail3.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_forearm.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_indexbase.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_lil0.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_lil1.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_lil2.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_lil3.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_littlebase.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ml0.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ml1.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ml2.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ml3.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_palm.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ril0.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ril1.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ril2.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ril3.obj
      ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_arm/r_ringbase.obj
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
