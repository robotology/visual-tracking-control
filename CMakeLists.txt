cmake_minimum_required(VERSION 3.0)

project(hand-tracking)

### --- FILES::packages --- ###
find_package(assimp             REQUIRED)
find_package(EIGEN3             REQUIRED)
find_package(GLEW               REQUIRED)
find_package(GLFW3              REQUIRED)
find_package(glm                REQUIRED)
find_package(ICUB               REQUIRED)
find_package(ICUBcontrib        REQUIRED)
find_package(OpenCV             REQUIRED)
find_package(OPENGL             REQUIRED)
find_package(YARP               REQUIRED)

### --- FILES --- ###
set(SOURCE_DIRS)
set(SOURCE_FILES)
set(INCLUDE_DIRS)
set(INCLUDE_FILES)
set(RESOURCE_FILES)
set(LIBS)

### --- FILES::BayesFiltersLib --- ###
find_package(BayesFiltersLib REQUIRED)

### --- FILES::superimpose-hand --- ###
set(SH_DIR                  ${CMAKE_CURRENT_SOURCE_DIR}/../superimpose-hand)

set(SH_INCLUDE_DIR          ${SH_DIR}/include)
set(SH_INCLUDE_FILES        ${SH_INCLUDE_DIR}/Shader.h
                            ${SH_INCLUDE_DIR}/Mesh.h
                            ${SH_INCLUDE_DIR}/Model.h
                            ${SH_INCLUDE_DIR}/SuperImpose.h
                            ${SH_INCLUDE_DIR}/SICAD.h
                            ${SH_INCLUDE_DIR}/SISkeleton.h
                            ${SH_INCLUDE_DIR}/CADSuperimposer.h
                            ${SH_INCLUDE_DIR}/SkeletonSuperimposer.h
                            ${SH_INCLUDE_DIR}/SuperimposerFactory.h)

LIST(APPEND INCLUDE_DIRS    ${SH_INCLUDE_DIR})
LIST(APPEND INCLUDE_FILES   ${SH_INCLUDE_FILES})


set(SH_SOURCE_DIR           ${SH_DIR}/src)
set(SH_SOURCE_FILES         ${SH_SOURCE_DIR}/Shader.cpp
                            ${SH_SOURCE_DIR}/Mesh.cpp
                            ${SH_SOURCE_DIR}/Model.cpp
                            ${SH_SOURCE_DIR}/SICAD.cpp
                            ${SH_SOURCE_DIR}/SISkeleton.cpp
                            ${SH_SOURCE_DIR}/CADSuperimposer.cpp
                            ${SH_SOURCE_DIR}/SkeletonSuperimposer.cpp
                            ${SH_SOURCE_DIR}/SuperimposerFactory.cpp)

LIST(APPEND SOURCE_DIRS     ${SH_SOURCE_DIR})
LIST(APPEND SOURCE_FILES    ${SH_SOURCE_FILES})

set(SHADERS_DIR             ${SH_DIR}/shader)
set(SHADERS_VERT_FILES      ${SHADERS_DIR}/shader_model.vert
                            ${SHADERS_DIR}/shader_background.vert)
set(SHADERS_FRAG_FILES      ${SHADERS_DIR}/shader_model.frag
                            ${SHADERS_DIR}/shader_model_simple.frag
                            ${SHADERS_DIR}/shader_background.frag)

LIST(APPEND RESOURCE_FILES  ${SHADERS_VERT_FILES}
                            ${SHADERS_FRAG_FILES})

### --- FILES::hand-tracking --- ###
set(HT_INCLUDE_DIR          ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(HT_INCLUDE_FILES        ${HT_INCLUDE_DIR}/BrownianMotion.h
                            ${HT_INCLUDE_DIR}/Proprioception.h
                            ${HT_INCLUDE_DIR}/VisualParticleFilterCorrection.h
                            ${HT_INCLUDE_DIR}/VisualSIRParticleFilter.h)

LIST(APPEND INCLUDE_DIRS    ${HT_INCLUDE_DIR})
LIST(APPEND INCLUDE_FILES   ${HT_INCLUDE_FILES})


set(HT_SOURCE_DIR           ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(HT_SOURCE_FILES         ${HT_SOURCE_DIR}/main.cpp
                            ${HT_SOURCE_DIR}/BrownianMotion.cpp
                            ${HT_SOURCE_DIR}/Proprioception.cpp
                            ${HT_SOURCE_DIR}/VisualParticleFilterCorrection.cpp
                            ${HT_SOURCE_DIR}/VisualSIRParticleFilter.cpp)

LIST(APPEND SOURCE_DIRS     ${HT_SOURCE_DIR})
LIST(APPEND SOURCE_FILES    ${HT_SOURCE_FILES})


set(ICUB_MESH_EXT           obj)
set(ICUB_HAND_DIR           resource/icub_right_hand/${ICUB_MESH_EXT})
set(ICUB_HAND_FILES         ${ICUB_HAND_DIR}/r_palm.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_ail0.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_ail1.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_ail2.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_ail3.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_indexbase.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_ml0.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_ml1.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_ml2.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_ml3.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_palm.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_tl0.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_tl1.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_tl2.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_tl3.${ICUB_MESH_EXT}
                            ${ICUB_HAND_DIR}/r_tl4.${ICUB_MESH_EXT})

LIST(APPEND RESOURCE_FILES  ${ICUB_HAND_FILES})

set(IMAGE_DIR               ${CMAKE_CURRENT_SOURCE_DIR}/resource/log/camera/left)
file(GLOB   IMAGE_FILES     ${IMAGE_DIR}/*.ppm)

# --- Thirft files need to be in the current source dir --- #
# --- At the moment, idl_rfm.thrift is copied from SH to HT folder --- #
set(THIRFT_FILES            ${HT_SOURCE_DIR}/idl_rfm.thrift)

LIST(APPEND CMAKE_MODULE_PATH   ${YARP_MODULE_PATH}
                                ${ICUB_MODULE_PATH}
                                ${ICUBCONTRIB_MODULE_PATH})

include(YarpInstallationHelpers)
include(ICUBcontribHelpers)
include(ICUBcontribOptions)

icubcontrib_set_default_prefix()

yarp_add_idl(IDL_GEN_SOURCE     ${THIRFT_FILES})

LIST(APPEND INCLUDE_DIRS        ${YARP_INCLUDE_DIRS}
                                ${ICUB_INCLUDE_DIRS}
                                ${OPENGL_INCLUDE_DIR}
                                ${ASSIMP_INCLUDE_DIRS}
                                ${EIGEN3_INCLUDE_DIRS})

LIST(APPEND LIBS                ${YARP_LIBRARIES}
                                ${ICUB_LIBRARIES}
                                ctrlLib
                                iKin
                                ${OpenCV_LIBS}
                                ${OPENGL_LIBRARIES}
                                glew
                                glm
                                glfw
                                ${ASSIMP_LIBRARIES} -L${ASSIMP_LIBRARY_DIRS}
                                BayesFiltersLib)

LIST(APPEND SOURCE_FILES ${IDL_GEN_SOURCE})


### --- OPTIONS --- ###
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-D GLFW_MAC)
    option(IS_APPLE_RETINA
           "Select this option if the Apple machine has a Retina display" ON)
    if(IS_APPLE_RETINA)
        add_definitions(-D GLFW_RETINA)
    endif()
endif()

### --- MAIN --- ###
if(${CMAKE_GENERATOR} MATCHES "Xcode")
    add_executable(${PROJECT_NAME}                        ${SOURCE_FILES} ${INCLUDE_FILES}
                                                          ${SHADERS_VERT_FILES} ${SHADERS_FRAG_FILES}
                                                          ${ICUB_HAND_FILES}
                                                          ${IMAGE_FILES})

    source_group("Source Files\\FilteringContext"   FILES ${FC_SOURCE_FILES})
    source_group("Source Files\\FilteringAlgorithm" FILES ${FA_SOURCE_FILES})
    source_group("Source Files\\FilteringFunction"  FILES ${FF_SOURCE_FILES})
    source_group("Header Files\\FilteringContext"   FILES ${FC_INCLUDE_FILES})
    source_group("Header Files\\FilteringAlgorithm" FILES ${FA_INCLUDE_FILES})
    source_group("Header Files\\FilteringFunction"  FILES ${FF_INCLUDE_FILES})
    source_group("Resource\\Shader Files"           FILES ${SHADERS_VERT_FILES} ${SHADERS_FRAG_FILES})
    source_group("Resource\\CAD Files"              FILES ${ICUB_HAND_FILES})
    source_group("Resource\\Image"                  FILES ${IMAGE_FILES})
else()
    add_executable(${PROJECT_NAME}                        ${SOURCE_FILES}
                                                          ${INCLUDE_FILES})
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}   ${LIBS})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_nullptr cxx_constexpr)


### --- CUSTOM --- ###
add_custom_command(TARGET  ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory                       $<TARGET_FILE_DIR:${PROJECT_NAME}>
                   COMMAND ${CMAKE_COMMAND} -E copy             ${RESOURCE_FILES}   $<TARGET_FILE_DIR:${PROJECT_NAME}>)
