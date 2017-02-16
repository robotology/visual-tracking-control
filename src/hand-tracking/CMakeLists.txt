set(EXE_TARGET_NAME hand-tracking)

# Dependencies
find_package(BayesFiltersLib REQUIRED)
find_package(EIGEN3          REQUIRED)
find_package(GLEW            REQUIRED)
find_package(GLFW3           REQUIRED)
find_package(ICUB            REQUIRED)
find_package(OpenCV          REQUIRED)
find_package(OPENGL          REQUIRED)
find_package(SuperImpose     REQUIRED)
find_package(YARP            REQUIRED)

LIST(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH}
                              ${ICUB_MODULE_PATH}
                              ${ICUBCONTRIB_MODULE_PATH})

include(YarpInstallationHelpers)

# Project source and header files
set(${EXE_TARGET_NAME}_HDR
        include/BrownianMotion.h
        include/VisualProprioception.h
        include/VisualParticleFilterCorrection.h
        include/VisualSIRParticleFilter.h)

set(${EXE_TARGET_NAME}_SRC
        src/main.cpp
        src/BrownianMotion.cpp
        src/VisualProprioception.cpp
        src/VisualParticleFilterCorrection.cpp
        src/VisualSIRParticleFilter.cpp)

set(${EXE_TARGET_NAME}_THRIFT_HDR
        thrift/idl_visualsirpf.thrift)

# Project target calls
yarp_add_idl(${EXE_TARGET_NAME}_THRIFT_SRC ${${EXE_TARGET_NAME}_THRIFT_HDR})

add_executable(${EXE_TARGET_NAME}   ${${EXE_TARGET_NAME}_HDR} ${${EXE_TARGET_NAME}_SRC}
                                    ${${EXE_TARGET_NAME}_THRIFT_SRC})

target_include_directories(${EXE_TARGET_NAME} PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                      ${EIGEN3_INCLUDE_DIRS}
                                                      ${ICUB_INCLUDE_DIRS}
                                                      ${OPENGL_INCLUDE_DIR}
                                                      ${YARP_INCLUDE_DIRS})

target_link_libraries(${EXE_TARGET_NAME} BayesFiltersLib::BayesFiltersLib
                                         ctrlLib
                                         ${GLEW_LIBRARIES}
                                         glfw
                                         ${ICUB_LIBRARIES}
                                         iKin
                                         ${OpenCV_LIBS}
                                         ${OPENGL_LIBRARIES}
                                         SuperImpose::SuperImpose
                                         ${YARP_LIBRARIES})

target_compile_features(${EXE_TARGET_NAME} PUBLIC cxx_nullptr cxx_constexpr)


# Custom command for RESOURCE/EXTRA files
set(SHADERS_VERT_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_model.vert
                        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_background.vert)

set(SHADERS_FRAG_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_model.frag
                        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_model_simple.frag
                        ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader_background.frag)

set(ICUB_HAND_FILES     ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_palm.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ail0.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ail1.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ail2.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ail3.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_forearm.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_forearm_v1.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_forearm_full_v1.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_forearm_scaled_v1.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_forearm_simpl.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_indexbase.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ml0.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ml1.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ml2.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_ml3.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_palm.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl0.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl1.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl2.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl3.obj
                        ${CMAKE_CURRENT_SOURCE_DIR}/mesh/icub_right_hand/r_tl4.obj)

add_custom_command(TARGET  ${EXE_TARGET_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory             $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>
                   COMMAND ${CMAKE_COMMAND} -E copy ${SHADERS_VERT_FILES} $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>
                   COMMAND ${CMAKE_COMMAND} -E copy ${SHADERS_FRAG_FILES} $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>
                   COMMAND ${CMAKE_COMMAND} -E copy ${ICUB_HAND_FILES}    $<TARGET_FILE_DIR:${EXE_TARGET_NAME}>)
