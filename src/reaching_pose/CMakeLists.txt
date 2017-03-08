set(EXE_TARGET_NAME reaching_pose)

find_package(ICUB        REQUIRED)
find_package(OpenCV      REQUIRED)
find_package(SuperImpose REQUIRED)
find_package(YARP        REQUIRED)

LIST(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH}
                              ${ICUB_MODULE_PATH}
                              ${ICUBCONTRIB_MODULE_PATH})

include(YarpInstallationHelpers)

# Application source and header files
set(${EXE_TARGET_NAME}_SRC
        main.cpp)

add_executable(${EXE_TARGET_NAME} ${${EXE_TARGET_NAME}_SRC})

target_include_directories(${EXE_TARGET_NAME} PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                      ${ICUB_INCLUDE_DIRS}
                                                      ${YARP_INCLUDE_DIRS})

target_link_libraries(${EXE_TARGET_NAME} ctrlLib
                                         ${ICUB_LIBRARIES}
                                         iKin
                                         ${OpenCV_LIBS}
                                         SuperImpose::SuperImpose
                                         ${YARP_LIBRARIES})

target_compile_features(${EXE_TARGET_NAME} PUBLIC cxx_nullptr cxx_constexpr)

set(${EXE_TARGET_NAME}_APP
        ${PROJECT_SOURCE_DIR}/app/reaching_pose.xml
        ${PROJECT_SOURCE_DIR}/app/reaching_pose-SIM.xml)

install(TARGETS ${EXE_TARGET_NAME}        DESTINATION bin)
install(FILES   ${${EXE_TARGET_NAME}_APP} DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})
