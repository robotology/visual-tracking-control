set(PLUGIN_NAME visualservoing)

find_package(YARP REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH})
include(YarpPlugin)
yarp_configure_external_installation(${PLUGIN_NAME} WITH_PLUGINS)

yarp_begin_plugin_library(visualservoingplugin)
    option(BUILD_VISUAL_SERVOING_CLIENT "Build visualservoing-client YARP plugin" OFF)
    if(BUILD_VISUAL_SERVOING_CLIENT)
        add_subdirectory(visualservoingclient)
        add_subdirectory(visualservoingclient-app)
    endif()

    option(BUILD_VISUAL_SERVOING_SERVER "Build visualservoing-server YARP plugin" OFF)
    if(BUILD_VISUAL_SERVOING_SERVER)
        add_subdirectory(visualservoingserver)
        add_subdirectory(visualservoingserver-app)
    endif()
yarp_end_plugin_library(visualservoingplugin)
