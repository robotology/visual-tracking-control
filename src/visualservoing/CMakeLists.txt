#===============================================================================
#
# Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# BSD 3-Clause license. See the accompanying LICENSE file for details.
#
#===============================================================================

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


    # Application XML app files
    if(BUILD_VISUAL_SERVOING_CLIENT OR BUILD_VISUAL_SERVOING_SERVER)
        set(${PLUGIN_NAME}_APP
                visualservoingcommon/app/BackgroundApps-SIM.xml
                visualservoingcommon/app/visualservoing.xml
                visualservoingcommon/app/visualservoing-click.xml
                visualservoingcommon/app/visualservoing-SIM.xml
                visualservoingcommon/app/visualservoing-datadump.xml)

        install(FILES ${${PLUGIN_NAME}_APP} DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})
    endif()
yarp_end_plugin_library(visualservoingplugin)
