#===============================================================================
#
# Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# BSD 3-Clause license. See the accompanying LICENSE file for details.
#
#===============================================================================

set(APP_TARGET_NAME visualservoingclient_app)

find_package(YARP REQUIRED
             COMPONENTS OS dev math sig)

# Application source and header files
set(${APP_TARGET_NAME}_SRC
        src/main.cpp)

# Application target calls
add_executable(${APP_TARGET_NAME} ${${APP_TARGET_NAME}_SRC})

target_link_libraries(${APP_TARGET_NAME}
                      PRIVATE
                          YARP::YARP_OS
                          YARP::YARP_dev
                          YARP::YARP_init
                          YARP::YARP_math
                          YARP::YARP_sig
                          visualservoingplugin)

install(TARGETS ${APP_TARGET_NAME} DESTINATION bin)
