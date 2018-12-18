set(PLUGIN_TARGET_NAME visualservoingclient)
string(TOUPPER ${PLUGIN_NAME} UC_PLUGIN_NAME)

yarp_prepare_plugin(${PLUGIN_TARGET_NAME}
                    CATEGORY
                        device
                    TYPE
                        VisualServoingClient
                    INCLUDE
                        VisualServoingClient.h
                    DEFAULT ON
                    ADVANCED
                    INTERNAL)


find_package(ICUB   REQUIRED)
find_package(OpenCV REQUIRED)

LIST(APPEND CMAKE_MODULE_PATH ${ICUB_MODULE_PATH})

# Plugin source and header files
set(${PLUGIN_TARGET_NAME}_HDR
        ../visualservoingcommon/include/VisualServoingCommon.h
        include/VisualServoingClient.h)

set(${PLUGIN_TARGET_NAME}_SRC
        src/VisualServoingClient.cpp)

set(${PLUGIN_TARGET_NAME}_CONF
        conf/visualservoingclient.ini)

set(${PLUGIN_TARGET_NAME}_THRIFT_HDR
        thrift/visualservoing.thrift)

# Plugin target calls
yarp_add_idl(${PLUGIN_TARGET_NAME}_THRIFT_SRC ${${PLUGIN_TARGET_NAME}_THRIFT_HDR})

yarp_add_plugin(${PLUGIN_TARGET_NAME} ${${PLUGIN_TARGET_NAME}_HDR} ${${PLUGIN_TARGET_NAME}_SRC}
                                      ${${PLUGIN_TARGET_NAME}_THRIFT_SRC})

target_include_directories(${PLUGIN_TARGET_NAME}
                           PRIVATE
                               "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                               ${ICUB_INCLUDE_DIRS})

target_link_libraries(${PLUGIN_TARGET_NAME}
                      PRIVATE
                          ctrlLib
                          ${ICUB_LIBRARIES}
                          iKin
                          ${OpenCV_LIBS}
                          YARP::YARP_OS
                          YARP::YARP_init
                          YARP::YARP_math
                          YARP::YARP_dev
                          YARP::YARP_sig
                          YARP::YARP_gsl)

yarp_install(TARGETS ${PLUGIN_TARGET_NAME}
             COMPONENT
                 plugin
             LIBRARY DESTINATION
                 ${ICUBCONTRIB_DYNAMIC_PLUGINS_INSTALL_DIR})

yarp_install(FILES ${${PLUGIN_TARGET_NAME}_CONF}
             COMPONENT
                 plugin
             DESTINATION
                 ${ICUBCONTRIB_PLUGIN_MANIFESTS_INSTALL_DIR})
