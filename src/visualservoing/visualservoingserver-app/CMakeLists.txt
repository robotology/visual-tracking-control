set(APP_TARGET_NAME visualservoingserver_app)

find_package(YARP REQUIRED
             COMPONENTS OS dev sig math)

include(YarpInstallationHelpers)

# Application source and header files
set(${APP_TARGET_NAME}_SRC
        src/main.cpp)

# Application target calls
add_executable(${APP_TARGET_NAME} ${${APP_TARGET_NAME}_SRC})

target_link_libraries(${APP_TARGET_NAME}
                      PRIVATE
                          YARP::YARP_OS
                          YARP::YARP_dev
                          YARP::YARP_math
                          YARP::YARP_sig
                          visualservoingplugin)

install(TARGETS ${APP_TARGET_NAME} DESTINATION bin)
