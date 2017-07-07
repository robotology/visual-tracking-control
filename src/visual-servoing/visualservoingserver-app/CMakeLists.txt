set(APP_TARGET_NAME visualservoingserver_app)

find_package(YARP REQUIRED)

include(YarpInstallationHelpers)

LIST(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH})

include(YarpInstallationHelpers)

# Application source and header files
set(${APP_TARGET_NAME}_SRC
        src/main.cpp)

# Application target calls
add_executable(${APP_TARGET_NAME} ${${APP_TARGET_NAME}_SRC})

target_include_directories(${APP_TARGET_NAME} PRIVATE ${YARP_INCLUDE_DIRS})

target_link_libraries(${APP_TARGET_NAME} ${YARP_LIBRARIES}
                                         visualservoingplugin)

install(TARGETS ${APP_TARGET_NAME}          DESTINATION bin)
