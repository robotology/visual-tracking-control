set(APP_TARGET_NAME visualservoingclient_app)

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

# Application XML app files
set(${APP_TARGET_NAME}_APP
        ${PROJECT_SOURCE_DIR}/app/visualservoing.xml
        ${PROJECT_SOURCE_DIR}/app/visualservoing-SIM.xml
        ${PROJECT_SOURCE_DIR}/app/visualservoing-datadump.xml
        ${PROJECT_SOURCE_DIR}/app/BackgroundApps-SIM.xml)

install(TARGETS ${APP_TARGET_NAME}          DESTINATION bin)
install(FILES   ${${APP_TARGET_NAME}_APP}   DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})
