set(EXE_TARGET_NAME reaching)

find_package(glew            REQUIRED)
find_package(ICUB            REQUIRED)
find_package(OpenCV          REQUIRED)
find_package(SuperimposeMesh REQUIRED)
find_package(YARP            REQUIRED)

LIST(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH}
                              ${ICUB_MODULE_PATH})

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
                                         SuperimposeMesh::SuperimposeMesh
                                         ${YARP_LIBRARIES})

set(${EXE_TARGET_NAME}_APP
        ${PROJECT_SOURCE_DIR}/app/reaching.xml
        ${PROJECT_SOURCE_DIR}/app/reaching-SIM.xml)

install(TARGETS ${EXE_TARGET_NAME}        DESTINATION bin)
install(FILES   ${${EXE_TARGET_NAME}_APP} DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})
