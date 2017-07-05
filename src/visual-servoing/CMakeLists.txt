option(BUILD_CLIENT_VISUAL_SERVOING "Build client-visual-servoing YARP plugin" OFF)
if(BUILD_CLIENT_VISUAL_SERVOING)
    add_subdirectory(client-visual-servoing)
endif()

option(BUILD_SERVER_VISUAL_SERVOING "Build server-visual-servoing YARP plugin" OFF)
if(BUILD_SERVER_VISUAL_SERVOING)
    add_subdirectory(server-visual-servoing)
endif()
