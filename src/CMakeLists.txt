option(BUILD_HAND_TRACKING "Build hand-tracking application" OFF)
if(BUILD_HAND_TRACKING)
    add_subdirectory(hand-tracking)
endif()

option(BUILD_REACHING "Build reaching application" OFF)
if(BUILD_REACHING)
    add_subdirectory(reaching)
endif()

add_subdirectory(visualservoing)
