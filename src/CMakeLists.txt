option(VTC_BUILD_HAND_TRACKING "Build hand-tracking application" OFF)
if(VTC_BUILD_HAND_TRACKING)
  add_subdirectory(hand-tracking)
endif()

option(VTC_BUILD_REACHING "Build reaching application" OFF)
if(VTC_BUILD_REACHING)
  add_subdirectory(reaching)
endif()

add_subdirectory(visualservoing)
