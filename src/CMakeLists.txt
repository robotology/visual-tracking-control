#===============================================================================
#
# Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
#
# This software may be modified and distributed under the terms of the
# BSD 3-Clause license. See the accompanying LICENSE file for details.
#
#===============================================================================

option(BUILD_HAND_TRACKING "Build hand-tracking application" OFF)
if(BUILD_HAND_TRACKING)
    add_subdirectory(hand-tracking)
endif()

option(BUILD_REACHING "Build reaching application" OFF)
if(BUILD_REACHING)
    add_subdirectory(reaching)
endif()

add_subdirectory(visualservoing)
