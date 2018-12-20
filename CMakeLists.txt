if(WIN32)
    cmake_minimum_required(VERSION 3.4)
else()
    cmake_minimum_required(VERSION 3.0)
endif()

project(visual-tracking-control
        LANGUAGES CXX
        VERSION 0.5.101)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(WIN32)
    add_definitions(-D_USE_MATH_DEFINES)
    if(MSVC)
        set(CMAKE_DEBUG_POSTFIX "d")
        set(CMAKE_CXX_FLAGS "/MP /EHsc")
        set(CMAKE_C_FLAGS "/MP /EHsc")
    endif()
endif()

### iCub-contrib project
find_package(ICUBcontrib REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})
include(ICUBcontribHelpers)
include(ICUBcontribOptions)
icubcontrib_set_default_prefix()
icubcontrib_add_uninstall_target()

# Support RPATH?
option(ENABLE_RPATH "Enable RPATH for this library" ON)
mark_as_advanced(ENABLE_RPATH)
include(AddInstallRPATHSupport)
add_install_rpath_support(BIN_DIRS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
                          LIB_DIRS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
                          INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}"
                          DEPENDS ENABLE_RPATH
                          USE_LINK_PATH)

add_subdirectory(src)
