# Copyright (C) 2018 Claudio Fantacci
# Copyright (C) 2006-2018 Istituto Italiano di Tecnologia (IIT)
# Copyright (C) 2006-2010 RobotCub Consortium
# All rights reserved.
#
# This software may be modified and distributed under the terms of the
# BSD-3-Clause license. See the accompanying LICENSE file for details.
#
# Original version available online:
# https://github.com/robotology/yarp/blob/master/cmake/YarpFindDependencies.cmake
# https://github.com/robotology/yarp/blob/master/cmake/YarpPrintFeature.cmake

function(colorize_string _out_var _color _bold _string)
    if(${ARGC} GREATER 4)
        set(_alt "${ARGN}")
    else()
        set(_alt "${_string}")
    endif()

    if($ENV{CLICOLOR_FORCE})
        unset(_bold_arg)
        if(_bold)
            set(_bold_arg "--bold")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --no-newline --${_color} ${_bold_arg} "${_string}"
        OUTPUT_VARIABLE ${_out_var})
    else()
        set(${_out_var} "${_alt}")
    endif()

    set(${_out_var} ${${_out_var}} PARENT_SCOPE)
endfunction()


function(checkbox _var _out_var)
    if(${_var})
        colorize_string(_on green 1 "✔" "x")
        set(${_out_var} "[${_on}]")
    else()
        colorize_string(_off red 1 "✘" " ")
        set(${_out_var} "[${_off}]")
    endif()

    set(${_out_var} ${${_out_var}} PARENT_SCOPE)
endfunction()


function(print_with_checkbox _var _doc)
    checkbox(${_var} _check)
    message(STATUS " ${_check} ${_doc}")
endfunction()


function(print_feature _var _lev _doc)
    set(_indent "")
    foreach(i RANGE 0 ${_lev} 1)
        if(NOT ${i} EQUAL ${_lev})
            set(_indent "${_indent}  ")
        endif()
    endforeach()
    colorize_string(_help black 1 "${_var}")
    print_with_checkbox(${_var} "${_indent}${_doc} (${_help})")
endfunction()


macro(print_dependency package)
    string(TOUPPER ${package} PKG)
    set(DEPENDENCY_FOUND FALSE)

    if(DEFINED ${package}_REQUIRED_VERSION)
        set(_required_version " (>= ${${package}_REQUIRED_VERSION})")
    endif()

    if(DEFINED ${package}_VERSION)
        set(_version " ${${package}_VERSION}")
    endif()

    if(NOT DEFINED ${PKG}_FOUND AND NOT DEFINED ${package}_FOUND)
        set(_reason "disabled")
    elseif((DEFINED ${PKG}_FOUND AND NOT ${PKG}_FOUND) OR (DEFINED ${package}_FOUND AND NOT ${package}_FOUND))
        set(_reason "not found")
    else()
        set(DEPENDENCY_FOUND TRUE)

        unset(_where)
        if(${package}_DIR)
            set(_where " (${${package}_DIR})")
        elseif(${package}_LIBRARIES)
            list(GET ${package}_LIBRARIES 0 _lib)
            if(_lib MATCHES "^(optimized|debug)$")
                list(GET ${package}_LIBRARIES 1 _lib)
            endif()
            set(_where " (${_lib})")
        elseif(${package}_INCLUDE_DIRS)
            list(GET ${package}_INCLUDE_DIRS 0 _incl)
            set(_where " (${_incl})")
        elseif(${package}_LIBRARY)
            set(_where " (${${package}_LIBRARY})")
        elseif(${package}_INCLUDE_DIR)
            set(_where " (${${package}_INCLUDE_DIR})")
        elseif(${PKG}_LIBRARY)
            set(_where " (${${PKG}_LIBRARY})")
        elseif(${PKG}_INCLUDE_DIR)
            set(_where " (${${PKG}_INCLUDE_DIR})")
        endif()
        set(_reason "found${_version}${_where}")
    endif()

    print_with_checkbox(DEPENDENCY_FOUND "${package}${_required_version}: ${_reason}")

    unset(_lib)
    unset(_where)
    unset(_version)
    unset(_required_version)
    unset(_reason)
endmacro()
