/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#version 330 core

out vec4 color;

void main()
{
    // color = vec4(1.0f, 0.5f, 0.2f, 1.0f); // RGB orange-like color
    color = vec4(0.2f, 0.5f, 1.0f, 1.0f); // BGR orange-like color
}
