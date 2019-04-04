/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#version 330 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec2 texCoord;

out vec3 ourColor;
out vec2 TexCoord;

uniform mat4 projection;

void main()
{
    //gl_Position = vec4(position, 0.99, 1.0f);
    gl_Position = projection * vec4(position, -99999.99, 1.0f);
    ourColor = color;
    TexCoord = vec2(texCoord.x, 1.0f - texCoord.y);
}
