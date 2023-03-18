#version 330

layout (location = 0) in vec2 vertex_position;
layout (location = 1) in vec2 vertex_texcoord;
layout (location = 2) in vec4 vertex_color;
uniform mat4 projection_mat, view_mat, model_mat;
out vec2 texture_coordinates;
out vec4 colorVal;

void main () {
	vec3 position_world = vec3 (model_mat * vec4 (vertex_position, 0.0, 1.0));
	texture_coordinates = vertex_texcoord;
	colorVal = vertex_color;
	gl_Position = projection_mat * view_mat * vec4 (position_world, 1.0);
}