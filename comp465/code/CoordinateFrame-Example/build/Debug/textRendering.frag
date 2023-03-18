#version 330

in vec2 texture_coordinates;
in vec4 colorVal;
uniform sampler2D lambertian_texture;

out vec4 fragment_colour; // final colour of surface

void main () {

	vec4 lambertianColor = colorVal;
    lambertianColor.a = texture(lambertian_texture, texture_coordinates).r;
	fragment_colour = lambertianColor;
		
}