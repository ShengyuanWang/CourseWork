#include "ExampleApp.h"
#include <config/VRDataIndex.h>

#include <glm/gtx/orthonormalize.hpp>

using namespace basicgraphics;
using namespace std;
using namespace glm;

ExampleApp::ExampleApp(int argc, char** argv) : VRApp(argc, argv)
{
	_lastTime = 0.0;
    _curFrameTime = 0.0;
    
}

ExampleApp::~ExampleApp()
{
	shutdown();
}

void ExampleApp::onAnalogChange(const VRAnalogEvent &event) {
    // This routine is called for all Analog_Change events.  Check event->getName()
    // to see exactly which analog input has been changed, and then access the
    // new value with event->getValue().
    
	if (event.getName() == "FrameStart") {
		_lastTime = _curFrameTime;
		_curFrameTime = event.getValue();
	}


}

void ExampleApp::onButtonDown(const VRButtonEvent &event) {
    // This routine is called for all Button_Down events.  Check event->getName()
    // to see exactly which button has been pressed down.
	
	//std::cout << "ButtonDown: " << event.getName() << std::endl;
}

void ExampleApp::onButtonUp(const VRButtonEvent &event) {
    // This routine is called for all Button_Up events.  Check event->getName()
    // to see exactly which button has been released.

	//std::cout << "ButtonUp: " << event.getName() << std::endl;
}

void ExampleApp::onCursorMove(const VRCursorEvent &event) {
	// This routine is called for all mouse move events. You can get the absolute position
	// or the relative position within the window scaled 0--1.
	
	//std::cout << "MouseMove: "<< event.getName() << " " << event.getPos()[0] << " " << event.getPos()[1] << std::endl;
}

void ExampleApp::onTrackerMove(const VRTrackerEvent &event) {
    // This routine is called for all Tracker_Move events.  Check event->getName()
    // to see exactly which tracker has moved, and then access the tracker's new
    // 4x4 transformation matrix with event->getTransform().

	// We will use trackers when we do a virtual reality assignment. For now, you can ignore this input type.
}

void ExampleApp::onRenderGraphicsContext(const VRGraphicsState &renderState) {
    // This routine is called once per graphics context at the start of the
    // rendering process.  So, this is the place to initialize textures,
    // load models, or do other operations that you only want to do once per
    // frame.
    
	// Is this the first frame that we are rendering after starting the app?
    if (renderState.isInitialRenderCall()) {

		//For windows, we need to initialize a few more things for it to recognize all of the
		// opengl calls.
		#ifndef __APPLE__
			glewExperimental = GL_TRUE;
			GLenum err = glewInit();
			if (GLEW_OK != err)
			{
				std::cout << "Error initializing GLEW." << std::endl;
			}
		#endif     


        glEnable(GL_DEPTH_TEST);
        glClearDepth(1.0f);
        glDepthFunc(GL_LEQUAL);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		glEnable(GL_MULTISAMPLE);

		// This sets the background color that is used to clear the canvas
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

		// This load shaders from disk, we do it once when the program starts up.
		reloadShaders();
        
        xAxis.reset(new Cylinder(vec3(0, 0, 0), vec3(5, 0, 0), 0.125, vec4(1.0, 0.647, 0, 1.0)));
        
        xPoint.reset(new Cone(vec3(5, 0, 0), vec3(5.5, 0, 0), 0.5, vec4(1.0, 0.647, 0, 1.0)));
        
        yAxis.reset(new Cylinder(vec3(0, 0, 0), vec3(0, 5, 0), 0.125, vec4(0, 0.4, 0, 1.0)));
        
        yPoint.reset(new Cone(vec3(0, 5, 0), vec3(0, 5.5, 0), 0.5, vec4(0, 0.4, 0, 1.0)));
        
        
        zAxis.reset(new Cylinder(vec3(0, 0, 0), vec3(0, 0, 5), 0.125, vec4(0, 0, 0.4, 1.0)));
        
        zPoint.reset(new Cone(vec3(0, 0, 5), vec3(0, 0, 5.5), 0.5, vec4(0, 0, 0.4, 1.0)));
        
        sphere.reset(new Sphere(vec3(0.5), 0.1, vec4(1,0,0,1)));


    }
}

void ExampleApp::onRenderGraphicsScene(const VRGraphicsState &renderState) {
    // This routine is called once per eye/camera.  This is the place to actually
    // draw the scene.
    
	// clear the canvas and other buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	// Setup the view matrix to set where the camera is located in the scene
    glm::vec3 eye_world = glm::vec3(0,5,20);
    glm::mat4 view = glm::lookAt(eye_world, glm::vec3(0,0,0), glm::vec3(0,1,0));

	// Setup the projection matrix so that things are rendered in perspective
	GLfloat windowHeight = renderState.index().getValue("WindowHeight");
	GLfloat windowWidth = renderState.index().getValue("WindowWidth");
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), windowWidth / windowHeight, 0.01f, 100.0f);
	
	// Setup the model matrix
	glm::mat4 model = glm::mat4(1.0);
    
	// Tell opengl we want to use this specific shader.
	_shader.use();
	
	_shader.setUniform("view_mat", view);
	_shader.setUniform("projection_mat", projection);
	
	_shader.setUniform("model_mat", model);
	_shader.setUniform("normal_mat", mat3(transpose(inverse(model))));
	_shader.setUniform("eye_world", eye_world);


    // This is the main coordinate system for the entire application
    drawAxes(model);
    
    
    mat4 Rx = toMat4(angleAxis(glm::pi<float>()/8.0f, vec3(1.0f,0.0,0.0)));
    
    mat4 Ry = toMat4(angleAxis(glm::pi<float>()/8.0f, vec3(0.0,1.0f,0.0)));
    
    mat4 Tx = translate(mat4(1.0), vec3(1,0,0));
    
    mat4 M = Tx * Ry * Rx;
    
    
    
    // This is the new coordinate system that we defined by combining two rotation and
    // and one translation transformations.
    drawAxes(M);
    
    
    // If you want to draw geometry relative to a CoordinateFrame, then the best way is to
    // set the model(objectToWorld) matrix in the shader.  The vertices of any geometry
    // that you draw automatically get multipled by this matrix.  By default, the objectToWorld
    // matrix is the identity, but you can change it using _shader.setUniform("ModelMatrix", mat4).
    // Here's an example.
    
    // Set to identity matrix
    _shader.setUniform("model_mat", model);
    sphere->draw(_shader, model);
    
    // The same point in the M coordinate system
    _shader.setUniform("model_mat", M);
    sphere->draw(_shader, M);
    
}

void ExampleApp::drawAxes(const mat4 &transform){
    xAxis->draw(_shader, transform);
    xPoint->draw(_shader, transform);
    yAxis->draw(_shader, transform);
    yPoint->draw(_shader, transform);
    zAxis->draw(_shader, transform);
    zPoint->draw(_shader, transform);
    
    _shader.setUniform("model_mat", mat4(1.0));
}


void ExampleApp::reloadShaders()
{
	_shader.compileShader("texture.vert", GLSLShader::VERTEX);
	_shader.compileShader("texture.frag", GLSLShader::FRAGMENT);
	_shader.link();
	_shader.use();
}
