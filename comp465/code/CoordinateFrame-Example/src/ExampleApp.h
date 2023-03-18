#ifndef EXAMPLEAPP_H
#define EXAMPLEAPP_H

#include <api/MinVR.h>
using namespace MinVR;

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

#ifdef _WIN32
#include "GL/glew.h"
#include "GL/wglew.h"
#elif (!defined(__APPLE__))
#include "GL/glxew.h"
#endif

// OpenGL Headers
#if defined(WIN32)
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#elif defined(__APPLE__)
#define GL_GLEXT_PROTOTYPES
#include <OpenGL/gl3.h>
#include <OpenGL/glext.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#endif

#include <BasicGraphics.h>


class ExampleApp : public VRApp {
public:
    
    /** The constructor passes argc, argv, and a MinVR config file on to VRApp.
     */
	ExampleApp(int argc, char** argv);
    virtual ~ExampleApp();

    
    /** USER INTERFACE CALLBACKS **/
    virtual void onAnalogChange(const VRAnalogEvent &state);
    virtual void onButtonDown(const VRButtonEvent &state);
    virtual void onButtonUp(const VRButtonEvent &state);
	virtual void onCursorMove(const VRCursorEvent &state);
    virtual void onTrackerMove(const VRTrackerEvent &state);
    
    
    /** RENDERING CALLBACKS **/
    virtual void onRenderGraphicsScene(const VRGraphicsState& state);
    virtual void onRenderGraphicsContext(const VRGraphicsState& state);
    
private:

	double _lastTime;
	double _curFrameTime;
    
    std::unique_ptr<basicgraphics::Sphere> sphere;
    
    std::unique_ptr<basicgraphics::Cylinder> xAxis;
    std::unique_ptr<basicgraphics::Cone> xPoint;
    std::unique_ptr<basicgraphics::Cylinder> yAxis;
    std::unique_ptr<basicgraphics::Cone> yPoint;
    std::unique_ptr<basicgraphics::Cylinder> zAxis;
    std::unique_ptr<basicgraphics::Cone> zPoint;
    
    void drawAxes(const glm::mat4 &transform);
    
    virtual void reloadShaders();
    basicgraphics::GLSLProgram _shader;
};


#endif //EXAMPLEAPP_H
