#include <chrono>
#include <future>
#include <iostream>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <yarp/os/LogStream.h>
#include <yarp/os/ConstString.h>

#include <BayesFiltersLib/FilteringContext.h>
#include <BayesFiltersLib/FilteringFunction.h>
#include <BayesFiltersLib/SIRParticleFilter.h>

#include "BrownianMotion.h"
#include "Proprioception.h"
#include "VisualParticleFilterCorrection.h"
#include "VisualSIRParticleFilter.h"

#define WINDOW_WIDTH  320
#define WINDOW_HEIGHT 240

using namespace bfl;
using namespace Eigen;
using namespace yarp::os;

/* DEBUG ONLY */
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
/* ********** */


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    /* When a user presses the escape key, we set the WindowShouldClose property to true, closing the application. */
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GL_TRUE);
}


bool openglSetUp(GLFWwindow *& window, const int width, const int height)
{
    ConstString log_ID = "[OpenGL]";
    yInfo() << log_ID << "Start setting up...";

    /* Initialize GLFW. */
    if (glfwInit() == GL_FALSE)
    {
        yError() << log_ID << "Failed to initialize GLFW.";
        return false;
    }

    /* Set context properties by "hinting" specific (property, value) pairs. */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE,             GL_FALSE);
    glfwWindowHint(GLFW_VISIBLE,               GL_TRUE);
#ifdef GLFW_MAC
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    /* Create a window. */
    window = glfwCreateWindow(width, height, "OpenGL Window", nullptr, nullptr);
    if (window == nullptr)
    {
        yError() << log_ID << "Failed to create GLFW window.";
        glfwTerminate();
        return false;
    }
    /* Make the OpenGL context of window the current one handled by this thread. */
    glfwMakeContextCurrent(window);

    /* Set window callback functions. */
    glfwSetKeyCallback(window, key_callback);

    /* Initialize GLEW to use the OpenGL implementation provided by the videocard manufacturer. */
    /* Note: remember that the OpenGL are only specifications, the implementation is provided by the manufacturers. */
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        yError() << log_ID << "Failed to initialize GLEW.";
        return false;
    }

    /* Set OpenGL rendering frame for the current window. */
    /* Note that the real monitor width and height may differ w.r.t. the choosen one in hdpi monitors. */
    int hdpi_width;
    int hdpi_height;
    glfwGetFramebufferSize(window, &hdpi_width, &hdpi_height);
    glViewport(0, 0, hdpi_width, hdpi_height);
    yInfo() << log_ID << "Viewport set to "+std::to_string(hdpi_width)+"x"+std::to_string(hdpi_height)+".";

    /* Set GL property. */
    glEnable(GL_DEPTH_TEST);

    glfwPollEvents();

    yInfo() << log_ID << "Succesfully set up!";
    
    return true;
}


int main(int argc, char const *argv[])
{
    ConstString log_ID = "[Main]";
    yInfo() << log_ID << "Configuring and starting module...";

    /* DEBUG ONLY */
    namedWindow("Superimposed Edges", WINDOW_NORMAL | WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
    /* ********** */

    /* Initialize OpenGL context */
    GLFWwindow * window = nullptr;
    if (!openglSetUp(window, WINDOW_WIDTH, WINDOW_HEIGHT)) return EXIT_FAILURE;

    std::shared_ptr<BrownianMotion> brown(new BrownianMotion());
    std::shared_ptr<ParticleFilterPrediction> pf_prediction(new ParticleFilterPrediction(brown));
    std::shared_ptr<Proprioception> proprio(new Proprioception(window));
    std::shared_ptr<VisualParticleFilterCorrection> vpf_correction(new VisualParticleFilterCorrection(proprio));
    std::shared_ptr<Resampling> resampling(new Resampling());
    VisualSIRParticleFilter vsir_pf(brown, pf_prediction, proprio, vpf_correction, resampling);

    std::future<void> thr_vpf = vsir_pf.spawn();
    while (vsir_pf.isRunning())
    {
        if (glfwWindowShouldClose(window))
        {
            std::chrono::milliseconds span(1);
            vsir_pf.stopThread();
            yInfo() << log_ID << "Joining filthering thread...";
            while (thr_vpf.wait_for(span) == std::future_status::timeout) glfwPollEvents();
        }
        else glfwPollEvents();
    }

    glfwMakeContextCurrent(NULL);
    glfwTerminate();

    /* DEBUG ONLY */
    destroyWindow("Superimposed Edges");
    /* ********** */

    yInfo() << log_ID << "Main returning.";
    yInfo() << log_ID << "Application closed.";

    return EXIT_SUCCESS;
}
