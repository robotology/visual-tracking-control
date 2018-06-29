#ifndef CAMERA_H
#define CAMERA_H

#include <BayesFilters/Process.h>

#include <array>
#include <tuple>

namespace bfl {
    class Camera;
}


class bfl::Camera : public bfl::Process
{
public:
    virtual ~Camera() noexcept { };

    struct CameraParameters
    {
        unsigned int width;
        unsigned int height;
        double fx;
        double fy;
        double cx;
        double cy;
    };

    virtual CameraParameters getCameraParameters() const = 0;
};

#endif /* CAMERA_H */
