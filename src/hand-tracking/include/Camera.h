#ifndef CAMERA_H
#define CAMERA_H

#include <array>
#include <tuple>

namespace bfl {
    class Camera;
}


class bfl::Camera
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

    virtual std::tuple<bool, CameraParameters> readCameraParameters() = 0;

    virtual std::tuple<bool, std::array<double, 3>, std::array<double, 4>> readCameraPose() = 0;
};

#endif /* CAMERA_H */
