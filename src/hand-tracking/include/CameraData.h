#ifndef CAMERATDATA_H
#define CAMERATDATA_H

#include <CVMatData.h>

namespace bfl {
    class CameraData;
}


class bfl::CameraData : public bfl::CVMatData
{
public:
    CameraData() noexcept :
        CVMatData()
    {
        position_ = std::make_shared<std::array<double, 3>>();

        orientation_ = std::make_shared<std::array<double, 4>>();
    }

    std::shared_ptr<std::array<double, 3>> position_ = nullptr;

    std::shared_ptr<std::array<double, 4>> orientation_ = nullptr;
};


#endif /* CAMERATDATA_H */
