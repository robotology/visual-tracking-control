#ifndef OCVMATDATA_H
#define OCVMATDATA_H

#include <BayesFilters/GenericData.h>

#include <memory>

#include <opencv2/core/mat.hpp>

namespace bfl {
    class CVMatData;
}


class bfl::CVMatData : public bfl::GenericData
{
public:
    CVMatData() noexcept
    {
        image_ = std::make_shared<cv::Mat>();
    }

    std::shared_ptr<cv::Mat> image_ = nullptr;
};


#endif /* OCVMATDATA_H */
