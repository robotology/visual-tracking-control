#ifndef AUTOCANNY_H
#define AUTOCANNY_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void CumSum(const cv::Mat & histogram, cv::Mat & cumsum)
{
    cumsum = cv::Mat::zeros(histogram.rows, histogram.cols, histogram.type());

    cumsum.at<float>(0) = histogram.at<float>(0);
    for (int i = 1; i < histogram.total(); ++i)
    {
        cumsum.at<float>(i) = cumsum.at<float>(i - 1) + histogram.at<float>(i);
    }
}


void ImgGradient(const cv::Mat & src, cv::Mat & GX, cv::Mat & GY)
{
    const double sigma = 1.4142;
//    const double sigma = 4;

    const int filter_length = 8 * static_cast<int>(ceil(sigma));
    const int n             = (filter_length - 1)/2;
    cv::Mat gauss_filter_transpose;
    cv::Mat gradient_gauss_filter_transpose;

    cv::Mat gauss_filter = cv::getGaussianKernel(filter_length, sigma);
    cv::normalize(gauss_filter, gauss_filter, 1.0, 0.0, cv::NORM_L1);

    cv::Mat gradient_gauss_filter(gauss_filter.size(), gauss_filter.type());
    gradient_gauss_filter.at<double>(0)               = gauss_filter.at<double>(1) - gauss_filter.at<double>(0);
    gradient_gauss_filter.at<double>(filter_length-1) = -gradient_gauss_filter.at<double>(0);
    for (size_t i = 1; i < filter_length/2; ++i)
    {
        gradient_gauss_filter.at<double>(i)                 = (gauss_filter.at<double>(i+1) - gauss_filter.at<double>(i-1))/2.0;
        gradient_gauss_filter.at<double>(filter_length-1-i) = -gradient_gauss_filter.at<double>(i);
    }
    normalize(gradient_gauss_filter, gradient_gauss_filter, 2.0, 0.0, cv::NORM_L1);

    flip(gauss_filter,          gauss_filter,          0);
    flip(gradient_gauss_filter, gradient_gauss_filter, 0);

    transpose(gauss_filter,          gauss_filter_transpose);
    transpose(gradient_gauss_filter, gradient_gauss_filter_transpose);

    filter2D(src, GX, CV_32F, gauss_filter,                    cv::Point(0, n), 0, cv::BORDER_REPLICATE);
    filter2D(GX,  GX, CV_32F, gradient_gauss_filter_transpose, cv::Point(n, 0), 0, cv::BORDER_REPLICATE);

    filter2D(src, GY, CV_32F, gauss_filter_transpose,          cv::Point(n, 0), 0, cv::BORDER_REPLICATE);
    filter2D(GY,  GY, CV_32F, gradient_gauss_filter ,          cv::Point(0, n), 0, cv::BORDER_REPLICATE);
}


void AutoThreshold(const cv::Mat & GX, const cv::Mat & GY, double & high, double & low)
{
    const double   percent_of_pixels_not_edges = 0.8;
    const double   highlow_threshold_ratio     = 0.4;

    cv::Mat        mag_gradient;
    cv::Mat        mag_hist;
    cv::Mat        mag_hist_cumsum;
    const int      channels[]   = {0, 1, 2};
    const int      hist_size[]  = {64};
    const float    range_r[]    = {0.0, 255.01};
    const float    range_g[]    = {0.0, 255.01};
    const float    range_b[]    = {0.0, 255.01};
    const float  * hist_range[] = {range_r, range_g, range_b};
    const double   percentile   = percent_of_pixels_not_edges * GX.cols * GX.rows;


    /* Hypot intensity gradient computation [1] */
    /* [1] https://en.wikipedia.org/wiki/Hypot  */
    cv::Mat GX_abs = cv::abs(GX);
    cv::Mat GY_abs = cv::abs(GY);
    cv::Mat G_t    = cv::min(GX_abs, GY_abs);
    cv::Mat G_x    = cv::max(GX_abs, GY_abs);
    cv::divide(G_t, G_x, G_t);
    cv::pow(G_t, 2.0, G_t);
    cv::sqrt(cv::Scalar(1.0, 1.0, 1.0) + G_t, mag_gradient);
    cv::multiply(G_x, mag_gradient, mag_gradient);

    cv::calcHist(&mag_gradient, 1, channels, cv::Mat(), mag_hist, 1, hist_size, hist_range);

    CumSum(mag_hist, mag_hist_cumsum);
    
    cv::MatIterator_<float> up_bgr;
    up_bgr = std::upper_bound(mag_hist_cumsum.begin<float>(), mag_hist_cumsum.end<float>(), percentile);
    high = (static_cast<double>(up_bgr.lpos()) + 1.0) / static_cast<double>(hist_size[0]) * 255.0;
    low  = highlow_threshold_ratio * high;
}


void AutoCanny(const cv::Mat & src, cv::Mat & dst)
{
    cv::Mat GX;
    cv::Mat GY;
    double  high_threshold;
    double  low_threshold;


    ImgGradient(src, GX, GY);

    AutoThreshold(GX, GY, high_threshold, low_threshold);

    /* OpenCV Canny edge of an image using custom image gradient [1] */
    /* [1] http://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga2a671611e104c093843d7b7fc46d24af */
    GX.convertTo(GX, CV_16SC3);
    GY.convertTo(GY, CV_16SC3);
    cv::Canny(GX, GY, dst, low_threshold, high_threshold, true);
}


void AutoDirCanny(const cv::Mat & src, cv::Mat & dst)
{
    cv::Mat GX;
    cv::Mat GY;
    cv::Mat dir;
    cv::Mat dst_cart;
    double  high_threshold;
    double  low_threshold;


    ImgGradient(src, GX, GY);

    AutoThreshold(GX, GY, high_threshold, low_threshold);

    /* OpenCV Canny directed edge of an image using custom image gradient [1] */
    /* [1] http://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga2a671611e104c093843d7b7fc46d24af */
    if (GX.channels() > 1)
    {
        cv::Mat GX_avg(GX.rows, GX.cols, CV_32FC1);
        cv::Mat GY_avg(GY.rows, GY.cols, CV_32FC1);
        for (size_t i = 0; i < GX.rows; ++i)
        {
            for (size_t j = 0; j < GX.cols; ++j)
            {
                GX_avg.at<float>(i, j) = (GX.at<cv::Vec3f>(i, j)[0] + GX.at<cv::Vec3f>(i, j)[1] + GX.at<cv::Vec3f>(i, j)[2]) / 3.0;
                GY_avg.at<float>(i, j) = (GY.at<cv::Vec3f>(i, j)[0] + GY.at<cv::Vec3f>(i, j)[1] + GY.at<cv::Vec3f>(i, j)[2]) / 3.0;
            }
        }
        cv::phase(GX_avg, GY_avg, dir);
    }
    else cv::phase(GX, GY, dir);

    GX.convertTo(GX, CV_16SC3);
    GY.convertTo(GY, CV_16SC3);
    cv::Canny(GX, GY, dst_cart, low_threshold, high_threshold, true);

    dir.copyTo(dst, dst_cart);
}

#endif /* AUTOCANNY_H */
