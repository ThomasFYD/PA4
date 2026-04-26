// SYSTEM INCLUDES
#include <585/common/types.h>
#include <585/vbow/vbow.h>
#include <585/grad/grad.h>

// C++ PROJECT INCLUDES
#include "ms/ms.h"

#include <vector>
#include <cmath>

namespace ivc
{
namespace student
{

    const ivc::PointCloud preprocess(const ivc::GrayscaleByteImg& img)
    {
        const size_t h = img.rows();
        const size_t w = img.cols();

        ivc::PointCloud pts(h * w, 3);

        size_t idx = 0;

        for(size_t y = 0; y < h; y++)
        {
            for(size_t x = 0; x < w; x++)
            {
                pts(idx, 0) = x;
                pts(idx, 1) = y;
                pts(idx, 2) = img(y, x);

                idx++;
            }
        }

        return pts;
    }  

    const ivc::PointCloud preprocess(const ivc::ColorByteImg& img)
    {
        const size_t h = img.rows();
        const size_t w = img.cols();

        ivc::PointCloud pts(h * w, 5);

        size_t idx = 0;

        for(size_t y = 0; y < h; y++)
        {
            for(size_t x = 0; x < w; x++)
            {
                const unsigned int color = img(y, x);

                pts(idx, 0) = x;
                pts(idx, 1) = y;
                pts(idx, 2) = color & 0xFF;
                pts(idx, 3) = (color >> 8) & 0xFF;
                pts(idx, 4) = (color >> 16) & 0xFF;

                idx++;
            }
        }

        return pts;
    }

    const ivc::PointCloud get_window(const ivc::PointCloud& img,
                                 const ivc::Point& center,
                                 const float_t radius)
    {
        std::vector<Eigen::Index> keep;

        for(Eigen::Index i = 0; i < img.rows(); i++)
        {
            const float_t dist =
                (img.row(i).transpose().eval() - center).norm();

            if(dist <= radius)
            {
                keep.push_back(i);
            }
        }

        ivc::PointCloud window(static_cast<Eigen::Index>(keep.size()), img.cols());

        for(Eigen::Index i = 0; i < static_cast<Eigen::Index>(keep.size()); i++)
        {
            window.row(i) = img.row(keep[i]);
        }

        return window;
    }

    const ivc::Point update_center(const ivc::PointCloud& window,
                               const ivc::Point& center)
    {
        if(window.rows() == 0)
            return center;

        const Eigen::VectorXf distances =
            (window.rowwise() - center.transpose()).rowwise().squaredNorm();

        const Eigen::VectorXf weights = ivc::kernel(distances);

        const float_t denominator = weights.sum();

        if(denominator == 0)
            return center;

        const Eigen::VectorXf normalized = weights / denominator;

        return window.transpose() * normalized;
    }

    const ivc::Point mean_shift(const ivc::PointCloud& all_points,
                            const ivc::Point& center,
                            const float_t radius,
                            const float_t epsilon)
    {
        ivc::Point cur = center;

        while(true)
        {
            ivc::PointCloud window = get_window(all_points, cur, radius);
            ivc::Point next = update_center(window, cur);

            if((next - cur).norm() <= epsilon)
            {
                return next;
            }

            cur = next;
        }
    }

    const ivc::GrayscaleByteImg mean_shift_segment(const ivc::GrayscaleByteImg& img,
                                               const float_t radius,
                                               const float_t epsilon,
                                               const ivc::GrayscalePreprocessFunction& feature_func,
                                               const ivc::GrayscaleLabelFunction& label_func)
    {
        ivc::PointCloud points = feature_func(img);

        std::map<std::tuple<size_t, size_t>, ivc::Point> pixel_to_center;

        for(size_t i = 0; i < points.rows(); i++)
        {
            ivc::Point start = points.row(i).transpose();
            ivc::Point center = mean_shift(points, start, radius, epsilon);

            size_t x = static_cast<size_t>(points(i, 0));
            size_t y = static_cast<size_t>(points(i, 1));

            pixel_to_center[std::make_tuple(y, x)] = center;
        }

        std::vector<ivc::Point> centers;

        for(const auto& item : pixel_to_center)
        {
            ivc::Point c = item.second;

            bool found = false;

            for(const auto& existing : centers)
            {
                if((c - existing).norm() <= epsilon)
                {
                    found = true;
                    break;
                }
            }

            if(!found)
            {
                centers.push_back(c);
            }
        }

        ivc::PointCloud unique_centers(centers.size(), points.cols());

        for(size_t i = 0; i < centers.size(); i++)
        {
            unique_centers.row(i) = centers[i].transpose();
        }

        ivc::GrayscaleByteImg output(img.rows(), img.cols());

        for(size_t y = 0; y < img.rows(); y++)
        {
            for(size_t x = 0; x < img.cols(); x++)
            {
                std::map<std::tuple<size_t, size_t>, ivc::Point> single_pixel;

                auto key = std::make_tuple(y, x);
                single_pixel[key] = pixel_to_center[key];

                output(y, x) = label_func(single_pixel, unique_centers);
            }
        }

        return output;
    }

    const ivc::ColorByteImg mean_shift_segment(const ivc::ColorByteImg& img,
                                           const float_t radius,
                                           const float_t epsilon,
                                           const ivc::ColorPreprocessFunction& feature_func,
                                           const ivc::ColorLabelFunction& label_func)
    {
        ivc::PointCloud points = feature_func(img);

        std::map<std::tuple<size_t, size_t>, ivc::Point> pixel_to_center;

        for(size_t i = 0; i < points.rows(); i++)
        {
            ivc::Point start = points.row(i).transpose();
            ivc::Point center = mean_shift(points, start, radius, epsilon);

            size_t x = static_cast<size_t>(points(i, 0));
            size_t y = static_cast<size_t>(points(i, 1));

            pixel_to_center[std::make_tuple(y, x)] = center;
        }

        std::vector<ivc::Point> centers;

        for(const auto& item : pixel_to_center)
        {
            ivc::Point c = item.second;

            bool found = false;

            for(const auto& existing : centers)
            {
                if((c - existing).norm() <= epsilon)
                {
                    found = true;
                    break;
                }
            }

            if(!found)
            {
                centers.push_back(c);
            }
        }

        ivc::PointCloud unique_centers(centers.size(), points.cols());

        for(size_t i = 0; i < centers.size(); i++)
        {
            unique_centers.row(i) = centers[i].transpose();
        }

        ivc::ColorByteImg output(img.rows(), img.cols());

        const size_t h = img.rows();
        const size_t w = img.cols() / 3;

        for(size_t y = 0; y < h; y++)
        {
            for(size_t x = 0; x < w; x++)
            {
                std::map<std::tuple<size_t, size_t>, ivc::Point> single_pixel;

                auto key = std::make_tuple(y, x);
                single_pixel[key] = pixel_to_center[key];

                ivc::Vec3b color = label_func(single_pixel, unique_centers);

                output(y, 3*x)     = color(0);
                output(y, 3*x + 1) = color(1);
                output(y, 3*x + 2) = color(2);
            }
        }

        return output;
    }

    const ivc::GrayscaleByteImg segment(const ivc::GrayscaleByteImg& img,
                                    const float_t radius,
                                    const float_t epsilon)
    {
        ivc::GrayscalePreprocessFunction feature_func =
            [](const ivc::GrayscaleByteImg& input)
            {
                return preprocess(input);
            };

        ivc::GrayscaleLabelFunction label_func =
            [](const std::map<std::tuple<size_t, size_t>, ivc::Point>& pixel_to_center,
            const ivc::PointCloud& unique_centers)
            {
                auto it = pixel_to_center.begin();
                ivc::Point center = it->second;

                return static_cast<uint8_t>(std::round(center(2)));
            };

        return mean_shift_segment(img, radius, epsilon, feature_func, label_func);
    }

    const ivc::ColorByteImg segment(const ivc::ColorByteImg& img,
                                const float_t radius,
                                const float_t epsilon)
    {
        ivc::ColorPreprocessFunction feature_func =
            [](const ivc::ColorByteImg& input)
            {
                return preprocess(input);
            };

        ivc::ColorLabelFunction label_func =
            [](const std::map<std::tuple<size_t, size_t>, ivc::Point>& pixel_to_center,
            const ivc::PointCloud& unique_centers)
            {
                auto it = pixel_to_center.begin();
                ivc::Point center = it->second;

                ivc::Vec3b color;
                color(0) = static_cast<uint8_t>(std::round(center(2)));
                color(1) = static_cast<uint8_t>(std::round(center(3)));
                color(2) = static_cast<uint8_t>(std::round(center(4)));

                return color;
            };

        return mean_shift_segment(img, radius, epsilon, feature_func, label_func);
    }

    const ivc::PointCloud preprocess_with_grads(const ivc::GrayscaleByteImg& img)
    {
        ivc::GrayscaleFloatImg mags = ivc::get_sobel_3x3_gradient_magnitudes(img);
        ivc::GrayscaleFloatImg angles = ivc::get_sobel_3x3_gradient_angles(img);

        const size_t h = img.rows();
        const size_t w = img.cols();

        ivc::PointCloud pts(h * w, 5);

        size_t idx = 0;

        for(size_t y = 0; y < h; y++)
        {
            for(size_t x = 0; x < w; x++)
            {
                pts(idx, 0) = x;
                pts(idx, 1) = y;
                pts(idx, 2) = img(y, x);
                pts(idx, 3) = mags(y, x);
                pts(idx, 4) = angles(y, x);

                idx++;
            }
        }

        return pts;
    }

    const ivc::PointCloud preprocess_with_grads(const ivc::ColorByteImg& img)
    {
        ivc::ColorFloatImg mags =
            ivc::get_sobel_3x3_gradient_magnitudes(img);

        ivc::ColorFloatImg angles =
            ivc::get_sobel_3x3_gradient_angles(img);

        const size_t h = img.rows();
        const size_t w = img.cols();

        ivc::PointCloud pts(h * w, 11);

        size_t idx = 0;

        for(size_t y = 0; y < h; y++)
        {
            for(size_t x = 0; x < w; x++)
            {
                ivc::Vec3b color = ivc::get_pixel(img, x, y);
                ivc::Vec3f mag   = ivc::get_pixel(mags, x, y);
                ivc::Vec3f ang   = ivc::get_pixel(angles, x, y);

                pts(idx, 0)  = x;
                pts(idx, 1)  = y;

                pts(idx, 2)  = color(0);
                pts(idx, 3)  = color(1);
                pts(idx, 4)  = color(2);

                pts(idx, 5)  = mag(0);
                pts(idx, 6)  = mag(1);
                pts(idx, 7)  = mag(2);

                pts(idx, 8)  = ang(0);
                pts(idx, 9)  = ang(1);
                pts(idx,10)  = ang(2);

                idx++;
            }
        }

        return pts;
    }

    const ivc::GrayscaleByteImg segment_grads(const ivc::GrayscaleByteImg& img,
                                          const float_t radius,
                                          const float_t epsilon)
    {
        ivc::GrayscalePreprocessFunction feature_func =
            [](const ivc::GrayscaleByteImg& input)
            {
                return preprocess_with_grads(input);
            };

        ivc::GrayscaleLabelFunction label_func =
            [](const std::map<std::tuple<size_t, size_t>, ivc::Point>& pixel_to_center,
            const ivc::PointCloud& unique_centers)
            {
                auto it = pixel_to_center.begin();
                ivc::Point center = it->second;

                return static_cast<uint8_t>(std::round(center(2)));
            };

        return mean_shift_segment(img, radius, epsilon, feature_func, label_func);
    }

    const ivc::ColorByteImg segment_grads(const ivc::ColorByteImg& img,
                                      const float_t radius,
                                      const float_t epsilon)
    {
        ivc::ColorPreprocessFunction feature_func =
            [](const ivc::ColorByteImg& input)
            {
                return preprocess_with_grads(input);
            };

        ivc::ColorLabelFunction label_func =
            [](const std::map<std::tuple<size_t, size_t>, ivc::Point>& pixel_to_center,
            const ivc::PointCloud& unique_centers)
            {
                auto it = pixel_to_center.begin();
                ivc::Point center = it->second;

                ivc::Vec3b color;
                color(0) = static_cast<uint8_t>(std::round(center(2)));
                color(1) = static_cast<uint8_t>(std::round(center(3)));
                color(2) = static_cast<uint8_t>(std::round(center(4)));

                return color;
            };

        return mean_shift_segment(img, radius, epsilon, feature_func, label_func);
    }

    const ivc::GrayscaleByteImg mean_shift_segment(const ivc::GrayscaleByteImg& img,
                                                   const float_t radius,
                                                   const float_t epsilon,
                                                   const size_t height_stride,
                                                   const size_t width_stride,
                                                   const ivc::GrayscalePreprocessFunction& feature_func,
                                                   const ivc::GrayscaleLabelFunction& label_func)
    {
        ivc::PointCloud points = feature_func(img);

        std::map<std::tuple<size_t, size_t>, ivc::Point> pixel_to_center;

        for(size_t y = 0; y < img.rows(); y += height_stride)
        {
            for(size_t x = 0; x < img.cols(); x += width_stride)
            {
                size_t idx = y * img.cols() + x;

                ivc::Point start = points.row(idx).transpose();
                ivc::Point center = mean_shift(points, start, radius, epsilon);

                pixel_to_center[std::make_tuple(y, x)] = center;
            }
        }

        std::vector<ivc::Point> centers;

        for(const auto& item : pixel_to_center)
        {
            centers.push_back(item.second);
        }

        ivc::PointCloud unique_centers(centers.size(), points.cols());

        for(size_t i = 0; i < centers.size(); i++)
        {
            unique_centers.row(i) = centers[i].transpose();
        }

        ivc::GrayscaleByteImg output(img.rows(), img.cols());

        for(size_t y = 0; y < img.rows(); y++)
        {
            for(size_t x = 0; x < img.cols(); x++)
            {
                size_t sy = (y / height_stride) * height_stride;
                size_t sx = (x / width_stride) * width_stride;

                std::map<std::tuple<size_t, size_t>, ivc::Point> single_pixel;
                auto key = std::make_tuple(sy, sx);

                single_pixel[key] = pixel_to_center[key];

                output(y, x) = label_func(single_pixel, unique_centers);
            }
        }

        return output;
    }

    const ivc::ColorByteImg mean_shift_segment(const ivc::ColorByteImg& img,
                                               const float_t radius,
                                               const float_t epsilon,
                                               const size_t height_stride,
                                               const size_t width_stride,
                                               const ivc::ColorPreprocessFunction& feature_func,
                                               const ivc::ColorLabelFunction& label_func)
    {
        ivc::PointCloud points = feature_func(img);

        std::map<std::tuple<size_t, size_t>, ivc::Point> pixel_to_center;

        const size_t h = img.rows();
        const size_t w = img.cols() / 3;

        for(size_t y = 0; y < h; y += height_stride)
        {
            for(size_t x = 0; x < w; x += width_stride)
            {
                size_t idx = y * w + x;

                ivc::Point start = points.row(idx).transpose();
                ivc::Point center = mean_shift(points, start, radius, epsilon);

                pixel_to_center[std::make_tuple(y, x)] = center;
            }
        }

        std::vector<ivc::Point> centers;

        for(const auto& item : pixel_to_center)
        {
            centers.push_back(item.second);
        }

        ivc::PointCloud unique_centers(centers.size(), points.cols());

        for(size_t i = 0; i < centers.size(); i++)
        {
            unique_centers.row(i) = centers[i].transpose();
        }

        ivc::ColorByteImg output(img.rows(), img.cols());

        for(size_t y = 0; y < h; y++)
        {
            for(size_t x = 0; x < w; x++)
            {
                size_t sy = (y / height_stride) * height_stride;
                size_t sx = (x / width_stride) * width_stride;

                auto key = std::make_tuple(sy, sx);

                std::map<std::tuple<size_t, size_t>, ivc::Point> single_pixel;
                single_pixel[key] = pixel_to_center[key];

                ivc::Vec3b color = label_func(single_pixel, unique_centers);

                output(y, 3*x)     = color(0);
                output(y, 3*x + 1) = color(1);
                output(y, 3*x + 2) = color(2);
            }
        }

        return output;
    }

} // end of namespace student
} // end of namespace ivc

