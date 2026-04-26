#pragma once
#ifndef _SUB_MS_MS_H_
#define _SUB_MS_MS_H_

// SYSTEM INCLUDES
#include <list>                 // std::list is a linked list
#include <map>                  // std::map
#include <585/common/types.h>
#include <585/ms/ms.h>


// C++ PROJECT INCLUDES


namespace ivc
{
namespace student
{

    // ----------------------------- REQUIRED BY ALL STUDENTS -------------------------------------
    // mean shifting algorithm w/ grayscale/rgb-color features

    // converts each pixel into a 3d point: (x, y, pixel_val) where pixel_val is [0, 256)
    const ivc::PointCloud preprocess(const ivc::GrayscaleByteImg& img);

    // converts each pixel into a 5d point: (x, y, r, g, b)
    const ivc::PointCloud preprocess(const ivc::ColorByteImg& img);

    // extracts all points within a "circular" ball of radius r from the center point
    // if you wanted to speed this up you wouldn't pass the entire image, instead you would build
    // some nearest-neighbors structure (like a ball-tree) to query faster.
    const ivc::PointCloud get_window(const ivc::PointCloud& img,
                                     const ivc::Point& center,
                                     const float_t radius);         // distance should be <=

    // recalculates the center of the window using ivc::kernel in <585/ms/ms.h>
    const ivc::Point update_center(const ivc::PointCloud& window,
                               const ivc::Point& center);

    // uses the mean-shift algorithm to have 'pt' climb the steepest hill it converges within tolerance 'epsilon'
    const ivc::Point mean_shift(const ivc::PointCloud& all_points,
                            const ivc::Point& center,
                            const float_t radius,
                            const float_t epsilon);            // converge when <=

    // performs image segmentation on grayscale images via mean-shifting
    // this algorithm does this by:
    //      1) converting each pixel into its feature representation (leave this up to the 'feature_func')
    //      2) for each point run the mean-shift algorithm to climb the hill (e.g. call 'mean_shift')
    //      3) collect the peak of the hill reached by each pixel
    //      4) merge together hill peaks that are close enough (e.g. duplicate hills, etc.)
    //      5) decide what grayscale value to set each pixel in the image to (leave this up to the 'label_func')
    const ivc::GrayscaleByteImg mean_shift_segment(const ivc::GrayscaleByteImg& img,
                                                   const float_t radius,
                                                   const float_t epsilon,
                                                   const ivc::GrayscalePreprocessFunction& feature_func,
                                                   const ivc::GrayscaleLabelFunction& label_func);


    // performs image segmentation on color images via mean-shifting
    // this algorithm does this with the same 5 steps as the grayscale except the 'label_func' returns a triple of bytes
    const ivc::ColorByteImg mean_shift_segment(const ivc::ColorByteImg& img,
                                               const float_t radius,
                                               const float_t epsilon,
                                               const ivc::ColorPreprocessFunction& feature_func,
                                               const ivc::ColorLabelFunction& label_func);

    // you will have to create a label function and a feature function here:
    //      feature_func: call 'preprocess'
    //      label_func: each unique hill peak has a grayscale value inside it...put that grayscale value in the
    //                  output image at each pixel who arrived at that hill
    const ivc::GrayscaleByteImg segment(const ivc::GrayscaleByteImg& img,
                                        const float_t radius,
                                        const float_t epsilon);

    // you will have to create a label function and a feature function here:
    //      feature_func: call 'preprocess'
    //      label_func: each unique hill peak has a (r,g,b) triple inside it...put that triple in the
    //                  output image at each pixel who arrived at that hill
    const ivc::ColorByteImg segment(const ivc::ColorByteImg& img,
                                    const float_t radius,
                                    const float_t epsilon);

    // ---------------------- REQUIRED BY GRAD / BONUS FOR UNDERGRAD ------------------------------
    // mean shifting algorithm with additional features

    // What happens if you include pixel gradient information as part of the features?
    // I want you to use the sobel gradient magnitudes & angles from <585/grad/grad.h> as well as the
    // grayscale/rgb values
    const ivc::PointCloud preprocess_with_grads(const ivc::GrayscaleByteImg& img);
    const ivc::PointCloud preprocess_with_grads(const ivc::ColorByteImg& img);

    // you will have to create a label function and a feature function here:
    //      feature_func: call 'preprocess_with_grads'
    //      label_func: each unique hill peak has a grayscale value inside it...put that grayscale value in the
    //                  output image at each pixel who arrived at that hill
    const ivc::GrayscaleByteImg segment_grads(const ivc::GrayscaleByteImg& img,
                                              const float_t radius,
                                              const float_t epsilon);

    // you will have to create a label function and a feature function here:
    //      feature_func: call 'preprocess_with_grads'
    //      label_func: each unique hill peak has a (r,g,b) triple inside it...put that triple in the
    //                  output image at each pixel who arrived at that hill
    const ivc::ColorByteImg segment_grads(const ivc::ColorByteImg& img,
                                          const float_t radius,
                                          const float_t epsilon);

    // ---------------------- BONUS FOR GRAD / NO CREDIT FOR UNDERGRAD -----------------------------
    // mean shift with striding (e.g. not getting clusters for every pixel)

    // performs image segmentation on grayscale images via mean-shifting
    // this algorithm does this by:
    //      1) converting each pixel into its feature representation (leave this up to the 'feature_func')
    //      2) for each point run the mean-shift algorithm to climb the hill (e.g. call 'mean_shift')
    //      3) collect the peak of the hill reached by each pixel
    //      4) merge together hill peaks that are close enough (e.g. duplicate hills, etc.)
    //      5) decide what grayscale value to set each pixel in the image to (leave this up to the 'label_func')
    //              one note here is that you will be missing some pixels (e.g. you did not cluster them)
    //              I want you to linearly interpolate their values from the two clusters of the two nearest
    //              pixels (in feature space) to a missing pixel
    const ivc::GrayscaleByteImg mean_shift_segment(const ivc::GrayscaleByteImg& img,
                                                   const float_t radius,
                                                   const float_t epsilon,
                                                   const size_t height_stride,  // how many pixels to skip
                                                   const size_t width_stride,   // how many pixels to skip
                                                   const ivc::GrayscalePreprocessFunction& feature_func,
                                                   const ivc::GrayscaleLabelFunction& label_func);


    // performs image segmentation on color images via mean-shifting
    // this algorithm does this with the same 5 steps as the grayscale except the 'label_func' returns a triple of bytes
    const ivc::ColorByteImg mean_shift_segment(const ivc::ColorByteImg& img,
                                               const float_t radius,
                                               const float_t epsilon,
                                               const size_t height_stride,      // how many pixels to skip
                                               const size_t width_stride,       // how many pixels to skip
                                               const ivc::ColorPreprocessFunction& feature_func,
                                               const ivc::ColorLabelFunction& label_func);
    
    

} // end of namespace student
} // end of namespace ivc


#endif // end of _SUB_MS_MS_H_  

