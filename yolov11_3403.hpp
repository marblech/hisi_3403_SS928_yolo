#ifndef YOLOV11_3403_HPP

#define YOLOV11_3403_HPP
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


#ifdef WIN32
#include <windows.h>
#else
#include <fstream>
#include <string>
#endif

#include "object_detect_alg.hpp"

class YOLOV11 : public  object_detect_alg{
    public:        
        YOLOV11()=default;
        int init(const char* model_path, const char* model_config, const char* labels) override;
        int detect(const void* picture,std::vector<bbox_t> &result_list) override;       
        // std::vector<cv::Rect> detect(const cv::Mat& frame, float confThreshold = 0.5, float nmsThreshold = 0.4);
        ~YOLOV11();

    private:
        int post_process(void *output_data, std::vector<bbox_t> &result_list, float factor_x, float factor_y, 
                         int input_width, int input_height);
        void yolov11n_post_process(const cv::Mat& output, float conf_threshold, float nms_threshold, 
                          std::vector<bbox_t>& detections, float x_factor, int y_factor, int img_width, int img_height);

};

extern "C" object_detect_alg* create_object_detect_alg() {
    return new YOLOV11();
}

#endif // YOLOV11_3403_HPP