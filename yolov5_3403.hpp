#ifndef YOLOV5_3403_HPP

#define YOLOV5_3403_HPP
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

class YOLOV5 : public  object_detect_alg{
    public:        
        YOLOV5()=default;
        int init(const char* model_path, const char* model_config, const char* labels) override;
        int detect(const void* picture,std::vector<bbox_t> &result_list) override;
        // std::vector<cv::Rect> detect(const cv::Mat& frame, float confThreshold = 0.5, float nmsThreshold = 0.4);
        ~YOLOV5();

    private:
        int post_process(void *output_data, std::vector<bbox_t> &result_list, float factor_x, float factor_y, 
                         int input_width, int input_height);
        std::vector<bbox_t> post_process_v62(void* output_, std::vector<bbox_t> &result_list,
                                     float x_factor, float y_factor,int input_width, int input_height);
        std::vector<bbox_t> post_process_cu(std::vector<cv::Mat>& det_outputs, 
                               float x_factor = 1.0, float y_factor = 1.0);

};

extern "C" object_detect_alg* create_object_detect_alg() {
    return new YOLOV5();
}

#endif // YOLOV5_3403_HPP