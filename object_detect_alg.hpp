#ifndef OBJECT_DETECT_ALG_HPP
#define OBJECT_DETECT_ALG_HPP

#include <opencv2/opencv.hpp>
#include <iostream>

class object_detect_alg{
    public:
        struct bbox_t{
            cv::Rect rect;
            std::string label;
            double prob;
            int obj_id;
        };
        virtual ~object_detect_alg()=default;
        virtual int init(const char* model_path, const char* model_config, const char* labels)=0;
        virtual int detect(const void* picture,std::vector<bbox_t> &result_list) =0;
};

extern "C" object_detect_alg* create_object_detect_alg();

#endif // OBJECT_DETECT_ALG_HPP