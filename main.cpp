#include "yolov5_3403.hpp"
#include "acl/acl.h"

void draw_boxes(cv::Mat& image, const std::vector<YOLOV5::bbox_t>& boxes) {
    for(size_t i = 0; i < boxes.size(); i++) {        
        int idx = boxes[i].obj_id;
        // cv::Rect rect={(int)boxes[i].x,(int)boxes[i].y,(int)boxes[i].w,(int)boxes[i].h};
        cv::rectangle(image, boxes[i].rect, cv::Scalar(128, 0, 255), 2, 8);
        cv::rectangle(image, cv::Point(boxes[i].rect.x, boxes[i].rect.y - 20),
            cv::Point(boxes[i].rect.x, boxes[i].rect.y), cv::Scalar(0, 255, 255), -1);
        putText(image, "fire", cv::Point(boxes[i].rect.x, boxes[i].rect.y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 255, 255), 2, 8);
    }
}

int main(){
    YOLOV5 yolo;
    std::string model_path = "model/yolov5s.om";
    std::string model_config = "model/yolov5s.cfg";
    std::string labels = "models/obj.names";

    yolo.init(model_path.c_str(), "model_config", "labels");
    
    cv::Mat image = cv::imread("output2.jpg");
    std::vector<YOLOV5::bbox_t> result_list;
    
    yolo.detect(&image, result_list);
    
    // for (const auto& bbox : result_list) {
    //     std::cout << "Detected object: " << bbox.label << ", Probability: " << bbox.prob << std::endl;
    // }

    // cv::Mat img = cv::imread("resu.jpg");
  
    draw_boxes(image,result_list);
    cv::imwrite("result.jpg",image);
    
    return 0;
}