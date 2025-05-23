#include <iostream>
#include <map>
#include <sstream>
#include <algorithm>
#include <functional>
#include <sys/stat.h>
#include <fstream>
#include <cstring>
#include <sys/time.h>

using namespace std;

#include "acl/acl.h"
#include "opencv2/opencv.hpp"
#include "yolov5_3403.hpp"
#include <experimental/filesystem>

// sigmoid激活函数
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

#define INFO_LOG(fmt, ...)  fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...)  fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR]  " fmt "\n", ##__VA_ARGS__)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

bool g_isDevice = false;

aclmdlDesc* modelDesc_ = nullptr;
aclrtStream stream_;
aclrtContext context_; 

uint32_t modelId_=0;
int32_t deviceId_=0; 

// 定义检测框结构体
struct Box {
    float x, y, w, h,x1,y1,x2,y2;
    float score;
    float confidence;
    float class_score;
    int class_id;
};

// 定义NMS输出结构
struct DetectionResult {
    float x, y, w, h;    // 中心点坐标和宽高
    float prob;          // 最终概率
    int obj_id;          // 类别ID
};

static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

int YOLOV5::init(const char *model_path, const char *model_config, const char *labels)
{
    std::string parent_path = std::experimental::filesystem::path(model_path).parent_path().string();
    std::cout<<"->model path is "<<parent_path<<std::endl;
    std::string aclConfigPath = parent_path + "/acl.json";
    /***************************************************/
    /*****************Init ACL**************************/
    /***************************************************/
    cout<<"->ACL INIT "<<endl;
    aclError ret = aclInit(aclConfigPath.c_str());
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl init failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    /***************************************************/
    /*****************apply resource********************/
    /***************************************************/
    // set device only one device
   
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl set device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    cout<<"->set device "<<deviceId_<<endl;
    // create context (set current)
    cout<<"->create context"<<endl; 
    
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl create context failed, deviceId = %d, errorCode = %d",
            deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    // create stream
    cout<<"->create stream"<<endl;  
    
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl create stream failed, deviceId = %d, errorCode = %d",
            deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    // get run mode
    cout<<"->get run mode"<<endl; 
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl get run mode failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    g_isDevice=(runMode==ACL_DEVICE) ;
    
    /***************************************************/
    /********load model and get infos of model**********/
    /***************************************************/
   
    ret = aclmdlLoadFromFile(model_path,&modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
            model_path, static_cast<int32_t>(ret));
        return FAILED;
    }
    cout<<"->load mode "<<"\""<<model_path<<"\""<<" model id is "<<modelId_<<endl; 
    //get model describe
    cout<<"->create model describe"<<endl; 
    
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }
    cout<<"->get model describe"<<endl; 
    ret = aclmdlGetDesc(modelDesc_, modelId_); 
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get model description failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    deviceId_=0;
    return 0;
}

/// <summary>
/// preprocess image
/// </summary>
/// <param name="image"></param>
/// <param name="target_size"></param>
/// <returns></returns>
cv::Mat preprocess_image(const cv::Mat& frame,float &x_factor, float &y_factor) {
    // Format frame
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    // Fix bug, boxes consistency!
    x_factor = image.cols / static_cast<float>(640);
    y_factor = image.rows / static_cast<float>(640);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);    

    // size_t tpixels = model_session.input_model_height * model_session.input_model_width * 3;
    // std::array<int64_t, 4> input_shape_info{ 1, 3, model_session.input_model_height, model_session.input_model_width };
    // return { blob, tpixels, input_shape_info, x_factor, y_factor };
    return blob;
}

aclmdlDataset* prepare_input_data(const cv::Mat& resized_frame) {
    aclmdlDataset *input_ = nullptr;
    void* inputDataBuffer = nullptr;
    size_t modelInputSize = 0;

    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return nullptr;
    }           

    // aclmdlDataset *input_;
    // void * inputDataBuffer = nullptr;
    modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    cout<<"->get input size "<<modelInputSize<<endl;

    cout<<"->apply input mem "<<endl;
    aclError aclRet = aclrtMalloc(&inputDataBuffer, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %zu, errorCode is %d",
            modelInputSize, static_cast<int32_t>(aclRet));
        return nullptr;
    }

    cout<<"->copy data to device "<<endl;
    aclError ret = aclrtMemcpy(inputDataBuffer, modelInputSize, resized_frame.data, modelInputSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("copy data to device failed, errorCode is %d", static_cast<int32_t>(ret));
        (void)aclrtFree(inputDataBuffer);
        inputDataBuffer = nullptr;
        return nullptr;
    }
    cout<<"->copy data to device success "<<endl;
   

    cout<<"->create input dataset "<<endl;
    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return nullptr;
    }
    cout<<"->create databuffer"<<endl; 
    aclDataBuffer *inputData = aclCreateDataBuffer(inputDataBuffer, modelInputSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return nullptr;
    }

    cout<<"->get input data buffer"<<endl;
    size_t inputNum = aclmdlGetDatasetNumBuffers(input_);
    cout<<"->get input dataset num "<<inputNum<<endl;
    if (inputNum != 0) {
        ERROR_LOG("dataset buffer num is not 0, create input failed");
        (void)aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return nullptr;
    }
    cout<<"->get input data buffer success "<<endl;

    cout<<"->add data to datasetbuffer"<<endl;
    ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        (void)aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return nullptr;
    }
    INFO_LOG("create model input success");
    return input_;   
}

aclmdlDataset* prepare_output_data_buffer(){
    aclmdlDataset *output_ = nullptr;
    /***************************************************/
    /************prepare output data buffer*************/
    /***************************************************/
    cout<<"->create dataset"<<endl;
    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return nullptr;
    }
    size_t output_num= aclmdlGetNumOutputs(modelDesc_); 
    cout<<"->get num of output "<<output_num<<endl;
    for (size_t i = 0; i < output_num; ++i) {
        size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        cout<<"-> output size["<<i<<"] :"<<modelOutputSize<<endl;
        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed, errorCode is %d",
                modelOutputSize, static_cast<int32_t>(ret));
            return nullptr;
        }
        //apply output buffer
        cout<<"->apply output buffer"<<endl;
        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            (void)aclrtFree(outputBuffer);
            return nullptr;
        }
        cout<<"->AddDatasetBuffer"<<endl;
        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed, errorCode is %d",
                static_cast<int32_t>(ret));
            (void)aclrtFree(outputBuffer);
            (void)aclDestroyDataBuffer(outputData);
            return nullptr;
        }

        cout<<"-> get original output test"<<endl;
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
        cout<<"-> getDataBufferSizeV2["<<i<<"] :"<<len<<endl;    
        float *outData = NULL;  
        outData = reinterpret_cast<float*>(data); 
        for(int num=0;num<10;num++){
            cout<<outData[num]<<endl;
        }
    }
    cout<<"->create model output success "<<endl;
    return output_;
}

int model_inference(aclmdlDataset* input_, aclmdlDataset* output_){
    /***************************************************/
    /******************inference************************/
    /***************************************************/
    // for(int i=0;i<100000;i++){
    cout<<"input data num is "<<aclmdlGetDatasetNumBuffers(input_)<<endl;
    cout<<"output data num is "<<aclmdlGetDatasetNumBuffers(output_)<<endl;
    cout<<"->begin inference "<<"model id is "<<modelId_<<endl;
     
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return FAILED;
    } 
        
   
    return SUCCESS;
}

void processFeatureMap(const float* feature_map, std::vector<Box>& detections,
                      int grid_h, int grid_w, int num_anchors, int num_outputs,
                      float conf_threshold, float x_factor, float y_factor,int input_width, int input_height) {
    
    // YOLOv5 v6.2 anchors和strides
    std::vector<std::vector<int>> anchors = {
        {10, 13, 16, 30, 33, 23},      // 80x80
        {30, 61, 62, 45, 59, 119},     // 40x40
        {116, 90, 156, 198, 373, 326}  // 20x20
    };
    
    int strides[] = {8, 16, 32}; // 对应80x80, 40x40, 20x20
    
    // 确定当前特征图对应的索引
    int feature_idx = 0;
    if (grid_h == 80) feature_idx = 0;
    else if (grid_h == 40) feature_idx = 1;
    else if (grid_h == 20) feature_idx = 2;
    else {
        std::cout << "Unsupported feature map size: " << grid_h << "x" << grid_w << std::endl;
        return;
    }
    
    int stride = strides[feature_idx];
    int num_classes = num_outputs - 5; // 85 - 5 = 80类
    
    // 调试信息
    std::cout << "处理特征图: " << grid_h << "x" << grid_w 
              << " stride=" << stride 
              << " anchor_group=" << feature_idx << std::endl;
    
    int debug_count = 0; // 用于限制调试输出
    
    // 遍历每个anchor、每行、每列
    for (int a = 0; a < num_anchors; ++a) {
        for (int i = 0; i < grid_h; ++i) {
            for (int j = 0; j < grid_w; ++j) {
                // 计算当前网格点在特征图中的索引
                // 对应 record 变量的计算
                float* record = const_cast<float*>(feature_map) + 
                              a * grid_h * grid_w * num_outputs +
                              i * grid_w * num_outputs + 
                              j * num_outputs;
                
                // 指向类别分数的指针
                float* cls_ptr = record + 5;
                
                // 遍历所有类别
                for (int cls = 0; cls < num_classes; ++cls) {
                    // 计算类别置信度 = sigmoid(类别分数) * sigmoid(objectness)
                    float score = sigmoid(cls_ptr[cls]) * sigmoid(record[4]);
                    
                    // 只处理高于阈值的检测结果
                    if (score > conf_threshold) {
                        // 解码边界框坐标
                        float cx = (sigmoid(record[0]) * 2.0f - 0.5f + j) * stride;
                        float cy = (sigmoid(record[1]) * 2.0f - 0.5f + i) * stride;
                        float w = pow(sigmoid(record[2]) * 2.0f, 2) * anchors[feature_idx][2 * a];
                        float h = pow(sigmoid(record[3]) * 2.0f, 2) * anchors[feature_idx][2 * a + 1];
                        
                        // 创建Box对象并保存检测结果
                        Box box;
                        box.x = std::max(0.0f,(cx - w /2.0f)*x_factor);                     // 中心x坐标
                        box.y = std::max(0.0f,(cy - h /2.0f)*y_factor);                     // 中心y坐标
                        box.w = w * x_factor;                      // 宽度
                        box.h = h * y_factor;                      // 高度
                        box.confidence = sigmoid(record[4]); // objectness
                        box.class_id = cls;            // 类别ID
                        box.class_score = sigmoid(cls_ptr[cls]); // 类别置信度
                        
                        // 添加到检测结果列表
                        detections.push_back(box);
                        
                        // 输出部分检测结果用于调试
                        if (debug_count < 4) {
                            std::cout << "找到目标: 类别=" << cls 
                                      << " 置信度=" << score 
                                      << " 边界框=[" << cx << ", " << cy << ", " << w << ", " << h << "]" 
                                      << std::endl;
                            debug_count++;
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "特征图 " << grid_h << "x" << grid_w << " 共检测到 " 
              << detections.size() << " 个目标" << std::endl;
}

// NMS 非极大值抑制
std::vector<DetectionResult> nms_boxes(const std::vector<Box>& boxes, 
                                      float nms_threshold, 
                                      float conf_threshold) {
    std::vector<DetectionResult> result;
    
    // 准备OpenCV NMS所需的数据结构
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> cv_boxes;
    
    // 遍历所有检测框
    for (const auto& box : boxes) {
        // 计算最终得分
        float score = box.confidence * box.class_score;
        
        // 应用置信度阈值
        if (score > conf_threshold) {
            // 转换为左上角和宽高表示
            float x1 = box.x; 
            float y1 = box.y;
            float w = box.w;
            float h = box.h;
            
            // // 确保在有效范围内
            // x1 = std::max(0.0f, x1);
            // y1 = std::max(0.0f, y1);
            
            // 添加到列表
            classIds.push_back(box.class_id);
            confidences.push_back(score);
            cv_boxes.push_back(cv::Rect(x1, y1, w, h));
        }
    }
    
    // 应用NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(cv_boxes, confidences, conf_threshold, nms_threshold, indices);
    
    std::cout << "NMS前有效框数量: " << cv_boxes.size() << ", NMS后保留: " << indices.size() << std::endl;
    
    // 构建输出结果
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        
        DetectionResult det;
        det.x = cv_boxes[idx].x; // 转回中心点表示
        det.y = cv_boxes[idx].y;
        det.w = cv_boxes[idx].width;
        det.h = cv_boxes[idx].height;
        det.prob = confidences[idx];
        det.obj_id = classIds[idx];
        
        result.push_back(det);
    }
    
    return result;
}

std::vector<YOLOV5::bbox_t> YOLOV5::post_process_v62(void* output_data, std::vector<bbox_t> &result_list,
                                     float x_factor, float y_factor,int input_width, int input_height) {
    
    aclmdlDataset* output_ = static_cast<aclmdlDataset*>(output_data);
    // 这里添加处理YOLOv5结果集的代码
    // 获取三个输出特征图
    const float* feature_map1 = nullptr;
    const float* feature_map2 = nullptr;
    const float* feature_map3 = nullptr;   
    
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* buffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(buffer);       

        if (i == 0) {
            feature_map1 = reinterpret_cast<float*>(data);  // 80x80            
        } else if (i == 1) {
            feature_map2 = reinterpret_cast<float*>(data);  // 40x40
        } else if (i == 2) {
            feature_map3 = reinterpret_cast<float*>(data);  // 20x20
        }       
    }
    
    // 处理参数
    const float conf_threshold = 0.25f;  // 置信度阈值
    const float nms_threshold = 0.45f;   // NMS阈值
    // const int input_width = 640;        // 输入图像宽度
    // const int input_height = 640;       // 输入图像高度

   // 确保创建一个全局向量存储所有检测结果
    std::vector<Box> all_detections;

    // 处理三个特征图
    std::vector<Box> detections_80x80;
    processFeatureMap(feature_map1, detections_80x80, 80, 80, 3, 85, conf_threshold, x_factor, y_factor, 
                      input_width, input_height);

    std::vector<Box> detections_40x40;
    processFeatureMap(feature_map2, detections_40x40, 40, 40, 3, 85, conf_threshold, x_factor, y_factor, 
                      input_width, input_height);

    std::vector<Box> detections_20x20;
    processFeatureMap(feature_map3, detections_20x20, 20, 20, 3, 85, conf_threshold, x_factor, y_factor, 
                      input_width, input_height);

    // 合并所有检测结果
    all_detections.insert(all_detections.end(), detections_80x80.begin(), detections_80x80.end());
    all_detections.insert(all_detections.end(), detections_40x40.begin(), detections_40x40.end());
    all_detections.insert(all_detections.end(), detections_20x20.begin(), detections_20x20.end());

    std::cout << "所有特征图总共检测到 " << all_detections.size() << " 个目标" << std::endl;

    // 执行NMS
    std::vector<DetectionResult> final_detections = nms_boxes(all_detections, nms_threshold, conf_threshold);

    // 输出最终结果
    std::cout << "最终检测到 " << final_detections.size() << " 个目标" << std::endl;

    for(auto item:final_detections){
        bbox_t bbox;
        bbox.rect = cv::Rect(item.x, item.y, item.w, item.h);
        bbox.prob = item.prob;
        bbox.obj_id = item.obj_id;
        result_list.push_back(bbox);
    }
    return result_list;
}

/// @brief [1,25200,6] 的输出的后处理
/// @param output_data 模型输出数据
/// @param batch_size 批次大小
/// @param num_anchors 锚点数量
/// @param num_classes 类别数量
/// @param conf_threshold 物体置信度阈值
/// @param nms_threshold 非极大值抑制阈值
/// @param x_factor 宽度缩放因子
/// @param y_factor 高度缩放因子
/// @return 
std::vector<YOLOV5::bbox_t> YOLOV5::post_process_cu(std::vector<cv::Mat>& det_outputs, 
                               float x_factor, float y_factor) {
    std::vector<bbox_t> result;         
    const float conf_threshold = 0.25f;  // 置信度阈值
    const float nms_threshold = 0.45f;   // NMS阈值

    // 用于存储NMS前的所有框
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    // 处理所有检测输出
    // 假设det_outputs[0]包含所有检测结果，格式为[1, 25200, 6]
    cv::Mat detection = det_outputs[0];
    
    // 检查输出维度
    CV_Assert(detection.dims == 3);
    
    // 获取检测结果的总数（通常是num_anchors，如25200）
    int num_anchors = detection.size[1];
    // 获取每个检测结果的维度（通常是5+num_classes，这里是6）
    int item_size = detection.size[2];
    
    // 获取数据指针
    float* data = (float*)detection.data;
    
    // 处理所有预测框
    for (int i = 0; i < num_anchors; ++i) {
        // 计算当前检测结果在data中的偏移
        float* detection_ptr = data + i * item_size;
        
        float confidence = detection_ptr[4];  // objectness 分数
        
        // 过滤低置信度的框
        if (confidence < conf_threshold) continue;
        
        // 获取类别ID和分数
        int class_id = static_cast<int>(detection_ptr[5]);
        float class_score = confidence;  // YOLOv5已经将objectness和类别分数相乘
        
        // 创建边界框，应用缩放因子
        float x = detection_ptr[0] * x_factor;  // 中心 x
        float y = detection_ptr[1] * y_factor;  // 中心 y
        float w = detection_ptr[2] * x_factor;  // 宽度
        float h = detection_ptr[3] * y_factor;  // 高度
        
        // 转换为左上角坐标
        int left = static_cast<int>(x - w / 2);
        int top = static_cast<int>(y - h / 2);
        
        // 添加到临时列表
        boxes.push_back(cv::Rect(left, top, static_cast<int>(w), static_cast<int>(h)));
        confidences.push_back(class_score);
        class_ids.push_back(class_id);
    }
    
    // 使用OpenCV内置的NMS函数
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    
    // 将NMS后的框添加到结果列表
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        
        bbox_t bbox;
        bbox.rect = boxes[idx];
        bbox.prob = confidences[idx];
        bbox.obj_id = class_ids[idx];
        
        result.push_back(bbox);
    }
    
    return result;
}

int YOLOV5::post_process(void *output_data, std::vector<bbox_t> &result_list, 
                         float x_factor, float y_factor,int input_width, int input_height) {
    /***************************************************/
    /******************post process*********************/
    /***************************************************/
    aclmdlDataset* output_ = static_cast<aclmdlDataset*>(output_data);

    size_t output_num = aclmdlGetDatasetNumBuffers(output_);
    if(output_num == 3){
        result_list = post_process_v62(output_, result_list, x_factor, y_factor, input_width, input_height);
    }else if(output_num == 1){
        // Extract data from aclmdlDataset* and convert it to std::vector<cv::Mat>
        std::vector<cv::Mat> det_outputs;
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
            aclDataBuffer* buffer = aclmdlGetDatasetBuffer(output_, i);
            void* data = aclGetDataBufferAddr(buffer);
            const float* feature_map1 = reinterpret_cast<float*>(data); 

            // size_t size = aclGetDataBufferSizeV2(buffer);
            int size[] = {1, 25200, 6}; // Assuming the output is [1, 25200, 6]

            // Assuming the output data is in a format compatible with cv::Mat
            cv::Mat mat(3, size, CV_32F, (float*)feature_map1);
            det_outputs.push_back(mat.clone()); // Clone to ensure data is managed by cv::Mat
        }

        // Call post_process_cu with the converted data
        result_list = post_process_cu(det_outputs, x_factor, y_factor);
    }else{    
        ERROR_LOG("output num is %zu, post process failed", output_num);
        return FAILED;
    }
    return SUCCESS;
}

void destory_data(aclmdlDataset* output_, aclmdlDataset* input_) {
    /***************************************************/
    /*********************destroy model output*********/
    /***************************************************/
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
    INFO_LOG("destroy model output success");

    /***************************************************/
    /*******************destroy model input*************/
    /***************************************************/
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    (void)aclmdlDestroyDataset(input_);
    input_ = nullptr;
    INFO_LOG("destroy model input success");
}

int YOLOV5::detect(const void *picture, std::vector<bbox_t> &result_list)
{
    cv::Mat image = *(cv::Mat*)picture;
    int64_t start_time=0;
    int64_t end_time=0;
    int64_t eclipes_time=0; 
    start_time = getCurrentTimeUs();  

    float x_factor = 0.0f;
    float y_factor = 0.0f;
    cv::Mat resized_frame=preprocess_image(image,x_factor,y_factor);

    aclmdlDataset *input_ = prepare_input_data(resized_frame);
    if (input_ == nullptr) {
        ERROR_LOG("prepare input data failed");
        return FAILED;
    }
    aclmdlDataset *output_ = prepare_output_data_buffer();
    if (output_ == nullptr) {
        ERROR_LOG("prepare output data buffer failed");
        return FAILED;
    }
    aclError ret = model_inference(input_, output_);
    if (ret != SUCCESS) {
        ERROR_LOG("model inference failed");
        return FAILED;
    }

    std::cout<<"x_factor="<<x_factor<<std::endl;
    std::cout<<"y_factor="<<y_factor<<std::endl;

    post_process(output_, result_list,x_factor,y_factor,image.cols,image.rows);

    destory_data(output_, input_);
    
    end_time = getCurrentTimeUs();
    eclipes_time=end_time-start_time;
    printf("------------------use time %.2f ms\n", eclipes_time/1000.f);
    return 0;
}

YOLOV5::~YOLOV5()
{
    /***************************************************/
    /******uninstall model and release resource*********/
    /***************************************************/
    cout<<"->unload model id is "<<modelId_<<endl;
    aclError ret = aclmdlUnload(modelId_);
     if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return;
    } 
    INFO_LOG("unload model success, modelId is %u", modelId_);
    // releasemodelDesc_
    if (modelDesc_ != nullptr) {
        aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
    INFO_LOG("release modelDesc_ success, modelId is %u", modelId_);
    //release resorce
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy stream failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        stream_ = nullptr;
    }
    cout<<"->destroy stream done"<<endl;

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy context failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        context_ = nullptr;
    }
    cout<<"->destroy context done "<<endl;
    
    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("reset device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
    }
    cout<<"->reset device id is "<<deviceId_<<endl;

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("  failed, errorCode = %d", static_cast<int32_t>(ret));
    }
    INFO_LOG("end to finalize acl");
}
