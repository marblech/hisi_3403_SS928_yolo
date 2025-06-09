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
#include "yolov11_3403.hpp"
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

std::string make_acl_json(std::string path){
    std::string acl_json = R"({})";
    // get date time timestamp   
    int timestamp = static_cast<int>(time(nullptr));
    std::ofstream out(path+"acl"+to_string(timestamp)+".json");
    if (!out) {
        std::cerr << "Failed to create acl.json" << std::endl;
        return "";
    }
    out << acl_json;
    out.close();
    return "acl"+to_string(timestamp)+".json";
}

int YOLOV11::init(const char *model_path, const char *model_config, const char *labels)
{
    std::string parent_path = std::experimental::filesystem::path(model_path).parent_path().string();
    std::cout<<"->model path is "<<parent_path<<std::endl;
    std::string aclConfigPath = make_acl_json(parent_path);
    std::cout<<"acl path: "<<aclConfigPath<<std::endl;
    if (aclConfigPath.empty()) {
        ERROR_LOG("Failed to create acl.json");
        return FAILED;
    }
    // std::string aclConfigPath = parent_path + "/acl.json";
    // std::string aclConfigPath = aclConfigPath;
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

void YOLOV11::yolov11n_post_process(const cv::Mat& output, float conf_threshold, float nms_threshold, 
                          std::vector<bbox_t>& detections, float x_factor, int y_factor, int img_width, int img_height) {
    // // 验证输出维度
    // if (output.dims != 3 || output.size[0] != 1 || output.size[1] != 84 || output.size[2] != 8400) {
    //     std::cerr << "输出维度不符合预期, 实际维度: [" << output.size[0] << ", " 
    //               << output.size[1] << ", " << output.size[2] << "]" << std::endl;
    //     return;
    // }
    
    int num_classes = 80;
    int num_boxes = output.size[2]; // 8400
    const float* data = (const float*)output.data;
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    // 遍历所有候选框
    for (int i = 0; i < num_boxes; i++) {
        // 找出最高置信度的类别
        float max_score = 0.0f;
        int max_class_id = -1;
        
        for (int c = 0; c < num_classes; c++) {
            float score = data[(c + 4) * num_boxes + i];
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }
        
        // 只处理高于阈值的检测
        if (max_score >= conf_threshold) {
            // 获取边界框坐标(YOLOv11n输出的是基于640x640的像素坐标，而非归一化坐标)
            float x_center = data[0 * num_boxes + i];
            float y_center = data[1 * num_boxes + i];
            float width = data[2 * num_boxes + i];
            float height = data[3 * num_boxes + i];                     
            
            // 计算边界框在原始图像上的位置
            int left = static_cast<int>((x_center - width/2) * x_factor);
            int top = static_cast<int>((y_center - height/2) * y_factor);
            int box_width = static_cast<int>(width * x_factor);
            int box_height = static_cast<int>(height * y_factor);
            
            // 确保边界框在图像范围内
            left = std::max(0, left);
            top = std::max(0, top);
            box_width = std::max(1, std::min(box_width, img_width - left));
            box_height = std::max(1, std::min(box_height, img_height - top));
            
            // 添加到候选列表
            boxes.push_back(cv::Rect(left, top, box_width, box_height));
            confidences.push_back(max_score);
            class_ids.push_back(max_class_id);
        }
    }
    
    // std::cout << "检测到 " << boxes.size() << " 个候选目标" << std::endl;
    
    // 应用非极大值抑制
    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    }
    
    // 生成最终检测结果
    detections.clear();
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        bbox_t det;
        det.rect = boxes[idx];
        det.prob = confidences[idx];
        det.obj_id = class_ids[idx];
              
        
        detections.push_back(det);
    }
    
    // std::cout << "NMS后检测到 " << detections.size() << " 个目标" << std::endl;
    
    // // 打印检测结果
    // for (const auto& det : detections) {
    //     std::cout << "box X: " << det.box.x << " Y: " << det.box.y 
    //               << " W: " << det.box.width << " H: " << det.box.height 
    //               << " score: " << det.score << " class_id: " << det.class_id << std::endl;
    // }
}


int YOLOV11::post_process(void *output_data, std::vector<bbox_t> &result_list, 
                         float x_factor, float y_factor,int input_width, int input_height) {
    /***************************************************/
    /******************post process*********************/
    /***************************************************/
    aclmdlDataset* output_ = static_cast<aclmdlDataset*>(output_data);

    size_t output_num = aclmdlGetDatasetNumBuffers(output_);
    if(output_num == 1){
        // Extract data from aclmdlDataset* and convert it to std::vector<cv::Mat>
        cv::Mat det_outputs;
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
            aclDataBuffer* buffer = aclmdlGetDatasetBuffer(output_, i);
            void* data = aclGetDataBufferAddr(buffer);
            const float* feature_map1 = reinterpret_cast<float*>(data); 

            // size_t size = aclGetDataBufferSizeV2(buffer);
            int size[] = {1, 84, 8400}; // Assuming the output is [1, 25200, 6]

            // Assuming the output data is in a format compatible with cv::Mat
            cv::Mat mat(3, size, CV_32F, (float*)feature_map1);
            det_outputs = mat.clone(); // Clone to ensure data is managed by cv::Mat
        }

        // Call post_process_cu with the converted data
        yolov11n_post_process(det_outputs, 0.25,0.45,result_list, x_factor, y_factor,input_width, input_height);
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


int YOLOV11::detect(const void *picture, std::vector<bbox_t> &result_list)
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

YOLOV11::~YOLOV11()
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
