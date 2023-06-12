#include "NvInfer.h"
#include <iostream>
#include "NvOnnxParser.h"
#include <fstream>
#include "cuda_runtime_api.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace nvonnxparser;
using namespace nvinfer1;
using namespace std;
using namespace cv;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}




struct  Object{
  Rect_<float> rect;
  int label;
  float prob;

};


void qsort_descent_inplace(vector<Object>&faceobjects,int left, int right){
    int i = left;
    int j = right;
    float p = faceobjects[(left+right)/2].prob;
    while (i<=j){
        while (faceobjects[i].prob>p ){
            i++;
        }
        while (faceobjects[j].prob<p){
            j--;
        }
        if(i<=j){
            swap(faceobjects[i],faceobjects[j]);
            i++;
            j--;

        }

    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }

    }
}



void  qsort_descent_inplace(vector<Object>&faceobjects){
    if(faceobjects.empty()){
        return ;
    }
    qsort_descent_inplace(faceobjects,0,faceobjects.size()-1);

}

float intersection_area(Object & a,Object&b) {
    Rect2f inter = a.rect&b.rect;
    return inter.area();

}


void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
         Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
          Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}



float data[3*640*640];
float prob[25200*85];
float out422[3*80*80*85];
float out481[3*40*40*85];
float out540[3*20*20*85];


static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    imwrite("/newhome/wangjd/tensortest/out.jpg",image);

}


int main(int argc,char ** argv){

    if(*argv[1]== 's') {
        IBuilder *builder = createInferBuilder(logger);

        uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        INetworkDefinition *network = builder->createNetworkV2(flag);

        IParser *parser = createParser(*network, logger);

        parser->parseFromFile("/newhome/wangjd/model/yolov3.onnx",
                              static_cast<int32_t>(ILogger::Severity::kWARNING));
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            std::cout << parser->getError(i)->desc() << std::endl;
        }


        IBuilderConfig *config = builder->createBuilderConfig();
        //config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,1U<<30);
    config->setMaxWorkspaceSize(1 << 30); 
         IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);

        ofstream p("/newhome/wangjd/engine/yolov3.engine", ios::binary);
        if (!p.good()) {
            cout << "open failed" << endl;
        }
        p.write(reinterpret_cast<const char *>(serializedModel->data()), serializedModel->size());

       delete parser;
       delete network;
       delete config;
       delete builder;
       delete serializedModel;

    }else if(*argv[1]=='d'){
        size_t size{0};
        char * trtModelStream{nullptr};
        ifstream file("/newhome/wangjd/engine/yolov3.engine", ios::binary);
        if(file.good()){
            file.seekg(0,ios::end);
            size = file.tellg();
            file.seekg(0,ios::beg);
            trtModelStream = new char[size];
            file.read(trtModelStream,size);
            file.close();

        }


        IRuntime * runtime = createInferRuntime(logger);
        ICudaEngine * engine = runtime->deserializeCudaEngine(trtModelStream,size);
        IExecutionContext *context = engine->createExecutionContext();
        delete[] trtModelStream;

        int BATCH_SIZE=1;
        int INPUT_H=640;
        int INPUT_W=640;

        const char * images = "images";
        const char * output = "output";
        const char * output422 = "422";
        const char * output481 = "481";
        const char * output540 = "540";


        int32_t images_index = engine->getBindingIndex(images);
        int32_t output_index = engine->getBindingIndex(output);
        int32_t output422_index = engine->getBindingIndex(output422);
        int32_t output481_index = engine->getBindingIndex(output481);
        int32_t output540_index = engine->getBindingIndex(output540);



        cout<<    images_index<<" "
                <<output_index<<" "
                <<output422_index<<" "
                <<output481_index<<" "
                <<output540_index<<" "
                <<endl;

            cout<<engine->getNbBindings()<<endl;

        void * buffers[5];
        cudaMalloc(&buffers[images_index],BATCH_SIZE*3*INPUT_W*INPUT_H*sizeof (float));

        cudaMalloc(&buffers[output422_index],BATCH_SIZE*3*80*80*85*sizeof (float));
        cudaMalloc(&buffers[output481_index],BATCH_SIZE*3*40*40*85*sizeof (float));
        cudaMalloc(&buffers[output540_index],BATCH_SIZE*3*20*20*85*sizeof (float));
        cudaMalloc(&buffers[output_index],BATCH_SIZE*25200*85*sizeof (float));



        Mat img = imread("/newhome/wangjd/tensortest/123.jpg");

        Mat pr_img = preprocess_img(img,INPUT_H,INPUT_W);


        for(int i = 0 ; i < INPUT_W*INPUT_H;++i){
            data[i] = pr_img.at<Vec3b>(i)[2]/255.0;
            data[i+INPUT_W*INPUT_H] = pr_img.at<Vec3b>(i)[1]/255.0;
            data[i+2*INPUT_W*INPUT_H]=pr_img.at<Vec3b>(i)[0]/255.0;

        }



        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMemcpyAsync(buffers[images_index],data,BATCH_SIZE*3*INPUT_W*INPUT_H* sizeof(float ),cudaMemcpyHostToDevice,stream);
        context->enqueueV2(buffers,stream, nullptr);


        cudaMemcpyAsync(prob,buffers[output_index],1*25200*85* sizeof(float ),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(out422,buffers[output422_index],1*3*80*80*85* sizeof(float ),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(out481,buffers[output481_index],1*3*40*40*85* sizeof(float ),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(out540,buffers[output540_index],1*3*20*20*85* sizeof(float ),cudaMemcpyDeviceToHost,stream);
        
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        cudaFree(buffers[images_index]);
        cudaFree(buffers[output_index]);
        cudaFree(buffers[output422_index]);
        cudaFree(buffers[output481_index]);
        cudaFree(buffers[output540_index]);
        delete context;
        delete runtime;
        delete engine;

        vector<Object> objects;
        for(int i = 0 ; i<25200;++i){
            if(prob[85*i+4]<=0.5) continue;

            int l ,r,t,b;
            float r_w = INPUT_W/(img.cols*1.0);
            float r_h = INPUT_H/(img.rows*1.0);


            float x = prob[85*i+0];
            float y = prob[85*i+1];
            float w = prob[85*i+2];
            float h = prob[85*i+3];
            float score = prob[85*i+4];
            if(r_h>r_w){
                l = x-w/2.0;
                r = x+w/2.0;
                t = y-h/2.0-(INPUT_H-r_w*img.rows)/2;
                b = y+h/2.0-(INPUT_H-r_w*img.rows)/2;
                l=l/r_w;
                r=r/r_w;
                t=t/r_w;
                b=b/r_w;
            }else{

                l = x-w/2.0-(INPUT_W-r_h*img.cols)/2;
                r = x+w/2.0-(INPUT_W-r_h*img.cols)/2;
                t = y-h/2.0;
                b = y+h/2.0;
                l=l/r_h;
                r=r/r_h;
                t=t/r_h;
                b=b/r_h;
            }


            int label_index = max_element(prob+85*i+5,prob+85*(i+1))-(prob+85*i+5);



            Object obj;
            obj.rect.x = l;
            obj.rect.y = t;
            obj.rect.width=r-l;
            obj.rect.height=b-t;
            obj.label = label_index;
            obj.prob = score;

            objects.push_back(obj);


        }

        qsort_descent_inplace(objects);

        vector<int> picked;
        nms_sorted_bboxes(objects,picked,0.45);



        int count = picked.size();
        vector<Object>obj_out(count);
        for(int i = 0 ; i <count ; ++i){
            obj_out[i] = objects[picked[i]];

        }

        draw_objects(img,obj_out);


   }


}
