// Yolov8 CPP inference implementation for seed counting 
#include<onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <chrono>
#include <unordered_map>
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

#define pi acos(-1)
float modelScoreThreshold=0.4;
float modelNMSThreshold=0.4;
std::vector<std::string> labels1 = {"rice"};
std::vector<std::string> labels2 = {"corn","sorghum","soybean","wheat"};
std::vector<std::string> labels3 = {"millet"};


int findMostFrequentElement(std::vector<int>& classIds) {
    std::unordered_map<int, int> frequencyMap;
    
    // 計算每個元素的出現次數
    for (int id : classIds) {
        frequencyMap[id]++;
    }
    
    // 找出出現次數最多的元素
    int mostFrequentElement = classIds[0];
    int maxFrequency = 0;
    
    for (const auto& pair : frequencyMap) {
        if (pair.second > maxFrequency) {
            maxFrequency = pair.second;
            mostFrequentElement = pair.first;
        }
    }
    
    return mostFrequentElement;
}
// each picture only has one category
void keepMostFrequentElement(std::vector<int>& classIds, std::vector<cv::Rect>& boxes, std::vector<float>& confidences) {
    int mostFrequent = findMostFrequentElement(classIds);
    
    // 遍歷vector並保留出現次數最多的元素，同時刪除其他對應的元素
    auto itClassIds = classIds.begin();
    auto itBoxes = boxes.begin();
    auto itConfidences = confidences.begin();
    
    while (itClassIds != classIds.end()) {
        if (*itClassIds != mostFrequent) {
            itClassIds = classIds.erase(itClassIds);
            itBoxes = boxes.erase(itBoxes);
            itConfidences = confidences.erase(itConfidences);
        } else {
            ++itClassIds;
            ++itBoxes;
            ++itConfidences;
        }
    }
}







// oriented bounding boxes
// for long-shape seeds

std::pair<cv::Mat, int> main_detectprocess_with_yolov8(std::string& onnx_path_name, std::vector<std::string>& labels,Mat &frame,bool draw_point)
{
	
	Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	Ort::Session session_(env, onnx_path_name.c_str(), session_options);


	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;

	size_t numInputNodes = session_.GetInputCount();
	size_t numOutputNodes = session_.GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	input_node_names.reserve(numInputNodes);

	// get the input information
	int input_w = 0;
	int input_h = 0;
	for (int i = 0; i < numInputNodes; i++) {
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_w = input_dims[3];
		input_h = input_dims[2];
		std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
	}

	// get the output information
	int output_h = 0;
	int output_w = 0;
	Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();
	output_h = output_dims[1]; // 84
	output_w = output_dims[2]; // 8400
	std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
	for (int i = 0; i < numOutputNodes; i++) {
		auto out_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(out_name.get());
	}
	std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

	// format frame
	
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));

	// fix bug, boxes consistence!
	float x_factor = image.cols / static_cast<float>(input_w);
	float y_factor = image.rows / static_cast<float>(input_h);

	cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
	size_t tpixels = input_h * input_w * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

	// set input data and inference
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	// output data
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 8400x84

	// post-process
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;

	

	for (int i = 0; i < det_output.rows; i++) {
		
		cv::Mat classes_scores = det_output.row(i).colRange(4, 4+labels.size());
		
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		// confidence between 0 and 1
		if (score > modelScoreThreshold)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}
	keepMostFrequentElement( classIds,  boxes,  confidences);

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, indexes);
	
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
        if (draw_point)
        {
            cv::Rect box=boxes[index];
            float cx,cy,area;
            area=box.width*box.height;
            cx=box.x+box.width/2;
            cy=box.y+box.height/2;
            int raius;
            if (area<600)
            {
                raius=3;
            }
            else if (area>1500)
            {
                raius=8;
            }
            else
            {
                raius=5;
            }
            cv::circle(frame,cv::Point2f(cx,cy),raius,cv::Scalar(0, 0, 255),-1);
        }
        else
        {
            cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
		    cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
		    putText(frame, labels[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        }
		
		
	}
	
	session_options.release();
	session_.release();

	return {frame, indexes.size()};

}





cv::Mat formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

//define a struct to save some information
typedef struct {
	cv::RotatedRect box;
	float score;
	int Classindex;
}RotatedBOX;

std::vector<RotatedBOX> main_detectprocess_with_yolov8_obb(std::string& onnx_path_name, std::vector<std::string>& labels,Mat &frame)
{
	
	// format frame

    cv::Mat image=formatToSquare(frame);

	Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	
	Ort::Session session_(env, onnx_path_name.c_str(), session_options);

	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;

	size_t numInputNodes = session_.GetInputCount();
	size_t numOutputNodes = session_.GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	input_node_names.reserve(numInputNodes);

	// get the input information
	int input_w = 0;
	int input_h = 0;
	for (int i = 0; i < numInputNodes; i++) {
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_w = input_dims[3];
		input_h = input_dims[2];
		//std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
	}

	// get the output information
	int output_h = 0;
	int output_w = 0;
	Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();
	output_h = output_dims[1]; // 84
	output_w = output_dims[2]; // 8400

	std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
	for (int i = 0; i < numOutputNodes; i++) {
		auto out_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(out_name.get());
	}
	//std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

	
	float x_factor = image.cols / static_cast<float>(input_w);
	float y_factor = image.rows / static_cast<float>(input_h);

	cv::Mat blob;
    cv::dnn::blobFromImage(image,blob, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
	size_t tpixels = input_h * input_w * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

	// set input data and inference
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	// output data
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 8400x84

	// post-process
	std::vector<cv::RotatedRect> boxes;
	std::vector<float> confidences;
    std::vector<RotatedBOX>BOXES;
    std::vector<int>class_list;

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, 4+labels.size());
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
        RotatedBOX BOX;

		// confidence between 0 and 1
		if (score > modelScoreThreshold)
		{
			float cx = det_output.at<float>(i, 0)*x_factor;
			float cy = det_output.at<float>(i, 1)* y_factor;
			float ow = det_output.at<float>(i, 2)*x_factor;
			float oh = det_output.at<float>(i, 3)* y_factor;
            float angle=det_output.at<float>(i,4+labels.size());
            //angle in [-pi/4,3/4 pi) --》 [-pi/2,pi/2)
            if (angle>=0.5*pi && angle <= 0.75*pi)
            {
                angle=angle-pi;
            }
           
            BOX.Classindex=classIdPoint.x;
            class_list.push_back(classIdPoint.x);
            BOX.score=score; 
            cv::RotatedRect box=cv::RotatedRect(cv::Point2f(cx,cy),cv::Size2f(ow,oh),angle*180/pi);
            BOX.box=box;
            boxes.push_back(box);
            BOXES.push_back(BOX); 
			confidences.push_back(score);
		}
	}

    // NMS accoding to each class
    std::vector<RotatedBOX> Remain_boxes;
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes,confidences,modelScoreThreshold,modelNMSThreshold, nms_result);
   


	for (int i=0;i< nms_result.size();i++)
	{
		int index=nms_result[i];
		RotatedBOX Box_=BOXES[index];
		Remain_boxes.push_back(Box_);

	}

 
	session_options.release();
	session_.release();
    return Remain_boxes;

}

std::pair<cv::Mat, int> Draw(cv::Mat& image,std::vector<RotatedBOX>& detect_boxes,bool draw_point)
{
    for (unsigned long i = 0; i < detect_boxes.size(); ++i)
    {
       
        RotatedBOX Box_=detect_boxes[i];
        cv::RotatedRect box_=Box_.box;
        cv::Point2f points[4];
        box_.points(points);
        int class_id=Box_.Classindex;

        if (draw_point)
        {
            float x1=points[0].x;
            float y1=points[0].y;
            float x2=points[1].x;
            float y2=points[1].y;
            float x3=points[2].x;
            float y3=points[2].y;
            float x4=points[3].x;
            float y4=points[3].y;
            float w=pow(pow(x1-x2,2)+pow(y1-y2,2),0.5);
            float h=pow(pow(x3-x2,2)+pow(y3-y2,2),0.5);
            float area=w*h;
            cout<<area<<endl;
            float cx,cy;
            cx=(x1+x2+x3+x4)/4;
            cy=(y1+y2+y3+y4)/4;
            int raius;
            if (area<600)
            {
                raius=3;
            }
            else if (area>1500)
            {
                raius=8;
            }
            else
            {
                raius=5;
            }
            cv::circle(image,cv::Point2f(cx,cy),raius,cv::Scalar(0, 0, 255),-1);

        }
        else
        {
            for (int i = 0; i < 4; ++i) 
            {
                cv::line(image, points[i], points[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);  
            }
            cv::putText(image, labels1[class_id], points[0], cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 1);
        }

        
    }
	int count_number=(int) detect_boxes.size();
	
	return {image, count_number};

}


//delete a specific folder
void delete_folder(fs::path FolderPath)
{
    if (fs::exists(FolderPath)&& fs::is_directory(FolderPath))
    {
        fs::remove_all(FolderPath);
        std::cout << "Folder deleted successfully."<<std::endl;
    }
}



//CUT the big image into smaller patches
void slice_img(const cv::Mat& img, const std::string& out_folder = "scratch_file",
               int sliceHeight = 640, int sliceWidth = 640,
               float overlap = 0.1, bool skip_highly_overlapped_tiles = false) {
    // 創建輸出文件夾
    if (!fs::exists(out_folder)) {
        std::cout << "創建輸出文件夾: " << out_folder << std::endl;
        fs::create_directory(out_folder);
    }

    int dx = static_cast<int>((1.0 - overlap) * sliceWidth);
    int dy = static_cast<int>((1.0 - overlap) * sliceHeight);

    int n_ims = 0;

    // 帶重疊率的滑窗
    for (int y0 = 0; y0 < img.rows; y0 += dy) {
        for (int x0 = 0; x0 < img.cols; x0 += dx) {
            n_ims++;

            int y = y0;
            int x = x0;

            // 遇到邊界，向內裁切
            if (y0 + sliceHeight > img.rows) {
                if (skip_highly_overlapped_tiles) {
                    if ((y0 + sliceHeight - img.rows) > (0.6 * sliceHeight)) {
                        continue;
                    } else {
                        y = img.rows - sliceHeight;
                    }
                } else {
                    y = img.rows - sliceHeight;
                }
            }

            if (x0 + sliceWidth > img.cols) {
                if (skip_highly_overlapped_tiles) {
                    if ((x0 + sliceWidth - img.cols) > (0.6 * sliceWidth)) {
                        continue;
                    } else {
                        x = img.cols - sliceWidth;
                    }
                } else {
                    x = img.cols - sliceWidth;
                }
            }

            // 要裁切的部分在整幅圖上的位置
            int xmin = x;
            int xmax = x + sliceWidth;
            int ymin = y;
            int ymax = y + sliceHeight;

            cv::Mat window_c = img(cv::Range(ymin, ymax), cv::Range(xmin, xmax));

            std::string out_final = out_folder + "/capture__" + std::to_string(y) + "_" + std::to_string(x) + "_"
                                    + std::to_string(sliceHeight) + "_" + std::to_string(sliceWidth) + "_0_"
                                    + std::to_string(img.cols) + "_" + std::to_string(img.rows) + ".jpg";

            cv::imwrite(out_final, window_c);
        }
    }
}





//images model prediction and save in txt file 
void predict_with_yolov8(std::string& onnx_path_name, std::string& img_path, std::string& save_path) {
   Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	Ort::Session session_(env, onnx_path_name.c_str(), session_options);

	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;

	size_t numInputNodes = session_.GetInputCount();
	size_t numOutputNodes = session_.GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	input_node_names.reserve(numInputNodes);

	// get the input information
	int input_w = 0;
	int input_h = 0;
	for (int i = 0; i < numInputNodes; i++) {
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_w = input_dims[3];
		input_h = input_dims[2];
		std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
	}

	// get the output information
	int output_h = 0;
	int output_w = 0;
	Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();
	output_h = output_dims[1]; // 84
	output_w = output_dims[2]; // 8400
	std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
	for (int i = 0; i < numOutputNodes; i++) {
		auto out_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(out_name.get());
	}
	std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;


    auto t0 =std::chrono::high_resolution_clock::now();
    for (const auto& entry : fs::directory_iterator(img_path)) {
        if (entry.path().extension() == ".jpg") {
            std::string img_name = entry.path().filename().string();
            std::string img_name_ = img_path + "/" + img_name;
            std::string img_title = img_name.substr(0, img_name.find_last_of('.'));
            cv::Mat img = cv::imread(img_name_);
           

            std::string txt_name = save_path + "/" + img_title + ".txt";
            std::ofstream txt_file(txt_name);
            
            int w = img.cols, h = img.rows;
            int _max = std::max(h, w);
            cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
            img.copyTo(image(cv::Rect(0, 0, w, h)));

            float x_factor = image.cols / static_cast<float>(input_w);
            float y_factor = image.rows / static_cast<float>(input_h);
            

            cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
            size_t tpixels = input_h * input_w * 3;
            std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

            auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), blob.total(), input_shape_info.data(), input_shape_info.size());
            
            const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	        const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
            std::vector<Ort::Value> ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor, 1, outNames.data(), outNames.size());

            const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
            cv::Mat det_output(output_h, output_w, CV_32F, (float*)pdata);
            cv::Mat det_output_t = det_output.t();

            std::vector<cv::Rect> boxes;
            std::vector<int> classIds;
            std::vector<float> confidences;
            
            for (int i = 0; i < det_output_t.rows; ++i) {
                cv::Mat classes_scores = det_output_t.row(i).colRange(4, 4+labels3.size());
                
                cv::Point classIdPoint;
                double score;
                minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

                if (score > 0.25) {
                    float cx = det_output_t.at<float>(i, 0);
                    float cy = det_output_t.at<float>(i, 1);
                    float ow = det_output_t.at<float>(i, 2);
                    float oh = det_output_t.at<float>(i, 3);
                    int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
                    int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
                    int width = static_cast<int>(ow * x_factor);
                    int height = static_cast<int>(oh * y_factor);
                    float area = width *height ;
                    float as_ratio = std::max(width /height, height /width);
                    
                    if (area >= 200 && as_ratio <= 1.5) {
                        boxes.emplace_back(x, y, width, height);
                        classIds.push_back(classIdPoint.x);
                        confidences.push_back(score);
                    }
                }
            }
            

            std::vector<int> indexes;
            cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, indexes);
            //保存歸一化座標
            for (int idx : indexes) {
                const cv::Rect& box = boxes[idx];
                txt_file << classIds[idx] << " " <<(float)(box.x+box.width/2)/img.cols << " " << (float)(box.y+box.height/2)/img.rows << " " << (float)box.width/img.cols << " " << (float)box.height/img.rows << " " << confidences[idx] << "\n";
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "detection+NMS time: " << elapsed.count() << "seconds" << std::endl;
}


//convert the box coordinate: cxcywh to x1y1x2y2
std::vector<int> convert_reverse(const std::pair<int, int>& size, const std::vector<float>& box) {
    float x = box[0];
    float y = box[1];
    float w = box[2];
    float h = box[3];

    float dw = 1.0f / size.first;
    float dh = 1.0f / size.second;

    float w0 = w / dw;
    float h0 = h / dh;
    float xmid = x / dw;
    float ymid = y / dh;

    int x0 = static_cast<int>(xmid - w0 / 2.0f);
    int x1 = static_cast<int>(xmid + w0 / 2.0f);
    int y0 = static_cast<int>(ymid - h0 / 2.0f);
    int y1 = static_cast<int>(ymid + h0 / 2.0f);

    return {x0, x1, y0, y1};
}
//get the global coordinate from local coordinate
std::pair<std::vector<int>, std::vector<std::vector<int>>> get_global_coords(
    const std::map<std::string, float>& row,
    float edge_buffer_test = 0,
    float max_edge_aspect_ratio = 2.5,
    float test_box_rescale_frac = 1.0,
    int max_bbox_size_pix = 100) 
{
    // 獲取參數
    float xmin0 = row.at("Xmin"), xmax0 = row.at("Xmax");
    float ymin0 = row.at("Ymin"), ymax0 = row.at("Ymax");
    float upper = row.at("Upper"), left = row.at("Left");
    float sliceHeight = row.at("Height"), sliceWidth = row.at("Width");
    float vis_w = row.at("Im_Width"), vis_h = row.at("Im_Height");
    float pad = row.at("Pad");
    float dx = xmax0 - xmin0;
    float dy = ymax0 - ymin0;

    // 框過大，視作檢測錯誤
    if ((dx > max_bbox_size_pix) || (dy > max_bbox_size_pix)) {
        return { {}, {} };
    }

    // 邊緣區域的處理
    if (edge_buffer_test > 0) {
        if ((xmin0 < edge_buffer_test) || (xmax0 > (sliceWidth - edge_buffer_test)) ||
            (ymin0 < edge_buffer_test) || (ymax0 > (sliceHeight - edge_buffer_test))) {
            if ((dx / dy > max_edge_aspect_ratio) || (dy / dx > max_edge_aspect_ratio)) {
                return { {}, {} };
            }
        }
    }

    if ((dx / dy > max_edge_aspect_ratio) || (dy / dx > max_edge_aspect_ratio)) {
        return { {}, {} };
    }

    // 轉換到全局座標，pad指的是滑窗時候填充的像素個數
    int xmin = std::max(0, static_cast<int>(std::round(xmin0+left)) - static_cast<int>(pad));
    int xmax = std::min(static_cast<int>(vis_w - 1), static_cast<int>(std::round(xmax0+left))  - static_cast<int>(pad));
    int ymin = std::max(0, static_cast<int>(std::round(ymin0+upper)) - static_cast<int>(pad));
    int ymax = std::min(static_cast<int>(vis_h - 1), static_cast<int>(std::round(ymax0+upper)) - static_cast<int>(pad));


    // 框是否缩放
    if (test_box_rescale_frac != 1.0) {
        float dl = test_box_rescale_frac;
        float xmid = (xmin + xmax) / 2.0f;
        float ymid = (ymin + ymax) / 2.0f;
        dx = dl * (xmax - xmin) / 2;
        dy = dl * (ymax - ymin) / 2;
        xmin = static_cast<int>(xmid - dx);
        xmax = static_cast<int>(xmid + dx);
        ymin = static_cast<int>(ymid - dy);
        ymax = static_cast<int>(ymid + dy);
    }

    // 設置邊界和座標
    std::vector<int> bounds = { xmin, xmax, ymin, ymax };
    std::vector<std::vector<int>> coords = { {xmin, ymin}, {xmax, ymin}, {xmax, ymax}, {xmin, ymax} };

    //檢測錯誤
    if (*std::min_element(bounds.begin(), bounds.end()) < 0) {
        std::cerr << " 预测框出現負值: ";
        for (const auto& b : bounds) std::cerr << b << " ";
        std::cerr << std::endl;
        std::cerr << " 出错: ";
        for (const auto& [key, value] : row) std::cerr << key << ": " << value << " ";
        std::cerr << std::endl;
        return { {}, {} };
    }
    if ((xmax > vis_w) || (ymax > vis_h)) {
        std::cerr << " 预测框大於原圖尺寸: ";
        for (const auto& b : bounds) std::cerr << b << " ";
        std::cerr << std::endl;
        std::cerr << " 出错: ";
        for (const auto& [key, value] : row) std::cerr << key << ": " << value << " ";
        std::cerr << std::endl;
        return { {}, {} };
    }

    return { bounds, coords };
}

// 定義結構體表示每一行數據
struct DataFrameRow {
    std::string im_name;
    float prob;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    int cat_int;
    std::string category;
    std::string im_name_root;
    std::string image_path;
    std::string Image_Root;
    std::string Slice_XY;
    float Upper;
    float Left;
    float Height;
    float Width;
    float Pad;
    float Im_Width;
    float Im_Height;
    float Xmin_Glob;
    float Xmax_Glob;
    float Ymin_Glob;
    float Ymax_Glob;
};



// implement the get_global_coords functions
std::vector<DataFrameRow> augment_data(
    const std::vector<DataFrameRow>& df,
    const std::string& slice_sep = "__",
    int max_box_size = 300,
    int edge_buffer_test = 0,
    float max_edge_aspect_ratio = 5.0,
    float test_box_rescale_frac = 1.0) {

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<DataFrameRow> df_new = df;

    for (auto& row : df_new) {
        std::string im_name = row.im_name;
        //__ 前面部分，沒什麼用
        std::string root_tmp = im_name.substr(0, im_name.find(slice_sep));
        //__後面部分，形如 '576_576_640_640_0_3840_2880'
        std::string coo_tmp = im_name.substr(im_name.find(slice_sep) + slice_sep.length());

        row.Slice_XY = coo_tmp;

        if (root_tmp.find('.') == std::string::npos) {
            row.Image_Root = root_tmp + "." + im_name;
        } else {
            row.Image_Root = root_tmp;
        }
        //轉化成string流
        std::istringstream ss(coo_tmp);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, '_')) {
            tokens.push_back(token);
        }
        //
        row.Upper = std::stof(tokens[0]);
        row.Left = std::stof(tokens[1]);
        row.Height = std::stof(tokens[2]);
        row.Width = std::stof(tokens[3]);
        row.Pad = std::stof(tokens[4].substr(0, tokens[4].find('.')));
        row.Im_Width = std::stof(tokens[5].substr(0, tokens[5].find('.')));
        row.Im_Height = std::stof(tokens[6].substr(0, tokens[6].find('.')));
    }

    std::cout << "圖名訊息已經導入" << std::endl;

    std::vector<int> bad_idxs;
    for (size_t index = 0; index < df_new.size(); ++index) {
        //以引用的方式
        auto& row = df_new[index];

        std::map<std::string, float> row_map = {
            {"Xmin", row.xmin}, {"Xmax", row.xmax},
            {"Ymin", row.ymin}, {"Ymax", row.ymax},
            {"Upper", row.Upper}, {"Left", row.Left},
            {"Height", row.Height}, {"Width", row.Width},
            {"Im_Width", row.Im_Width}, {"Im_Height", row.Im_Height},
            {"Pad", row.Pad}
        };

        auto result = get_global_coords(
            row_map,
            edge_buffer_test,
            max_edge_aspect_ratio,
            test_box_rescale_frac,
            max_box_size
        );

        if (result.first.empty() && result.second.empty()) {
            bad_idxs.push_back(index);
            row.Xmin_Glob = 0;
            row.Xmax_Glob = 0;
            row.Ymin_Glob = 0;
            row.Ymax_Glob = 0;
        } else {
            row.Xmin_Glob = result.first[0];
            row.Xmax_Glob = result.first[1];
            row.Ymin_Glob = result.first[2];
            row.Ymax_Glob = result.first[3];
        }
    }

    if (!bad_idxs.empty()) {
        std::cout << "不满足邊界、縱橫比等要求的预测框數量: " << bad_idxs.size() << std::endl;
        for (const auto& idx : bad_idxs) {
            df_new.erase(df_new.begin() + idx);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "get global coordinate time: " << elapsed.count() << "seconds" << std::endl;
    std::cout << "剩余: " << df_new.size() << std::endl;

    return df_new;
}


std::pair<cv::Mat, int> draw_for_millet(std::vector<DataFrameRow>& filter_data_list, cv::Mat& image,bool draw_point) {
    int count_number = 0;
    
    for (const auto& df_row : filter_data_list) 
    {
        if (draw_point)
        {
            float w,h,cx,cy,area;
            w=df_row.Xmax_Glob-df_row.Xmin_Glob;
            h=df_row.Ymax_Glob-df_row.Ymin_Glob;
            area=w*h;
            cout<<area<<endl;
            cx=(df_row.Xmax_Glob+df_row.Xmin_Glob)/2;
            cy=(df_row.Ymax_Glob+df_row.Ymin_Glob)/2;
            int raius;
            if (area<600)
            {
                raius=3;
            }
            else
            {
                raius=5;
            }
            cv::circle(image,cv::Point2f(cx,cy),raius,cv::Scalar(0, 0, 255),-1);
        }
        else
        {
            cv::rectangle(image, cv::Point((int) df_row.Xmin_Glob, (int) df_row.Ymin_Glob), cv::Point((int) df_row.Xmax_Glob,(int)df_row.Ymax_Glob), cv::Scalar(0, 255, 0), 2);
        }
        count_number++;
    

    }
    
    return {image, count_number};
}



//綜合函數：執行裁切大圖、預測、小圖預測結果轉爲大圖預測結果、拼接等步驟
std::pair<cv::Mat, int> execute(cv::Mat img, const std::string& labels_dir = "scratch_file",
            int subgraph_size = 640,
            const std::vector<std::string>& classes = {"millet"},
            float detect_thresh = 0.3,
            float max_edge_aspect_ratio = 5.0,
            int edge_buffer_test = 0,
            float global_NMS_thresh=0.25) 
{

    std::vector<DataFrameRow> data_list;

    // 遍歷labels_dir下的txt文件
    for (const auto& entry : std::filesystem::directory_iterator(labels_dir)) {
        //得到txt子文件名稱
        if (entry.path().extension() == ".txt") {
            std::string txt_path = entry.path().string();
            //得到類似 'capture__0_0_640_640_0_3840_2880'
            std::string prefix_name = entry.path().stem().string();
            //txt文件中所有內容讀取進inflie流中
            std::ifstream infile(txt_path);
            std::string line;
            std::vector<DataFrameRow> out_data;
            //inflie流逐行讀取進line中
            while (std::getline(infile, line)) {
                std::istringstream iss(line);
                int cat_int;
                float x_frac, y_frac, w_frac, h_frac, prob;
                iss >> cat_int >> x_frac >> y_frac >> w_frac >> h_frac >> prob;

                std::string cat_str = classes.empty() ? "" : classes[cat_int];
                std::vector<float> box = {x_frac, y_frac, w_frac, h_frac};
                auto pix_box = convert_reverse({640,640},box);
                float x0 = pix_box[0], x1 = pix_box[1], y0 = pix_box[2], y1 = pix_box[3];

                DataFrameRow out_data_list={prefix_name, prob, x0, y0, x1, y1, cat_int, cat_str, "", "","" ,"",0, 0, 0, 0,0,0,0,0,0,0,0};
                out_data.push_back(out_data_list);
                
            }

            data_list.insert(data_list.end(), out_data.begin(), out_data.end());
        }
    }
   
    if (data_list.empty()) {
        std::cerr << "没有檢測結果" << std::endl;
        return {img, 0};
    }

    // 提取圖像名稱
    for (auto& row : data_list) {
        row.im_name_root = row.im_name.substr(0, row.im_name.find('_'));
        row.image_path = "capture.jpg";
    }
    //獲取全局座標
    auto df_new = augment_data(data_list);
    
    std::vector<Rect> global_coordinate;
    std::vector<float> scores;
    for (const auto& df_new_row:df_new)
    {
        float prob_score=df_new_row.prob;
        int width=df_new_row.Xmax_Glob-df_new_row.Xmin_Glob;
        int height=df_new_row.Ymax_Glob-df_new_row.Ymin_Glob;
        int x_center=df_new_row.Xmin_Glob+ width/2;
        int y_center=df_new_row.Ymin_Glob+ height/2;
        scores.push_back(prob_score);
        Rect box;
        box.x=x_center;
        box.y=y_center;
        box.width=width;
        box.height=height;
        global_coordinate.push_back(box);

    }
    std::vector<int> keep_indices;
    cv::dnn::NMSBoxes(global_coordinate,scores,detect_thresh,global_NMS_thresh,keep_indices);
    std::cout<<"NMS後的數量:"<<keep_indices.size()<<std::endl; 
    std::vector<DataFrameRow>filter_data_list;
    for(int& idx:keep_indices )
    {
        filter_data_list.push_back(df_new[idx]);
    }

    auto [image, count_number]=draw_for_millet(filter_data_list,img,true);
    
    //delete the temporary folder
    delete_folder(labels_dir);
    return {image, count_number};


}



std::pair<cv::Mat, int> detect_img_and_count_yolov8(std::string &img_name,  int &task,bool& vis,std::string img_savepath="scratch_file", std::string predict_savepath="scratch_file")
{
	Mat input_image = cv::imread(img_name);
	int ih = input_image.rows;
	int iw = input_image.cols;
	int64 start = cv::getTickCount();
    std::string onnx_path_name;
	
	try {
		if (task==0)
		{
            onnx_path_name="/home/kingargroo/seed/ablation1/best.onnx";
			std::vector<RotatedBOX> detect_boxes=main_detectprocess_with_yolov8_obb(onnx_path_name, labels1, input_image);
			float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
			putText(input_image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
			auto [image, count_number]=Draw( input_image, detect_boxes,true);
            if (vis)
            {
                cv::imwrite("test.jpg",image);
            }
            
			cout<<count_number<<endl;
            if (vis)
            {
                cv::imwrite("test.jpg",image);
            }
			return {image, count_number};
		}
		else if (task==1)
		{
            onnx_path_name="/home/kingargroo/seed/vision6.onnx";
			auto [image, count_number]=main_detectprocess_with_yolov8(onnx_path_name, labels2,input_image,true);
            if (vis)
            {
                cv::imwrite("test.jpg",image);
            }
			return {image, count_number};
            
		}
		else if (task==2)
		{
            onnx_path_name="/home/kingargroo/seed/ablation1/v8n.onnx";
			slice_img(input_image);
			predict_with_yolov8(onnx_path_name,img_savepath,predict_savepath);
			auto [image, count_number]=execute(input_image,predict_savepath);
			std::cout<<"finsh"<<std::endl;
            if (vis)
            {
                cv::imwrite("test.jpg",image);
            }
			return {image, count_number};

		}
		else
		{
			throw task;
		}
	}
	catch(int)
	{ 
		cout<<"task value error, task value must be one of 0,1 or 2"<<endl;
		exit(1);
	}
	

}




int main()
{
   
	std::string img_name="/home/kingargroo/seed/ablation1/allimages/WIN_20231031_15_35_35_Pro.jpg";
	
	// task=0 for rice seed; task=1 for regular seed like corn, soybean 
	// task=2 for extremely small seed like millet	
	int task=1;
    // save the visualized image or not
    bool vis=true;
	auto [image, count_number]=detect_img_and_count_yolov8(img_name,task,vis);
	cout<<count_number<<endl;
 


}
