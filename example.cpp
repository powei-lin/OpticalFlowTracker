#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "optical_flow/OpticalFlowFrameToFrame.h"

using namespace std;
using namespace cv;

int main(){
    // constexpr char dataset[] = "/home/powei/Documents/dataset/EuRoC/V1_01_easy/mav0/cam0/data/*.png";
    constexpr char dataset[] = "/Users/powei/Documents/mav0/cam0/data/*.png";
    vector<string> filenames;
    glob(dataset, filenames);

    vo::OpticalFlowFrameToFrame<float, vo::Pattern51> flow;

    for(const auto &name:filenames){
        Mat img = imread(name, cv::IMREAD_UNCHANGED);
        flow.processFrame({img});
        // imshow("img", img);
        waitKey(3);
    }
    return 0;
}