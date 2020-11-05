#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "image/image.h"

using namespace std;
using namespace cv;

int main(){
    constexpr char dataset[] = "/home/powei/Documents/dataset/EuRoC/V1_01_easy/mav0/cam0/data/*.png";
    vector<string> filenames;
    glob(dataset, filenames);
    for(const auto &name:filenames){
        Mat img = imread(name, cv::IMREAD_UNCHANGED);
        imshow("img", img);
        waitKey(30);
    }
    return 0;
}