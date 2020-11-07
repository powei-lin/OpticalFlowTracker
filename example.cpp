#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "json.hpp"

#include "optical_flow/OpticalFlowFrameToFrame.h"

using namespace std;
using namespace cv;

nlohmann::json fromFile(const string &filename) {
  ifstream i(filename);
  nlohmann::json j;
  i >> j;
  i.close();
  return j;
}

Sophus::SE3f T_1_0(const nlohmann::json &j) {
  Sophus::SE3f T_imu_0;
  auto &cam0 = j.at("value0").at("T_imu_cam").at(0);
  T_imu_0.translation() =
      Eigen::Vector3f(cam0.at("px"), cam0.at("py"), cam0.at("pz"));
  T_imu_0.so3().setQuaternion(Eigen::Quaternionf(cam0.at("qw"), cam0.at("qx"),
                                                 cam0.at("qy"), cam0.at("qz")));
  Sophus::SE3f T_imu_1;
  auto &cam1 = j.at("value0").at("T_imu_cam").at(1);
  T_imu_1.translation() =
      Eigen::Vector3f(cam1.at("px"), cam1.at("py"), cam1.at("pz"));
  T_imu_1.so3().setQuaternion(Eigen::Quaternionf(cam1.at("qw"), cam1.at("qx"),
                                                 cam1.at("qy"), cam1.at("qz")));
  return T_imu_1.inverse() * T_imu_0;
}

int main() {
  constexpr char dataset0[] =
      "/home/powei/Documents/dataset/EuRoC/V1_01_easy/mav0/cam0/data/*.png";
  constexpr char dataset1[] =
      "/home/powei/Documents/dataset/EuRoC/V1_01_easy/mav0/cam1/data/*.png";
  // constexpr char dataset[] = "/Users/powei/Documents/mav0/cam0/data/*.png";
  vector<vector<string>> filenames(2);
  glob(dataset0, filenames[0]);
  glob(dataset1, filenames[1]);
  const nlohmann::json j = fromFile("../data/euroc_ds_calib.json");
  cout << T_1_0(j).matrix() << endl;

  vo::OpticalFlowFrameToFrame<float, vo::Pattern51> flow(2, T_1_0(j).inverse());

  VideoWriter video("./outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(1504, 480));
  for (size_t i = 0; i < filenames[0].size(); i++) {
    Mat img0 = imread(filenames[0][i], cv::IMREAD_UNCHANGED);
    Mat img1 = imread(filenames[1][i], cv::IMREAD_UNCHANGED);
    Mat show  = flow.processFrame({img0, img1});
    if(i>100){
      cout << show.size() << endl;
      imshow("track", show);
      video.write(show);
    }
    waitKey(1);
    if(i>1000)
      break;
  }
  video.release();
  return 0;
}