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

Sophus::SE3d T_1_0(const nlohmann::json &j){
  Sophus::SE3d T_imu_0;
  auto &cam0 = j.at("value0").at("T_imu_cam").at(0);
  T_imu_0.translation() = Eigen::Vector3d(cam0.at("px"),cam0.at("py"),cam0.at("pz"));
  T_imu_0.so3().setQuaternion(Eigen::Quaterniond(cam0.at("qw"),cam0.at("qx"),cam0.at("qy"),cam0.at("qz")));
  Sophus::SE3d T_imu_1;
  auto &cam1 = j.at("value0").at("T_imu_cam").at(1);
  T_imu_1.translation() = Eigen::Vector3d(cam1.at("px"),cam1.at("py"),cam1.at("pz"));
  T_imu_1.so3().setQuaternion(Eigen::Quaterniond(cam1.at("qw"),cam1.at("qx"),cam1.at("qy"),cam1.at("qz")));
  return T_imu_1.inverse() * T_imu_0;
}

int main() {
  constexpr char dataset[] =
      "/home/powei/Documents/dataset/EuRoC/V1_01_easy/mav0/cam0/data/*.png";
  // constexpr char dataset[] = "/Users/powei/Documents/mav0/cam0/data/*.png";
  vector<string> filenames;
  glob(dataset, filenames);
  const nlohmann::json j = fromFile("../data/euroc_ds_calib.json");
  cout << T_1_0(j).matrix() << endl;

  vo::OpticalFlowFrameToFrame<float, vo::Pattern51> flow;

  for (const auto &name : filenames) {
    Mat img = imread(name, cv::IMREAD_UNCHANGED);
    flow.processFrame({img});
    // imshow("img", img);
    waitKey(3);
  }
  return 0;
}