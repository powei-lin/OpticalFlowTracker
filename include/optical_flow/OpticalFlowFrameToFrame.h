#pragma once

#include <thread>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>
#include <sophus/se2.hpp>
#include <opencv2/highgui.hpp>

#include "image/ImageData.h"
#include "optical_flow/OpticalFlowBase.h"
#include "optical_flow/patch.h"
#include "utils/sophus_utils.hpp"

namespace vo {

template <typename Scalar, template <typename> typename Pattern>
class OpticalFlowFrameToFrame : public OpticalFlowBase {
 public:
  typedef OpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Sophus::SE2<Scalar> SE2;

  OpticalFlowFrameToFrame()
      : t_ns(-1), frame_counter(0), last_keypoint_id(0), cam_num(1) {
    // input_queue.set_capacity(10);

    patch_coord = PatchT::pattern2.template cast<float>();

    // if (calib.intrinsics.size() > 1) {
    //   Eigen::Matrix4d Ed;
    //   Sophus::SE3d T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[1];
    //   computeEssential(T_i_j, Ed);
    //   E = Ed.cast<Scalar>();
    // }

    // processing_thread.reset(
    //     new std::thread(&FrameToFrameOpticalFlow::processingLoop, this));
  }

  void processFrame(const std::vector<cv::Mat>& imgs) {

    cv::imshow("sss", imgs[0]);

    std::vector<ImageData> imgData;
    MatToImageData(imgs, imgData);
    if (!initialized) {
      pyramid.reset(new std::vector<ManagedImagePyr<u_int16_t>>);
      pyramid->resize(cam_num);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, cam_num),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(*imgData[i].img,
                                                        optical_flow_levels);
                          }
                        });

      // addPoints();
      // filterPoints();
      initialized = true;
    } else {
    }
  }

  void addPoints() {
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    for (const auto& kv : observations.at(0)) {
      std::cout << kv.second.translation() << std::endl;
      // pts0.emplace_back(kv.second.translation().cast<double>());
    }

    KeypointsData kd;

    // detectKeypoints(pyramid->at(0).lvl(0), kd, optical_flow_detection_grid_size,
    //                 1, pts0);

    // Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> new_poses0,
    //     new_poses1;

    // for (size_t i = 0; i < kd.corners.size(); i++) {
    //   Eigen::AffineCompact2f transform;
    //   transform.setIdentity();
    //   transform.translation() = kd.corners[i].cast<Scalar>();

    //   observations.at(0)[last_keypoint_id] = transform;
    //   new_poses0[last_keypoint_id] = transform;

    //   last_keypoint_id++;
    // }

    // if (cam_num > 1) {
    //   trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);

    //   for (const auto& kv : new_poses1) {
    //     transforms->observations.at(1).emplace(kv);
    //   }
    // }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  static constexpr uint8_t optical_flow_detection_grid_size = 50;
  static constexpr float optical_flow_max_recovered_dist2 = 0.04;
  static constexpr uint8_t optical_flow_max_iterations = 5;
  static constexpr uint8_t optical_flow_levels = 3;
  int64_t t_ns;

  uint64_t frame_counter;

  KeypointId last_keypoint_id;

  bool initialized = false;

  const uint8_t cam_num;

  //[cam][point id]
  std::vector<Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>>
      observations;
  // OpticalFlowResult::Ptr transforms;
  std::shared_ptr<std::vector<ManagedImagePyr<u_int16_t>>> old_pyramid, pyramid;

  Matrix4 E;

  std::shared_ptr<std::thread> processing_thread;
};
}  // namespace vo