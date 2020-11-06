#pragma once

#define VO_DEBUG

#include <thread>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>
#include <opencv2/highgui.hpp>
#include <sophus/se2.hpp>

#include "image/ImageData.h"
#include "optical_flow/OpticalFlowBase.h"
#include "optical_flow/patch.h"
#include "utils/sophus_utils.hpp"

#include <random>

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
    // cv::imshow("sss", imgs[0]);

    std::vector<ImageData> imgData;
    MatToImageData(imgs, imgData);
    if (!initialized) {
      observations.resize(cam_num);
      pyramid.reset(new std::vector<ManagedImagePyr<u_int16_t>>);
      pyramid->resize(cam_num);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, cam_num),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(*imgData[i].img,
                                                        optical_flow_levels);
                          }
                        });

      addPoints();
      // filterPoints();
      initialized = true;
    } else {
      old_pyramid = pyramid;

      pyramid.reset(new std::vector<vo::ManagedImagePyr<u_int16_t>>);
      pyramid->resize(cam_num);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, cam_num),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(*imgData[i].img,
                                                        optical_flow_levels);
                          }
                        });

      std::vector<Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>>
          new_observations(cam_num);

      for (size_t i = 0; i < cam_num; i++) {
        trackPoints(old_pyramid->at(i), pyramid->at(i), observations[i],
                    new_observations[i]);
      }

      observations = new_observations;

#ifdef VO_DEBUG
      cv::Mat img_show;
      cv::cvtColor(imgs[0], img_show, cv::COLOR_GRAY2BGR);
      std::uniform_int_distribution<int> dis(1, 255);
      for (const auto& ob : observations.at(0)) {
        std::mt19937 gen(ob.first);
        cv::Scalar color(dis(gen), dis(gen), dis(gen));
        cv::Point2f pt(ob.second.translation().x(),
                       ob.second.translation().y());
        cv::circle(img_show, pt, 3, color, -1);
        cv::putText(img_show, std::to_string(ob.first), pt, 1, 1, color);
      }
      cv::imshow("add", img_show);
#endif

      addPoints();
      // filterPoints();
    }

    frame_counter++;
  }

  void trackPoints(const vo::ManagedImagePyr<u_int16_t>& pyr_1,
                   const vo::ManagedImagePyr<u_int16_t>& pyr_2,
                   const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_1,
                   Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_2) const {
    size_t num_points = transform_map_1.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);

    for (const auto& kv : transform_map_1) {
      ids.push_back(kv.first);
      init_vec.push_back(kv.second);
    }

    tbb::concurrent_unordered_map<KeypointId, Eigen::AffineCompact2f,
                                  std::hash<KeypointId>>
        result;

    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];

        const Eigen::AffineCompact2f& transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = transform_1;

        bool valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2);

        if (valid) {
          Eigen::AffineCompact2f transform_1_recovered = transform_2;

          valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);

          if (valid) {
            Scalar dist2 = (transform_1.translation() -
                            transform_1_recovered.translation())
                               .squaredNorm();

            if (dist2 < optical_flow_max_recovered_dist2) {
              result[id] = transform_2;
            }
          }
        }
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);

    tbb::parallel_for(range, compute_func);
    // compute_func(range);

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
  }

  inline bool trackPoint(const vo::ManagedImagePyr<uint16_t>& old_pyr,
                         const vo::ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform,
                         Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    transform.linear().setIdentity();

    for (int level = optical_flow_levels; level >= 0 && patch_valid; level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      // Perform tracking on current level
      patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);

      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const u_int16_t>& img_2,
                                const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    for (int iteration = 0;
         patch_valid && iteration < optical_flow_max_iterations; iteration++) {
      typename PatchT::VectorP res;

      typename PatchT::Matrix2P transformed_pat =
          transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      bool valid = dp.residual(img_2, transformed_pat, res);

      if (valid) {
        Vector3 inc = -dp.H_se2_inv_J_se2_T * res;
        transform *= SE2::exp(inc).matrix();

        const int filter_margin = 2;

        if (!img_2.InBounds(transform.translation(), filter_margin))
          patch_valid = false;
      } else {
        patch_valid = false;
      }
    }

    return patch_valid;
  }
  void addPoints() {
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    for (const auto& kv : observations.at(0)) {
      pts0.emplace_back(kv.second.translation().template cast<double>());
    }

    KeypointsData kd;

    detectKeypoints(pyramid->at(0).lvl(0), kd, optical_flow_detection_grid_size,
                    1, pts0);

    Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> new_poses0,
        new_poses1;

    for (size_t i = 0; i < kd.corners.size(); i++) {
      Eigen::AffineCompact2f transform;
      transform.setIdentity();
      transform.translation() = kd.corners[i].cast<Scalar>();

      observations.at(0)[last_keypoint_id] = transform;
      new_poses0[last_keypoint_id] = transform;

      last_keypoint_id++;
    }

    if (cam_num > 1) {
      trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);

      for (const auto& kv : new_poses1) {
        observations.at(1).emplace(kv);
      }
    }
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
  // OpticalFlowResult::Ptr transforms;
  std::shared_ptr<std::vector<ManagedImagePyr<u_int16_t>>> old_pyramid, pyramid;

  Matrix4 E;

  std::shared_ptr<std::thread> processing_thread;
};
}  // namespace vo