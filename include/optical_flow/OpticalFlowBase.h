#pragma once

#include <Eigen/Core>

#include "image/image_pyr.h"
#include "utils/common_types.h"

namespace vo {

using KeypointId = uint64_t;
class OpticalFlowBase {
 protected:
  static constexpr int EDGE_THRESHOLD = 19;
  std::vector<Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>>
      observations;
  void detectKeypoints(
      const Image<const uint16_t>& img_raw, KeypointsData& kd, int PATCH_SIZE,
      int num_points_cell,
      const Eigen::aligned_vector<Eigen::Vector2d>& current_points) {
    kd.corners.clear();
    kd.corner_angles.clear();
    kd.corner_descriptors.clear();

    const size_t x_start = (img_raw.w % PATCH_SIZE) / 2;
    const size_t x_stop = x_start + img_raw.w - PATCH_SIZE;

    const size_t y_start = (img_raw.h % PATCH_SIZE) / 2;
    const size_t y_stop = y_start + img_raw.h - PATCH_SIZE;

    // std::cerr << "x_start " << x_start << " x_stop " << x_stop << std::endl;
    // std::cerr << "y_start " << y_start << " y_stop " << y_stop << std::endl;

    Eigen::MatrixXi cells;
    cells.setZero(img_raw.h / PATCH_SIZE + 1, img_raw.w / PATCH_SIZE + 1);

    for (const Eigen::Vector2d& p : current_points) {
      if (p[0] >= x_start && p[1] >= y_start) {
        int x = (p[0] - x_start) / PATCH_SIZE;
        int y = (p[1] - y_start) / PATCH_SIZE;

        cells(y, x) += 1;
      }
    }

    for (size_t x = x_start; x < x_stop; x += PATCH_SIZE) {
      for (size_t y = y_start; y < y_stop; y += PATCH_SIZE) {
        if (cells((y - y_start) / PATCH_SIZE, (x - x_start) / PATCH_SIZE) > 0)
          continue;

        const vo::Image<const uint16_t> sub_img_raw =
            img_raw.SubImage(x, y, PATCH_SIZE, PATCH_SIZE);

        cv::Mat subImg(PATCH_SIZE, PATCH_SIZE, CV_8U);

        for (int y = 0; y < PATCH_SIZE; y++) {
          uchar* sub_ptr = subImg.ptr(y);
          for (int x = 0; x < PATCH_SIZE; x++) {
            sub_ptr[x] = (sub_img_raw(x, y) >> 8);
          }
        }

        int points_added = 0;
        int threshold = 40;

        while (points_added < num_points_cell && threshold >= 5) {
          std::vector<cv::KeyPoint> points;
          cv::FAST(subImg, points, threshold);

          std::sort(points.begin(), points.end(),
                    [](const cv::KeyPoint& a, const cv::KeyPoint& b) -> bool {
                      return a.response > b.response;
                    });

// #define USE_CORNER_SUBPIX
#ifdef USE_CORNER_SUBPIX
          if (points.size() > 0) {
            std::vector<cv::Point2f> subcorner_pts;
            cv::KeyPoint::convert(points, subcorner_pts);
            cv::TermCriteria criteria = cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.01);
            cv::Size winSize = cv::Size(5, 5);
            cv::Size zeroZone = cv::Size(-1, -1);
            cv::cornerSubPix(subImg, subcorner_pts, winSize, zeroZone,
                             criteria);
      
          for (size_t i = 0;
               i < subcorner_pts.size() && points_added < num_points_cell; i++)
            if (img_raw.InBounds(x + subcorner_pts[i].x, y + subcorner_pts[i].y,
                                 EDGE_THRESHOLD)) {
              kd.corners.emplace_back(x + subcorner_pts[i].x, y + subcorner_pts[i].y);
              points_added++;
            }
          }
#else
          for (size_t i = 0;
               i < points.size() && points_added < num_points_cell; i++)
            if (img_raw.InBounds(x + points[i].pt.x, y + points[i].pt.y,
                                 EDGE_THRESHOLD)) {
              kd.corners.emplace_back(x + points[i].pt.x, y + points[i].pt.y);
              points_added++;
            }
#endif

          threshold /= 2;
        }
      }
    }

    // std::cout << "Total points: " << kd.corners.size() << std::endl;

    //  cv::TermCriteria criteria =
    //      cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
    //  cv::Size winSize = cv::Size(5, 5);
    //  cv::Size zeroZone = cv::Size(-1, -1);
    //  cv::cornerSubPix(image, points, winSize, zeroZone, criteria);

    //  for (size_t i = 0; i < points.size(); i++) {
    //    if (img_raw.InBounds(points[i].pt.x, points[i].pt.y, EDGE_THRESHOLD))
    //    {
    //      kd.corners.emplace_back(points[i].pt.x, points[i].pt.y);
    //    }
    //  }
  }

 public:
  Eigen::MatrixXf patch_coord;
};
}  // namespace vo