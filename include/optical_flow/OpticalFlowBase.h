#pragma once

#include <Eigen/Core>
namespace vo {

using KeypointId = uint64_t;
class OpticalFlowBase {
 public:
  Eigen::MatrixXf patch_coord;
};
}  // namespace vo