#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "image/image_pyr.h"

namespace vo {
struct ImageData {
  ImageData() : exposure(0) {}

  ManagedImage<uint16_t>::Ptr img;
  double exposure;
};

inline void MatToImageData(const std::vector<cv::Mat> &imgs,
                           std::vector<ImageData> &imgDatas) {
  // imgDatas.clear();
  imgDatas.resize(imgs.size());
  for (size_t i = 0; i < imgs.size(); ++i) {
    if (imgs[i].type() == CV_8UC1) {
      imgDatas[i].img.reset(
          new ManagedImage<uint16_t>(imgs[i].cols, imgs[i].rows));

      const uint8_t *data_in = imgs[i].ptr();
      uint16_t *data_out = imgDatas[i].img->ptr;

      size_t full_size = imgs[i].cols * imgs[i].rows;
      for (size_t i = 0; i < full_size; i++) {
        int val = data_in[i];
        val = val << 8;
        data_out[i] = val;
      }
    } else if (imgs[i].type() == CV_8UC3) {
      imgDatas[i].img.reset(
          new ManagedImage<uint16_t>(imgs[i].cols, imgs[i].rows));

      const uint8_t *data_in = imgs[i].ptr();
      uint16_t *data_out = imgDatas[i].img->ptr;

      size_t full_size = imgs[i].cols * imgs[i].rows;
      for (size_t i = 0; i < full_size; i++) {
        int val = data_in[i * 3];
        val = val << 8;
        data_out[i] = val;
      }
    } else if (imgs[i].type() == CV_16UC1) {
      imgDatas[i].img.reset(
          new ManagedImage<uint16_t>(imgs[i].cols, imgs[i].rows));
      std::memcpy(imgDatas[i].img->ptr, imgs[i].ptr(),
                  imgs[i].cols * imgs[i].rows * sizeof(uint16_t));
    }
  }
}

}  // namespace vo