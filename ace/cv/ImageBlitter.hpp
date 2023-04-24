//
//  ImageBlitter.hpp
//  MNN
//
//  Created by MNN on 2018/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ImageBlitter_hpp
#define ImageBlitter_hpp

#include <stdio.h>

#include <ace/ImageProcess.hpp>
namespace ace {
namespace CV {
class ImageBlitter {
 public:
  typedef void (*BLITTER)(const unsigned char* source, unsigned char* dest,
                          size_t count);

  static BLITTER choose(ImageFormat source, ImageFormat dest);
};
}  // namespace CV
}  // namespace ace

#endif /* ImageBlitter_hpp */
