#include <assert.h>
#include <stdlib.h>

#include "matrix.h"

void RandomizeMatrix(float* matrix, const std::vector<int>& shape) {
  // shape is a (M, N) value pair
  //   assert(shape.size() == 2);

  const int row = shape[0];
  const int col = shape[1];

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      matrix[i * col + j] = 1.0f * (rand() / float(RAND_MAX)) - 0.5f;
    }
  }
}