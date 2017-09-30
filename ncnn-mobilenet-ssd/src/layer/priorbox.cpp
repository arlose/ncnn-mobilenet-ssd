// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "priorbox.h"
#include <math.h>
#include <algorithm>
#include <vector>

namespace ncnn {

DEFINE_LAYER_CREATOR(PriorBox)

PriorBox::PriorBox()
{
    one_blob_only = true;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int PriorBox::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d %d %d %d %f %f %f %f %f  %d %d",
                       &image_width, &image_height, &min_size, &max_size, &ar[0], &ar[1], &var[0], &var[1], &var[2], &var[3], &offset, &flip, &clip);
    if (nscan != 13)
    {
        printf("%d %d %d %d %d %d %f %f %f %f %f  %d %d\n",
                       &image_width, &image_height, &min_size, &max_size, &ar[0], &ar[1], &var[0], &var[1], &var[2], &var[3], &offset, &flip, &clip);
        fprintf(stderr, "PriorBox load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int PriorBox::load_param_bin(FILE* paramfp)
{
    fread(&image_width, sizeof(int), 1, paramfp);
    fread(&image_height, sizeof(int), 1, paramfp);

    fread(&min_size, sizeof(int), 1, paramfp);
    fread(&max_size, sizeof(int), 1, paramfp);

    fread(&ar[0], sizeof(int), 1, paramfp);
    fread(&ar[1], sizeof(int), 1, paramfp);

    fread(&var[0], sizeof(float), 1, paramfp);
    fread(&var[1], sizeof(float), 1, paramfp);
    fread(&var[2], sizeof(float), 1, paramfp);
    fread(&var[3], sizeof(float), 1, paramfp);

    fread(&offset, sizeof(float), 1, paramfp);

    fread(&flip, sizeof(int), 1, paramfp);
    fread(&clip, sizeof(int), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int PriorBox::load_param(const unsigned char*& mem)
{
    image_width = *(int*)(mem);
    mem += 4;

    image_height = *(int*)(mem);
    mem += 4;

    min_size = *(int*)(mem);
    mem += 4;

    max_size = *(int*)(mem);
    mem += 4;

    ar[0] = *(int*)(mem);
    mem += 4;

    ar[1] = *(int*)(mem);
    mem += 4;

    var[0] = *(float*)(mem);
    mem += 4;
    var[1] = *(float*)(mem);
    mem += 4;
    var[2] = *(float*)(mem);
    mem += 4;
    var[3] = *(float*)(mem);
    mem += 4;

    offset = *(float*)(mem);
    mem += 4;

    flip = *(int*)(mem);
    mem += 4;
    clip = *(int*)(mem);
    mem += 4;

    return 0;
}

int PriorBox::forward(const Mat& bottom_blobs, Mat& top_blobs) const
{

  const int layer_width = bottom_blobs.w;
  const int layer_height = bottom_blobs.h;
  const int img_width = image_width;
  const int img_height = image_height;

  float step_w, step_h;
  step_w = static_cast<float>(img_width) / layer_width;
  step_h = static_cast<float>(img_height) / layer_height;

  int num_priors = 6;
  if(ar[1]==-233)
    num_priors = 3;
  int dim = layer_height * layer_width * num_priors * 4;
  //printf("priorbox num: %d, w: %d, h: %d\n", dim, layer_width, layer_height);
  
  //top_blobs.create(1, 2, dim);
  top_blobs.create(dim,2);
  float* top_data = top_blobs;

  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + offset) * step_w;
      float center_y = (h + offset) * step_h;
      float box_width, box_height;

        // first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_size;
        // xmin
        top_data[idx++] = (center_x - box_width / 2.) / img_width;
        // ymin
        top_data[idx++] = (center_y - box_height / 2.) / img_height;
        // xmax
        top_data[idx++] = (center_x + box_width / 2.) / img_width;
        // ymax
        top_data[idx++] = (center_y + box_height / 2.) / img_height;

        if (max_size!=-233) {
          // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
          box_width = box_height = sqrt(min_size * max_size);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }

        for(int j=0; j<2; j++)
        {
            if(ar[j]!=-233)
            {
              box_width = min_size * sqrt(ar[j]);
              box_height = min_size / sqrt(ar[j]);
              // xmin
              top_data[idx++] = (center_x - box_width / 2.) / img_width;
              // ymin
              top_data[idx++] = (center_y - box_height / 2.) / img_height;
              // xmax
              top_data[idx++] = (center_x + box_width / 2.) / img_width;
              // ymax
              top_data[idx++] = (center_y + box_height / 2.) / img_height;

              box_width = min_size * sqrt(1.0/ar[j]);
              box_height = min_size / sqrt(1.0/ar[j]);
              // xmin
              top_data[idx++] = (center_x - box_width / 2.) / img_width;
              // ymin
              top_data[idx++] = (center_y - box_height / 2.) / img_height;
              // xmax
              top_data[idx++] = (center_x + box_width / 2.) / img_width;
              // ymax
              top_data[idx++] = (center_y + box_height / 2.) / img_height;
            }
        }
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip==1) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min(std::max(top_data[d], float(0.0)), float(1.0));
    }
  }
  // set the variance.
  top_data += dim;
  int count = 0;
  for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
        for (int i = 0; i < num_priors; ++i) {
          for (int j = 0; j < 4; ++j) {
            top_data[count] = var[j];
            ++count;
          }
        }
      }
    }


  return 0;
}

} // namespace ncnn
