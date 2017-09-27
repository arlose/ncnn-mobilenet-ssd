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

#include "detectionoutput.h"
#include <math.h>
#include <algorithm>
#include <vector>

namespace ncnn {

DEFINE_LAYER_CREATOR(DetectionOutput)



DetectionOutput::DetectionOutput()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int DetectionOutput::load_param(FILE* paramfp)
{
//     float ratio;
//     float scale;
    /*
    int nscan = fscanf(paramfp, "%d %d %d %d %f %d",
                       &feat_stride, &base_size, &pre_nms_topN, &after_nms_topN,
                       &nms_thresh, &min_size);
    if (nscan != 6)
    {
        fprintf(stderr, "PriorBox load_param failed %d\n", nscan);
        return -1;
    }
    */

    return 0;
}
#endif // NCNN_STRING
int DetectionOutput::load_param_bin(FILE* paramfp)
{
    /*
    fread(&feat_stride, sizeof(int), 1, paramfp);

    fread(&base_size, sizeof(int), 1, paramfp);

//     float ratio;
//     float scale;

    fread(&pre_nms_topN, sizeof(int), 1, paramfp);

    fread(&after_nms_topN, sizeof(int), 1, paramfp);

    fread(&nms_thresh, sizeof(float), 1, paramfp);

    fread(&min_size, sizeof(int), 1, paramfp);
    */

    return 0;
}
#endif // NCNN_STDIO

int DetectionOutput::load_param(const unsigned char*& mem)
{
    /*
    feat_stride = *(int*)(mem);
    mem += 4;

    base_size = *(int*)(mem);
    mem += 4;

//     float ratio;
//     float scale;

    pre_nms_topN = *(int*)(mem);
    mem += 4;

    after_nms_topN = *(int*)(mem);
    mem += 4;

    nms_thresh = *(float*)(mem);
    mem += 4;

    min_size = *(int*)(mem);
    mem += 4;
    */

    return 0;
}

int DetectionOutput::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    
    return 0;
}

} // namespace ncnn
