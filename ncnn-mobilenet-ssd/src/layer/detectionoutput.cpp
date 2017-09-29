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
    one_blob_only = false;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int DetectionOutput::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %f %d %d %f",
                       &num_classes, &nms_threshold, &nms_top_k, &keep_top_k, &confidence_threshold);
    if (nscan != 5)
    {
        fprintf(stderr, "DetectionOutput load_param failed %d\n", nscan);
        return -1;
    }
    return 0;
}
#endif // NCNN_STRING
int DetectionOutput::load_param_bin(FILE* paramfp)
{
    fread(&num_classes, sizeof(int), 1, paramfp);
    fread(&nms_threshold, sizeof(float), 1, paramfp);
    fread(&nms_top_k, sizeof(int), 1, paramfp);
    fread(&keep_top_k, sizeof(int), 1, paramfp);
    fread(&confidence_threshold, sizeof(float), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int DetectionOutput::load_param(const unsigned char*& mem)
{
    num_classes = *(int*)(mem);
    mem += 4;

    nms_threshold = *(float*)(mem);
    mem += 4;

    nms_top_k = *(int*)(mem);
    mem += 4;

    keep_top_k = *(int*)(mem);
    mem += 4;

    confidence_threshold = *(float*)(mem);
    mem += 4;
    
    return 0;
}

int DetectionOutput::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    
    return 0;
}

} // namespace ncnn
