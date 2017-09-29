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

#include "concat.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Concat)

Concat::Concat()
{
}

int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
#ifdef MobileNetSSD
    int h = bottom_blobs[0].h;
    int c = bottom_blobs[0].c;

    // total widths
    int top_width = 0;
    for (size_t b=0; b<bottom_blobs.size(); b++)
    {
        const Mat& bottom_blob = bottom_blobs[b];
        top_width += bottom_blob.w;
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(top_width, h, c);

    if (top_blob.empty())
        return -100;

    int cur_w=0;

    for (size_t b=0; b<bottom_blobs.size(); b++)
    {
        const Mat& bottom_blob = bottom_blobs[b];
        const float* ptr = bottom_blob;
        int w = bottom_blob.w;
        float* outptr = top_blob;
        for(int ic=0;ic<c;ic++)
            for(int ih=0;ih<h;ih++)
                for(int iw=0;iw<w;iw++)
                {
                    outptr[cur_w+iw+ih*top_width+ic*top_width*h] = ptr[iw+ih*w+ic*h*w];
                }
        cur_w += w;
    }
#else
    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;

    // total channels
    int top_channels = 0;
    for (size_t b=0; b<bottom_blobs.size(); b++)
    {
        const Mat& bottom_blob = bottom_blobs[b];
        top_channels += bottom_blob.c;
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, top_channels);
    if (top_blob.empty())
        return -100;

    int q = 0;
    for (size_t b=0; b<bottom_blobs.size(); b++)
    {
        const Mat& bottom_blob = bottom_blobs[b];

        int channels = bottom_blob.c;
        int size = bottom_blob.cstep * channels;

        const float* ptr = bottom_blob;
        float* outptr = top_blob.channel(q);
        for (int i=0; i<size; i++)
        {
            outptr[i] = ptr[i];
        }

        q += channels;
    }
#endif
    return 0;
}

} // namespace ncnn
