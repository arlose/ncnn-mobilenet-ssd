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

#include "permute.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Permute)

Permute::Permute()
{
    one_blob_only = true;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int Permute::load_param(FILE* paramfp)
{
    /*
    int nscan = fscanf(paramfp, "%d %d %d",
                       &w, &h, &c);
    if (nscan != 3)
    {
        fprintf(stderr, "String Permute load_param failed %d\n", nscan);
        return -1;
    }

    ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;

    */

    return 0;
}
#endif // NCNN_STRING
int Permute::load_param_bin(FILE* paramfp)
{
    /*
    fread(&w, sizeof(int), 1, paramfp);

    fread(&h, sizeof(int), 1, paramfp);

    fread(&c, sizeof(int), 1, paramfp);

    ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;
    */

    fprintf(stderr, "Permute load_param bin \n");
    return 0;
}
#endif // NCNN_STDIO

int Permute::load_param(const unsigned char*& mem)
{
    /*
    w = *(int*)(mem);
    mem += 4;

    h = *(int*)(mem);
    mem += 4;

    c = *(int*)(mem);
    mem += 4;

    ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;
    */
    fprintf(stderr, "Permute load_param bin \n");
    return 0;
}

int Permute::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c;

    top_blob.create(total);
    if (top_blob.empty())
        return -100;

    // c-h-w to h-w-c
    float* ptr = top_blob;
    for (int i=0; i<bottom_blob.h; i++)
    {
        for (int j=0; j<bottom_blob.w; j++)
        {
            for (int p=0; p<bottom_blob.c; p++)
            {
                const float* bptr = bottom_blob.channel(p);
                *ptr++ = bptr[i*bottom_blob.w + j];
            }
        }
    }
  

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
