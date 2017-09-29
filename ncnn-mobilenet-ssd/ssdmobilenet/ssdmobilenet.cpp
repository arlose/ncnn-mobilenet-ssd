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

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sys/time.h>
#include <unistd.h>
#include "net.h"

static int detect_mobilenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net mobilenet;
    mobilenet.load_param("ssdmobilenet.param");
    mobilenet.load_model("ssdmobilenet.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 300, 300);

    //const float mean_vals[3] = {104.f, 117.f, 123.f};
    //in.substract_mean_normalize(mean_vals, 0);

    // ncnn::Extractor ex = mobilenet.create_extractor();
    // ex.set_light_mode(true);
    // ex.set_num_threads(4);

    // ex.input("data", in);

    ncnn::Mat out;

    for(int i=0;i<1;i++)
    {
        ncnn::Extractor ex = mobilenet.create_extractor();
        ex.set_light_mode(true);
        //ex.set_num_threads(4);
        ex.input("data", in);
        ex.extract("mbox_priorbox",out);
        //ex.extract("conv6/dw_conv6/dw/relu", out);    

    }

    printf("%d %d %d\n", out.w, out.h, out.c);

    for (int l=0;l<400;l++)
        {
                printf("%f ", out[l]);
                if((l+1)%21==0)
                    printf("\n");
        }
    // for(int c=0;c<1;c++) // out.c
    // {
    //     for(int h=0;h<10;h++) // out.h
    //     {
    //         for(int w=0;w<10;w++) // out.w
    //         {
    //             printf("%f ", out[c*out.w*out.h+h*out.w+w]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n\n");
    // }

/*
    cls_scores.resize(out.c);
    for (int j=0; j<out.c; j++)
    {
        const float* prob = out.data + out.cstep * j;
        cls_scores[j] = prob[0];
    }
*/
    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_mobilenet(m, cls_scores);

    //print_topk(cls_scores, 3);

    return 0;
}

