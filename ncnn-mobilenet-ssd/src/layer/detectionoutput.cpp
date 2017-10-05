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
#include <map>

using namespace std;

#define CHECK_GT(x,y) (x)>(y)?true:false
#define CHECK_EQ(x,y) (x)==(y)?true:false
#define CHECK_LT(x,y) (x)<(y)?true:false

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

typedef struct
{
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float size;
    int difficult;
}NormalizedBBox;

typedef float Dtype;
typedef map<int, std::vector<NormalizedBBox> > LabelBBox;

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox) {
  if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin ||
      bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) {
    // Return [0, 0, 0, 0] if there is no intersection.
    intersect_bbox->xmin = 0;
    intersect_bbox->ymin = 0;
    intersect_bbox->xmax = 0;
    intersect_bbox->ymax = 0;
  } else {
    intersect_bbox->xmin = (std::max(bbox1.xmin, bbox2.xmin));
    intersect_bbox->ymin = (std::max(bbox1.ymin, bbox2.ymin));
    intersect_bbox->xmax = (std::min(bbox1.xmax, bbox2.xmax));
    intersect_bbox->ymax = (std::min(bbox1.ymax, bbox2.ymax));
  }
}

float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true) {
  if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0;
  } else {
      float width = bbox.xmax - bbox.xmin;
      float height = bbox.ymax - bbox.ymin;
      if (normalized) {
        return width * height;
      } else {
        // If bbox is not within range [0, 1].
        return (width + 1) * (height + 1);
      }
  }
}

Dtype BBoxSize(const Dtype* bbox, const bool normalized = true) {
  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return Dtype(0.);
  } else {
    const Dtype width = bbox[2] - bbox[0];
    const Dtype height = bbox[3] - bbox[1];
    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1) * (height + 1);
    }
  }
}

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int>& pair1,
                                   const pair<float, int>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> >& pair1,
                                   const pair<float, pair<int, int> >& pair2);


void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox) {
  clip_bbox->xmin = (std::max(std::min(bbox.xmin, 1.f), 0.f));
  clip_bbox->ymin = (std::max(std::min(bbox.ymin, 1.f), 0.f));
  clip_bbox->xmax = (std::max(std::min(bbox.xmax, 1.f), 0.f));
  clip_bbox->ymax = (std::max(std::min(bbox.ymax, 1.f), 0.f));

  clip_bbox->size = (BBoxSize(*clip_bbox));
  clip_bbox->difficult = bbox.difficult;
}

void ClipBBox(const NormalizedBBox& bbox, const float height, const float width,
              NormalizedBBox* clip_bbox) {
  clip_bbox->xmin = (std::max(std::min(bbox.xmin, width), 0.f));
  clip_bbox->ymin = (std::max(std::min(bbox.ymin, height), 0.f));
  clip_bbox->xmax = (std::max(std::min(bbox.xmax, width), 0.f));
  clip_bbox->ymax = (std::max(std::min(bbox.ymax, height), 0.f));
  clip_bbox->size = (BBoxSize(*clip_bbox));
  clip_bbox->difficult = (bbox.difficult);
}

void GetLocPredictions(const Dtype* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, std::vector<LabelBBox>* loc_preds) {
  loc_preds->clear();
  //num_loc_classes = 1;
  loc_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    LabelBBox& label_bbox = (*loc_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4;
      for (int c = 0; c < num_loc_classes; ++c) {
        int label = share_location ? -1 : c;
        if (label_bbox.find(label) == label_bbox.end()) {
          label_bbox[label].resize(num_preds_per_class);
        }
        label_bbox[label][p].xmin = (loc_data[start_idx + c * 4]);
        label_bbox[label][p].ymin = (loc_data[start_idx + c * 4 + 1]);
        label_bbox[label][p].xmax = (loc_data[start_idx + c * 4 + 2]);
        label_bbox[label][p].ymax = (loc_data[start_idx + c * 4 + 3]);
      }
    }
    loc_data += num_preds_per_class * num_loc_classes * 4;
  }
}

void GetConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      vector<map<int, vector<float> > >* conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    map<int, vector<float> >& label_scores = (*conf_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        label_scores[c].push_back(conf_data[start_idx + c]);
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}

void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances) {
  prior_bboxes->clear();
  prior_variances->clear();
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    NormalizedBBox bbox;
    bbox.xmin = (prior_data[start_idx]);
    bbox.ymin = (prior_data[start_idx + 1]);
    bbox.xmax = (prior_data[start_idx + 2]);
    bbox.ymax = (prior_data[start_idx + 3]);
    float bbox_size = BBoxSize(bbox);
    bbox.size = (bbox_size);
    prior_bboxes->push_back(bbox);
  }

  for (int i = 0; i < num_priors; ++i) {
    int start_idx = (num_priors + i) * 4;
    vector<float> var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_idx + j]);
    }
    prior_variances->push_back(var);
  }
}

enum PriorBoxParameter_CodeType {
  PriorBoxParameter_CodeType_CORNER = 1,
  PriorBoxParameter_CodeType_CENTER_SIZE = 2,
  PriorBoxParameter_CodeType_CORNER_SIZE = 3
};
typedef PriorBoxParameter_CodeType CodeType;

void DecodeBBox(
    const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const NormalizedBBox& bbox,
    NormalizedBBox* decode_bbox) {
  if (code_type == PriorBoxParameter_CodeType_CORNER) {
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      decode_bbox->xmin = (prior_bbox.xmin + bbox.xmin);
      decode_bbox->ymin = (prior_bbox.ymin + bbox.ymin);
      decode_bbox->xmax = (prior_bbox.xmax + bbox.xmax);
      decode_bbox->ymax = (prior_bbox.ymax + bbox.ymax);
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox->xmin = (
          prior_bbox.xmin + prior_variance[0] * bbox.xmin);
      decode_bbox->ymin = (
          prior_bbox.ymin + prior_variance[1] * bbox.ymin);
      decode_bbox->xmax = (
          prior_bbox.xmax + prior_variance[2] * bbox.xmax);
      decode_bbox->ymax = (
          prior_bbox.ymax + prior_variance[3] * bbox.ymax);
    }
  } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
    float prior_width = prior_bbox.xmax - prior_bbox.xmin;
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax - prior_bbox.ymin;
    CHECK_GT(prior_height, 0);
    float prior_center_x = (prior_bbox.xmin + prior_bbox.xmax) / 2.;
    float prior_center_y = (prior_bbox.ymin + prior_bbox.ymax) / 2.;

    float decode_bbox_center_x, decode_bbox_center_y;
    float decode_bbox_width, decode_bbox_height;
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decode_bbox_center_x = bbox.xmin * prior_width + prior_center_x;
      decode_bbox_center_y = bbox.ymin * prior_height + prior_center_y;
      decode_bbox_width = exp(bbox.xmax) * prior_width;
      decode_bbox_height = exp(bbox.ymax) * prior_height;
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox_center_x =
          prior_variance[0] * bbox.xmin * prior_width + prior_center_x;
      decode_bbox_center_y =
          prior_variance[1] * bbox.ymin * prior_height + prior_center_y;
      decode_bbox_width =
          exp(prior_variance[2] * bbox.xmax) * prior_width;
      decode_bbox_height =
          exp(prior_variance[3] * bbox.ymax) * prior_height;
    }

    decode_bbox->xmin = (decode_bbox_center_x - decode_bbox_width / 2.);
    decode_bbox->ymin = (decode_bbox_center_y - decode_bbox_height / 2.);
    decode_bbox->xmax = (decode_bbox_center_x + decode_bbox_width / 2.);
    decode_bbox->ymax = (decode_bbox_center_y + decode_bbox_height / 2.);
  } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
    float prior_width = prior_bbox.xmax - prior_bbox.xmin;
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax - prior_bbox.ymin;
    CHECK_GT(prior_height, 0);
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      decode_bbox->xmin = (prior_bbox.xmin + bbox.xmin * prior_width);
      decode_bbox->ymin = (prior_bbox.ymin + bbox.ymin * prior_height);
      decode_bbox->xmax = (prior_bbox.xmax + bbox.xmax * prior_width);
      decode_bbox->ymax = (prior_bbox.ymax + bbox.ymax * prior_height);
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox->xmin = (
          prior_bbox.xmin + prior_variance[0] * bbox.xmin * prior_width);
      decode_bbox->ymin = (
          prior_bbox.ymin + prior_variance[1] * bbox.ymin * prior_height);
      decode_bbox->xmax = (
          prior_bbox.xmax + prior_variance[2] * bbox.xmax * prior_width);
      decode_bbox->ymax = (
          prior_bbox.ymax + prior_variance[3] * bbox.ymax * prior_height);
    }
  } else {
    fprintf(stderr, "Unknown LocLossType.\n");
  }
  float bbox_size = BBoxSize(*decode_bbox);
  decode_bbox->size = (bbox_size);
  if (clip_bbox) {
    ClipBBox(*decode_bbox, decode_bbox);
  }
}

void DecodeBBoxes(
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const vector<NormalizedBBox>& bboxes,
    vector<NormalizedBBox>* decode_bboxes) {
  CHECK_EQ(prior_bboxes.size(), prior_variances.size());
  CHECK_EQ(prior_bboxes.size(), bboxes.size());
  int num_bboxes = prior_bboxes.size();
  if (num_bboxes >= 1) {
    CHECK_EQ(prior_variances[0].size(), 4);
  }
  decode_bboxes->clear();
  for (int i = 0; i < num_bboxes; ++i) {
    NormalizedBBox decode_bbox;
    DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
               variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
    decode_bboxes->push_back(decode_bbox);
  }
}

void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_preds,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, vector<LabelBBox>* all_decode_bboxes) {
  //CHECK_EQ(all_loc_preds.size(), num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  for (int i = 0; i < num; ++i) {
    // Decode predictions into bboxes.
    LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        // Something bad happened if there are no predictions for current label.
        fprintf(stderr, "Could not find location predictions for label %d\n", label);
      }
      const vector<NormalizedBBox>& label_loc_preds =
          all_loc_preds[i].find(label)->second;
      DecodeBBoxes(prior_bboxes, prior_variances,
                   code_type, variance_encoded_in_target, clip,
                   label_loc_preds, &(decode_bboxes[label]));
    }
  }
}

void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
      const int top_k, vector<pair<Dtype, int> >* score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < num; ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::sort(score_index_vec->begin(), score_index_vec->end(),
            SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized = true) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_width, intersect_height;
  if (normalized) {
    intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
    intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;
  } else {
    intersect_width = intersect_bbox.xmax - intersect_bbox.xmin + 1;
    intersect_height = intersect_bbox.ymax - intersect_bbox.ymin + 1;
  }
  if (intersect_width > 0 && intersect_height > 0) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = BBoxSize(bbox1);
    float bbox2_size = BBoxSize(bbox2);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}

Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
      bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else {
    const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
    const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
    const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
    const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

    const Dtype inter_width = inter_xmax - inter_xmin;
    const Dtype inter_height = inter_ymax - inter_ymin;
    const Dtype inter_size = inter_width * inter_height;

    const Dtype bbox1_size = BBoxSize(bbox1);
    const Dtype bbox2_size = BBoxSize(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      vector<int>* indices) {
  // Sanity check.
  if(!CHECK_EQ(bboxes.size(), scores.size()))
      fprintf(stderr, "bboxes and scores have different size.\n");

  // Get top_k scores (with corresponding indices).
  vector<pair<float, int> > score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices) {
  // Get top_k scores (with corresponding indices).
  vector<pair<Dtype, int> > score_index_vec;
  GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap(bboxes + idx * 4, bboxes + kept_idx * 4);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

int DetectionOutput::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
  const Dtype* loc_data = bottom_blobs[0];
  const Dtype* conf_data = bottom_blobs[1];
  const Dtype* prior_data = bottom_blobs[2];
  const int num = 1; // only one image

  int num_priors_ = bottom_blobs[0].w/4;
  bool share_location_ = true;
  int num_classes_ = num_classes;
  int num_loc_classes_ = share_location_ ? 1 : num_classes_;
  int background_label_id_ = 0;
  CodeType code_type_ = PriorBoxParameter_CodeType_CENTER_SIZE;
  float confidence_threshold_ = confidence_threshold;
  float nms_threshold_ = nms_threshold;
  int top_k_ = nms_top_k;
  int keep_top_k_ = keep_top_k;
  bool variance_encoded_in_target_ = false;
  int eta_ = 1;

  // Retrieve all location predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                    share_location_, &all_loc_preds);

  // Retrieve all confidences.
  vector<map<int, vector<float> > > all_conf_scores;
  GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                      &all_conf_scores);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Decode all loc predictions to bboxes.
  vector<LabelBBox> all_decode_bboxes;
  const bool clip_bbox = false;
  DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                  share_location_, num_loc_classes_, background_label_id_,
                  code_type_, variance_encoded_in_target_, clip_bbox,
                  &all_decode_bboxes);

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    map<int, vector<int> > indices;
    int num_det = 0;
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        fprintf(stderr, "Could not find confidence predictions for label %d\n",c);
      }
      const vector<float>& scores = conf_scores.find(c)->second;
      int label = share_location_ ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        fprintf(stderr,  "Could not find location predictions for label %d\n",label);
        continue;
      }
      const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
      ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_, eta_,
          top_k_, &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        if (conf_scores.find(label) == conf_scores.end()) {
          // Something bad happened for current label.
          fprintf(stderr, "Could not find location predictions for %d\n",label);
          continue;
        }
        const vector<float>& scores = conf_scores.find(label)->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          CHECK_LT(idx, scores.size());
          score_index_pairs.push_back(std::make_pair(
                  scores[idx], std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(7);

  //printf("%d %d %d %d %d\n", num_kept, top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
  Dtype* top_data;
  if (num_kept == 0) {
    fprintf(stderr, "Couldn't find any detections\n");
    top_shape[2] = num;
    Mat& top_blob = top_blobs[0];
    top_blob.create(top_shape[1], top_shape[2], top_shape[3]);
    top_data = top_blob;
    // top[0]->Reshape(top_shape);
    // top_data = top[0]->mutable_cpu_data();
    // caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      top_data += 7;
    }
    //printf("top_data: %d %d %d", top_blob.w, top_blob.h, top_blob.c);
  } else {
    // top[0]->Reshape(top_shape);
    // top_data = top[0]->mutable_cpu_data();
    Mat& top_blob = top_blobs[0];
    top_blob.create(top_shape[3], top_shape[2]);
    top_data = top_blob;
    //printf("top_data: %d %d %d", top_blob.w, top_blob.h, top_blob.c);
  }

  int count = 0;
  int i = 0;
  for (int i = 0; i < num; ++i) {
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      if (conf_scores.find(label) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        fprintf(stderr, "Could not find confidence predictions for %d\n");
        continue;
      }
      const vector<float>& scores = conf_scores.find(label)->second;
      int loc_label = share_location_ ? -1 : label;
      if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        fprintf(stderr, "Could not find location predictions for %d\n", loc_label);
        continue;
      }
      const vector<NormalizedBBox>& bboxes =
          decode_bboxes.find(loc_label)->second;
      vector<int>& indices = it->second;
      //printf("%d\n", indices.size());
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * 7] = i;
        top_data[count * 7 + 1] = label;
        top_data[count * 7 + 2] = scores[idx];
        const NormalizedBBox& bbox = bboxes[idx];
        top_data[count * 7 + 3] = bbox.xmin;
        top_data[count * 7 + 4] = bbox.ymin;
        top_data[count * 7 + 5] = bbox.xmax;
        top_data[count * 7 + 6] = bbox.ymax;
        ++count;
        //printf("%d %d %f %f %f %f %f \n",i, label, scores[idx], bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax());
      }
    }
  }
  //printf("\n");
    return 0;
}

} // namespace ncnn
