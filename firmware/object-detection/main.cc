#include <cstdio>
#include <vector>
#include <cmath>

#include "libs/base/filesystem.h"
#include "libs/base/http_server.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/freertos_kernel/include/semphr.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "metadata.hpp"

#define ENABLE_HTTP_SERVER 1
#define DEBUG 1

namespace coralmicro {
namespace {

// Image result struct
typedef struct {
  std::string info;
  std::vector<uint8_t> *jpeg;
} ImgResult;

// Bounding box struct
typedef struct {
  float id;
  float score;
  float ymin;
  float xmin;
  float ymax;
  float xmax;
} BBox;

// Camera settings
constexpr auto camRotation = CameraRotation::k270; // Default: CameraRotation::k0

// Globals
constexpr char kIndexFileName[] = "/index.html";
constexpr char kCameraStreamUrlPrefix[] = "/camera_stream";
constexpr char kBoundingBoxPrefix[] = "/bboxes";
constexpr char kModelPath[] =
    "/../../exported_models/apple_detector/model_int8_edgetpu.tflite";
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
static std::vector<uint8_t> *img_ptr;
static int img_width;
static int img_height;
static constexpr float score_threshold = 0.5f; //def 0.5
static constexpr float iou_threshold = 0.3f; //def 0.3
static constexpr size_t max_bboxes = 5; //def 5
static constexpr unsigned int bbox_buf_size = 100 + (max_bboxes * 200) + 1;
static char bbox_buf[bbox_buf_size];


/*******************************************************************************
 * Functions
 */

void Blink(unsigned int num, unsigned int delay_ms);
bool CalculateAnchorBox(unsigned int idx, float *anchor);
float CalculateIOU(BBox *bbox1, BBox *bbox2);

/**
 * Loop forever taking images from the camera and performing inference
 */
[[noreturn]] void Main() {

  // Initialize camera
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

  // Used for calculating FPS
  unsigned long dtime;
  unsigned long timestamp;
  unsigned long timestamp_prev = xTaskGetTickCount() * 
    (1000 / configTICK_RATE_HZ);

  // x_center, y_center, w, h
  float anchor[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Load model
  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  // Initialize TPU
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice(PerformanceMode::kHigh);
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }

  // Initialize ops
  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  // Initialize TFLM interpreter for inference
  tflite::MicroInterpreter interpreter(
    tflite::GetModel(model.data()), 
    resolver,
    tensor_arena, 
    kTensorArenaSize,
    &error_reporter
  );
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  // Check model input tensor size
  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  // Configure model inputs and outputs
  auto* input_tensor = interpreter.input_tensor(0);
  img_height = input_tensor->dims->data[1];
  img_width = input_tensor->dims->data[2];
  img_ptr = new std::vector<uint8>(img_height * img_width * 
    CameraFormatBpp(CameraFormat::kRgb));
  std::vector<tensorflow::Object> results;

  // Get output tensor shapes
  TfLiteTensor* tensor_bboxes = interpreter.output_tensor(0);
  TfLiteTensor* tensor_scores = interpreter.output_tensor(1);
  unsigned int num_boxes = tensor_bboxes->dims->data[1];
  unsigned int num_coords = tensor_bboxes->dims->data[2];
  unsigned int num_classes = tensor_scores->dims->data[2];

  // Get quantization parameters
  const float input_scale = input_tensor->params.scale;
  const int input_zero_point = input_tensor->params.zero_point;
  const float locs_scale = tensor_bboxes->params.scale;
  const int locs_zero_point = tensor_bboxes->params.zero_point;
  const float scores_scale = tensor_scores->params.scale;
  const int scores_zero_point = tensor_scores->params.zero_point;

  // Convert threshold to fixed point
  uint8_t score_threshold_quantized = 
    static_cast<uint8_t>(score_threshold * 256);

  // Print input/output details
#if DEBUG
  printf("num_boxes: %d\r\n", num_boxes);
  printf("num_coords: %d\r\n", num_coords);
  printf("num_classes: %d\r\n", num_classes);
  printf("bytes in tensor_bboxes: %d\r\n", tensor_bboxes->bytes);
  if (tensor_scores->data.data == nullptr) {
    printf("tensor_scores.data is empty!\r\n");
  }
  printf("input_scale: %f\r\n", input_scale);
  printf("input_zero_point: %d\r\n", input_zero_point);
  printf("locs_scale: %f\r\n", locs_scale);
  printf("locs_zero_point: %d\r\n", locs_zero_point);
  printf("scores_scale: %f\r\n", scores_scale);
  printf("scores_zero_point: %d\r\n", scores_zero_point);
  printf("score_threshold_quantized: %d\r\n", score_threshold_quantized);
#endif

  // Do forever
  while (true) {

    std::vector<std::vector<float>> bbox_list;

    // Calculate time between inferences
    timestamp = xTaskGetTickCount() * (1000 / configTICK_RATE_HZ);
    dtime = timestamp - timestamp_prev;
    timestamp_prev = timestamp;

    // Turn status LED on to let the user know we're taking a photo
    // LedSet(Led::kUser, true);


    // Configure camera image
    CameraFrameFormat fmt{
      CameraFormat::kRgb,   
      CameraFilterMethod::kBilinear,
      camRotation,
      img_height,
      img_width,         
      false,            // Preserve ratio
      img_ptr->data(),  // Where the image is saved
      true              // Auto white balance
    };

    // Take a photo
    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
      printf("ERROR: Could not capture frame from camera\r\n");
      continue;
    }    

    // Turn status LED off to let the user know we're done taking a photo
    // LedSet(Led::kUser, false);

    // Copy image to input tensor (~6 ms)
    std::memcpy(
      tflite::GetTensorData<uint8_t>(input_tensor), 
      img_ptr->data(),
      img_ptr->size()
    );

    // Perform inference (~65 ms)
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("ERROR: Inference failed\r\n");
      continue;
    }

    // Get data
    uint8_t *scores = tensor_scores->data.uint8;
    uint8_t *raw_boxes = tensor_bboxes->data.uint8;

    // Find bounding boxes with scores above threshold
    for (unsigned int i = 0; i < metadata::num_anchors; ++i) {
      for (unsigned int c = 0; c < num_classes; ++c) {
          
        // Only keep boxes above a particular score threshold
        if (scores[i * num_classes + c] > score_threshold_quantized) {
            
          // Calculate anchor box coordinates based on index
          if (!CalculateAnchorBox(i, anchor)) {
            printf("ERROR: Could not calculate anchor box\r\n");
            continue;
          }

          // Assume raw box output tensor is given as YXHW format
          float y_center = raw_boxes[(i * num_coords) + 0];
          float x_center = raw_boxes[(i * num_coords) + 1];
          float h = raw_boxes[(i * num_coords) + 2];
          float w = raw_boxes[(i * num_coords) + 3];

          // De-quantize the output boxes (move to x, y, w, h format)
          x_center = (x_center - locs_zero_point) * locs_scale;
          y_center = (y_center - locs_zero_point) * locs_scale;
          w = (w - locs_zero_point) * locs_scale;
          h = (h - locs_zero_point) * locs_scale;

          // Scale the output boxes from anchor coordinates
          x_center = x_center / metadata::x_scale * anchor[2] + anchor[0];
          y_center = y_center / metadata::y_scale * anchor[3] + anchor[1];
          if (metadata::apply_exp_scaling) {
            w = exp(w / metadata::w_scale) * anchor[2];
            h = exp(h / metadata::h_scale) * anchor[3];
          } else {
            w = w / metadata::w_scale * anchor[2];
            h = h / metadata::h_scale * anchor[3];
          }

          // Convert box coordinates to top left and bottom right
          float x_min = x_center - w / 2.0f;
          float y_min = y_center - h / 2.0f;
          float x_max = x_center + w / 2.0f;
          float y_max = y_center + h / 2.0f;

          // Clamp values to between 0 and 1
          x_min = std::max(std::min(x_min, 1.0f), 0.0f);
          y_min = std::max(std::min(y_min, 1.0f), 0.0f);
          x_max = std::max(std::min(x_max, 1.0f), 0.0f);
          y_max = std::max(std::min(y_max, 1.0f), 0.0f);

          // De-quantize the score
          float score = (scores[(i * num_classes) + c] - scores_zero_point) *
            scores_scale;

          // Add to list of bboxes
          bbox_list.push_back({(float)c, score,
            y_min, x_min, y_max, x_max});
          }
      }
    }

    // Sort bboxes by score
    std::sort(bbox_list.begin(), bbox_list.end(), 
      [](const std::vector<float>& a, const std::vector<float>& b) {
        return a[1] > b[1];
      }
    );

    // Perform non-maximum suppression
    for (unsigned int i = 0; i < bbox_list.size(); ++i) {
      for (unsigned int j = i + 1; j < bbox_list.size(); ++j) {
        BBox bbox1 = {
          bbox_list[i][0], 
          bbox_list[i][1], 
          bbox_list[i][2], 
          bbox_list[i][3], 
          bbox_list[i][4], 
          bbox_list[i][5]
        };
        BBox bbox2 = {
          bbox_list[j][0], 
          bbox_list[j][1], 
          bbox_list[j][2], 
          bbox_list[j][3], 
          bbox_list[j][4], 
          bbox_list[j][5]
        };
        float iou = CalculateIOU(&bbox1, &bbox2);
        if (iou > iou_threshold) {
          bbox_list.erase(bbox_list.begin() + j);
          --j;
        }
      }
    }

    // Determine number of bboxes to send
    size_t num_bboxes_output = (bbox_list.size() < max_bboxes) ? 
      bbox_list.size() : max_bboxes; 

    // Convert top k bboxes to JSON string
    std::string bbox_string = "{\"dtime\": " + std::to_string(dtime) + ", ";
      bbox_string += "\"bboxes\": [";
      for (unsigned int i = 0; i < num_bboxes_output; ++i) {
        int class_id = static_cast<int>(bbox_list[i][0]);
        bbox_string += "{\"id\": " + std::to_string(class_id) + ", ";
        bbox_string += "\"score\": " + std::to_string(bbox_list[i][1]) + ", ";
        bbox_string += "\"xmin\": " + std::to_string(bbox_list[i][3]) + ", ";
        bbox_string += "\"ymin\": " + std::to_string(bbox_list[i][2]) + ", ";
        bbox_string += "\"xmax\": " + std::to_string(bbox_list[i][5]) + ", ";
        bbox_string += "\"ymax\": " + std::to_string(bbox_list[i][4]) + "}";
        if (i != num_bboxes_output - 1) {
          bbox_string += ", ";
        }
      }
      bbox_string += "]}";
      
    bool person_detected = num_bboxes_output > 0;
    LedSet(coralmicro::Led::kUser, person_detected);

    // Check length of JSON string
    if (bbox_string.length() > bbox_buf_size) {
      printf("ERROR: Bounding box JSON string too long\r\n");
      continue;
    }


    // Print bounding box JSON string
    printf("%s\r\n", bbox_string.c_str());

    // Sleep to let other tasks run
    // vTaskDelay(pdMS_TO_TICKS(10));
  }
}

/**
* Blink error codes on the status LED
*/
void Blink(unsigned int num, unsigned int delay_ms) {
  static bool on = false;
  for (unsigned int i = 0; i < num * 2; i++) {
    on = !on;
    coralmicro::LedSet(Led::kStatus, on);
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
  }
}

/**
 * Calculate anchor box coordinates based on index and metadata
 */
bool CalculateAnchorBox(unsigned int idx, float *anchor) {

  unsigned int sector = 0;
  float x_idx;
  float x_center;
  float y_idx;
  float y_center;
  float w;
  float h;

  // Check index
  if (idx >= metadata::num_anchors) {
    return false;
  }

  // Find the sector that the index belongs in
  for (unsigned int s = 0; s < metadata::num_sectors; ++s) {
    if (idx >= metadata::reset_idxs[s]) {
      sector = s;
    }
  }

  // Find the X centert
  x_idx = (idx % metadata::num_xs_per_y[sector]) / 
    metadata::num_anchors_per_coord;
  x_center = (metadata::x_strides[sector] / 2.0f) + 
    (x_idx * metadata::x_strides[sector]);

  // Find the Y center
  y_idx = (idx - metadata::reset_idxs[sector]) / 
    metadata::num_xs_per_y[sector];
  y_center = (metadata::y_strides[sector] / 2.0f) +
    (y_idx * metadata::y_strides[sector]);

  // Find the width and height
  w = metadata::widths[sector][idx % metadata::num_anchors_per_coord];
  h = metadata::heights[sector][idx % metadata::num_anchors_per_coord];

  // Save anchor box coordinates
  anchor[0] = x_center;
  anchor[1] = y_center;
  anchor[2] = w;
  anchor[3] = h;

  return true;
}

/**
 * Calculate intersection over union (IOU) between two bounding boxes
 */
float CalculateIOU(BBox *bbox1, BBox *bbox2) {

  // Calculate intersection
  float x_min = std::max(bbox1->xmin, bbox2->xmin);
  float y_min = std::max(bbox1->ymin, bbox2->ymin);
  float x_max = std::min(bbox1->xmax, bbox2->xmax);
  float y_max = std::min(bbox1->ymax, bbox2->ymax);
  float intersection = std::max(0.0f, x_max - x_min) * 
    std::max(0.0f, y_max - y_min);

  // Calculate union
  float bbox1_area = (bbox1->xmax - bbox1->xmin) * 
    (bbox1->ymax - bbox1->ymin);
  float bbox2_area = (bbox2->xmax - bbox2->xmin) * 
    (bbox2->ymax - bbox2->ymin);
  float union_area = bbox1_area + bbox2_area - intersection;

  // Calculate IOU
  float iou = 0.0f;
  if (union_area > 0.0f) {
    iou = intersection / union_area;
  }

  return iou;
}

}  // namespace
}  // namespace coralmicro

/**
 * Entrypoint
 */
extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
