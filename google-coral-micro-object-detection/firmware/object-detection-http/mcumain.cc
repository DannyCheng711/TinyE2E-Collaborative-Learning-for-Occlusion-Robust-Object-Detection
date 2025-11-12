#include <cstdio>
#include <vector>
#include <cmath>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "libs/base/wifi.h"
#include "libs/base/network.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/freertos_kernel/include/semphr.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

// #include "metadata.hpp"
#include "yolo_mcunet_metadata.hpp"

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
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;
        float id;
    } BBox;

    constexpr char kModelPath[] = "/yolo_mcunet_model.tflite";
    constexpr int kTensorArenaSize = 1024 * 1024; // 1MB
    // 411_56081_108572_28_int8_160
    // constexpr char kImagePath[] = "examples/images/411_56081_108572_28_int8_160.rgb";
    STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize); 

    static SemaphoreHandle_t img_mutex;
    static int img_width;
    static int img_height;
    static constexpr float score_threshold = 0.05f;
    static constexpr float iou_threshold = 0.3f;
    static constexpr size_t max_bboxes = 60; // 5 
    static constexpr unsigned int bbox_buf_size = 100 + (max_bboxes * 200) + 1;
    static char bbox_buf[bbox_buf_size];

    // Performance metrics
    static unsigned long total_inference_time = 0;
    static unsigned long min_inference_time = ULONG_MAX;
    static unsigned long max_inference_time = 0;
    static unsigned long inference_count = 0;
    static unsigned long failed_inferences = 0;
    static unsigned long invoke_errors = 0;
    static unsigned long wifi_msg_id = 0;
    static unsigned long wifi_msg_sent = 0;
    static unsigned long wifi_send_errors = 0;

    // Ensure no padding for consistent binary format 
    // (align fields at 1-byte boundaries (no padding).)
    #pragma pack(push, 1) 
    struct BinaryBBox {
        float xmin, ymin, xmax, ymax, score;
        uint8_t id;
    };

    struct BinaryDetectionPacket {
        uint32_t msg_id;
        uint32_t total_expected;
        uint32_t dtime;
        uint32_t num_bboxes;
        uint32_t payload_size;
        char image_filename[32];  // Fixed size filename
        BinaryBBox bboxes[60];     // max_bboxes = 5 
    };
    #pragma pack(pop)

    /*******************************************************************************
    * Helper Functions
    */

    void Blink(unsigned int num, unsigned int delay_ms);
    float Sigmoid(float x);
    float CalculateIOU(BBox* bbox1, BBox* bbox2);
    void DecodeBBoxes(const uint8_t* raw_output, 
        std::vector<std::vector<float>>& bbox_list, float output_scale, int output_zero_point );
    
    bool SendBBoxViaWiFi(const char* json_data, const char* image_filename, 
        unsigned long msg_id, unsigned long dtime, 
        size_t num_bboxes, size_t payload_size, size_t total_expected);

    bool SendBBoxViaWiFiBinary(const std::vector<std::vector<float>>& bbox_list, const char* image_filename, 
        unsigned long msg_id, unsigned long dtime, 
        size_t num_bboxes, size_t total_expected);


    float Sigmoid(float x) {
        if (x >= 0) {
            return 1.0f / (1.0f + expf(-x));
        } else {
            float exp_x = expf(x);
            return exp_x / (1.0f + exp_x);
        }
    }

    // Pass container by reference
    
    void DecodeBBoxes(const uint8_t* raw_output, std::vector<std::vector<float>>& bbox_list, 
        float output_scale, int output_zero_point){

            // Dequantize and reshape the output 
            const int grid_size = metadata::grid_size;
            const int num_anchors = metadata::num_anchors;
            const int num_classes = 20;
            const int output_size = grid_size * grid_size * num_anchors * (5 + num_classes);

            // Store dequantized output
            std::vector<float> dequantized_output(output_size);
            for (int i = 0; i < output_size; ++i) {
                dequantized_output[i] = output_scale * (static_cast<float>(raw_output[i]) - output_zero_point);
            }

            for (int i = 0; i < grid_size; ++i) {
                for (int j = 0; j < grid_size; ++j){
                    for (int a = 0; a < num_anchors; ++ a){
                        // Calculate index in "flatten" array
                        int base_idx = (i * grid_size * num_anchors * (5 + num_classes)) +
                              (j * num_anchors * (5 + num_classes)) +
                              (a * (5 + num_classes));
                        
                        // Extract raw predictions
                        float tx = dequantized_output[base_idx + 0];
                        float ty = dequantized_output[base_idx + 1];
                        float tw = dequantized_output[base_idx + 2];
                        float th = dequantized_output[base_idx + 3];

                        float obj_logit = dequantized_output[base_idx + 4];

                        // Apply activation functions
                        float sig_tx = Sigmoid(tx);
                        float sig_ty = Sigmoid(ty);
                        float exp_tw = metadata::apply_exp_scaling ? expf(tw) : tw;
                        float exp_th = metadata::apply_exp_scaling ? expf(th) : th;
                        float obj_prob = Sigmoid(obj_logit);

                         // Compute normalized center coordinates and dimensions
                        float cx = (j + sig_tx) / grid_size;
                        float cy = (i + sig_ty) / grid_size;
                        float bw = (metadata::anchor_widths[a] * exp_tw) / grid_size;
                        float bh = (metadata::anchor_heights[a] * exp_th) / grid_size;
                        
                        // Convert to pixel coordinates
                        float x_min = (cx - bw / 2.0f) * metadata::image_size;
                        float y_min = (cy - bh / 2.0f) * metadata::image_size;
                        float x_max = (cx + bw / 2.0f) * metadata::image_size;
                        float y_max = (cy + bh / 2.0f) * metadata::image_size;
                        
                        // Clamp to image boundaries
                        x_min = std::max(0.0f, std::min(x_min, static_cast<float>(metadata::image_size)));
                        y_min = std::max(0.0f, std::min(y_min, static_cast<float>(metadata::image_size)));
                        x_max = std::max(0.0f, std::min(x_max, static_cast<float>(metadata::image_size)));
                        y_max = std::max(0.0f, std::min(y_max, static_cast<float>(metadata::image_size)));
                        
                        // Process class predictions
                        float best_cls_prob = 0.0f;
                        int best_cls_id = 0; 

                        // Apply softmax to class logits 
                        // Get max class logit as the predicted class 
                        float max_logit = dequantized_output[base_idx + 5];
                        for (int c = 1; c < num_classes ; c++){
                            max_logit = std::max(max_logit, dequantized_output[base_idx + 5 + c]);
                        }
                        // Get sum class logits
                        float sum_exp = 0.0f;
                        for (int c = 0; c < num_classes; ++c) {
                            sum_exp += expf(dequantized_output[base_idx + 5 + c] - max_logit);
                        }
                        
                        // Normalize and get class probability 
                        for (int c = 0; c < num_classes; ++c) {
                            float cls_prob = expf(dequantized_output[base_idx + 5 + c] - max_logit);
                            if (cls_prob > best_cls_prob) {
                                best_cls_prob = cls_prob;
                                best_cls_id = c; 
                            }
                        }
                        
                        // Compute final confidence
                        float final_conf = obj_prob * best_cls_prob;
                        
                        // Filter by confidence threshold
                        if (final_conf > score_threshold) {
                            bbox_list.push_back({x_min, y_min, x_max, y_max, static_cast<float>(best_cls_id), final_conf});
                        }
                    }
                }
            }
        }
    
    /* Calculate IOU between two bbox*/
    float CalculateIOU(BBox* bbox1, BBox* bbox2) {
        // Calculate intersection 
        float x_min = std::max(bbox1->xmin, bbox2->xmin);
        float y_min = std::max(bbox1->ymin, bbox2->ymin);
        float x_max = std::min(bbox1->xmax, bbox2->xmax);
        float y_max = std::min(bbox1->ymax, bbox2->ymax);
        float intersection = std::max(0.0f, x_max - x_min) * std::max(0.0f, y_max - y_min);
        
        // Calculate union
        float bbox1_area = (bbox1->xmax - bbox1->xmin) * (bbox1->ymax - bbox1->ymin);
        float bbox2_area = (bbox2->xmax - bbox2->xmin) * (bbox2->ymax - bbox2->ymin);
        float union_area = bbox1_area + bbox2_area - intersection;

        return (union_area > 0.0f) ? (intersection / union_area) : 0.0f; 

    }

    // Add this function before InferenceTask
    bool SendBBoxViaWiFi(const char* json_data, const char* image_filename, 
                        unsigned long msg_id, unsigned long dtime, 
                        size_t num_bboxes, size_t payload_size, size_t total_expected) {

        const char* target_ip = "192.168.0.14"; // "172.20.10.4";  
        constexpr int target_port = 5005;
        
        // Create comprehensive log message
        char wifi_log_buf[1024];
        int json_len = snprintf(wifi_log_buf, sizeof(wifi_log_buf),
            "{\"msg_id\":%lu,\"total_expected\":%lu,\"image\":\"%s\",\"dtime\":%lu,\"num_bboxes\":%lu,\"payload_size\":%lu,\"bboxes\":%s}",
            msg_id, (unsigned long)total_expected, image_filename, dtime,
            (unsigned long)num_bboxes, (unsigned long)payload_size,
            json_data
        );

        printf("=== DEBUG: JSON TO SEND ===\r\n");
        printf("JSON Length: %d bytes\r\n", json_len);
        printf("Buffer Size: %lu bytes\r\n", (unsigned long)sizeof(wifi_log_buf));
        
        if (json_len >= sizeof(wifi_log_buf)) {
            printf("ERROR: JSON TRUNCATED! Need %d bytes but have %lu\r\n", 
                json_len, (unsigned long)sizeof(wifi_log_buf));
        }
        
        printf("JSON Content:\r\n%s\r\n", wifi_log_buf);
        printf("===========================\r\n");




        // Send over Wi-Fi
        bool success = coralmicro::UdpSend(target_ip, target_port, wifi_log_buf, strlen(wifi_log_buf));
        
        if (success) {
            wifi_msg_sent++;
            printf("WiFi msg sent: ID=%lu, image=%s, size=%lu bytes\r\n", 
                msg_id, image_filename, (unsigned long)strlen(wifi_log_buf));
        } else {
            wifi_send_errors++;
            printf("ERROR: WiFi send failed for msg ID=%lu, image=%s\r\n", 
                msg_id, image_filename);
        }
        
        return success;
    }

    void SendTestMessageViaUDP() {
        const char* message = "Hello from Coral Dev Micro!";
        const char* target_ip = "192.168.0.14"; // "172.20.10.4";  
        constexpr int target_port = 5005;

        bool success = coralmicro::UdpSend(target_ip, target_port, message, strlen(message));
        if (success) {
            printf("UDP message sent successfully!\r\n");
        } else {
            printf("Failed to send UDP message\r\n");
        }
    }

    bool SendBBoxViaWiFiBinary(const std::vector<std::vector<float>>& bbox_list, const char* image_filename, 
                     unsigned long msg_id, unsigned long dtime, 
                     size_t num_bboxes, size_t total_expected) {

        const char* target_ip = "192.168.0.14";
        constexpr int target_port = 5005;
        
        // Create binary packet
        BinaryDetectionPacket packet = {0};
        packet.msg_id = msg_id;
        packet.total_expected = (uint32_t)total_expected;
        packet.dtime = dtime;
        packet.num_bboxes = (uint32_t)num_bboxes;
        packet.payload_size = sizeof(packet);
        
        // Copy filename (truncate if too long)
        strncpy(packet.image_filename, image_filename, sizeof(packet.image_filename) - 1);
        packet.image_filename[sizeof(packet.image_filename) - 1] = '\0';
        
        // Copy bounding boxes
        for (size_t i = 0; i < num_bboxes && i < max_bboxes; ++i) {
            packet.bboxes[i].xmin = bbox_list[i][0];
            packet.bboxes[i].ymin = bbox_list[i][1];
            packet.bboxes[i].xmax = bbox_list[i][2];
            packet.bboxes[i].ymax = bbox_list[i][3];
            packet.bboxes[i].score = bbox_list[i][5];
            packet.bboxes[i].id = (uint8_t)bbox_list[i][4];
        }
        printf("=== DEBUG: BINARY PACKET ===\r\n");
        printf("Packet size: %lu bytes\r\n", (unsigned long)sizeof(packet));
        printf("msg_id: %lu, dtime: %lu, num_bboxes: %lu\r\n", 
            (unsigned long)packet.msg_id, (unsigned long)packet.dtime, (unsigned long)packet.num_bboxes);
        printf("image: %s\r\n", packet.image_filename);
        printf("============================\r\n");

        // Send binary packet
        bool success = coralmicro::UdpSend(target_ip, target_port, (char*)&packet, sizeof(packet));
        
        if (success) {
            wifi_msg_sent++;
            printf("Binary msg sent: ID=%lu, image=%s, size=%lu bytes\r\n", 
                msg_id, image_filename, (unsigned long)sizeof(packet));
        } else {
            wifi_send_errors++;
            printf("ERROR: Binary send failed for msg ID=%lu, image=%s\r\n", 
                msg_id, image_filename);
        }
        
        return success;
    }


    /**
    * Loop forever performing inference
    */
    [[noreturn]] void InferenceTask(void* param) {

        printf("YOLO MCUNet Object Detection Example with CPU!\r\n");
        // Turn on Status LED to show the board is on.
        LedSet(Led::kStatus, true);

        // Used for calculating FPS
        // printf("Set Time Config!\r\n");
        unsigned long dtime;
        unsigned long timestamp;
        unsigned long timestamp_prev = xTaskGetTickCount() * 
            (1000 / configTICK_RATE_HZ);

        // Load model
        // printf("Loading model...\r\n");
        std::vector<uint8_t> model;
        if (!LfsReadFile(kModelPath, &model)) {
            printf("ERROR: Failed to load %s\r\n", kModelPath);
            xSemaphoreGive(img_mutex); // Ensure semaphore is released
            vTaskSuspend(nullptr); 
        }
        // printf("Model loaded successfully!\r\n");

        // Check model schema version
        // printf("Check model schema version!\r\n");
        auto model_data = tflite::GetModel(model.data());
        if (model_data->version() != TFLITE_SCHEMA_VERSION) {
            printf(
                "Model provided is schema version %lu not equal to supported version "
                "%d\r\n",
                model_data->version(), TFLITE_SCHEMA_VERSION);
            vTaskSuspend(nullptr);
        }
        // printf("Model schema version is valid!\r\n");


        // Initialize TPU
        // auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
        // if (!tpu_context) {
        //   printf("ERROR: Failed to get EdgeTpu context\r\n");
        //   vTaskSuspend(nullptr);
        // }

        // Initialize ops
        // printf("Initializing TensorFlow Lite Micro ops...\r\n");
        tflite::MicroErrorReporter error_reporter;
        tflite::MicroMutableOpResolver<16> resolver;
        resolver.AddAdd();
        resolver.AddPad();
        resolver.AddConv2D();
        resolver.AddDepthwiseConv2D();
        resolver.AddReshape();
        resolver.AddAveragePool2D();
        resolver.AddLogistic(); // For sigmoid activation
        resolver.AddMul();
        resolver.AddConcatenation();
        resolver.AddSpaceToDepth();
        resolver.AddShape();
        resolver.AddStridedSlice(); 
        resolver.AddPack();
        // printf("Ops initialized successfully!\r\n");

        
        // Initialize TFLM interpreter for inference
        // printf("Initializing TensorFlow Lite Micro interpreter...\r\n");
        tflite::MicroInterpreter interpreter(
            model_data, resolver, tensor_arena, kTensorArenaSize, &error_reporter
        );
        if (interpreter.AllocateTensors() != kTfLiteOk) {
            printf("ERROR: AllocateTensors() failed\r\n");
            vTaskSuspend(nullptr);
        }
        // printf("Interpreter initialized successfully!\r\n");
        printf("=== TENSOR ARENA INFO ===\r\n");
        printf("Arena size: %d bytes (%.1f KB)\r\n", kTensorArenaSize, kTensorArenaSize / 1024.0f);
        // Get arena usage after allocation
        size_t arena_used = interpreter.arena_used_bytes();
        printf("Arena used: %lu bytes (%.1f KB)\r\n", (unsigned long)arena_used, arena_used / 1024.0f);
        printf("Arena utilization: %.1f%%\r\n", (float)arena_used / kTensorArenaSize * 100.0f);
        printf("Arena unused: %lu bytes (%.1f KB)\r\n", (unsigned long)(kTensorArenaSize - arena_used), (kTensorArenaSize - arena_used) / 1024.0f);

        // Calculate known tensor sizes for comparison
        auto* input = interpreter.input_tensor(0);
        auto* output = interpreter.output_tensor(0);
        size_t input_bytes = input->bytes;
        size_t output_bytes = output->bytes;
        printf("Input tensor: %lu bytes\r\n", (unsigned long)input_bytes);
        printf("Output tensor: %lu bytes\r\n", (unsigned long)output_bytes);
        printf("Intermediate/scratch: %lu bytes\r\n", (unsigned long)(arena_used - input_bytes - output_bytes));
        printf("========================\r\n");


        // Check model input tensor size
        // printf("Checking model input tensor size...\r\n");
        if (interpreter.inputs().size() != 1) {
            printf("ERROR: Model must have only one input tensor\r\n");
            vTaskSuspend(nullptr);
        }
        // printf("Model input tensor size is valid!\r\n");


        // Configure model inputs and outputs
        auto* input_tensor = interpreter.input_tensor(0);// Get the first input tensor
        auto* output_tensor = interpreter.output_tensor(0); // Get the first output tensor

        img_height = input_tensor->dims->data[1];
        img_width = input_tensor->dims->data[2];    
        // Get quantization parameters 
        const float output_scale = output_tensor->params.scale;
        const int output_zero_point = output_tensor->params.zero_point;
        
        //printf("Input dimensions: %dx%d\r\n", img_width, img_height);
        //printf("Output scale: %f, zero point: %d\r\n", output_scale, output_zero_point);


        // List all files in the images directory
        std::vector<std::string> image_files;
        //if (!LfsReadFile(kImagePath, image_data.data(), image_data.size())) {
        if (!ListDirectory("examples/images", &image_files)) {
            printf("ERROR: Failed to list images directory\r\n");
            vTaskSuspend(nullptr);
        }

        // Filter for .rgb files
        std::vector<std::string> rgb_files;
        for (const auto& fname : image_files) {
            rgb_files.push_back("examples/images/" + fname);
        }

        printf("Found %lu RGB images.\r\n", (unsigned long)rgb_files.size());

        // while (true) {
        // Read all images in flash
        for (const auto& img_path : rgb_files) {
            printf("Processing image: %s\r\n", img_path.c_str());

            std::vector<uint8_t> image_data(img_width * img_height * 3);
            if (!LfsReadFile(img_path.c_str(), image_data.data(), image_data.size())) {
                printf("ERROR: Failed to read image file: %s\r\n", img_path.c_str());
                failed_inferences++;
                continue;     
            }

            std::vector<std::vector<float>> bbox_list;

            // Start timing inference
            unsigned long inference_start = xTaskGetTickCount() * (1000 / configTICK_RATE_HZ);
    
            // Calculate time between inferences
            timestamp = xTaskGetTickCount() * (1000 / configTICK_RATE_HZ);
            dtime = timestamp - timestamp_prev;
            timestamp_prev = timestamp;

            // Get frame from camera using the configuration we set (~38 ms)
            // printf("Waiting to take img_mutex...\r\n");
            if (xSemaphoreTake(img_mutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
                // printf("Semaphore taken successfully.\r\n");

                if (input_tensor->type == kTfLiteInt8) {
                    // printf("Input type is Int8 !");
                    std::memcpy(
                        tflite::GetTensorData<int8_t>(input_tensor),
                        image_data.data(),
                        image_data.size()
                    );
                    // printf("Image copied to input tensor.\r\n");
                } else {
                    printf("ERROR: Unsupported input tensor type\r\n");
                    failed_inferences++;
                    invoke_errors++;
                    Blink(5, 100); // Blink rapidly to indicate an error
                    vTaskSuspend(nullptr);
                }

                // Perform inference
                printf("Performing inference...\r\n");
                LedSet(Led::kUser, true); // Turn on User LED during inference
                if (interpreter.Invoke() != kTfLiteOk) {
                    printf("ERROR: Inference failed\r\n");
                    failed_inferences++;
                    invoke_errors++;
                    xSemaphoreGive(img_mutex);
                    LedSet(Led::kUser, false); // Turn off User LED on failure
                    continue;
                } 
                // printf("Inference completed successfully.\r\n");
                
                // End timing inference
                unsigned long inference_end = xTaskGetTickCount() * (1000 / configTICK_RATE_HZ);
                unsigned long inference_time = inference_end - inference_start;
                
                // Update timing statistics
                total_inference_time += inference_time;
                min_inference_time = std::min(min_inference_time, inference_time);
                max_inference_time = std::max(max_inference_time, inference_time);
                inference_count++;

                // Decode bounding boxes
                uint8_t *raw_output = output_tensor->data.uint8;
                // printf("Decoding bounding boxes...\r\n");
                DecodeBBoxes(raw_output, bbox_list, output_scale, output_zero_point);
                // printf("Decoding bounding boxes is successful!\r\n");
                xSemaphoreGive(img_mutex);
                // printf("Semaphore released.\r\n");

                // Turn off User LED after inference
                LedSet(Led::kUser, false);

                // Sort bboxes by score(highest first) 
                std::sort(bbox_list.begin(), bbox_list.end(),
                    [](const std::vector<float>& a, const std::vector<float>& b) {
                        return a[5] > b[5];  // Compare scores
                    }
                );

                // Perform non-maximum suppression 
                if (!bbox_list.empty()){
                    for (size_t i = 0; i < bbox_list.size(); ++i) {
                        for (size_t j = i + 1; j < bbox_list.size(); ++j){
                            // class id comparison
                            if (static_cast<int>(bbox_list[i][4]) != static_cast<int>(bbox_list[j][4])) {
                                continue;
                            }
                        
                            BBox bbox1 = {bbox_list[i][0], bbox_list[i][1], bbox_list[i][2], bbox_list[i][3], bbox_list[i][4], bbox_list[i][5]};
                            BBox bbox2 = {bbox_list[j][0], bbox_list[j][1], bbox_list[j][2], bbox_list[j][3], bbox_list[j][4], bbox_list[j][5]};
                            
                            float iou = CalculateIOU(&bbox1, &bbox2);
                            if (iou > iou_threshold) {
                                bbox_list.erase(bbox_list.begin() + j);
                                --j;
                            }
                        }
                    }
                } else {
                    printf("No bounding boxes detected\r\n");
                }

                // Limit number of output bboxes
                size_t num_bboxes_output = std::min(bbox_list.size(), max_bboxes);
                // size_t num_bboxes_output = bbox_list.size();
                
                // Enhanced logging with your format
                printf("=== INFERENCE METRICS ===\r\n");
                printf("image: %s\r\n", img_path.c_str());
                printf("inference_time: %lu ms\r\n", inference_time);
                printf("num_bboxes: %lu\r\n", static_cast<unsigned long>(num_bboxes_output));
                
                // dtime: 
                // Includes inference time + post-processing time + file I/O time

                printf("dtime: %lu, num_bboxes: %lu\n", dtime, static_cast<unsigned long>(num_bboxes_output));
                for (size_t i = 0; i < num_bboxes_output; ++i) {
                    int class_id = static_cast<int>(bbox_list[i][4]);
                    float score = bbox_list[i][5];
                    float xmin = bbox_list[i][0];
                    float ymin = bbox_list[i][1];
                    float xmax = bbox_list[i][2];
                    float ymax = bbox_list[i][3];

                    printf("bbox %lu: class=%d, score=%.2f, xmin=%.1f, ymin=%.1f, xmax=%.1f, ymax=%.1f\n",
                        static_cast<unsigned long>(i), class_id, score, xmin, ymin, xmax, ymax);
                }

                printf("=========================\r\n\r\n");

                // // Convert to JSON string
                
                // std::string bbox_string = "[";
                // for (size_t i = 0; i < num_bboxes_output; ++i) {
                //     int class_id = static_cast<int>(bbox_list[i][4]);
                //     char bbox_buf[128];
                //     snprintf(bbox_buf, sizeof(bbox_buf),
                //         "{\"id\": %d, \"score\": %.2f, \"xmin\": %.1f, \"ymin\": %.1f, \"xmax\": %.1f, \"ymax\": %.1f}",
                //         class_id, bbox_list[i][5], bbox_list[i][0], bbox_list[i][1], bbox_list[i][2], bbox_list[i][3]);
                //     bbox_string += bbox_buf;
                //     if (i != num_bboxes_output - 1) {
                //         bbox_string += ", ";
                //     }
                // }
                // bbox_string += "]";

                // Add increment message ID
                wifi_msg_id++;

                // Calculate payload characteristics
                // size_t payload_size = bbox_string.size();
                const char* image_filename = strrchr(img_path.c_str(), '/');
                image_filename = image_filename ? image_filename + 1 : img_path.c_str();

                // Send via Wi-Fi with image filename
                // SendBBoxViaWiFi(bbox_string.c_str(), image_filename, wifi_msg_id, 
                //                dtime, num_bboxes_output, payload_size, rgb_files.size());
               
                SendBBoxViaWiFiBinary(bbox_list, image_filename, wifi_msg_id, dtime, 
                    num_bboxes_output, rgb_files.size());


            } else {
                printf("ERROR: Timeout while waiting for img_mutex\r\n");
                failed_inferences++;
                continue;
            }
    }
     // Calculate statistics
    unsigned long avg_inference_time = (inference_count > 0) ? (total_inference_time / inference_count) : 0;
    unsigned long jitter = max_inference_time - min_inference_time;
    float miss_rate = (inference_count > 0) ? ((float)failed_inferences / (float)(inference_count + failed_inferences)) * 100.0f : 0.0f;
    
    printf("\r\n=== FINAL INFERENCE STATISTICS ===\r\n");
    printf("total_images_processed: %lu\r\n", inference_count);
    printf("total_failures: %lu\r\n", failed_inferences);
    printf("avg_latency: %lu ms\r\n", avg_inference_time);
    printf("min_latency: %lu ms\r\n", min_inference_time);
    printf("max_latency: %lu ms\r\n", max_inference_time);
    printf("jitter: %lu ms\r\n", jitter);
    printf("invoke_errors: %lu\r\n", invoke_errors);
    printf("miss_rate: %.2f%%\r\n", miss_rate);
    // Wi-Fi statistics
    printf("wifi_msgs_sent: %lu\r\n", wifi_msg_sent);
    printf("wifi_send_errors: %lu\r\n", wifi_send_errors);
    float wifi_success_rate = (wifi_msg_id > 0) ? ((float)wifi_msg_sent / (float)wifi_msg_id) * 100.0f : 0.0f;
    printf("wifi_success_rate: %.2f%%\r\n", wifi_success_rate);
    printf("===================================\r\n");

    vTaskSuspend(nullptr);
}
            
/**
 * Blink error codes on the status LED
 */
void Blink(unsigned int num, unsigned int delay_ms) {
    static bool on = false;
    for (unsigned int i = 0; i < num * 2; i++) {
        on = !on;
        LedSet(Led::kStatus, on);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
}


/*******************************************************************************
 * Main
 */

 
 void Main() {
    // Say hello 
    Blink(3, 500);
    #if DEBUG
    printf("YOLO MCUNet Object Detection Starting...\r\n");
    #endif

    // Try to connect to Wi-Fi
    printf("Turning on Wi-Fi...\r\n");
    if (!WiFiTurnOn(/*default_iface=*/true)) {
        printf("ERROR: Failed to turn on Wi-Fi\r\n");
        Blink(5, 200);
        return;
    }

    printf("Connecting to Wi-Fi...\r\n");
    if (!WiFiConnect()) {
        printf("ERROR: Failed to connect to Wi-Fi\r\n");
        Blink(4, 200);
        return;
    }

    // Success!
    const auto& our_ip_addr = WiFiGetIp();
    if (our_ip_addr.has_value()) {
        printf("DHCP succeeded, our IP is %s.\r\n", our_ip_addr.value().c_str());
    } else {
        printf("We didn't get an IP via DHCP, not progressing further.\r\n");
        return;
    }

    // Initialize mutexes
    img_mutex = xSemaphoreCreateMutex();
    if (img_mutex == NULL) {
        printf("ERROR: Failed to create mutexes\r\n");
        while (true) {
            Blink(2, 100);
        }
    }

    // Start inference task
    #if DEBUG
    printf("Starting inference task\r\n");
    #endif
    xTaskCreate(
        &InferenceTask,
        "InferenceTask",
        configMINIMAL_STACK_SIZE * 40,
        nullptr,
        kAppTaskPriority - 1,
        nullptr
    );

    // Main will go to sleep if the inference done
    vTaskSuspend(nullptr);

}

// void Main() {
//   // Blink to show board booted
//   Blink(3, 300);
//   printf("Starting Coral Dev Board Micro Wi-Fi + UDP test...\r\n");

//   // Try to connect to Wi-Fi
//   printf("Turning on Wi-Fi...\r\n");
//   if (!WiFiTurnOn(/*default_iface=*/true)) {
//     printf("ERROR: Failed to turn on Wi-Fi\r\n");
//     Blink(5, 200);
//     return;
//   }

//   printf("Connecting to Wi-Fi...\r\n");
//   if (!WiFiConnect()) {
//     printf("ERROR: Failed to connect to Wi-Fi\r\n");
//     Blink(4, 200);
//     return;
//   }

//   // Success!
//   const auto& our_ip_addr = WiFiGetIp();
//   if (our_ip_addr.has_value()) {
//     printf("DHCP succeeded, our IP is %s.\r\n", our_ip_addr.value().c_str());
//   } else {
//     printf("We didn't get an IP via DHCP, not progressing further.\r\n");
//     return;
//   }

//   // Send UDP test message
//   SendTestMessageViaUDP();

//   // Blink slowly to indicate success
//   while (true) {
//     Blink(1, 500);  // Single blink every 1 sec
//     vTaskDelay(pdMS_TO_TICKS(1000));
//   }
// }
    
// test
/*
void Main() {
    printf("Starting Coral Dev Board Micro demo...\n");

    while (true) {
        printf("Hello from Coral Dev Micro!\n");
        Blink(1, 500); // 1 blink, 500 ms each ON/OFF
        vTaskDelay(pdMS_TO_TICKS(1000)); // Wait 1 second
    }
}
*/

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
    (void)param;
    setvbuf(stdout, nullptr, _IONBF, 0); // no printf buffering
    printf("BOOT app_main\r\n");
    coralmicro::Main();
}



            
            
