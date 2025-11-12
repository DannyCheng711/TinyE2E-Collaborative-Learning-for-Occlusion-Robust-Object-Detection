// Filename: yolo_mcunet_metadata.hpp
// Template for YOLO MCUNet model metadata
// You need to replace these values with your specific model parameters

#ifndef YOLO_MCUNET_METADATA_HPP
#define YOLO_MCUNET_METADATA_HPP

namespace metadata {

    // Number of total anchor boxes across all detection layers
    constexpr unsigned int num_anchors = 5;
    
    // Anchor widths and heights (from metadata.json)
    constexpr float anchor_widths[num_anchors] = {
        0.27662376191, 0.85550725756, 1.34005545401, 2.11501002585, 4.09882813018
    };
    constexpr float anchor_heights[num_anchors] = {
        0.64734107032, 1.77290703713, 3.0700760405, 4.14907654168, 4.28586514869
    };

    
    // Coordinate scaling factors (usually 1.0 for YOLO)
    constexpr float x_scale = 10.0;
    constexpr float y_scale = 10.0;
    constexpr float w_scale = 5.0;
    constexpr float h_scale = 5.0;
    
    // Whether to apply exponential scaling to width/height predictions
    constexpr bool apply_exp_scaling = true;

    // Number of values per keypoint (typically 4 for bounding boxes: x, y, w, h)
    constexpr unsigned int num_values_per_keypoint = 4;

    // Grid size (if fixed, otherwise this can be passed dynamically)
    constexpr unsigned int grid_size = 5;

    // Image size (if fixed, otherwise this can be passed dynamically)
    constexpr unsigned int image_size = 160;
}

#endif // YOLO_MCUNET_METADATA_HPP
