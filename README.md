# Tiny End-to-End Collaborative Learning for Occlusion-Robust Object Detection.

This project explores collaborative object detection on resource-constrained devices.
It includes five main components:

1. **Model Training:** A lightweight detector (MCUNet + YOLOv2) trained on the VOC2012 dataset.
2. **Quantisation:** Model compression using TensorFlow Lite for MCU deployment.
3. **Collaborative Inference:** Multi-view fusion at both feature and decision levels.
4. **Decentralised Federated Learning (DFL):** Proof-of-concept implementation using FedAvg across nodes.
5. **Deployment:** On-device inference and Wi-Fi-based communication on the Coral Dev Board Micro.

## Background  
Ultra–low-end MCUs face three primary challenges in real-world object detection:
1. limited computational capacity,
2. communication overhead in collaborative settings, and
3. accuracy degradation under occlusion and domain shift.

This work addresses these challenges through a compact model design based on **MCUNet and YOLOv2**, combined with **TFLite quantisation** for embedded deployment. 

Two collaborative inference schemes were implemented and compared:
- **Feature-level fusion:** exchanges intermediate feature maps across devices.
- **Decision-level fusion (WBF):** merges detection results across multiple views.

We also evaluate the trade-offs between accuracy improvements and communication cost as additional views are introduced.

Finally, **DFL with FedAvg** was implemented to examine distributed learning stability under non-iid local data.


## Project Structure 
```text
├── README.md
├── docs                  
├── models                     # Model architecture
│   ├── dethead
│   │   ├── mvdet_test.py
│   │   ├── mvyolodet.py
│   │   └── yolodet.py
│   └── mcunetYolo
│       ├── tinynas
│       └── utils
├── dataset                    # VOC and CO3D datasets (with occlusion simulation)
├── vocmain                    # Pretraining (MCUNet + YOLOv2)
│   ├── main.py
│   └── trainer.py
├── quanmain                   # Quantisation and performance profiling
│   ├── main.py
│   ├── model_config
│   │   └── mcunetYolo_config.json
│   ├── profiling.py
│   ├── verification.py
│   └── visualization.py
├── colmain                    # Collaborative inference (feature & decision fusion)
│   ├── decision_fusion
│   │   └── fusion.py
│   ├── feature_fusion
│   │   ├── data.py
│   │   ├── main.py
│   │   └── trainer.py
│   ├── main
│   │   ├── twoviews.py
│   │   └── threeviews.py
│   └── visualization.py
├── dflmain                    # Decentralised federated learning (FedAvg prototype)
│   ├── fedavg_logs.json
│   └── main.py
├── google-coral-micro-object-detection  # MCU deployment and Wi-Fi transmission
├── utils                      # Utility scripts (weight conversion, feature extraction, etc.)
├── validation                 # Bounding box metrics (mAP@0.5), evaluation, visualisation
├── config.py
└── run_vocmain.sh
```

## Model Experiment Usage 
- **Pretraining:** 
    ```bash
    python -m vocmain.main
    ```
    The `Trainer` class accepts parameters such as anchor count, number of classes, image size, and grid size.

- **Quantisation:** 
    ```bash
    python -m quanmain.main
    ```
    Quantises the model and evaluates accuracy (mAP@0.5), size, memory footprint, and inference latency.

- **Collaborative Inference**
    ```bash
    python -m colmain.main.twoviews
    python -m colmain.main.threeviews
    ```
    - Two-view mode: Compares single-view baseline, feature-level, and decision-level (WBF) fusion.
    - Three-view mode: Evaluates incremental improvements when introducing a third view.

- **Decentralised Federated Learning**
    ```bash
    python -m dflmain.main
    ```
    - Runs a two-node FedAvg implementation for decentralised training.

## MCU Deployment Usage (Coral Dev Board Micro)

### Clone the Repository (including Coral Micro SDK)
```bash
git clone https://github.com/DannyCheng711/Individual-Project.git
cd Individual-Project
git clone https://github.com/google-coral/coralmicro.git firmware/coralmicro
```

### Directory Structure
```text

google-coral-micro-object-detection/
├── firmware/
│   ├── coralmicro/                  # Coral Micro SDK (RTOS, drivers, Wi-Fi stack, build system)
│   └── object-detection-http/       # MCU deployment code (currently being refactored)
├── notebooks/
│   ├── generate_rgb_image.ipynb     # Preprocesses inference images into .rgb format with scaling considerations
│   └── tflite-runtime-test-object-detection.ipynb  
│                                    # Validates TFLite runtime behaviour and checks inference logic
└── pyserver/
    └── udp_server.py                # Host-side Python UDP server for receiving bounding-box messages from MCUs

```

### Build Firmware
```bash
cd firmware/object-detection-http
rm -rf build
cmake -B build/ -S .
make -C build -j4
```
You should see output ending with a compiled image:
```bash
build/
└── coralmicro-app    # Firmware ELF to flash onto the Coral Dev Board Micro
```

### Flash the Coral Dev Board Micro
```bash
cd firmware/object-detection-http
python3 ../coralmicro/scripts/flashtool.py \
  --build_dir build/ \
  --elf_path build/coralmicro-app \
  --wifi_ssid "YOUR_SSID" \
  --wifi_psk "YOUR_PASSWORD"
```

This step contains flashing firmware, writing Wi-Fi credentials and rebooting into application mode. If successful, your device should connect to the network and wait for incoming UDP messages.

### Start the Host-Side Python UDP Server
Navigate to your server folder:
```bash
cd pyserver
python -m udp_server
```
This server listens for MCU detections and prints or logs incoming messages such as bounding boxes, scores, and latency.

### Additional MCU Documentation
Detailed MCU-side behaviour, feature descriptions, and example logs are provided in [google-coral-micro-object-detection/README.md￼](google-coral-micro-object-detection/README.md)


## Key Findings
- **MCUNet + YOLOv2** achieved the best mAP among lightweight baselines (0.2575) with excellent memory efficiency.
- **Quantisation** reduced model size and peak RAM usage by ≈71 % and 83 % respectively, with only a –0.003 mAP drop.
- **Decision-level fusion (WBF)** yielded consistent gains under occlusion (+0.2736 mAP at 30–50 % occlusion).
- **Feature-level fusion** showed potential (+0.2625 mAP) but instability (drops up to –0.1769 mAP).
- **Three-view fusion** improved performance by +0.07–0.15 mAP on average, up to +0.3827 mAP in asymmetric occlusion.
- Each additional view added ≈6 KB of Wi-Fi payload; packet loss appeared near the 1500 B MTU limit.
- **DFL (FedAvg)** achieved stable convergence without central coordination but exhibited high absolute loss (~23 800) under non-iid data.

## Conclusion

Decision-level fusion (WBF) is a practical and robust strategy for handling occlusion.
It requires no backbone retraining, minimal calibration, and offers low communication overhead, which is ideal for **edge-based computer vision systems** such as post-disaster search and rescue.

## Documentation
- [Project Slides (PDF)](docs/tiny_collab_learning.pdf)

## Notes
The project report details all methods, evaluation metrics, and analysis.  
Due to project constraints, some scripts are experimental prototypes and may require adaptation for different environments.  

---

*Imperial College London, MSc Computing Individual Project (2025).*
