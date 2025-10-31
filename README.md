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

## Usage 
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


## Key Findings
- **MCUNet + YOLOv2** achieved the best mAP among lightweight baselines (0.2575) with excellent memory efficiency.
- **Quantisation** reduced model size and peak RAM usage by ≈71 % and 83 % respectively, with only a –0.003 mAP drop.
- **Decision-level fusion (WBF)** yielded consistent gains under occlusion (+0.2736 mAP at 30–50 % occlusion).
- **Feature-level fusion** showed potential (+0.2625 mAP) but instability (drops up to –0.1769 mAP).
- **Three-view fusion** improved performance by +0.07–0.15 mAP on average, up to +0.3827 mAP in asymmetric occlusion.
- Each additional view added ≈6 KB of Wi-Fi payload; packet loss appeared near the 1500 B MTU limit.
- **DFL (FedAvg)** achieved stable convergence without central coordination but exhibited high absolute loss (~23 800) under non-iid data.

### Conclusion

Decision-level fusion (WBF) is a practical and robust strategy for handling occlusion.
It requires no backbone retraining, minimal calibration, and offers low communication overhead, which is ideal for **edge-based computer vision systems** such as post-disaster search and rescue.

## Notes
The project report details all methods, evaluation metrics, and analysis.  
Due to project constraints, some scripts are experimental prototypes and may require adaptation for different environments.  

> The Coral Dev Board Micro deployment code (`google-coral-micro-object-detection/`) is currently under code cleaning and reconstruction to improve readability, modularity, and compatibility with future firmware updates.

---

*Imperial College London, MSc Computing Individual Project (2025).*
