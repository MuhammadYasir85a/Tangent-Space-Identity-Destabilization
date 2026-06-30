<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=06B6M6&height=200&section=header&text=TSID&fontSize=90&fontColor=ffffff&animation=fadeIn" width="100%" />
</div>

<div align="center">
  <h3>Tangent-Space Identity Destabilization</h3>
  <h4>An Adversarial Privacy Filter that Protects Faces from AI Recognition while Staying Visible to Humans</h4>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Adversarial%20ML-06B6D4?style=for-the-badge" alt="Adversarial ML" />
</div>

---

## Overview

TSID is a novel two-stage adversarial attack algorithm that protects face images from AI-based facial recognition systems while remaining visually identical to humans. The system adds mathematically-designed imperceptible noise to a face such that any standard face recognition model (FaceNet, ArcFace, VGG-Face) sees a completely different identity, but the image looks unchanged to human viewers.

The motivation comes from real privacy concerns: surveillance cameras, social media auto-tagging, and AI-powered tracking systems can identify individuals from photos without consent. Existing privacy solutions either destroy image quality (blurring, pixelation) or are impractical (physical masks, disabling cameras). TSID solves this elegantly by exploiting the mathematics of embedding spaces.

This project was developed as part of CSC-361 Machine Learning at Namal University Mianwali, achieving cosine similarity of -0.6076 (3x stronger than published baseline of -0.1893) while maintaining SSIM 0.93 and PSNR 35.7 dB. The implementation also surpasses standard adversarial attacks including FGSM, PGD, and Random Noise baselines using the same perturbation budget.

---

## Key Features

- **Two-Stage TSID Algorithm** — Novel approach combining identity shift with orthogonal manifold dispersion for maximum embedding destabilization
- **Ensemble Attack Model** — Targets both VGGFace2 and CASIA-WebFace simultaneously, producing universal adversarial perturbations
- **Face-Region-Only Protection** — MTCNN face detection ensures noise is applied ONLY to the face region while background remains mathematically untouched
- **Multi-Scale Optimization** — Operates at 320x320 resolution for visual quality while interfacing with FaceNet at native 160x160
- **Resolution-Aware Hyperparameters** — Custom lambda tuning prevents Stage 2 drift at higher resolutions
- **Best-Snapshot Safety Mechanism** — Tracks lowest cosine similarity and returns the optimal snapshot, guaranteeing best result
- **Real-Time Video Processing** — Processes 232 frames in 50 seconds (40x speedup over per-frame attack)
- **Comprehensive Baseline Comparison** — Outperforms Random Noise, Targeted FGSM, and Targeted PGD on the same budget

---

## Tech Stack

**Deep Learning Frameworks:**
- PyTorch 2.4.1 with CUDA 12.1
- TorchVision 0.19.1
- facenet-pytorch 2.6.0

**Face Recognition Models:**
- InceptionResnetV1 (VGGFace2 weights)
- InceptionResnetV1 (CASIA-WebFace weights)
- MTCNN face detector

**Image Processing:**
- PIL/Pillow 10.2
- scikit-image 0.24.0
- imageio with ffmpeg

**Evaluation:**
- DeepFace 0.0.93 (6 black-box models)
- NumPy 2.0.2
- Pandas, Matplotlib, Seaborn

**Tools and Platforms:**
- Google Colab (Tesla T4 GPU)
- Jupyter Notebook
- Google Drive (artifact storage)

---

## System Architecture
<img width="993" height="992" alt="TSID_Architecture" 
src="https://raw.githubusercontent.com/MuhammadYasir85a/Tangent-Space-Identity-Destabilization/main/TSID_Architecture.png" />

---

## Project Structure

```
Tangent-Space-Identity-Destabilization/
│
├── comparison/
│   ├── attack_comparison.png
│   └── baseline_comparison.csv
│
├── evaluation/
│   └── whitebox_metrics.csv
│
├── logs/
│   ├── final_report.txt
│   ├── final_report.md
│   └── config_snapshot.json
│
├── plots/
│   ├── master_dashboard.png
│   ├── whitebox_evaluation.png
│   └── face_only_FINAL_*.png
│
├── results/
│   └── protected_*.png
│
├── video/
│   └── protected_video.mp4
│
├── ML_Final_Implementation.ipynb
├── .gitignore
├── LICENSE
└── README.md
```

---

## How It Works

### 1. Face Detection and Cropping
MTCNN detects the face bounding box at native resolution using a three-stage cascade (P-Net, R-Net, O-Net). The face is cropped at full resolution to preserve maximum detail before resizing.

### 2. Multi-Scale Preparation
The face is resized to 320x320 using Lanczos resampling for the attack optimization. The model wrapper internally downsizes to 160x160 only during forward pass, preserving 4x more visual detail in the optimization space.

### 3. Stage 1 - Identity Shift
PGD-style gradient descent pushes the face embedding away from its original position. The loss combines cosine similarity minimization with L2 regularization. After 40 iterations, cosine similarity drops from +1.0 to approximately -0.77.

### 4. Stage 1.5 - Geometric Handover
The identity direction vector v = normalize(f(x1) - f(x)) is computed and frozen. This vector represents the axis along which Stage 1 moved the embedding, serving as the reference for Stage 2's orthogonal dispersion.

### 5. Stage 2 - Orthogonal Manifold Dispersion
Using Expectation Over Transformation (EOT) with K=10 random transformations per iteration, the algorithm maximizes embedding variance perpendicular to v while keeping the centroid far from the original. The best-snapshot mechanism tracks the optimal result throughout this process.

### 6. Feathered Paste-Back
The protected face is resized back to native dimensions using Lanczos resampling. A Gaussian-feathered mask blends the protected face into the original image with no visible boundary, leaving the background mathematically untouched.

---

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (Tesla T4 or better recommended)
- Google Colab account (for cloud execution) OR local GPU machine
- Google Drive (for artifact persistence)

### Setup Steps

1. Clone the repository:

```bash
git clone https://github.com/MuhammadYasir85a/Tangent-Space-Identity-Destabilization.git
cd Tangent-Space-Identity-Destabilization
```

2. Open the notebook in Google Colab:

```bash
# Upload ML_Final_Implementation.ipynb to Google Colab
# Or use direct link from your local copy
```

3. Run Cell 1 to install dependencies:

```python
# Cell 1 installs:
# torch==2.4.1, torchvision==0.19.1
# facenet-pytorch==2.6.0 (with --no-deps)
# scikit-image==0.24.0
# deepface==0.0.93
# numpy==2.0.2
```

4. After installation, restart the Colab runtime and run Cells 2 through 12 in order.

The notebook will mount Google Drive, load models, and create the project folder structure automatically.

---

## Usage

### Method 1: Single Image Protection
1. Run Cells 1 through 6 (one-time setup, takes 5 minutes)
2. Run Cell 7C and upload your face image when prompted
3. Wait 6-10 seconds for TSID to complete
4. View the protected image with full visualization
5. Output is saved automatically to Google Drive

### Method 2: Video Protection
1. Complete Cells 1 through 7C first
2. Run Cell 12 and upload a short video (5-15 seconds recommended)
3. TSID computes perturbation once, applies to all frames
4. Download the protected MP4 file

### Method 3: Multi-Model Verification
1. Run any cell that includes face_verification_system function
2. Upload two images (or use original + protected)
3. View verdicts from 4+ different face recognition models

---

## Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Cosine Similarity to Original | **-0.6076** | < -0.10 | EXCEEDED 6x |
| SSIM (Structural Similarity) | **0.9351** | > 0.90 | PASSED |
| PSNR (Image Quality) | **35.76 dB** | > 30 dB | PASSED |
| Attack Time on T4 GPU | **6-10 sec** | Real-time | PASSED |
| Improvement Over Baseline | **3x stronger** | Beat baseline | EXCEEDED |
| Video Processing Speedup | **40x** | Practical | EXCEEDED |

### Comparison with Baseline Attacks

| Rank | Attack | Cosine Sim | SSIM | Verdict |
|------|--------|-----------|------|---------|
| 1 | **TSID (ours)** | **-0.6076** | **0.9351** | **FOOLED** |
| 2 | Targeted PGD | +0.3055 | 0.8950 | Partial |
| 3 | Targeted FGSM | +0.7496 | 0.8460 | Failed |
| 4 | Random Noise | +0.9978 | 0.9340 | Failed |

TSID is the only attack achieving negative cosine similarity while maintaining the highest visual quality.

---

## Use Cases

- **Privacy-Conscious Content Creators** — Share photos and videos online without enabling facial recognition tracking
- **Activists and Journalists** — Protect identity in published material while keeping faces visible for human readers
- **Video Conferencing Privacy** — Protect against AI-powered identity capture during online meetings
- **Social Media Users** — Prevent unauthorized facial recognition by platforms and third parties
- **Research and Education** — Demonstrate adversarial machine learning concepts in academic settings

---

## Future Enhancements

- Real-time webcam integration with live TSID protection
- Per-frame video TSID for stronger protection at cost of processing time
- Mobile application deployment (iOS and Android)
- Adversarial training defense evaluation against TSID
- Facial landmark-targeted attacks for finer control
- Browser extension for automatic photo protection before upload
- Distributed processing for batch image protection

---

## Learning Outcomes

Through this project, the following concepts and skills were developed:

- Deep learning model architecture (CNN, InceptionResNetV1)
- Adversarial machine learning theory (FGSM, PGD, EOT)
- Embedding space geometry and orthogonal decomposition
- PyTorch autograd and gradient-based optimization
- Multi-scale image processing and Lanczos resampling
- Face detection and recognition pipelines (MTCNN, FaceNet)
- Hyperparameter tuning and resolution-aware optimization
- End-to-end ML system design and evaluation methodology

---

## Project Status

**Status:** Completed

This project successfully meets all academic requirements and exceeds published baselines. The implementation is reproducible, fully documented, and ready for deployment or further research extension.

---

## Academic Information

| Field | Detail |
|-------|--------|
| Course | CSC-361 Machine Learning |
| Track | B (Software-Based AI System) |
| Institution | Namal University Mianwali |
| Department | Computer Science |
| Project Type | Academic Team Project |
| Semester | Spring 2025 |
| Instructor | Dr. Shafiq Ur Rehman Khan |

---

## Contributors

- **Muhammad Yasir (NUM-BSCS-2023-37)** — TSID algorithm design, multi-scale optimization, hyperparameter tuning, evaluation pipeline
- **Rehan Ali (NUM-BSCS-2023-15)** — Face detection integration, feathered paste-back, baseline comparison, video processing

---

## Author

**Muhammad Yasir**

Computer Science Undergraduate at Namal University Mianwali  
Aspiring AI and Computer Vision Engineer

<div>
  <a href="https://www.linkedin.com/in/muhammad-yasir-6a9500343/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  <a href="mailto:muhammadyasir85a@gmail.com">
    <img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>
  <a href="https://github.com/MuhammadYasir85a">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
</div>

---

## Acknowledgments

- **Namal University Mianwali** for academic guidance and computational resources
- **Dr. Shafiq Ur Rehman Khan** for project supervision and valuable feedback on face-region attack approach
- **facenet-pytorch community** for the excellent pretrained models and MTCNN implementation
- **PyTorch team** for the robust autograd engine that made gradient-based attacks possible
- **Google Colab** for providing free GPU access throughout development

---

## References

- Schroff, F., Kalenichenko, D., and Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR.
- Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks. IEEE Signal Processing Letters.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR.
- Goodfellow, I., Shlens, J., and Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. ICLR.
- Athalye, A., Engstrom, L., Ilyas, A., and Kwok, K. (2018). Synthesizing Robust Adversarial Examples. ICML.
- Cao, Q., Shen, L., Xie, W., Parkhi, O. M., and Zisserman, A. (2018). VGGFace2: A dataset for recognising faces across pose and age. FG.
- Yi, D., Lei, Z., Liao, S., and Li, S. Z. (2014). Learning Face Representation from Scratch. arXiv:1411.7923.

---

## License

This project is licensed under the **MIT License**.

Free for academic and personal use. Commercial use requires attribution to the original authors.

---

<div align="center">
  <i>"Privacy is not about hiding wrongdoing. It is about preserving the human right to control your own identity."</i>
</div>

<br/>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=06B6D4&height=100&section=footer" width="100%" />
</div>
