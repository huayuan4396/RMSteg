# 🌟RMSteg

Implementation for the paper [**Robust Message Embedding via Attention Flow-Based Steganography**](https://arxiv.org/pdf/2405.16414v2). 

<div align='center'>
  <img width="512" alt="teaser" src="https://github.com/user-attachments/assets/c465dd48-910f-4ef7-9006-3bfe20191434"/>
</div>
Our approach allows for message embedding into image with strong robustness against real-world image distortions (printing, JPEG compression, etc.).

<div align='center'>
  <img width="1024" alt="evl_quality" src="https://github.com/user-attachments/assets/395b0892-174f-4987-8c45-091544716a7f" />
</div>


## 📸 News

### [Feburary 27, 2025] - Our paper is accepted at CVPR 2025!
### [April 8, 2025] - Source code is released!
### [April 16, 2025] - Pre-trained model is available!

## 🔧 Quick Start

### From Pre-trained Model
You can download our pre-trained mode from [here](https://drive.google.com/file/d/1iMowiDN8T2Rm6nlI_vDZqeNCvn-zeTkZ/view?usp=sharing) and place it in the `src/pretrained/` folder. Then, you can run the following command for a quick start:
```bash
cd src
python single_test.py
```
