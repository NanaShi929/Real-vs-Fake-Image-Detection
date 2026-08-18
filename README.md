# AI-Generated vs. Real Image Detection

## 📌 Project Overview
The rise of advanced generative models has made it increasingly challenging to differentiate between real and synthetic imagery. This project focuses on building deep learning classifiers to detect whether an image is **Real** or **Fake** (AI-Generated).

We trained and evaluated three computer vision architectures:
* **ResNet50:** A deep residual convolutional neural network.
* **EfficientNet-B0:** A lightweight, highly scaled convolutional network.
* **Vision Transformer (ViT - `vit_base_patch16_224`):** A self-attention-based transformer model applied directly to image patches.

---

## 📊 Dataset
* **Source:** [AI-Generated Images vs Real Images Dataset (Kaggle)](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)
* **Dataset Size:** ~48.4 GB
* **Resolution / Input Size:** Resized to $224 \times 224$ pixels
* **Data Splits:**
  * **Train Set:** 38,400 images (19,200 Real / 19,200 Fake)
  * **Validation Set:** 9,600 images (4,800 Real / 4,800 Fake)
  * **Test Set:** 12,000 images (6,000 Real / 6,000 Fake)

---

## 📈 Benchmark Test Results (12,000 Images)

Each model was evaluated on the unseen 12,000-image test set. Below is the performance breakdown across key evaluation metrics:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Log Loss | Parameters | Epochs Used | Best Val Accuracy | Final Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | **0.9567** | 0.9386 | **0.9775** | **0.9576** | **0.9922** | 0.1390 | 23,512,130 | 6 | **0.9616** | 0.1525 |
| **EfficientNet-B0** | 0.9557 | **0.9522** | 0.9595 | 0.9558 | 0.9918 | **0.1164** | **4,010,110** | 6 | 0.9576 | **0.1554** |
| **ViT (Vision Transformer)** | 0.9451 | 0.9287 | 0.9642 | 0.9461 | 0.9877 | 0.1420 | 85,800,194 | 4 | 0.9482 | 0.1683 |

### Key Observations:
1. **ResNet50** achieved the highest overall test accuracy (**95.67%**) and F1 Score (**0.9576**), making it the strongest general detector on this benchmark.
2. **EfficientNet-B0** provided nearly identical accuracy (**95.57%**) with the lowest Log Loss (**0.1164**), while utilizing only **4 million parameters** (~6x fewer than ResNet50).
3. **ViT** converged within 4 epochs, delivering a test accuracy of **94.51%**.

---

## 🔍 Custom Data Testing (100% AI-Generated Samples)

To evaluate individual model behavior outside the training distribution, we tested a custom batch consisting **entirely of AI-generated images**:

| # | Image | ResNet50 Prediction (Conf) | EfficientNet Prediction (Conf) | ViT Prediction (Conf) |
| :-: | :--- | :---: | :---: | :---: |
| 0 | `coa9_actor_Volodymyr_Pielikh...` | **fake** (97.64%) | **fake** (99.21%) | **fake** (92.11%) |
| 1 | `arquitectosfonseca_Daytime_photograph...` | **fake** (98.81%) | **fake** (99.86%) | **fake** (99.44%) |
| 2 | `u4514512553_Seoul_Gangnam_Morning...` | **fake** (97.31%) | **fake** (100.00%) | **fake** (97.75%) |
| 3 | `jqnotu_A_vintage_oil_painting...` | **real** (96.09%) | **real** (60.71%) | **real** (74.41%) |
| 4 | `designthewayyoulive_the_luxurious...` | **fake** (87.99%) | **fake** (99.92%) | **fake** (66.56%) |
| 5 | `domkazz0607_Full-body_fashion...` | **real** (79.05%) | **fake** (89.20%) | **fake** (83.72%) |
| 6 | `vladislav282828_vertical_poster...` | **real** (92.97%) | **fake** (98.60%) | **fake** (65.04%) |

### Custom Test Analysis :
* **EfficientNet-B0:** Exhibited the strongest individual performance on the custom batch, correctly classifying 6 out of 7 AI-generated images with high confidence.
* **ViT (Vision Transformer):** Correctly identified 6 out of 7 AI-generated samples, performing with slightly more conservative confidence scores than EfficientNet.
* **ResNet50:** Correctly classified 4 out of 7 images, misclassifying 3 AI-generated samples as `real`.
---
## 🎯 Conclusion
* All three models successfully differentiate real and AI-generated images with high accuracy on the benchmark dataset (94.5% to 95.6%).
* **ResNet50** achieved the highest overall accuracy (**95.67%**) on the standard test set.
* **EfficientNet-B0** and **ViT** performed better on the custom AI-generated images, correctly classifying 6 out of 7 samples, whereas ResNet50 only classified 4 correctly.
* **EfficientNet-B0** offers the best balance of size and performance, achieving near-identical accuracy to ResNet50 while using approximately 6 times fewer parameters.