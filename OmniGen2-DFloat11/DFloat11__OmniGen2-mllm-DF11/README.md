---
base_model:
  - OmniGen2/OmniGen2
base_model_relation: quantized
pipeline_tag: any-to-any
tags:
- dfloat11
- df11
- lossless compression
- 70% size, 100% accuracy
---

# DFloat11 Compressed Model: `OmniGen2/OmniGen2` MLLM

This is a **DFloat11 losslessly compressed** version of the original `OmniGen2/OmniGen2` model. It reduces model size by **32%** compared to the original BFloat16 model, while maintaining **bit-identical outputs** and supporting **efficient GPU inference**.

### 📊 Performance Comparison

| Metric                                          | OmniGen2 (BFloat16) | OmniGen2 (DFloat11) |
| ----------------------------------------------- | ------------------- | ------------------- |
| Model Size                                      | 16.23 GB            | 11.11 GB            |
| Peak GPU Memory<br>(1024×1024 image generation) | 18.41 GB            | 14.36 GB            |
| Generation Time<br>(A100 GPU)                   | 25 seconds          | 27 seconds          |

### 🔧 How to Use

A complete usage guide is available in our GitHub repository (forked from the official OmniGen2 repository).

👉 [https://github.com/LeanModels/OmniGen2-DFloat11](https://github.com/LeanModels/OmniGen2-DFloat11) 👈

### 🔍 How It Works

We apply **Huffman coding** to losslessly compress the exponent bits of BFloat16 model weights, which are highly compressible (their 8 bits carry only ~2.6 bits of actual information). To enable fast inference, we implement a highly efficient CUDA kernel that performs on-the-fly weight decompression directly on the GPU.

The result is a model that is **~32% smaller**, delivers **bit-identical outputs**, and achieves performance **comparable to the original** BFloat16 model.

Learn more in our [research paper](https://arxiv.org/abs/2504.11651).

### 📄 Learn More

* **Paper**: [70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float](https://arxiv.org/abs/2504.11651)
* **GitHub**: [https://github.com/LeanModels/DFloat11](https://github.com/LeanModels/DFloat11)
* **HuggingFace**: [https://huggingface.co/DFloat11](https://huggingface.co/DFloat11)
