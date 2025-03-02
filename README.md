# Image Compression Using 2D DCT and Sparse Orthonormal Transforms

This project explores and implements various image compression techniques, starting with the traditional 2D DCT JPEG algorithm and enhancing it with modern approaches like sparse orthonormal transforms.

## Project Overview

We've implemented several compression techniques:

1. **JPEG Compression**: A full implementation of the standard JPEG algorithm including 2D DCT, quantization, and Huffman encoding
2. **PCA-Based Compression**: An alternative approach using Principal Component Analysis for dimensionality reduction
3. **Sparse Orthonormal Transforms (SOTs)**: An advanced technique that adapts to different signal characteristics

## Features

- Complete JPEG compression pipeline for grayscale images
- Detailed analysis through RMSE vs BPP metrics
- Comparative analysis between implemented and existing JPEG compression
- PCA-based compression with variance retention control
- SOT implementation based on classification and annealing techniques

## Results

The project includes comprehensive performance analysis:
- Compression ratio evaluation
- RMSE (Root Mean Square Error) vs BPP (Bits Per Pixel) curves
- PSNR (Peak Signal-to-Noise Ratio) analysis
- Visual comparisons of compressed images at different quality levels

## Implementation Details

### JPEG Compression
- Image division into 8×8 blocks
- 2D Discrete Cosine Transform
- Quantization with configurable quality factors
- Entropy coding (Run-Length + Huffman)

### PCA-Based Compression
- Dimensionality reduction through principal components
- Configurable variance retention
- Analysis of compression ratio vs components used

### Sparse Orthonormal Transforms
- Classification-based approach for signal grouping
- Annealing process for transform optimization
- Adaptive to various signal structures without predefined models

## Usage

The main implementation includes functions for:
- Image preprocessing
- Block-wise transformation
- Quantization and encoding
- Image reconstruction

## Dependencies

- NumPy
- SciPy
- Matplotlib (for visualization)
- (Other libraries as needed by your specific implementation)

## Authors

- Asmith Reddy - 22B0663
- Vishal Gautam - 22B0065
- Ananya Nawale - 21D170004

## Acknowledgments

This project was developed under the guidance of Prof. Ajit Rajwade at the Indian Institute of Technology Bombay.

## References

O. G. Sezer, O. G. Guleryuz, and Y. Altunbasak, "Approximation and Compression With Sparse Orthonormal Transforms," in IEEE Transactions on Image Processing, vol. 24, no. 8, pp. 2328-2343, Aug. 2015. doi: 10.1109/TIP.2015.2414879.
