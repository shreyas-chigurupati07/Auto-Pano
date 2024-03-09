# RBE/CS 549 Computer Vision: AutoPano Project

## Project Overview

This project introduces two methodologies for stitching images to create a seamless panorama. It begins with a traditional approach using feature matching and homography estimation, followed by innovative supervised and unsupervised deep learning methods to predict homography between images.

## Key Components

1. **Traditional Image Stitching:** Utilizes corner detection, feature descriptors, adaptive non-maximal suppression (ANMS), and RANSAC to compute homography, followed by image blending for panorama creation.
2. **Deep Learning Approaches:** Implements supervised and unsupervised neural networks to estimate homography, leveraging a dataset synthesized from the MSCOCO dataset for training.
3. **Homography Estimation:** Both deep learning approaches aim to refine the process of homography estimation, crucial for accurate image alignment in panoramic stitching.

## Implementation Details

- Dataset: Synthetic data generated from MSCOCO, with labeled patches for the supervised approach and only image pairs for the unsupervised method.
- Feature Matching: Utilizes Harris and Shi-Tomashi corner detection methods followed by feature description and matching for traditional stitching.
- Neural Networks: Employ deep learning models to predict homography, with the supervised approach focusing on regression and the unsupervised on direct transformation learning.
- Stitching and Blending: Applies calculated or predicted homographies to warp and stitch images, ensuring seamless transitions between frames.

## Results

The project demonstrates successful panoramic image stitching using traditional methods and explores the potential of deep learning to enhance the homography estimation process, aiming for faster and more robust stitching solutions.
![panaroma](https://github.com/shreyas-chigurupati07/Auto-Pano/assets/84034817/11b4b66d-8b29-4208-a6e7-b820f4031757)
![ANMS](https://github.com/shreyas-chigurupati07/Auto-Pano/assets/84034817/999e0fa5-2017-4d98-8f57-17fcc401f86e)

## How to Run

1. Clone the repository: `git clone [repository-link]`
2. Navigate to the project directory: `cd [project-directory]`
3. Execute the scripts for traditional and deep learning-based stitching: `python traditional_stitch.py`, `python supervised_stitch.py`, `python unsupervised_stitch.py`

## Dependencies

- Python
- NumPy
- OpenCV
- PyTorch
- Kornia (for the unsupervised method)


## References

- Detailed references to the methodologies and tools used throughout the project, including academic papers and software documentation.
