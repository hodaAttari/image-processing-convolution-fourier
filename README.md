# Image Processing — Convolution & Fourier Analysis  

A concise educational project in Python demonstrating foundational **image processing** techniques that underlie **Convolutional Neural Networks (CNNs)**.  
It includes manual convolution, max pooling, Gaussian blur, Fourier transform filtering, and Laplacian edge detection — all implemented from scratch using `NumPy`, `OpenCV`, and `Matplotlib`.

---

## Features  

This project shows how individual image filters and transformations work in both spatial and frequency domains.  
It performs step‑by‑step:
- Convolution with edge and blur kernels  
- Gaussian smoothing and edge enhancement  
- Max pooling for down‑sampling  
- Fourier Transform for frequency mask filtering (low‑pass, high‑pass, band‑pass)  
- Laplacian operator for boundary detection  

---

## Files  

`project_image.py` — Main Python script that executes all processing stages.  
`project_image.ipynb` — Notebook version with visualizations and explanations.  
`yann_lecun.jpg` — Sample grayscale image required for Fourier and Laplacian sections.  

---

## Run

Install dependencies:
```bash
pip install numpy opencv-python matplotlib
```

Run script:
python project_image.py

Or open the notebook:
jupyter notebook project_image.ipynb


## Educational Goal

This mini‑project helps students understand how CNN building blocks actually manipulate images —

using kernel‑based filtering in the spatial domain and masking in the frequency domain.

It bridges mathematical operations and visual intuition without relying on deep‑learning frameworks.
