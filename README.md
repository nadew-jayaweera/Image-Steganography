# CyberMaths Steganography Tool

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-gray?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **A robust Tkinter GUI application for LSB image steganography, XOR encryption, and quantitative image analysis (MSE/PSNR).**

---

## Gallery

| **Encryption Tab** | **Analysis Tab** |
|:---:|:---:|
| ![Encode Preview](https://via.placeholder.com/400x300?text=Encode+Tab+Screenshot) | ![Analysis Preview](https://via.placeholder.com/400x300?text=Analysis+Tab+Screenshot) |


---

## Overview
**CyberMaths Steganography Tool** is a desktop application designed to bridge the gap between cryptography and digital image processing. It allows users to hide secret text messages inside images using **Least Significant Bit (LSB)** manipulation. 

Beyond simple hiding, this tool emphasizes educational value by including:
* **XOR Encryption** to demonstrate basic symmetric ciphers.
* **Mathematical Analysis** to mathematically quantify the visual distortion introduced by steganography.
* **Metadata Inspection** for checking file integrity and properties.

---

## Key Features
* **🔒 LSB Steganography:** Hide messages within the pixel data of PNG, JPG, or BMP images.
* **🔑 XOR Encryption:** Optional layer of security that encrypts your message before embedding it.
* **📊 Mathematical Analysis:** Built-in tools to calculate **MSE** and **PSNR** to verify image quality retention.
* **🖼️ Theme-Aware GUI:** Modern Interface that adapts to your system theme (Dark/Light modes supported).
* **📂 Smart Decoding:** Auto-detects and reveals hidden messages from loaded stego-images.
* **📝 EXIF Reader:** Lightweight metadata viewer to inspect image properties on the fly.

---

## Maths & Algorithms

### 1. LSB Encoding (The "Hiding" Process)
The tool converts your message into an 8-bit binary stream. It then iterates through the image's flattened pixel array, replacing the **Least Significant Bit (LSB)** of each color channel with a message bit. 
* **Sentinel:** A `$$STOP$$` marker is appended to the message so the decoder knows exactly where to stop reading.

### 2. Boolean Algebra (XOR Encryption)
If enabled, the message undergoes a bitwise exclusive-OR operation against a user-provided password before embedding.
* **Formula:** $C = M \oplus K$
* *Note: This acts as a symmetric cipher; the same operation is used to decrypt.*

### 3. Signal Processing Metrics
To ensure the stego-image looks identical to the original, we use two standard metrics:

* **Mean Squared Error (MSE):** Measures the average squared difference between the original ($I$) and modified ($K$) image pixels.
    $$MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i, j) - K(i, j)]^2$$

* **Peak Signal-to-Noise Ratio (PSNR):** Expressed in decibels (dB), this compares the maximum possible pixel power to the corrupting noise (MSE).
    $$PSNR = 20 \cdot \log_{10}\left(\frac{MAX_I}{\sqrt{MSE}}\right)$$
    *(Where $MAX_I$ is 255 for standard 8-bit images. Higher PSNR = Better Quality.)*

---

## ⚙️ Installation

### Prerequisites
* Python 3.9 or higher
* Git

### Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nadew-jayaweera/Image-Steganography.git](https://github.com/nadew-jayaweera/Image-Steganography.git)
    cd Image-Steganography
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage Guide

### Method 1: Running from Source
Execute the main script via your terminal:
```bash
python app.py
```

### Method 2: Running the Executable (Windows)
Download the standalone `app.exe` from our **Releases Page** and run it directly. No Python installation required.

---

## 📝 Step-by-Step Instructions

### Tab 1: Encode (Hide)
1. Click **Load Image** (Select PNG for lossless quality).
2. Type your secret message into the text area.
3. *(Optional)* Enter a password to enable **XOR Encryption**.
4. Click **ENCRYPT & SAVE**. Select a destination to save your new "Stego-Image".

### Tab 2: Decode (Reveal)
1. Click **Load Stego Image** and select the image containing the hidden text.
2. If you used a password during encoding, enter it in the **Decryption Password** field.
3. Click **REVEAL HIDDEN MESSAGE**. The text will appear on the screen.

### Tab 3: Maths Analysis
1. Ensure both the **Original Image** and **Stego Image** are loaded (this happens automatically after encoding).
2. Click **Calculate Metrics**.
3. Review the **MSE** (should be close to 0) and **PSNR** (should be high, typically >50dB) to verify that the changes are invisible to the naked eye.

---

## Contributing
Contributions, bug reports, and improvements are welcome. Suggested improvements:
- Replace XOR with standard authenticated encryption (e.g., AES-GCM) for confidentiality.
- Add adaptive embedding (randomized LSB positions using a PRNG keyed by password) to improve stealth.
- Support embedding binary files (images/documents) by using a length-prefix + sentinel scheme.

---

## ⚖️ License
This project is released under MIT-style permissive terms (add your preferred license header).

---
