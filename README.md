# Image Steganography Tool

**A Tkinter GUI application that demonstrates image steganography, simple XOR encryption, and mathematical analysis (MSE / PSNR).**

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Maths & Algorithms](#maths--algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [GUI Overview](#gui-overview)
- [Acknowledgements](#acknowledgements)

---

## Overview
`CyberMaths Steganography Tool` is a desktop GUI (Tkinter) application that hides text inside images using Least Significant Bit (LSB) steganography, optionally combined with a simple XOR-based encryption. It also computes image quality metrics (MSE and PSNR) to analyze the visual difference between the original and stego images, and provides a lightweight EXIF-style metadata reader.

---

## Features
- Load an image and hide a secret message using LSB steganography.
- Optional XOR-based encryption of the message before embedding.
- Save the generated stego image as a PNG.
- Load the stego image and reveal the hidden message (auto-decrypt if password used).
- Calculate MSE (Mean Squared Error) and PSNR (Peak Signal-to-Noise Ratio) between two images for quantitative analysis.
- Lightweight metadata / EXIF-like reader using `Pillow` and file system stats.
- Theme-aware GUI (dark/light/system) with a polished layout and user-friendly dialogs.

---

## Maths & Algorithms

### LSB Encoding / Decoding
- The message is converted to 8-bit binary and a sentinel `$$STOP$$` is appended to mark the end of the message.
- Bits are written into the least-significant-bit of each channel byte of the image array (flattened).
- During decode, LSBs are read and reassembled into bytes until the sentinel is detected.

### XOR Encryption (Optional)
- A simple bytewise XOR of the message with the provided key string. XOR is symmetric: the same operation both encrypts and decrypts.
- Implemented as a demonstration of Boolean algebra techniques (not cryptographically secure).

### Image Quality Metrics
- **MSE (Mean Squared Error)**: average squared difference per pixel between original and stego image arrays.
- **PSNR (Peak Signal-to-Noise Ratio)**: calculated as `20 * log10(MAX / sqrt(MSE))`, where `MAX = 255` for 8-bit images.
- MSE close to 0 indicates minimal change; higher PSNR indicates better perceived image quality.

---

## Installation

**Clone repository:**
```
git clone https://github.com/nadew-jayaweera/Image-Steganography.git
```

**Requirements** (tested with Python 3.9+):
- Python 3.x
- See `requirements.txt` for dependencies.

Check our [Releases](https://github.com/nadew-jayaweera/Image-Steganography/releases/tag/v1.0.0):
- Run `app.exe`

```bash
pip install -r requirements.txt
```

---

## Usage

1. Run the application:
```bash
python app.py
```

2. **Encode (Hide)** tab:
   - Click **Load Image** and choose a PNG/JPG/BMP (PNG is recommended for lossless output).
   - Enter the secret message in the text box.
   - (Optional) Enter a password and select the `LSB + XOR Encryption` technique.
   - Click **ENCRYPT & SAVE IMAGE** and choose a file path (PNG recommended).

3. **Decode (Reveal)** tab:
   - Click **Load Stego Image** (prefer PNG/BMP for accurate bit preservation).
   - If encryption was used, enter the same password in the Decryption Password box.
   - Click **REVEAL HIDDEN MESSAGE** to display the recovered message.

4. **Maths Analysis** tab:
   - Load or keep the original and the produced stego image paths set (the app auto-loads the saved stego for analysis after saving).
   - Click **Calculate Metrics** to compute MSE and PSNR; results display in the Analysis tab.

---

## GUI Overview
- Three main tabs: **Encode (Hide)**, **Decode (Reveal)**, and **Maths Analysis**.
- Theme selector (Dark / Light / System) detects the OS and allows switching.
- Image preview areas for both encode and decode flows.
- The app displays a small EXIF-like metadata output when a stego image is loaded.

---

## Contributing
Contributions, bug reports, and improvements are welcome. Suggested improvements:
- Replace XOR with standard authenticated encryption (e.g., AES-GCM) for confidentiality.
- Add adaptive embedding (randomized LSB positions using a PRNG keyed by password) to improve stealth.
- Support embedding binary files (images/documents) by using a length-prefix + sentinel scheme.

---

## Acknowledgements
- Project generated based on the provided `app.py`. fileciteturn0file0
- Uses `Pillow` and `numpy` for image processing and numerical work.
