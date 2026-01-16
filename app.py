#!/usr/bin/env python3
"""
CyberMaths Steganography Tool - Mathematics Edition
Standard LSB Steganography with comprehensive mathematical analysis.
"""

import tkinter as tk
from tkinter import filedialog
import math
import os
from datetime import datetime
import platform
import base64

# Check for required packages
try:
    from PIL import Image, ImageTk, ExifTags
    import numpy as np
except ImportError as e:
    print("=" * 50)
    print("Missing packages! Install with:")
    print("  pip install Pillow numpy")
    print("=" * 50)
    raise e


# -------------------------------------------------------------------------
# MATHEMATICS ENGINE
# -------------------------------------------------------------------------

class MathEngine:
    """Mathematical calculations for image analysis."""

    @staticmethod
    def calculate_entropy(image_path):
        """
        Calculate Shannon Entropy of an image.

        Formula: H(X) = -Σ p(x) * log2(p(x))

        Higher entropy = more randomness/information
        Max entropy for 8-bit image = 8.0
        """
        img = Image.open(image_path).convert('L')  # Grayscale
        histogram = img.histogram()
        total_pixels = sum(histogram)

        entropy = 0
        for count in histogram:
            if count > 0:
                probability = count / total_pixels
                entropy -= probability * math.log2(probability)

        return entropy

    @staticmethod
    def calculate_statistics(image_path):
        """
        Calculate comprehensive image statistics.

        Returns: mean, std_dev, variance, min, max, median
        """
        img = np.array(Image.open(image_path))

        stats = {
            'mean': np.mean(img),
            'std_dev': np.std(img),
            'variance': np.var(img),
            'min': np.min(img),
            'max': np.max(img),
            'median': np.median(img),
            'pixels': img.size,
            'unique_values': len(np.unique(img))
        }

        return stats

    @staticmethod
    def calculate_histogram(image_path):
        """
        Calculate histogram data for visualization.
        Returns histogram for each channel (R, G, B).
        """
        img = Image.open(image_path)

        if img.mode == 'L':
            return {'gray': img.histogram()}

        img = img.convert('RGB')
        r, g, b = img.split()

        return {
            'red': r.histogram(),
            'green': g.histogram(),
            'blue': b.histogram()
        }

    @staticmethod
    def chi_square_test(image_path):
        """
        Chi-Square Steganalysis Test.

        Detects LSB steganography by analyzing pixel pair distribution.

        Formula: χ² = Σ (Oi - Ei)² / Ei

        High χ² value = likely contains hidden data
        """
        img = np.array(Image.open(image_path).convert('L')).flatten()

        # Count pairs of values (2i, 2i+1)
        pairs = {}
        for pixel in img:
            pair_index = pixel // 2
            if pair_index not in pairs:
                pairs[pair_index] = [0, 0]
            pairs[pair_index][pixel % 2] += 1

        # Calculate chi-square
        chi_square = 0
        degrees_of_freedom = 0

        for pair_index, counts in pairs.items():
            expected = (counts[0] + counts[1]) / 2
            if expected > 0:
                chi_square += ((counts[0] - expected) ** 2) / expected
                chi_square += ((counts[1] - expected) ** 2) / expected
                degrees_of_freedom += 1

        # Calculate p-value approximation
        if degrees_of_freedom > 0:
            # Simplified p-value estimation
            z = (chi_square - degrees_of_freedom) / \
                math.sqrt(2 * degrees_of_freedom)
            p_value = 0.5 * (1 + math.erf(-z / math.sqrt(2)))
        else:
            p_value = 1.0

        return chi_square, degrees_of_freedom, p_value

    @staticmethod
    def analyze_lsb_distribution(image_path):
        """
        Analyze the distribution of Least Significant Bits.

        In a natural image, LSB should be roughly 50% zeros and 50% ones.
        After steganography, this ratio may change.
        """
        img = np.array(Image.open(image_path)).flatten()

        lsb_values = img & 1  # Extract LSB
        zeros = np.sum(lsb_values == 0)
        ones = np.sum(lsb_values == 1)
        total = len(lsb_values)

        return {
            'zeros': zeros,
            'ones': ones,
            'total': total,
            'zero_ratio': zeros / total,
            'one_ratio': ones / total,
            'balance': abs(0.5 - (zeros / total))  # 0 = perfectly balanced
        }

    @staticmethod
    def calculate_ssim(img1_path, img2_path):
        """
        Calculate Structural Similarity Index (SSIM).

        Formula: SSIM(x,y) = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)

        SSIM ranges from -1 to 1, where 1 = identical images
        """
        img1 = np.array(Image.open(img1_path).convert('L')).astype(np.float64)
        img2 = np.array(Image.open(img2_path).convert('L')).astype(np.float64)

        if img1.shape != img2.shape:
            raise ValueError("Images must have same dimensions")

        # Constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Statistics
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        # SSIM formula
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim = numerator / denominator

        return ssim

    @staticmethod
    def calculate_correlation(img1_path, img2_path):
        """
        Calculate Pearson Correlation Coefficient.

        Formula: r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)² * Σ(yi - ȳ)²]

        r = 1: Perfect positive correlation
        r = 0: No correlation
        r = -1: Perfect negative correlation
        """
        img1 = np.array(Image.open(img1_path).convert('L')
                        ).flatten().astype(np.float64)
        img2 = np.array(Image.open(img2_path).convert('L')
                        ).flatten().astype(np.float64)

        if len(img1) != len(img2):
            raise ValueError("Images must have same size")

        mean1, mean2 = np.mean(img1), np.mean(img2)

        numerator = np.sum((img1 - mean1) * (img2 - mean2))
        denominator = math.sqrt(
            np.sum((img1 - mean1) ** 2) * np.sum((img2 - mean2) ** 2))

        if denominator == 0:
            return 1.0

        return numerator / denominator

    @staticmethod
    def calculate_mse_psnr(img1_path, img2_path):
        """
        Calculate Mean Squared Error and Peak Signal-to-Noise Ratio.

        MSE = (1/mn) * Σ Σ [I(i,j) - K(i,j)]²
        PSNR = 10 * log10(MAX² / MSE) = 20 * log10(MAX / √MSE)

        For 8-bit images, MAX = 255
        """
        img1 = np.array(Image.open(img1_path)).astype(np.float64)
        img2 = np.array(Image.open(img2_path)).astype(np.float64)

        mse = np.mean((img1 - img2) ** 2)

        if mse == 0:
            return 0, float('inf')

        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

        return mse, psnr

    @staticmethod
    def text_to_binary(text):
        """Convert text to binary representation."""
        result = []
        for char in text:
            binary = format(ord(char), '08b')
            result.append((char, ord(char), binary))
        return result

    @staticmethod
    def demonstrate_xor(text, key):
        """Demonstrate XOR encryption step by step."""
        if not key:
            return []

        result = []
        for i, char in enumerate(text[:10]):  # Limit to 10 chars for display
            key_char = key[i % len(key)]
            char_bin = format(ord(char), '08b')
            key_bin = format(ord(key_char), '08b')
            xor_result = ord(char) ^ ord(key_char)
            xor_bin = format(xor_result, '08b')

            result.append({
                'char': char,
                'char_bin': char_bin,
                'key_char': key_char,
                'key_bin': key_bin,
                'xor_result': chr(xor_result) if 32 <= xor_result <= 126 else '?',
                'xor_bin': xor_bin
            })

        return result

    @staticmethod
    def calculate_capacity_details(image_path):
        """Calculate detailed capacity information."""
        img = Image.open(image_path)

        width, height = img.size
        channels = len(img.getbands())
        total_pixels = width * height
        total_subpixels = total_pixels * channels
        max_bits = total_subpixels
        max_bytes = max_bits // 8
        max_chars = max_bytes - 10  # Reserve for delimiter

        return {
            'width': width,
            'height': height,
            'channels': channels,
            'total_pixels': total_pixels,
            'total_subpixels': total_subpixels,
            'max_bits': max_bits,
            'max_bytes': max_bytes,
            'max_chars': max_chars,
            'bits_per_pixel': channels
        }

    

# -------------------------------------------------------------------------
# STEGANOGRAPHY ENGINE
# -------------------------------------------------------------------------

class StegoEngine:
    """Standard LSB Steganography Engine."""

    @staticmethod
    def encode(image_path, message, password=None):
        """Hide message using LSB method."""
        image = Image.open(image_path)
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')

        img_array = np.array(image)

        if password:
            message = StegoEngine._xor_cipher(message, password)

        message += "<<END>>"
        binary_message = ''.join(format(ord(char), '08b') for char in message)
        msg_length = len(binary_message)

        if msg_length > img_array.size:
            raise ValueError(
                f"Message too large! Max: {img_array.size // 8} chars")

        flat_img = img_array.flatten()

        for i in range(msg_length):
            flat_img[i] = (flat_img[i] & 0xFE) | int(binary_message[i])

        result_array = flat_img.reshape(img_array.shape)
        return Image.fromarray(result_array.astype('uint8'), image.mode)

    @staticmethod
    def decode(image_path, password=None):
        """Extract hidden message using LSB method."""
        image = Image.open(image_path)
        flat_img = np.array(image).flatten()

        binary_data = ""
        message = ""

        for pixel in flat_img:
            binary_data += str(pixel & 1)

            if len(binary_data) == 8:
                char_code = int(binary_data, 2)
                binary_data = ""

                if char_code == 0:
                    continue

                try:
                    char = chr(char_code)
                    message += char

                    if message.endswith("<<END>>"):
                        result = message[:-7]
                        if password:
                            result = StegoEngine._xor_cipher(result, password)
                        return result
                except:
                    continue

        return "No hidden message found."

    @staticmethod
    def _xor_cipher(text, key):
        """XOR encryption/decryption."""
        if not key:
            return text
        return ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(text))

    @staticmethod
    def get_metadata(image_path):
        """Extract image metadata."""
        try:
            img = Image.open(image_path)
            stats = os.stat(image_path)

            lines = [
                f"{'Filename':<16}: {os.path.basename(image_path)}",
                f"{'Size':<16}: {stats.st_size / 1024:.1f} KB",
                f"{'Dimensions':<16}: {img.width} × {img.height} px",
                f"{'Total Pixels':<16}: {img.width * img.height:,}",
                f"{'Format':<16}: {img.format or 'Unknown'}",
                f"{'Color Mode':<16}: {img.mode}",
                f"{'Modified':<16}: {datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M')}",
            ]

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"


# -------------------------------------------------------------------------
# THEME
# -------------------------------------------------------------------------

class Theme:
    BG_DARK = "#09090b"
    BG_MAIN = "#0c0c0e"
    BG_SECONDARY = "#131316"
    BG_CARD = "#18181b"
    BG_HOVER = "#1f1f23"
    BG_INPUT = "#1c1c20"

    SIDEBAR = "#0a0a0c"
    SIDEBAR_HOVER = "#151518"
    SIDEBAR_ACTIVE = "#1a1a1e"

    TEXT = "#fafafa"
    TEXT_SEC = "#a1a1aa"
    TEXT_MUTED = "#52525b"

    BLUE = "#3b82f6"
    BLUE_HOVER = "#2563eb"
    GREEN = "#22c55e"
    GREEN_HOVER = "#16a34a"
    RED = "#ef4444"
    PURPLE = "#a855f7"
    ORANGE = "#f97316"
    CYAN = "#06b6d4"
    YELLOW = "#eab308"

    BORDER = "#27272a"
    BORDER_HOVER = "#3f3f46"

    FONT = "Segoe UI"
    MONO = "Consolas"


# -------------------------------------------------------------------------
# THEMED MESSAGE BOXES
# -------------------------------------------------------------------------

class ThemedMessageBox:
    """Custom message boxes with theme support."""

    @staticmethod
    def showerror(title, message):
        """Themed error message box."""
        box = tk.Toplevel()
        box.title(title)
        box.geometry("400x200")
        box.configure(bg=Theme.BG_MAIN)
        box.resizable(False, False)
        
        # Center on screen
        box.update_idletasks()
        x = (box.winfo_screenwidth() // 2) - 200
        y = (box.winfo_screenheight() // 2) - 100
        box.geometry(f"+{x}+{y}")
        
        # Icon and title
        frame_top = tk.Frame(box, bg=Theme.BG_CARD)
        frame_top.pack(fill="x", padx=20, pady=(20, 10))
        
        tk.Label(frame_top, text="❌  " + title, font=(Theme.FONT, 12, "bold"),
                bg=Theme.BG_CARD, fg=Theme.RED).pack(anchor="w")
        
        # Message
        msg_frame = tk.Frame(box, bg=Theme.BG_MAIN)
        msg_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        tk.Label(msg_frame, text=message, font=(Theme.FONT, 10),
                bg=Theme.BG_MAIN, fg=Theme.TEXT, justify="left", wraplength=350).pack(anchor="w")
        
        # Button
        btn_frame = tk.Frame(box, bg=Theme.BG_MAIN)
        btn_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        def close():
            box.destroy()
        
        btn = tk.Button(btn_frame, text="OK", font=(Theme.FONT, 10),
                       bg=Theme.RED, fg=Theme.TEXT, relief="flat", 
                       command=close, activebackground=Theme.RED, padx=20, pady=8)
        btn.pack(side="right")
        
        box.focus()
        box.grab_set()
        box.wait_window()

    @staticmethod
    def showinfo(title, message):
        """Themed info message box."""
        box = tk.Toplevel()
        box.title(title)
        box.geometry("400x200")
        box.configure(bg=Theme.BG_MAIN)
        box.resizable(False, False)
        
        # Center on screen
        box.update_idletasks()
        x = (box.winfo_screenwidth() // 2) - 200
        y = (box.winfo_screenheight() // 2) - 100
        box.geometry(f"+{x}+{y}")
        
        # Icon and title
        frame_top = tk.Frame(box, bg=Theme.BG_CARD)
        frame_top.pack(fill="x", padx=20, pady=(20, 10))
        
        tk.Label(frame_top, text="ℹ️  " + title, font=(Theme.FONT, 12, "bold"),
                bg=Theme.BG_CARD, fg=Theme.BLUE).pack(anchor="w")
        
        # Message
        msg_frame = tk.Frame(box, bg=Theme.BG_MAIN)
        msg_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        tk.Label(msg_frame, text=message, font=(Theme.FONT, 10),
                bg=Theme.BG_MAIN, fg=Theme.TEXT, justify="left", wraplength=350).pack(anchor="w")
        
        # Button
        btn_frame = tk.Frame(box, bg=Theme.BG_MAIN)
        btn_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        def close():
            box.destroy()
        
        btn = tk.Button(btn_frame, text="OK", font=(Theme.FONT, 10),
                       bg=Theme.BLUE, fg=Theme.TEXT, relief="flat", 
                       command=close, activebackground=Theme.BLUE, padx=20, pady=8)
        btn.pack(side="right")
        
        box.focus()
        box.grab_set()
        box.wait_window()

    @staticmethod
    def showwarning(title, message):
        """Themed warning message box."""
        box = tk.Toplevel()
        box.title(title)
        box.geometry("400x200")
        box.configure(bg=Theme.BG_MAIN)
        box.resizable(False, False)
        
        # Center on screen
        box.update_idletasks()
        x = (box.winfo_screenwidth() // 2) - 200
        y = (box.winfo_screenheight() // 2) - 100
        box.geometry(f"+{x}+{y}")
        
        # Icon and title
        frame_top = tk.Frame(box, bg=Theme.BG_CARD)
        frame_top.pack(fill="x", padx=20, pady=(20, 10))
        
        tk.Label(frame_top, text="⚠️  " + title, font=(Theme.FONT, 12, "bold"),
                bg=Theme.BG_CARD, fg=Theme.ORANGE).pack(anchor="w")
        
        # Message
        msg_frame = tk.Frame(box, bg=Theme.BG_MAIN)
        msg_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        tk.Label(msg_frame, text=message, font=(Theme.FONT, 10),
                bg=Theme.BG_MAIN, fg=Theme.TEXT, justify="left", wraplength=350).pack(anchor="w")
        
        # Button
        btn_frame = tk.Frame(box, bg=Theme.BG_MAIN)
        btn_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        def close():
            box.destroy()
        
        btn = tk.Button(btn_frame, text="OK", font=(Theme.FONT, 10),
                       bg=Theme.ORANGE, fg=Theme.TEXT, relief="flat", 
                       command=close, activebackground=Theme.ORANGE, padx=20, pady=8)
        btn.pack(side="right")
        
        box.focus()
        box.grab_set()
        box.wait_window()


# -------------------------------------------------------------------------
# UI COMPONENTS
# -------------------------------------------------------------------------

class Button(tk.Canvas):
    """Animated button."""

    def __init__(self, parent, text="", command=None, style="primary",
                 width=180, height=42, icon=None):

        self.parent_bg = str(parent.cget('bg'))
        super().__init__(parent, width=width, height=height,
                         bg=self.parent_bg, highlightthickness=0)

        self.text = text
        self.command = command
        self.style = style
        self.w = width
        self.h = height
        self.icon = icon
        self.hover = 0.0
        self.animating = False

        self._setup_colors()
        self._draw()

        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<Button-1>", self._click)

    def _setup_colors(self):
        styles = {
            "primary": (Theme.BLUE, Theme.BLUE_HOVER, "#fff"),
            "success": (Theme.GREEN, Theme.GREEN_HOVER, "#fff"),
            "danger": (Theme.RED, "#dc2626", "#fff"),
            "secondary": (Theme.BG_CARD, Theme.BG_HOVER, Theme.TEXT),
            "ghost": (self.parent_bg, Theme.BG_HOVER, Theme.TEXT_SEC),
            "purple": (Theme.PURPLE, "#9333ea", "#fff"),
            "cyan": (Theme.CYAN, "#0891b2", "#fff"),
        }
        self.c1, self.c2, self.tc = styles.get(self.style, styles["primary"])

    def _lerp_color(self, c1, c2, t):
        if c1 == self.parent_bg:
            c1 = Theme.BG_CARD
        try:
            r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
            r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            return f'#{r:02x}{g:02x}{b:02x}'
        except:
            return c2 if t > 0.5 else c1

    def _draw(self):
        self.delete("all")
        color = self._lerp_color(self.c1, self.c2, self.hover)

        r = 8
        pts = [r, 0, self.w-r, 0, self.w, 0, self.w, r, self.w, self.h-r,
               self.w, self.h, self.w-r, self.h, r, self.h, 0, self.h,
               0, self.h-r, 0, r, 0, 0]
        self.create_polygon(pts, fill=color, smooth=True)

        txt = f"{self.icon}  {self.text}" if self.icon else self.text
        self.create_text(self.w//2, self.h//2, text=txt, fill=self.tc,
                         font=(Theme.FONT, 10, "bold"))

    def _animate(self):
        target = 1.0 if self.hovering else 0.0
        if abs(self.hover - target) > 0.02:
            self.hover += (target - self.hover) * 0.3
            self._draw()
            self.after(16, self._animate)
        else:
            self.hover = target
            self._draw()
            self.animating = False

    def _enter(self, e):
        self.config(cursor="hand2")
        self.hovering = True
        if not self.animating:
            self.animating = True
            self._animate()

    def _leave(self, e):
        self.hovering = False
        if not self.animating:
            self.animating = True
            self._animate()

    def _click(self, e):
        if self.command:
            self.command()


class Entry(tk.Frame):
    def __init__(self, parent, placeholder="", show=None):
        super().__init__(parent, bg=parent.cget('bg'))

        self.border = tk.Frame(self, bg=Theme.BORDER)
        self.border.pack(fill="x", ipady=1, ipadx=1)

        inner = tk.Frame(self.border, bg=Theme.BG_INPUT)
        inner.pack(fill="x", padx=1, pady=1)

        self.entry = tk.Entry(inner, bg=Theme.BG_INPUT, fg=Theme.TEXT,
                              insertbackground=Theme.BLUE, relief="flat",
                              font=(Theme.FONT, 11), show=show or "")
        self.entry.pack(fill="x", padx=14, pady=12)

        self.entry.bind(
            "<FocusIn>", lambda e: self.border.config(bg=Theme.BLUE))
        self.entry.bind(
            "<FocusOut>", lambda e: self.border.config(bg=Theme.BORDER))

    def get(self):
        return self.entry.get()

    def delete(self, a, b):
        self.entry.delete(a, b)

    def insert(self, i, s):
        self.entry.insert(i, s)


class TextArea(tk.Frame):
    def __init__(self, parent, height=5):
        super().__init__(parent, bg=parent.cget('bg'))

        self.border = tk.Frame(self, bg=Theme.BORDER)
        self.border.pack(fill="both", expand=True, ipady=1, ipadx=1)

        inner = tk.Frame(self.border, bg=Theme.BG_INPUT)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        # Scrollbar
        scrollbar = tk.Scrollbar(
            inner, bg=Theme.BG_INPUT, troughcolor=Theme.BG_INPUT)
        scrollbar.pack(side="right", fill="y")

        self.text = tk.Text(inner, bg=Theme.BG_INPUT, fg=Theme.TEXT,
                            insertbackground=Theme.BLUE, relief="flat",
                            font=(Theme.MONO, 10), height=height, wrap="word",
                            padx=12, pady=10, selectbackground=Theme.BLUE,
                            selectforeground="#fff", highlightthickness=0,
                            yscrollcommand=scrollbar.set)
        self.text.pack(fill="both", expand=True, side="left")
        scrollbar.config(command=self.text.yview)

        self.text.bind(
            "<FocusIn>", lambda e: self.border.config(bg=Theme.BLUE))
        self.text.bind(
            "<FocusOut>", lambda e: self.border.config(bg=Theme.BORDER))

    def get(self):
        return self.text.get("1.0", "end-1c")

    def delete(self):
        self.text.delete("1.0", "end")

    def insert(self, txt):
        self.text.insert("1.0", txt)

    def bind_key(self, key, fn):
        self.text.bind(key, fn)


class Card(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=Theme.BG_CARD,
                         highlightbackground=Theme.BORDER, highlightthickness=1)


class Dropzone(tk.Frame):
    def __init__(self, parent, on_load=None, on_reset=None):
        super().__init__(parent, bg=parent.cget('bg'))

        self.on_load = on_load
        self.on_reset = on_reset
        self.image_path = None
        self.photo = None

        self.border = tk.Frame(self, bg=Theme.BORDER)
        self.border.pack(fill="both", expand=True)

        self.inner = tk.Frame(self.border, bg=Theme.BG_CARD)
        self.inner.pack(fill="both", expand=True, padx=2, pady=2)

        self.content = tk.Frame(self.inner, bg=Theme.BG_CARD)
        self.content.pack(fill="both", expand=True, padx=20, pady=20)

        self.icon = tk.Label(self.content, text="🖼️", font=(Theme.FONT, 40),
                             bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED)
        self.icon.pack(pady=(20, 8))

        self.title = tk.Label(self.content, text="Click to select image",
                              font=(Theme.FONT, 12, "bold"),
                              bg=Theme.BG_CARD, fg=Theme.TEXT)
        self.title.pack()

        self.subtitle = tk.Label(self.content, text="PNG, JPG, BMP",
                                 font=(Theme.FONT, 10),
                                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED)
        self.subtitle.pack(pady=(4, 20))

        self.preview = tk.Label(self.content, bg=Theme.BG_CARD)
        self.info = tk.Label(self.content, bg=Theme.BG_CARD, fg=Theme.TEXT_SEC,
                             font=(Theme.FONT, 9))
        self.remove_btn = Button(self.content, "Remove", self.reset,
                                 "danger", 100, 30, "✕")

        for w in [self.inner, self.content, self.icon, self.title, self.subtitle]:
            w.bind("<Button-1>", self._browse)
            w.bind("<Enter>", lambda e: self.border.config(bg=Theme.BLUE))
            w.bind("<Leave>", lambda e: self.border.config(bg=Theme.BORDER))
            w.config(cursor="hand2")

    def _browse(self, e=None):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if path:
            self._load(path)

    def _load(self, path):
        try:
            self.image_path = path

            self.icon.pack_forget()
            self.title.pack_forget()
            self.subtitle.pack_forget()

            img = Image.open(path)
            dims = f"{img.width} × {img.height}"
            img.thumbnail((220, 220))
            self.photo = ImageTk.PhotoImage(img)

            self.preview.config(image=self.photo)
            self.preview.pack(pady=(5, 0))

            size_kb = os.path.getsize(path) / 1024
            name = os.path.basename(path)
            if len(name) > 25:
                name = name[:22] + "..."
            self.info.config(text=f"{name}\n{dims}  •  {size_kb:.1f} KB")
            self.info.pack(pady=(8, 0))

            self.remove_btn.pack(pady=(8, 0))

            if self.on_load:
                self.on_load(path)
        except Exception as e:
            ThemedMessageBox.showerror("Error", f"Could not load image:\n{e}")
            self.reset()

    def reset(self):
        self.image_path = None
        self.photo = None

        self.preview.pack_forget()
        self.info.pack_forget()
        self.remove_btn.pack_forget()

        self.icon.pack(pady=(20, 8))
        self.title.pack()
        self.subtitle.pack(pady=(4, 20))

        if self.on_reset:
            self.on_reset()


class NavItem(tk.Frame):
    def __init__(self, parent, icon, text, active=False, command=None):
        super().__init__(parent, bg=Theme.SIDEBAR, cursor="hand2")

        self.active = active
        self.command = command

        self.indicator = tk.Frame(self, width=3,
                                  bg=Theme.BLUE if active else Theme.SIDEBAR)
        self.indicator.pack(side="left", fill="y")

        bg = Theme.SIDEBAR_ACTIVE if active else Theme.SIDEBAR
        self.content = tk.Frame(self, bg=bg)
        self.content.pack(side="left", fill="both", expand=True)

        self.icon_lbl = tk.Label(self.content, text=icon, font=(Theme.FONT, 13),
                                 bg=bg, fg=Theme.BLUE if active else Theme.TEXT_MUTED)
        self.icon_lbl.pack(side="left", padx=(14, 8), pady=11)

        self.text_lbl = tk.Label(self.content, text=text,
                                 font=(Theme.FONT, 10, "bold"), bg=bg,
                                 fg=Theme.TEXT if active else Theme.TEXT_SEC)
        self.text_lbl.pack(side="left", pady=11)

        for w in [self, self.content, self.icon_lbl, self.text_lbl]:
            w.bind("<Button-1>", lambda e: command() if command else None)
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)

    def _enter(self, e):
        if not self.active:
            self._set_bg(Theme.SIDEBAR_HOVER)

    def _leave(self, e):
        if not self.active:
            self._set_bg(Theme.SIDEBAR)

    def _set_bg(self, bg):
        self.content.config(bg=bg)
        self.icon_lbl.config(bg=bg)
        self.text_lbl.config(bg=bg)

    def set_active(self, active):
        self.active = active
        bg = Theme.SIDEBAR_ACTIVE if active else Theme.SIDEBAR
        self.indicator.config(bg=Theme.BLUE if active else Theme.SIDEBAR)
        self._set_bg(bg)
        self.icon_lbl.config(fg=Theme.BLUE if active else Theme.TEXT_MUTED)
        self.text_lbl.config(fg=Theme.TEXT if active else Theme.TEXT_SEC)


class MetricCard(tk.Frame):
    def __init__(self, parent, icon, title, color=Theme.BLUE):
        super().__init__(parent, bg=Theme.BG_CARD,
                         highlightbackground=Theme.BORDER, highlightthickness=1)

        c = tk.Frame(self, bg=Theme.BG_CARD)
        c.pack(fill="both", expand=True, padx=14, pady=10)

        header = tk.Frame(c, bg=Theme.BG_CARD)
        header.pack(fill="x")

        tk.Label(header, text=icon, font=(Theme.FONT, 14),
                 bg=Theme.BG_CARD, fg=color).pack(side="left")
        tk.Label(header, text=title, font=(Theme.FONT, 9),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left", padx=(6, 0))

        self.value = tk.Label(c, text="--", font=(Theme.FONT, 18, "bold"),
                              bg=Theme.BG_CARD, fg=color)
        self.value.pack(anchor="w", pady=(6, 0))

    def set(self, val):
        self.value.config(text=val)


class ProgressBar(tk.Canvas):
    def __init__(self, parent, width=200, height=4):
        super().__init__(parent, width=width, height=height,
                         bg=Theme.BG_CARD, highlightthickness=0)
        self.w = width
        self.h = height
        self.progress = 0
        self._draw()

    def _draw(self):
        self.delete("all")
        self.create_rectangle(0, 0, self.w, self.h,
                              fill=Theme.BG_HOVER, outline="")
        if self.progress > 0:
            pw = (self.progress / 100) * self.w
            self.create_rectangle(
                0, 0, pw, self.h, fill=Theme.BLUE, outline="")

    def set(self, val):
        self.progress = max(0, min(100, val))
        self._draw()


class HistogramCanvas(tk.Canvas):
    """Canvas for drawing histograms."""

    def __init__(self, parent, width=300, height=150):
        super().__init__(parent, width=width, height=height,
                         bg=Theme.BG_CARD, highlightthickness=0)
        self.w = width
        self.h = height

    def draw_histogram(self, histogram_data, color=Theme.BLUE):
        self.delete("all")

        if not histogram_data:
            self.create_text(self.w//2, self.h//2, text="No data",
                             fill=Theme.TEXT_MUTED, font=(Theme.FONT, 10))
            return

        # Normalize data
        max_val = max(histogram_data) if max(histogram_data) > 0 else 1
        bar_width = self.w / 256

        for i, val in enumerate(histogram_data):
            if val > 0:
                bar_height = (val / max_val) * (self.h - 20)
                x = i * bar_width
                self.create_rectangle(
                    x, self.h - bar_height,
                    x + bar_width, self.h,
                    fill=color, outline=""
                )

        # Axis labels
        self.create_text(10, 10, text="Frequency", anchor="nw",
                         fill=Theme.TEXT_MUTED, font=(Theme.FONT, 8))
        self.create_text(self.w - 10, self.h - 5, text="Pixel Value (0-255)", anchor="se",
                         fill=Theme.TEXT_MUTED, font=(Theme.FONT, 8))


# -------------------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CyberMaths Steganography")
        self.root.geometry("1200x800")
        self.root.configure(bg=Theme.BG_DARK)
        self.root.minsize(1000, 700)

        self.src_path = None
        self.stego_path = None
        self.current_tab = 0

        self._build()

    def _build(self):
        main = tk.Frame(self.root, bg=Theme.BG_DARK)
        main.pack(fill="both", expand=True)

        self._sidebar(main)

        self.main_area = tk.Frame(main, bg=Theme.BG_MAIN)
        self.main_area.pack(side="left", fill="both", expand=True)

        self._header()

        self.content = tk.Frame(self.main_area, bg=Theme.BG_MAIN)
        self.content.pack(fill="both", expand=True)

        self.tabs = []
        self._encode_tab()
        self._decode_tab()
        self._analysis_tab()
        self._math_tab()

        self._statusbar()
        self._switch(0)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=Theme.SIDEBAR, width=200)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # Logo
        logo = tk.Frame(sb, bg=Theme.SIDEBAR)
        logo.pack(fill="x", pady=(20, 24))

        canvas = tk.Canvas(logo, width=44, height=44, bg=Theme.SIDEBAR,
                           highlightthickness=0)
        canvas.pack()
        canvas.create_oval(4, 4, 40, 40, fill=Theme.BLUE, outline="")
        canvas.create_text(22, 22, text="∑", font=(
            Theme.FONT, 16, "bold"), fill="#fff")

        tk.Label(logo, text="CyberMaths", font=(Theme.FONT, 14, "bold"),
                 bg=Theme.SIDEBAR, fg=Theme.TEXT).pack(pady=(6, 0))
        tk.Label(logo, text="Steganography", font=(Theme.FONT, 9),
                 bg=Theme.SIDEBAR, fg=Theme.TEXT_MUTED).pack()

        tk.Frame(sb, bg=Theme.BORDER, height=1).pack(fill="x", padx=14, pady=6)

        # Nav
        tk.Label(sb, text="MAIN", font=(Theme.FONT, 8, "bold"),
                 bg=Theme.SIDEBAR, fg=Theme.TEXT_MUTED).pack(anchor="w", padx=14, pady=(10, 6))

        self.nav = []
        items = [
            ("🔒", "Encode"),
            ("🔓", "Decode"),
            ("📊", "Analysis"),
            ("📐", "Mathematics")
        ]
        for i, (icon, text) in enumerate(items):
            item = NavItem(sb, icon, text, i == 0,
                           lambda idx=i: self._switch(idx))
            item.pack(fill="x")
            self.nav.append(item)

        tk.Frame(sb, bg=Theme.BORDER, height=1).pack(
            fill="x", padx=14, pady=10)

        # Formulas section
        tk.Label(sb, text="KEY FORMULAS", font=(Theme.FONT, 8, "bold"),
                 bg=Theme.SIDEBAR, fg=Theme.TEXT_MUTED).pack(anchor="w", padx=14, pady=(4, 6))

        formulas = [
            ("MSE", "Σ(I-K)²/mn"),
            ("PSNR", "20·log₁₀(MAX/√MSE)"),
            ("Entropy", "-Σp·log₂(p)"),
            ("XOR", "A ⊕ B"),
        ]

        for name, formula in formulas:
            row = tk.Frame(sb, bg=Theme.SIDEBAR)
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=name, font=(Theme.MONO, 8),
                     bg=Theme.SIDEBAR, fg=Theme.TEXT_MUTED, width=8, anchor="w").pack(side="left")
            tk.Label(row, text=formula, font=(Theme.MONO, 8),
                     bg=Theme.SIDEBAR, fg=Theme.CYAN).pack(side="left")

        # Version
        tk.Label(sb, text="v3.0 Math Edition", font=(Theme.FONT, 8),
                 bg=Theme.SIDEBAR, fg=Theme.TEXT_MUTED).pack(side="bottom", pady=14)

    def _header(self):
        hdr = tk.Frame(self.main_area, bg=Theme.BG_SECONDARY, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        self.title = tk.Label(hdr, text="🔒  Encode Message",
                              font=(Theme.FONT, 13, "bold"),
                              bg=Theme.BG_SECONDARY, fg=Theme.TEXT)
        self.title.pack(side="left", padx=18, pady=14)

        tk.Frame(self.main_area, bg=Theme.BORDER, height=1).pack(fill="x")

    def _statusbar(self):
        tk.Frame(self.main_area, bg=Theme.BORDER,
                 height=1).pack(side="bottom", fill="x")

        bar = tk.Frame(self.main_area, bg=Theme.BG_SECONDARY, height=26)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)

        self.status = tk.Label(bar, text="Ready", font=(Theme.FONT, 9),
                               bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED)
        self.status.pack(side="left", padx=14, pady=5)

    def _encode_tab(self):
        frame = tk.Frame(self.content, bg=Theme.BG_MAIN)
        self.tabs.append(frame)

        container = tk.Frame(frame, bg=Theme.BG_MAIN)
        container.pack(fill="both", expand=True, padx=18, pady=18)

        # Left
        left = tk.Frame(container, bg=Theme.BG_MAIN)
        left.pack(side="left", fill="both", expand=True, padx=(0, 9))

        card1 = Card(left)
        card1.pack(fill="both", expand=True)

        c1 = tk.Frame(card1, bg=Theme.BG_CARD)
        c1.pack(fill="both", expand=True, padx=14, pady=14)

        tk.Label(c1, text="📁  Cover Image", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w", pady=(0, 10))

        self.enc_drop = Dropzone(c1, self._enc_load, self._enc_reset)
        self.enc_drop.pack(fill="both", expand=True)

        # Capacity
        cap_frame = tk.Frame(c1, bg=Theme.BG_CARD)
        cap_frame.pack(fill="x", pady=(10, 0))

        cap_hdr = tk.Frame(cap_frame, bg=Theme.BG_CARD)
        cap_hdr.pack(fill="x")

        tk.Label(cap_hdr, text="Capacity", font=(Theme.FONT, 9),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left")

        self.cap_pct = tk.Label(cap_hdr, text="0%", font=(Theme.FONT, 9, "bold"),
                                bg=Theme.BG_CARD, fg=Theme.BLUE)
        self.cap_pct.pack(side="right")

        self.cap_bar = ProgressBar(cap_frame, 240, 4)
        self.cap_bar.pack(fill="x", pady=(5, 0))

        self.cap_info = tk.Label(cap_frame, text="Select an image",
                                 font=(Theme.FONT, 9), bg=Theme.BG_CARD,
                                 fg=Theme.TEXT_MUTED)
        self.cap_info.pack(anchor="w", pady=(3, 0))

        # Right
        right = tk.Frame(container, bg=Theme.BG_MAIN)
        right.pack(side="right", fill="both", expand=True, padx=(9, 0))

        # Message
        card2 = Card(right)
        card2.pack(fill="x")

        c2 = tk.Frame(card2, bg=Theme.BG_CARD)
        c2.pack(fill="both", padx=14, pady=14)

        hdr2 = tk.Frame(c2, bg=Theme.BG_CARD)
        hdr2.pack(fill="x", pady=(0, 6))

        tk.Label(hdr2, text="💬  Secret Message", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(side="left")

        self.char_lbl = tk.Label(hdr2, text="0 chars", font=(Theme.FONT, 9),
                                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED)
        self.char_lbl.pack(side="right")

        self.msg_txt = TextArea(c2, 4)
        self.msg_txt.pack(fill="x")
        self.msg_txt.bind_key("<KeyRelease>", self._msg_change)

        # Password
        card3 = Card(right)
        card3.pack(fill="x", pady=(10, 0))

        c3 = tk.Frame(card3, bg=Theme.BG_CARD)
        c3.pack(fill="both", padx=14, pady=14)

        tk.Label(c3, text="🔑  Password", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w")
        tk.Label(c3, text="XOR cipher: result = message ⊕ key",
                 font=(Theme.MONO, 9), bg=Theme.BG_CARD,
                 fg=Theme.CYAN).pack(anchor="w", pady=(2, 6))

        self.enc_pass = Entry(c3, show="●")
        self.enc_pass.pack(fill="x")

        # Button
        Button(right, "Encrypt & Save Image", self._encode, "primary",
               240, 44, "🔐").pack(pady=(14, 0))

    def _decode_tab(self):
        frame = tk.Frame(self.content, bg=Theme.BG_MAIN)
        self.tabs.append(frame)

        container = tk.Frame(frame, bg=Theme.BG_MAIN)
        container.pack(fill="both", expand=True, padx=18, pady=18)

        # Left
        left = tk.Frame(container, bg=Theme.BG_MAIN)
        left.pack(side="left", fill="both", expand=True, padx=(0, 9))

        card1 = Card(left)
        card1.pack(fill="both", expand=True)

        c1 = tk.Frame(card1, bg=Theme.BG_CARD)
        c1.pack(fill="both", expand=True, padx=14, pady=14)

        tk.Label(c1, text="🖼️  Stego Image", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w", pady=(0, 10))

        self.dec_drop = Dropzone(c1, self._dec_load, self._dec_reset)
        self.dec_drop.pack(fill="both", expand=True)

        # Right
        right = tk.Frame(container, bg=Theme.BG_MAIN)
        right.pack(side="right", fill="both", expand=True, padx=(9, 0))

        # Password
        card2 = Card(right)
        card2.pack(fill="x")

        c2 = tk.Frame(card2, bg=Theme.BG_CARD)
        c2.pack(fill="both", padx=14, pady=14)

        tk.Label(c2, text="🔑  Password", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w", pady=(0, 6))

        self.dec_pass = Entry(c2, show="●")
        self.dec_pass.pack(fill="x")

        Button(right, "Reveal Message", self._decode, "success",
               180, 40, "🔍").pack(pady=(10, 0))

        # Output
        card3 = Card(right)
        card3.pack(fill="both", expand=True, pady=(10, 0))

        c3 = tk.Frame(card3, bg=Theme.BG_CARD)
        c3.pack(fill="both", expand=True, padx=14, pady=14)

        hdr3 = tk.Frame(c3, bg=Theme.BG_CARD)
        hdr3.pack(fill="x", pady=(0, 6))

        tk.Label(hdr3, text="📝  Message", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(side="left")

        Button(hdr3, "Copy", self._copy, "ghost",
               65, 26, "📋").pack(side="right")

        self.output = TextArea(c3, 6)
        self.output.pack(fill="both", expand=True)

    def _analysis_tab(self):
        frame = tk.Frame(self.content, bg=Theme.BG_MAIN)
        self.tabs.append(frame)

        # Scrollable container
        canvas = tk.Canvas(frame, bg=Theme.BG_MAIN, highlightthickness=0)
        scrollbar = tk.Scrollbar(
            frame, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=Theme.BG_MAIN)

        scrollable.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        container = tk.Frame(scrollable, bg=Theme.BG_MAIN)
        container.pack(fill="both", expand=True, padx=18, pady=18)

        # Header
        tk.Label(container, text="📊  Quality Analysis",
                 font=(Theme.FONT, 14, "bold"),
                 bg=Theme.BG_MAIN, fg=Theme.TEXT).pack(anchor="w")
        tk.Label(container, text="Mathematical comparison of original vs stego image",
                 font=(Theme.FONT, 10), bg=Theme.BG_MAIN,
                 fg=Theme.TEXT_SEC).pack(anchor="w", pady=(2, 14))

        # Quality Metrics
        metrics = tk.Frame(container, bg=Theme.BG_MAIN)
        metrics.pack(fill="x", pady=(0, 12))

        self.mse = MetricCard(metrics, "📉", "MSE (Mean Squared Error)", Theme.BLUE)
        self.mse.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.psnr = MetricCard(metrics, "📈", "PSNR (Peak Signal-to-Noise Ratio)", Theme.GREEN)
        self.psnr.pack(side="left", fill="both", expand=True, padx=5)

        self.ssim_card = MetricCard(metrics, "🔗", "SSIM (Structural Similarity Index)", Theme.PURPLE)
        self.ssim_card.pack(side="left", fill="both", expand=True, padx=5)

        self.corr = MetricCard(metrics, "📊", "Correlation", Theme.CYAN)
        self.corr.pack(side="left", fill="both", expand=True, padx=(5, 0))

        Button(container, "Calculate All Metrics", self._analyze, "primary",
               200, 40, "🔬").pack(pady=(0, 14))

        # Entropy & Statistics
        stats_row = tk.Frame(container, bg=Theme.BG_MAIN)
        stats_row.pack(fill="x", pady=(0, 12))

        # Entropy card
        ent_card = Card(stats_row)
        ent_card.pack(side="left", fill="both", expand=True, padx=(0, 5))

        ent_c = tk.Frame(ent_card, bg=Theme.BG_CARD)
        ent_c.pack(fill="both", expand=True, padx=14, pady=14)

        tk.Label(ent_c, text="🎲  Shannon Entropy", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w")
        tk.Label(ent_c, text="H(X) = -Σ p(x) · log₂(p(x))",
                 font=(Theme.MONO, 9), bg=Theme.BG_CARD, fg=Theme.CYAN).pack(anchor="w", pady=(2, 8))

        self.entropy_orig = tk.Label(ent_c, text="Original: --", font=(Theme.FONT, 10),
                                     bg=Theme.BG_CARD, fg=Theme.TEXT_SEC)
        self.entropy_orig.pack(anchor="w")
        self.entropy_stego = tk.Label(ent_c, text="Stego: --", font=(Theme.FONT, 10),
                                      bg=Theme.BG_CARD, fg=Theme.TEXT_SEC)
        self.entropy_stego.pack(anchor="w")

        # Chi-Square card
        chi_card = Card(stats_row)
        chi_card.pack(side="left", fill="both", expand=True, padx=(5, 0))

        chi_c = tk.Frame(chi_card, bg=Theme.BG_CARD)
        chi_c.pack(fill="both", expand=True, padx=14, pady=14)

        tk.Label(chi_c, text="📐  Chi-Square Test", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w")
        tk.Label(chi_c, text="χ² = Σ (O - E)² / E",
                 font=(Theme.MONO, 9), bg=Theme.BG_CARD, fg=Theme.CYAN).pack(anchor="w", pady=(2, 8))

        self.chi_value = tk.Label(chi_c, text="χ² value: --", font=(Theme.FONT, 10),
                                  bg=Theme.BG_CARD, fg=Theme.TEXT_SEC)
        self.chi_value.pack(anchor="w")
        self.chi_result = tk.Label(chi_c, text="Detection: --", font=(Theme.FONT, 10),
                                   bg=Theme.BG_CARD, fg=Theme.TEXT_SEC)
        self.chi_result.pack(anchor="w")

        # LSB Analysis
        lsb_card = Card(container)
        lsb_card.pack(fill="x", pady=(0, 12))

        lsb_c = tk.Frame(lsb_card, bg=Theme.BG_CARD)
        lsb_c.pack(fill="both", padx=14, pady=14)

        tk.Label(lsb_c, text="🔢  LSB Distribution", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w")
        tk.Label(lsb_c, text="Analysis of Least Significant Bits (natural images ≈ 50/50)",
                 font=(Theme.FONT, 9), bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 8))

        self.lsb_info = tk.Label(lsb_c, text="Load an image to analyze LSB distribution",
                                 font=(Theme.MONO, 10), bg=Theme.BG_CARD, fg=Theme.TEXT_SEC)
        self.lsb_info.pack(anchor="w")

        # Histogram
        hist_card = Card(container)
        hist_card.pack(fill="x")

        hist_c = tk.Frame(hist_card, bg=Theme.BG_CARD)
        hist_c.pack(fill="both", padx=14, pady=14)

        tk.Label(hist_c, text="📊  Histogram", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w", pady=(0, 8))

        self.histogram = HistogramCanvas(hist_c, 400, 120)
        self.histogram.pack(anchor="w")

    def _math_tab(self):
        """Mathematics demonstration tab."""
        frame = tk.Frame(self.content, bg=Theme.BG_MAIN)
        self.tabs.append(frame)

        # Scrollable
        canvas = tk.Canvas(frame, bg=Theme.BG_MAIN, highlightthickness=0)
        scrollbar = tk.Scrollbar(
            frame, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=Theme.BG_MAIN)

        scrollable.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        container = tk.Frame(scrollable, bg=Theme.BG_MAIN)
        container.pack(fill="both", expand=True, padx=18, pady=18)

        # Header
        tk.Label(container, text="📐  Mathematics Explained",
                 font=(Theme.FONT, 14, "bold"),
                 bg=Theme.BG_MAIN, fg=Theme.TEXT).pack(anchor="w")
        tk.Label(container, text="Understanding the math behind steganography",
                 font=(Theme.FONT, 10), bg=Theme.BG_MAIN,
                 fg=Theme.TEXT_SEC).pack(anchor="w", pady=(2, 14))


        # Base64 Encoding
        b64_card = Card(container)
        b64_card.pack(fill="x", pady=(0, 12))

        b64_c = tk.Frame(b64_card, bg=Theme.BG_CARD)
        b64_c.pack(fill="both", padx=14, pady=14)

        tk.Label(b64_c, text="🔤  Base64 Encoding", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w")
        tk.Label(b64_c, text="Convert text to Base64 for data transmission",
                 font=(Theme.FONT, 9), bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 8))

        b64_input_frame = tk.Frame(b64_c, bg=Theme.BG_CARD)
        b64_input_frame.pack(fill="x", pady=(0, 8))

        tk.Label(b64_input_frame, text="Text:", font=(Theme.FONT, 10),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SEC).pack(side="left")

        self.b64_entry = Entry(b64_input_frame)
        self.b64_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))

        Button(b64_input_frame, "Encode", self._encode_base64,
               "cyan", 90, 34).pack(side="left")

        b64_result_frame = tk.Frame(b64_c, bg=Theme.BG_CARD)
        b64_result_frame.pack(fill="x", pady=(8, 0))

        self.b64_result = tk.Label(b64_result_frame, text="", font=(Theme.MONO, 9),
                                   bg=Theme.BG_CARD, fg=Theme.CYAN,
                                   justify="left", anchor="w", wraplength=400)
        self.b64_result.pack(side="left", fill="both", expand=True)

        Button(b64_result_frame, "Copy", self._copy_base64_result,
               "purple", 70, 34).pack(side="left", padx=(8, 0))

        # Binary Conversion
        bin_card = Card(container)
        bin_card.pack(fill="x", pady=(0, 12))

        bin_c = tk.Frame(bin_card, bg=Theme.BG_CARD)
        bin_c.pack(fill="both", padx=14, pady=14)

        tk.Label(bin_c, text="🔢  Binary Conversion", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w")
        tk.Label(bin_c, text="Each character → ASCII → 8-bit binary",
                 font=(Theme.FONT, 9), bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 8))

        bin_input_frame = tk.Frame(bin_c, bg=Theme.BG_CARD)
        bin_input_frame.pack(fill="x", pady=(0, 8))

        tk.Label(bin_input_frame, text="Text:", font=(Theme.FONT, 10),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SEC).pack(side="left")

        self.bin_entry = Entry(bin_input_frame)
        self.bin_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))

        Button(bin_input_frame, "Convert", self._convert_binary,
               "cyan", 90, 34).pack(side="left")

        bin_result_frame = tk.Frame(bin_c, bg=Theme.BG_CARD)
        bin_result_frame.pack(fill="x", pady=(8, 0))

        self.bin_result = tk.Label(bin_result_frame, text="", font=(Theme.MONO, 9),
                                   bg=Theme.BG_CARD, fg=Theme.CYAN,
                                   justify="left", anchor="w")
        self.bin_result.pack(side="left", fill="both", expand=True)

        Button(bin_result_frame, "Copy", self._copy_binary_result,
               "purple", 70, 34).pack(side="left", padx=(8, 0))

        # XOR Demonstration
        xor_card = Card(container)
        xor_card.pack(fill="x", pady=(0, 12))

        xor_c = tk.Frame(xor_card, bg=Theme.BG_CARD)
        xor_c.pack(fill="both", padx=14, pady=14)

        tk.Label(xor_c, text="⊕  XOR Encryption", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w")
        tk.Label(xor_c, text="XOR Truth Table: 0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0",
                 font=(Theme.MONO, 9), bg=Theme.BG_CARD, fg=Theme.CYAN).pack(anchor="w", pady=(2, 8))

        xor_input = tk.Frame(xor_c, bg=Theme.BG_CARD)
        xor_input.pack(fill="x", pady=(0, 8))

        tk.Label(xor_input, text="Message:", font=(Theme.FONT, 10),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SEC).pack(side="left")
        self.xor_msg = Entry(xor_input)
        self.xor_msg.pack(side="left", fill="x", expand=True, padx=8)

        tk.Label(xor_input, text="Key:", font=(Theme.FONT, 10),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SEC).pack(side="left")
        self.xor_key = Entry(xor_input)
        self.xor_key.pack(side="left", fill="x", expand=True, padx=8)

        Button(xor_input, "Encrypt", self._demo_xor,
               "purple", 90, 34).pack(side="left")

        xor_result_frame = tk.Frame(xor_c, bg=Theme.BG_CARD)
        xor_result_frame.pack(fill="x", pady=(8, 0))

        self.xor_result = tk.Label(xor_result_frame, text="", font=(Theme.MONO, 9),
                                   bg=Theme.BG_CARD, fg=Theme.PURPLE,
                                   justify="left", anchor="w")
        self.xor_result.pack(side="left", fill="both", expand=True)

        Button(xor_result_frame, "Copy", self._copy_xor_result,
               "purple", 70, 34).pack(side="left", padx=(8, 0))

        # LSB Explanation
        lsb_card = Card(container)
        lsb_card.pack(fill="x", pady=(0, 12))

        lsb_c = tk.Frame(lsb_card, bg=Theme.BG_CARD)
        lsb_c.pack(fill="both", padx=14, pady=14)

        tk.Label(lsb_c, text="🎨  LSB Steganography", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w")

        explanation = """
How LSB works:
┌─────────────────────────────────────────────────────────┐
│ Original pixel value:  156  =  10011100  (binary)       │
│ Message bit to hide:                  1                 │
│ ─────────────────────────────────────────               │
│ New pixel value:       157  =  10011101  (binary)       │
│                                      ↑                  │
│                               LSB changed               │
└─────────────────────────────────────────────────────────┘

• Only the last bit (LSB) is modified
• Change of ±1 in pixel value is invisible to human eye
• Each pixel can store 1 bit of secret message
• RGB image: 3 bits per pixel (one per channel)
"""
        tk.Label(lsb_c, text=explanation, font=(Theme.MONO, 9),
                 bg=Theme.BG_CARD, fg=Theme.GREEN,
                 justify="left", anchor="w").pack(anchor="w")

        # Formulas Reference
        form_card = Card(container)
        form_card.pack(fill="x")

        form_c = tk.Frame(form_card, bg=Theme.BG_CARD)
        form_c.pack(fill="both", padx=14, pady=14)

        tk.Label(form_c, text="📚  Mathematical Formulas", font=(Theme.FONT, 11, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT).pack(anchor="w", pady=(0, 8))

        formulas_text = """
IMAGE QUALITY METRICS
────────────────────────────────────────

Mean Squared Error (MSE):
  MSE = (1 / (m * n)) * ΣΣ [ I(i, j) - K(i, j) ]²

Peak Signal-to-Noise Ratio (PSNR):
  PSNR = 10 * log10(MAX² / MSE)
  PSNR = 20 * log10(MAX / sqrt(MSE))
  For 8-bit images: MAX = 255

Shannon Entropy:
  H(X) = - Σ p(xᵢ) * log2(p(xᵢ))
  Max entropy for 8-bit image = 8.0 bits

Structural Similarity Index (SSIM):
  SSIM = ((2 * μx * μy + C1) * (2 * σxy + C2)) /
         ((μx² + μy² + C1) * (σx² + σy² + C2))

Chi-Square Test:
  χ² = Σ (Oi - Ei)² / Ei
  O = Observed, E = Expected

Pearson Correlation Coefficient:
  r = Σ (xi - x̄)(yi - ȳ) /
      sqrt( Σ(xi - x̄)² * Σ(yi - ȳ)² )
"""

        tk.Label(form_c, text=formulas_text, font=(Theme.MONO, 9),
                 bg=Theme.BG_CARD, fg=Theme.ORANGE,
                 justify="left", anchor="w").pack(anchor="w")

    # -------------------------------------------------------------------------
    # HANDLERS
    # -------------------------------------------------------------------------

    def _enc_load(self, path):
        self.src_path = path
        self._set_status(f"Loaded: {os.path.basename(path)}")
        self._update_cap()

    def _enc_reset(self):
        self.src_path = None
        self._set_status("Ready")
        self.cap_bar.set(0)
        self.cap_pct.config(text="0%")
        self.cap_info.config(text="Select an image")

    def _dec_load(self, path):
        self.stego_path = path
        self._set_status(f"Loaded: {os.path.basename(path)}")

        # Update histogram
        hist_data = MathEngine.calculate_histogram(path)
        if 'gray' in hist_data:
            self.histogram.draw_histogram(hist_data['gray'], Theme.TEXT_SEC)
        else:
            self.histogram.draw_histogram(hist_data['red'], Theme.RED)

        # Update LSB analysis
        lsb = MathEngine.analyze_lsb_distribution(path)
        self.lsb_info.config(
            text=f"Zeros: {lsb['zeros']:,} ({lsb['zero_ratio']*100:.2f}%)  |  "
            f"Ones: {lsb['ones']:,} ({lsb['one_ratio']*100:.2f}%)  |  "
            f"Balance deviation: {lsb['balance']*100:.3f}%"
        )

        # Update entropy
        entropy = MathEngine.calculate_entropy(path)
        self.entropy_stego.config(text=f"Stego: {entropy:.4f} bits")

    def _dec_reset(self):
        self.stego_path = None
        self._set_status("Ready")

    def _msg_change(self, e=None):
        length = len(self.msg_txt.get())
        self.char_lbl.config(text=f"{length:,} chars")
        self._update_cap()

    def _update_cap(self):
        if not self.src_path:
            return

        cap = MathEngine.calculate_capacity_details(self.src_path)
        used = len(self.msg_txt.get())

        if cap['max_chars'] > 0:
            pct = min(100, (used / cap['max_chars']) * 100)
            self.cap_bar.set(pct)
            self.cap_pct.config(text=f"{pct:.1f}%")
            self.cap_info.config(text=f"{used:,} / {cap['max_chars']:,} chars  "
                                 f"({cap['width']}×{cap['height']} = {cap['total_pixels']:,} pixels)")

            if pct > 90:
                self.cap_pct.config(fg=Theme.RED)
            elif pct > 70:
                self.cap_pct.config(fg=Theme.ORANGE)
            else:
                self.cap_pct.config(fg=Theme.BLUE)

        # Update original entropy
        entropy = MathEngine.calculate_entropy(self.src_path)
        self.entropy_orig.config(text=f"Original: {entropy:.4f} bits")

    def _encode(self):
        if not self.src_path:
            ThemedMessageBox.showerror("Error", "Please select an image!")
            return

        msg = self.msg_txt.get().strip()
        if not msg:
            ThemedMessageBox.showerror("Error", "Please enter a message!")
            return

        pwd = self.enc_pass.get() or None

        try:
            self._set_status("Encoding...")
            self.root.update()

            stego = StegoEngine.encode(self.src_path, msg, pwd)

            save = filedialog.asksaveasfilename(
                title="Save Stego Image",
                defaultextension=".png",
                filetypes=[("PNG", "*.png")],
                initialfile="stego_image.png"
            )

            if save:
                stego.save(save)
                self.stego_path = save
                self._set_status(f"Saved: {os.path.basename(save)}")
                ThemedMessageBox.showinfo(
                    "Success ✓", f"Message hidden!\nSaved to: {save}")
        except Exception as e:
            self._set_status("Error")
            ThemedMessageBox.showerror("Error", str(e))

    def _decode(self):
        if not self.stego_path:
            ThemedMessageBox.showerror("Error", "Please select a stego image!")
            return

        pwd = self.dec_pass.get() or None

        try:
            self._set_status("Decoding...")
            self.root.update()

            msg = StegoEngine.decode(self.stego_path, pwd)

            self.output.delete()
            self.output.insert(msg)

            self._set_status("Done")

            if not msg.startswith("No hidden"):
                ThemedMessageBox.showinfo("Success ✓", "Message extracted!")
        except Exception as e:
            self._set_status("Error")
            ThemedMessageBox.showerror("Error", str(e))

    def _analyze(self):
        if not self.src_path or not self.stego_path:
            ThemedMessageBox.showwarning(
                "Warning", "Need both original and stego images!")
            return

        try:
            # MSE & PSNR
            mse_val, psnr_val = MathEngine.calculate_mse_psnr(
                self.src_path, self.stego_path)
            self.mse.set(f"{mse_val:.6f}")
            
            # Determine quality rating
            if psnr_val == float('inf'):
                psnr_display = "∞"
                quality = "Perfect"
            else:
                psnr_display = f"{psnr_val:.2f} dB"
                quality = "Excellent" if psnr_val > 50 else "Very Good" if psnr_val > 40 else "Good" if psnr_val > 30 else "Fair"
            
            self.psnr.set(f"{psnr_display}\n({quality})")

            # SSIM
            ssim_val = MathEngine.calculate_ssim(
                self.src_path, self.stego_path)
            self.ssim_card.set(f"{ssim_val:.6f}")

            # Correlation
            corr_val = MathEngine.calculate_correlation(
                self.src_path, self.stego_path)
            self.corr.set(f"{corr_val:.6f}")

            # Entropy
            ent_orig = MathEngine.calculate_entropy(self.src_path)
            ent_stego = MathEngine.calculate_entropy(self.stego_path)
            self.entropy_orig.config(text=f"Original: {ent_orig:.4f} bits")
            self.entropy_stego.config(text=f"Stego: {ent_stego:.4f} bits")

            # Chi-Square
            chi_sq, dof, p_val = MathEngine.chi_square_test(self.stego_path)
            self.chi_value.config(text=f"χ² = {chi_sq:.2f} (df={dof})")

            if p_val < 0.05:
                self.chi_result.config(
                    text=f"⚠ Likely contains hidden data (p={p_val:.4f})", fg=Theme.ORANGE)
            else:
                self.chi_result.config(
                    text=f"✓ Appears normal (p={p_val:.4f})", fg=Theme.GREEN)

            # LSB
            lsb = MathEngine.analyze_lsb_distribution(self.stego_path)
            self.lsb_info.config(
                text=f"Zeros: {lsb['zeros']:,} ({lsb['zero_ratio']*100:.2f}%)  |  "
                f"Ones: {lsb['ones']:,} ({lsb['one_ratio']*100:.2f}%)  |  "
                f"Balance: {lsb['balance']*100:.3f}%"
            )

            self._set_status("Analysis complete")

            # Summary
            ThemedMessageBox.showinfo("Analysis Results",
                                f"Quality Metrics:\n"
                                f"  • MSE: {mse_val:.6f}\n"
                                f"  • PSNR: {psnr_display} ({quality})\n"
                                f"  • SSIM: {ssim_val:.6f}\n"
                                f"  • Correlation: {corr_val:.6f}\n\n"
                                f"Entropy:\n"
                                f"  • Original: {ent_orig:.4f} bits\n"
                                f"  • Stego: {ent_stego:.4f} bits\n\n"
                                f"Chi-Square: {chi_sq:.2f} (p={p_val:.4f})"
                                )

        except Exception as e:
            ThemedMessageBox.showerror("Error", str(e))

    def _convert_binary(self):
        text = self.bin_entry.get()[:20]  # Limit
        if not text:
            return

        result = MathEngine.text_to_binary(text)
        lines = []
        for char, ascii_val, binary in result:
            display_char = char if char.isprintable() else '?'
            lines.append(f"'{display_char}' → {ascii_val:3d} → {binary}")

        self.bin_result_text = "\n".join(lines)
        self.bin_result.config(text=self.bin_result_text)

    def _copy_binary_result(self):
        if hasattr(self, 'bin_result_text') and self.bin_result_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.bin_result_text)
            self._set_status("Binary conversion copied!")

    def _demo_xor(self):
        msg = self.xor_msg.get()[:10]
        key = self.xor_key.get()

        if not msg or not key:
            return

        result = MathEngine.demonstrate_xor(msg, key)
        lines = ["Char  Binary      Key   Binary      XOR Result"]
        lines.append("─" * 50)

        for r in result:
            lines.append(
                f" {r['char']}    {r['char_bin']}   "
                f"{r['key_char']}     {r['key_bin']}   "
                f"{r['xor_bin']} ({r['xor_result']})"
            )

        self.xor_result_text = "\n".join(lines)
        self.xor_result.config(text=self.xor_result_text)

    def _copy_xor_result(self):
        if hasattr(self, 'xor_result_text') and self.xor_result_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.xor_result_text)
            self._set_status("XOR encryption copied!")

    def _copy(self):
        txt = self.output.get().strip()
        if txt:
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            self._set_status("Copied!")

    def _switch(self, idx):
        self.current_tab = idx

        for i, item in enumerate(self.nav):
            item.set_active(i == idx)

        titles = ["🔒  Encode", "🔓  Decode", "📊  Analysis", "📐  Mathematics"]
        self.title.config(text=titles[idx])

        for i, tab in enumerate(self.tabs):
            if i == idx:
                tab.pack(fill="both", expand=True)
            else:
                tab.pack_forget()

    def _encode_base64(self):
        text = self.b64_entry.get()
        if not text:
            return
        
        encoded = base64.b64encode(text.encode()).decode()
        self.b64_result_text = encoded
        self.b64_result.config(text=encoded)

    def _copy_base64_result(self):
        if hasattr(self, 'b64_result_text') and self.b64_result_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.b64_result_text)
            self._set_status("Base64 result copied!")

    def _set_status(self, txt):
        self.status.config(text=txt)
        self.root.update_idletasks()


# -------------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()

    try:
        if platform.system() == "Windows":
            root.iconbitmap("icon.ico")
    except:
        pass

    app = App(root)

    try:
        root.state('zoomed')
    except:
        try:
            root.attributes('-zoomed', True)
        except:
            pass

    root.mainloop()