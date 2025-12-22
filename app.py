#!/usr/bin/env python3
"""
CyberMaths Steganography Tool - Modern UI Edition
A sleek, modern steganography application with dark theme.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ExifTags, ImageDraw # type: ignore
import numpy as np # type: ignore
import math
import os
from datetime import datetime
import platform

# -------------------------------------------------------------------------
# BACKEND LOGIC (The Maths & Algorithms)
# -------------------------------------------------------------------------

class StegoEngine:
    """
    Handles the mathematical operations for image manipulation.
    """

    @staticmethod
    def to_binary(data):
        """Convert any data (string or integers) to 8-bit binary format."""
        if isinstance(data, str):
            return ''.join([format(ord(i), "08b") for i in data])
        elif isinstance(data, bytes) or isinstance(data, np.ndarray):
            return [format(i, "08b") for i in data]
        elif isinstance(data, int) or isinstance(data, np.uint8):
            return format(data, "08b")
        else:
            raise TypeError("Input type not supported")

    @staticmethod
    def calculate_metrics(original_img, stego_img):
        """
        Calculates MSE and PSNR for the Maths Assessment.
        """
        img1 = np.array(original_img).astype(np.float64)
        img2 = np.array(stego_img).astype(np.float64)

        mse = np.mean((img1 - img2) ** 2)

        if mse == 0:
            return 0, float('inf')

        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

        return mse, psnr

    @staticmethod
    def xor_encrypt(message, key):
        """
        Simple XOR encryption (Maths Concept: Boolean Algebra).
        """
        if not key:
            return message

        encrypted = []
        key_len = len(key)
        for i, char in enumerate(message):
            encrypted.append(chr(ord(char) ^ ord(key[i % key_len])))
        return ''.join(encrypted)

    @staticmethod
    def encode_lsb(image_path, message, password=None):
        """
        Hides data in the Least Significant Bit of the image matrix.
        """
        image = Image.open(image_path)
        img_array = np.array(image)

        if password:
            message = StegoEngine.xor_encrypt(message, password)

        message += "$$STOP$$"

        binary_message = StegoEngine.to_binary(message)
        data_len = len(binary_message)

        total_pixels = img_array.size
        if data_len > total_pixels:
            raise ValueError(f"Message too large. Need {data_len} bits, image has {total_pixels} pixels.")

        flat_img = img_array.flatten()

        for i in range(data_len):
            flat_img[i] = flat_img[i] & 254
            flat_img[i] = flat_img[i] | int(binary_message[i])

        stego_array = flat_img.reshape(img_array.shape)
        stego_image = Image.fromarray(stego_array.astype('uint8'), image.mode)

        return stego_image

    @staticmethod
    def decode_lsb(image_path, password=None):
        """
        Extracts LSBs to reconstruct the message.
        """
        image = Image.open(image_path)
        img_array = np.array(image)
        flat_img = img_array.flatten()

        binary_data = ""
        decoded_string = ""

        for i in range(len(flat_img)):
            binary_data += str(flat_img[i] & 1)

            if len(binary_data) >= 8:
                char_code = int(binary_data[:8], 2)
                char = chr(char_code)
                decoded_string += char
                binary_data = binary_data[8:]

                if decoded_string.endswith("$$STOP$$"):
                    final_msg = decoded_string[:-8]
                    if password:
                        return StegoEngine.xor_encrypt(final_msg, password)
                    return final_msg

        return "No hidden message found or delimiter missing."

    @staticmethod
    def get_exif_data(image_path):
        """
        Extracts detailed metadata from the image.
        """
        try:
            img = Image.open(image_path)
            stats = os.stat(image_path)
            info_list = []

            info_list.append(f"{'File Name':<20}: {os.path.basename(image_path)}")
            info_list.append(f"{'Directory':<20}: {os.path.dirname(image_path) or '.'}")

            size_kb = stats.st_size / 1024
            info_list.append(f"{'File Size':<20}: {size_kb:.1f} kB")

            mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y:%m:%d %H:%M:%S')
            acc_time = datetime.fromtimestamp(stats.st_atime).strftime('%Y:%m:%d %H:%M:%S')
            info_list.append(f"{'File Mod Date':<20}: {mod_time}")
            info_list.append(f"{'File Access Date':<20}: {acc_time}")

            info_list.append(f"{'File Type':<20}: {img.format}")
            info_list.append(f"{'MIME Type':<20}: {Image.MIME.get(img.format, 'image/' + img.format.lower())}")

            info_list.append(f"{'Image Width':<20}: {img.width}")
            info_list.append(f"{'Image Height':<20}: {img.height}")
            info_list.append(f"{'Image Size':<20}: {img.width}x{img.height}")

            mp = (img.width * img.height) / 1000000
            info_list.append(f"{'Megapixels':<20}: {mp:.3f}")

            bit_depth_map = {'1': 1, 'L': 8, 'P': 8, 'RGB': 8, 'RGBA': 8, 'CMYK': 8, 'YCbCr': 8, 'I': 32, 'F': 32}
            bit_depth = bit_depth_map.get(img.mode, 'Unknown')
            info_list.append(f"{'Bit Depth':<20}: {bit_depth}")
            info_list.append(f"{'Color Type':<20}: {img.mode}")

            info_list.append("─" * 45)

            has_extra = False

            if img.info:
                for key, value in img.info.items():
                    if key in ['dpi', 'transparency']:
                        continue
                    if isinstance(value, (str, int, float)):
                        info_list.append(f"{key:<20}: {value}")
                        has_extra = True

            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        value = "<Binary>"
                    info_list.append(f"{tag:<20}: {value}")
                    has_extra = True

            if not has_extra:
                info_list.append("No additional internal metadata found.")

            return "\n".join(info_list)

        except Exception as e:
            return f"Error reading metadata: {str(e)}"


# -------------------------------------------------------------------------
# MODERN UI COMPONENTS
# -------------------------------------------------------------------------

class ModernColors:
    """Modern color palette for dark theme."""
    BG_PRIMARY = "#0f0f0f"
    BG_SECONDARY = "#1a1a1a"
    BG_TERTIARY = "#252525"
    BG_CARD = "#1e1e1e"

    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a0a0a0"
    TEXT_MUTED = "#666666"

    ACCENT_BLUE = "#3b82f6"
    ACCENT_BLUE_HOVER = "#2563eb"
    ACCENT_PURPLE = "#8b5cf6"
    ACCENT_GREEN = "#10b981"
    ACCENT_GREEN_HOVER = "#059669"
    ACCENT_RED = "#ef4444"
    ACCENT_RED_HOVER = "#dc2626"
    ACCENT_ORANGE = "#f59e0b"

    BORDER = "#333333"
    BORDER_LIGHT = "#404040"

    GRADIENT_START = "#3b82f6"
    GRADIENT_END = "#8b5cf6"


class ModernButton(tk.Canvas):
    """A modern styled button with gradient, hover effects, and rounded corners."""

    def __init__(self, parent, text, command=None, variant="primary", width=200, height=44, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=ModernColors.BG_PRIMARY, highlightthickness=0, **kwargs)

        self.text = text
        self.command = command
        self.variant = variant
        self.width = width
        self.height = height
        self.is_hovered = False
        self.is_pressed = False

        self._set_colors()
        self._draw()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _set_colors(self):
        """Set colors based on variant."""
        if self.variant == "primary":
            self.bg_color = ModernColors.ACCENT_BLUE
            self.hover_color = ModernColors.ACCENT_BLUE_HOVER
            self.text_color = "#ffffff"
        elif self.variant == "success":
            self.bg_color = ModernColors.ACCENT_GREEN
            self.hover_color = ModernColors.ACCENT_GREEN_HOVER
            self.text_color = "#ffffff"
        elif self.variant == "danger":
            self.bg_color = ModernColors.ACCENT_RED
            self.hover_color = ModernColors.ACCENT_RED_HOVER
            self.text_color = "#ffffff"
        elif self.variant == "secondary":
            self.bg_color = ModernColors.BG_TERTIARY
            self.hover_color = ModernColors.BORDER_LIGHT
            self.text_color = ModernColors.TEXT_PRIMARY
        elif self.variant == "gradient":
            self.bg_color = ModernColors.ACCENT_BLUE
            self.hover_color = ModernColors.ACCENT_PURPLE
            self.text_color = "#ffffff"

    def _draw(self):
        """Draw the button."""
        self.delete("all")

        color = self.hover_color if self.is_hovered else self.bg_color
        if self.is_pressed:
            color = self.hover_color

        # Draw rounded rectangle
        radius = 10
        self._create_rounded_rect(2, 2, self.width - 2, self.height - 2, radius, color)

        # Draw text
        self.create_text(self.width // 2, self.height // 2, text=self.text,
                         fill=self.text_color, font=("Segoe UI Semibold", 11))

    def _create_rounded_rect(self, x1, y1, x2, y2, radius, color):
        """Create a rounded rectangle."""
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        self.create_polygon(points, fill=color, smooth=True)

    def _on_enter(self, event):
        self.is_hovered = True
        self.config(cursor="hand2")
        self._draw()

    def _on_leave(self, event):
        self.is_hovered = False
        self.is_pressed = False
        self._draw()

    def _on_press(self, event):
        self.is_pressed = True
        self._draw()

    def _on_release(self, event):
        self.is_pressed = False
        self._draw()
        if self.is_hovered and self.command:
            self.command()


class ModernEntry(tk.Frame):
    """A modern styled entry field with rounded corners."""

    def __init__(self, parent, placeholder="", show=None, **kwargs):
        super().__init__(parent, bg=ModernColors.BG_PRIMARY)

        self.placeholder = placeholder
        self.show_char = show
        self.has_focus = False

        # Container with border
        self.container = tk.Frame(self, bg=ModernColors.BORDER, padx=1, pady=1)
        self.container.pack(fill="x")

        self.inner = tk.Frame(self.container, bg=ModernColors.BG_TERTIARY)
        self.inner.pack(fill="x", padx=1, pady=1)

        self.entry = tk.Entry(self.inner, bg=ModernColors.BG_TERTIARY,
                             fg=ModernColors.TEXT_PRIMARY,
                             insertbackground=ModernColors.ACCENT_BLUE,
                             relief="flat", font=("Segoe UI", 11),
                             show=show if show else "")
        self.entry.pack(fill="x", padx=12, pady=10)

        self.entry.bind("<FocusIn>", self._on_focus_in)
        self.entry.bind("<FocusOut>", self._on_focus_out)

    def _on_focus_in(self, event):
        self.has_focus = True
        self.container.config(bg=ModernColors.ACCENT_BLUE)

    def _on_focus_out(self, event):
        self.has_focus = False
        self.container.config(bg=ModernColors.BORDER)

    def get(self):
        return self.entry.get()

    def delete(self, first, last):
        self.entry.delete(first, last)

    def insert(self, index, string):
        self.entry.insert(index, string)


class ModernText(tk.Frame):
    """A modern styled text area with rounded corners."""

    def __init__(self, parent, height=5, **kwargs):
        super().__init__(parent, bg=ModernColors.BG_PRIMARY)

        self.has_focus = False

        # Container with border
        self.container = tk.Frame(self, bg=ModernColors.BORDER, padx=1, pady=1)
        self.container.pack(fill="both", expand=True)

        self.inner = tk.Frame(self.container, bg=ModernColors.BG_TERTIARY)
        self.inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.text = tk.Text(self.inner, bg=ModernColors.BG_TERTIARY,
                           fg=ModernColors.TEXT_PRIMARY,
                           insertbackground=ModernColors.ACCENT_BLUE,
                           relief="flat", font=("Segoe UI", 11),
                           height=height, wrap="word")
        self.text.pack(fill="both", expand=True, padx=10, pady=10)

        self.text.bind("<FocusIn>", self._on_focus_in)
        self.text.bind("<FocusOut>", self._on_focus_out)

    def _on_focus_in(self, event):
        self.has_focus = True
        self.container.config(bg=ModernColors.ACCENT_BLUE)

    def _on_focus_out(self, event):
        self.has_focus = False
        self.container.config(bg=ModernColors.BORDER)

    def get(self, start, end):
        return self.text.get(start, end)

    def delete(self, start, end):
        self.text.delete(start, end)

    def insert(self, index, chars):
        self.text.insert(index, chars)

    def config(self, **kwargs):
        self.text.config(**kwargs)


class ModernCard(tk.Frame):
    """A modern card container with subtle border and shadow effect."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernColors.BG_CARD, **kwargs)

        self.config(highlightbackground=ModernColors.BORDER,
                   highlightthickness=1,
                   highlightcolor=ModernColors.BORDER)


class ModernRadioButton(tk.Frame):
    """A modern styled radio button."""

    def __init__(self, parent, text, variable, value, description="", **kwargs):
        super().__init__(parent, bg=ModernColors.BG_PRIMARY, **kwargs)

        self.variable = variable
        self.value = value
        self.is_selected = False

        # Main container
        self.container = tk.Frame(self, bg=ModernColors.BG_TERTIARY,
                                 cursor="hand2")
        self.container.pack(fill="x", pady=3)

        # Radio circle
        self.canvas = tk.Canvas(self.container, width=20, height=20,
                               bg=ModernColors.BG_TERTIARY, highlightthickness=0)
        self.canvas.pack(side="left", padx=(12, 8), pady=12)

        # Text container
        text_frame = tk.Frame(self.container, bg=ModernColors.BG_TERTIARY)
        text_frame.pack(side="left", fill="x", expand=True, pady=10)

        self.label = tk.Label(text_frame, text=text, bg=ModernColors.BG_TERTIARY,
                             fg=ModernColors.TEXT_PRIMARY, font=("Segoe UI Semibold", 10),
                             anchor="w")
        self.label.pack(fill="x")

        if description:
            self.desc_label = tk.Label(text_frame, text=description,
                                       bg=ModernColors.BG_TERTIARY,
                                       fg=ModernColors.TEXT_SECONDARY,
                                       font=("Segoe UI", 9), anchor="w")
            self.desc_label.pack(fill="x")

        self._draw_radio()

        # Bindings
        for widget in [self.container, self.canvas, self.label, self]:
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

        if hasattr(self, 'desc_label'):
            self.desc_label.bind("<Button-1>", self._on_click)
            self.desc_label.bind("<Enter>", self._on_enter)
            self.desc_label.bind("<Leave>", self._on_leave)

        # Track variable changes
        self.variable.trace_add("write", lambda *args: self._draw_radio())

    def _draw_radio(self):
        """Draw the radio button circle."""
        self.canvas.delete("all")
        self.is_selected = self.variable.get() == self.value

        # Outer circle
        outer_color = ModernColors.ACCENT_BLUE if self.is_selected else ModernColors.BORDER_LIGHT
        self.canvas.create_oval(2, 2, 18, 18, outline=outer_color, width=2)

        # Inner circle (when selected)
        if self.is_selected:
            self.canvas.create_oval(6, 6, 14, 14, fill=ModernColors.ACCENT_BLUE, outline="")

    def _on_click(self, event):
        self.variable.set(self.value)

    def _on_enter(self, event):
        self.container.config(bg=ModernColors.BG_CARD)
        self.canvas.config(bg=ModernColors.BG_CARD)
        self.label.config(bg=ModernColors.BG_CARD)
        if hasattr(self, 'desc_label'):
            self.desc_label.config(bg=ModernColors.BG_CARD)

    def _on_leave(self, event):
        self.container.config(bg=ModernColors.BG_TERTIARY)
        self.canvas.config(bg=ModernColors.BG_TERTIARY)
        self.label.config(bg=ModernColors.BG_TERTIARY)
        if hasattr(self, 'desc_label'):
            self.desc_label.config(bg=ModernColors.BG_TERTIARY)


class ModernModal(tk.Toplevel):
    """A modern styled modal dialog."""

    def __init__(self, parent, title, message, modal_type="info"):
        super().__init__(parent)

        self.title("")
        self.resizable(False, False)
        self.configure(bg=ModernColors.BG_PRIMARY)
        self.transient(parent)
        self.grab_set()

        # Icons and colors based on type
        icons = {
            "info": ("ℹ️", ModernColors.ACCENT_BLUE),
            "success": ("✓", ModernColors.ACCENT_GREEN),
            "error": ("✕", ModernColors.ACCENT_RED),
            "warning": ("⚠", ModernColors.ACCENT_ORANGE)
        }
        icon, accent = icons.get(modal_type, icons["info"])

        # Main container
        main_frame = tk.Frame(self, bg=ModernColors.BG_SECONDARY,
                             highlightbackground=ModernColors.BORDER,
                             highlightthickness=1)
        main_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # Header with icon
        header = tk.Frame(main_frame, bg=ModernColors.BG_SECONDARY)
        header.pack(fill="x", padx=24, pady=(24, 16))

        # Icon circle
        icon_canvas = tk.Canvas(header, width=48, height=48,
                               bg=ModernColors.BG_SECONDARY, highlightthickness=0)
        icon_canvas.pack(side="left", padx=(0, 16))
        icon_canvas.create_oval(4, 4, 44, 44, fill=accent, outline="")
        icon_canvas.create_text(24, 24, text=icon, fill="#ffffff",
                               font=("Segoe UI", 16, "bold"))

        # Title
        tk.Label(header, text=title, bg=ModernColors.BG_SECONDARY,
                fg=ModernColors.TEXT_PRIMARY, font=("Segoe UI Semibold", 14),
                anchor="w").pack(side="left", fill="x", expand=True)

        # Message
        msg_label = tk.Label(main_frame, text=message, bg=ModernColors.BG_SECONDARY,
                            fg=ModernColors.TEXT_SECONDARY, font=("Segoe UI", 11),
                            wraplength=350, justify="left", anchor="w")
        msg_label.pack(fill="x", padx=24, pady=(0, 24))

        # Button container
        btn_frame = tk.Frame(main_frame, bg=ModernColors.BG_SECONDARY)
        btn_frame.pack(fill="x", padx=24, pady=(0, 24))

        # Determine button variant based on modal type
        btn_variant = "primary"
        if modal_type == "success":
            btn_variant = "success"
        elif modal_type == "error":
            btn_variant = "danger"

        ok_btn = ModernButton(btn_frame, text="OK", command=self.destroy,
                             variant=btn_variant, width=100, height=40)
        ok_btn.pack(side="right")

        # Center the modal
        self.update_idletasks()
        width = 420
        height = self.winfo_reqheight()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (width // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Bindings
        self.bind('<Return>', lambda e: self.destroy())
        self.bind('<Escape>', lambda e: self.destroy())

        self.focus_set()
        self.wait_window()


class ModernImageDropzone(tk.Frame):
    """A modern image dropzone with preview and reset functionality."""

    def __init__(self, parent, on_image_load=None, on_image_reset=None, **kwargs):
        super().__init__(parent, bg=ModernColors.BG_PRIMARY, **kwargs)

        self.on_image_load = on_image_load
        self.on_image_reset = on_image_reset
        self.image_path = None
        self.photo = None

        # Container with dashed border effect
        self.container = tk.Frame(self, bg=ModernColors.BG_TERTIARY,
                                 highlightbackground=ModernColors.BORDER,
                                 highlightthickness=2)
        self.container.pack(fill="both", expand=True, padx=2, pady=2)

        # Content frame
        self.content = tk.Frame(self.container, bg=ModernColors.BG_TERTIARY)
        self.content.pack(fill="both", expand=True, padx=20, pady=20)

        # Icon
        self.icon_label = tk.Label(self.content, text="🖼️",
                                   font=("Segoe UI", 36),
                                   bg=ModernColors.BG_TERTIARY,
                                   fg=ModernColors.TEXT_MUTED)
        self.icon_label.pack(pady=(20, 10))

        # Text
        self.text_label = tk.Label(self.content, text="Click to select an image",
                                   font=("Segoe UI", 11),
                                   bg=ModernColors.BG_TERTIARY,
                                   fg=ModernColors.TEXT_SECONDARY)
        self.text_label.pack()

        self.subtext_label = tk.Label(self.content, text="Supports PNG, JPG, BMP",
                                      font=("Segoe UI", 9),
                                      bg=ModernColors.BG_TERTIARY,
                                      fg=ModernColors.TEXT_MUTED)
        self.subtext_label.pack(pady=(5, 20))

        # Image preview label (hidden initially)
        self.preview_label = tk.Label(self.content, bg=ModernColors.BG_TERTIARY)

        # Reset button (hidden initially)
        self.reset_btn = ModernButton(self.content, text="✕ Remove Image",
                                     command=self.reset, variant="danger",
                                     width=140, height=32)

        # Bindings
        for widget in [self, self.container, self.content, self.icon_label,
                      self.text_label, self.subtext_label]:
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)
            widget.config(cursor="hand2")

    def _on_click(self, event):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if path:
            self.load_image(path)

    def _on_enter(self, event):
        self.container.config(highlightbackground=ModernColors.ACCENT_BLUE)

    def _on_leave(self, event):
        self.container.config(highlightbackground=ModernColors.BORDER)

    def load_image(self, path):
        """Load and display an image."""
        self.image_path = path

        # Hide dropzone text
        self.icon_label.pack_forget()
        self.text_label.pack_forget()
        self.subtext_label.pack_forget()

        # Load and resize image
        img = Image.open(path)
        img.thumbnail((280, 280))
        self.photo = ImageTk.PhotoImage(img)

        # Show preview
        self.preview_label.config(image=self.photo)
        self.preview_label.pack(fill="both", expand=True)

        # Show reset button
        self.reset_btn.pack(pady=(10, 0))

        # Callback
        if self.on_image_load:
            self.on_image_load(path)

    def reset(self):
        """Reset the dropzone."""
        self.image_path = None
        self.photo = None
        self.preview_label.pack_forget()
        self.reset_btn.pack_forget()
        self.icon_label.pack(pady=(20, 10))
        self.text_label.pack()
        self.subtext_label.pack(pady=(5, 20))

        # Callback for reset
        if self.on_image_reset:
            self.on_image_reset()


class ModernTab(tk.Frame):
    """A modern tab button."""

    def __init__(self, parent, text, is_active=False, command=None, **kwargs):
        super().__init__(parent, bg=ModernColors.BG_SECONDARY, **kwargs)

        self.text = text
        self.is_active = is_active
        self.command = command

        self.label = tk.Label(self, text=text, font=("Segoe UI Semibold", 11),
                             bg=ModernColors.BG_SECONDARY,
                             fg=ModernColors.TEXT_PRIMARY if is_active else ModernColors.TEXT_MUTED,
                             padx=24, pady=12, cursor="hand2")
        self.label.pack()

        # Active indicator
        self.indicator = tk.Frame(self, height=3,
                                 bg=ModernColors.ACCENT_BLUE if is_active else ModernColors.BG_SECONDARY)
        self.indicator.pack(fill="x")

        self.label.bind("<Button-1>", self._on_click)
        self.label.bind("<Enter>", self._on_enter)
        self.label.bind("<Leave>", self._on_leave)

    def _on_click(self, event):
        if self.command:
            self.command()

    def _on_enter(self, event):
        if not self.is_active:
            self.label.config(fg=ModernColors.TEXT_SECONDARY)

    def _on_leave(self, event):
        if not self.is_active:
            self.label.config(fg=ModernColors.TEXT_MUTED)

    def set_active(self, active):
        self.is_active = active
        self.label.config(fg=ModernColors.TEXT_PRIMARY if active else ModernColors.TEXT_MUTED)
        self.indicator.config(bg=ModernColors.ACCENT_BLUE if active else ModernColors.BG_SECONDARY)


# -------------------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------------------

class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CyberMaths Steganography Tool")
        self.root.geometry("950x700")
        self.root.configure(bg=ModernColors.BG_PRIMARY)
        self.root.minsize(800, 600)

        # Variables
        self.src_image_path = None
        self.stego_image_object = None
        self.decoded_image_path = None
        self.current_tab = 0
        self.algo_var = tk.StringVar(value="LSB")

        self._setup_ui()

    def _setup_ui(self):
        """Setup the main UI."""
        # Header
        header = tk.Frame(self.root, bg=ModernColors.BG_SECONDARY, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        # Logo/Title
        title_frame = tk.Frame(header, bg=ModernColors.BG_SECONDARY)
        title_frame.pack(side="left", padx=24, pady=12)

        tk.Label(title_frame, text="🔐", font=("Segoe UI", 18),
                bg=ModernColors.BG_SECONDARY, fg=ModernColors.ACCENT_BLUE).pack(side="left")
        tk.Label(title_frame, text="CyberMaths Stego", font=("Segoe UI Semibold", 16),
                bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY).pack(side="left", padx=(8, 0))

        # Tab bar
        tab_bar = tk.Frame(self.root, bg=ModernColors.BG_SECONDARY)
        tab_bar.pack(fill="x")

        self.tabs = []
        tab_names = ["  🔒  Encode  ", "  🔓  Decode  ", "  📊  Analysis  "]
        for i, name in enumerate(tab_names):
            tab = ModernTab(tab_bar, name, is_active=(i == 0),
                           command=lambda idx=i: self._switch_tab(idx))
            tab.pack(side="left")
            self.tabs.append(tab)

        # Separator
        tk.Frame(self.root, bg=ModernColors.BORDER, height=1).pack(fill="x")

        # Content area
        self.content = tk.Frame(self.root, bg=ModernColors.BG_PRIMARY)
        self.content.pack(fill="both", expand=True)

        # Tab frames
        self.tab_frames = []
        self._setup_encode_tab()
        self._setup_decode_tab()
        self._setup_analysis_tab()

        # Show first tab
        self._switch_tab(0)

    def _setup_encode_tab(self):
        """Setup the encode tab."""
        frame = tk.Frame(self.content, bg=ModernColors.BG_PRIMARY)
        self.tab_frames.append(frame)

        # Two column layout
        left_col = tk.Frame(frame, bg=ModernColors.BG_PRIMARY)
        left_col.pack(side="left", fill="both", expand=True, padx=(24, 12), pady=24)

        right_col = tk.Frame(frame, bg=ModernColors.BG_PRIMARY)
        right_col.pack(side="right", fill="both", expand=True, padx=(12, 24), pady=24)

        # Left column - Image
        tk.Label(left_col, text="Cover Image", font=("Segoe UI Semibold", 13),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                anchor="w").pack(fill="x", pady=(0, 12))

        self.encode_dropzone = ModernImageDropzone(left_col, self._on_encode_image_load, self._on_encode_image_reset)
        self.encode_dropzone.pack(fill="both", expand=True)

        # Right column - Controls
        msg_header = tk.Frame(right_col, bg=ModernColors.BG_PRIMARY)
        msg_header.pack(fill="x", pady=(0, 8))

        tk.Label(msg_header, text="Secret Message", font=("Segoe UI Semibold", 13),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                anchor="w").pack(side="left")

        # Clear button for message
        self.btn_clear_msg = ModernButton(msg_header, text="Clear",
                                         command=self._clear_message,
                                         variant="secondary",
                                         width=80, height=28)
        self.btn_clear_msg.pack(side="right")

        self.txt_msg = ModernText(right_col, height=5)
        self.txt_msg.pack(fill="x", pady=(0, 16))

        # Password
        tk.Label(right_col, text="Encryption Password (Optional)",
                font=("Segoe UI Semibold", 13),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                anchor="w").pack(fill="x", pady=(0, 8))

        self.entry_pass_enc = ModernEntry(right_col, show="•")
        self.entry_pass_enc.pack(fill="x", pady=(0, 16))

        # Technique
        tk.Label(right_col, text="Technique", font=("Segoe UI Semibold", 13),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                anchor="w").pack(fill="x", pady=(0, 8))

        ModernRadioButton(right_col, "Standard LSB", self.algo_var, "LSB",
                         "Basic least significant bit embedding").pack(fill="x")
        ModernRadioButton(right_col, "LSB + XOR Encryption", self.algo_var, "XOR",
                         "Enhanced security with password encryption").pack(fill="x")

        # Encode button
        btn_frame = tk.Frame(right_col, bg=ModernColors.BG_PRIMARY)
        btn_frame.pack(fill="x", pady=(24, 0))

        self.btn_encode = ModernButton(btn_frame, text="🔐  Encrypt & Save Image",
                                       command=self.process_encode, variant="gradient",
                                       width=280, height=48)
        self.btn_encode.pack(fill="x")

    def _setup_decode_tab(self):
        """Setup the decode tab."""
        frame = tk.Frame(self.content, bg=ModernColors.BG_PRIMARY)
        self.tab_frames.append(frame)

        # Center container
        container = tk.Frame(frame, bg=ModernColors.BG_PRIMARY)
        container.pack(fill="both", expand=True, padx=24, pady=24)

        # Image dropzone
        tk.Label(container, text="Stego Image", font=("Segoe UI Semibold", 13),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                anchor="w").pack(fill="x", pady=(0, 12))

        dropzone_frame = tk.Frame(container, bg=ModernColors.BG_PRIMARY)
        dropzone_frame.pack(fill="x")

        self.decode_dropzone = ModernImageDropzone(dropzone_frame, self._on_decode_image_load, self._on_decode_image_reset)
        self.decode_dropzone.pack(fill="both", expand=True, pady=(0, 16))

        # Password
        tk.Label(container, text="Decryption Password (if encrypted)",
                font=("Segoe UI Semibold", 13),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                anchor="w").pack(fill="x", pady=(0, 8))

        self.entry_pass_dec = ModernEntry(container, show="•")
        self.entry_pass_dec.pack(fill="x", pady=(0, 16))

        # Decode button
        self.btn_decode = ModernButton(container, text="🔍  Reveal Hidden Message",
                                       command=self.process_decode, variant="success",
                                       width=280, height=48)
        self.btn_decode.pack(pady=(0, 16))

        # Output
        output_header = tk.Frame(container, bg=ModernColors.BG_PRIMARY)
        output_header.pack(fill="x", pady=(0, 8))

        tk.Label(output_header, text="Hidden Message", font=("Segoe UI Semibold", 13),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                anchor="w").pack(side="left")

        # Clear button for output
        self.btn_clear_output = ModernButton(output_header, text="Clear",
                                           command=self._clear_output,
                                           variant="secondary",
                                           width=80, height=28)
        self.btn_clear_output.pack(side="right")

        self.txt_output = ModernText(container, height=4)
        self.txt_output.config(fg=ModernColors.ACCENT_GREEN)
        self.txt_output.pack(fill="x")

    def _setup_analysis_tab(self):
        """Setup the analysis tab."""
        frame = tk.Frame(self.content, bg=ModernColors.BG_PRIMARY)
        self.tab_frames.append(frame)

        container = tk.Frame(frame, bg=ModernColors.BG_PRIMARY)
        container.pack(fill="both", expand=True, padx=24, pady=24)

        # Title
        tk.Label(container, text="Mathematical Analysis",
                font=("Segoe UI Semibold", 16),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                anchor="w").pack(fill="x", pady=(0, 8))

        tk.Label(container, text="Compare original and stego images to measure quality degradation",
                font=("Segoe UI", 11),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_SECONDARY,
                anchor="w").pack(fill="x", pady=(0, 20))

        # Metrics cards
        metrics_frame = tk.Frame(container, bg=ModernColors.BG_PRIMARY)
        metrics_frame.pack(fill="x", pady=(0, 20))

        # MSE Card
        mse_card = ModernCard(metrics_frame)
        mse_card.pack(side="left", fill="both", expand=True, padx=(0, 8))

        mse_inner = tk.Frame(mse_card, bg=ModernColors.BG_CARD)
        mse_inner.pack(fill="both", expand=True, padx=20, pady=16)

        tk.Label(mse_inner, text="MSE", font=("Segoe UI", 10),
                bg=ModernColors.BG_CARD, fg=ModernColors.TEXT_SECONDARY).pack(anchor="w")
        self.lbl_mse = tk.Label(mse_inner, text="N/A", font=("Segoe UI Semibold", 18),
                               bg=ModernColors.BG_CARD, fg=ModernColors.ACCENT_BLUE)
        self.lbl_mse.pack(anchor="w", pady=(4, 0))
        tk.Label(mse_inner, text="Mean Squared Error", font=("Segoe UI", 9),
                bg=ModernColors.BG_CARD, fg=ModernColors.TEXT_MUTED).pack(anchor="w")

        # PSNR Card
        psnr_card = ModernCard(metrics_frame)
        psnr_card.pack(side="left", fill="both", expand=True, padx=(8, 0))

        psnr_inner = tk.Frame(psnr_card, bg=ModernColors.BG_CARD)
        psnr_inner.pack(fill="both", expand=True, padx=20, pady=16)

        tk.Label(psnr_inner, text="PSNR", font=("Segoe UI", 10),
                bg=ModernColors.BG_CARD, fg=ModernColors.TEXT_SECONDARY).pack(anchor="w")
        self.lbl_psnr = tk.Label(psnr_inner, text="N/A", font=("Segoe UI Semibold", 18),
                                bg=ModernColors.BG_CARD, fg=ModernColors.ACCENT_GREEN)
        self.lbl_psnr.pack(anchor="w", pady=(4, 0))
        tk.Label(psnr_inner, text="Signal-to-Noise Ratio (dB)", font=("Segoe UI", 9),
                bg=ModernColors.BG_CARD, fg=ModernColors.TEXT_MUTED).pack(anchor="w")

        # Calculate button
        self.btn_analysis = ModernButton(container, text="📊  Calculate Metrics",
                                        command=self.run_analysis, variant="primary",
                                        width=200, height=44)
        self.btn_analysis.pack(pady=(0, 20))

        # EXIF Data
        tk.Label(container, text="Image Metadata (EXIF)", font=("Segoe UI Semibold", 13),
                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                anchor="w").pack(fill="x", pady=(0, 8))

        self.txt_exif = ModernText(container, height=10)
        self.txt_exif.pack(fill="both", expand=True)

    # -------------------------------------------------------------------------
    # EVENT HANDLERS
    # -------------------------------------------------------------------------

    def _on_encode_image_load(self, path):
        """Handle image load for encoding."""
        self.src_image_path = path

    def _on_encode_image_reset(self):
        """Handle image reset for encoding."""
        self.src_image_path = None

    def _on_decode_image_load(self, path):
        """Handle image load for decoding."""
        self.decoded_image_path = path

        # Show EXIF data
        exif_info = StegoEngine.get_exif_data(path)
        self.txt_exif.delete("1.0", tk.END)
        self.txt_exif.insert("1.0", exif_info)

    def _on_decode_image_reset(self):
        """Handle image reset for decoding."""
        self.decoded_image_path = None
        # Clear EXIF data when image is removed
        self.txt_exif.delete("1.0", tk.END)

    def process_encode(self):
        """Process the encoding."""
        if not self.src_image_path:
            ModernModal(self.root, "Error", "Please select an image first.", "error")
            return

        msg = self.txt_msg.get("1.0", tk.END).strip()
        if not msg:
            ModernModal(self.root, "Error", "Please enter a secret message.", "error")
            return

        password = self.entry_pass_enc.get() if self.entry_pass_enc.get() else None

        try:
            stego_img = StegoEngine.encode_lsb(self.src_image_path, msg, password)
            self.stego_image_object = stego_img

            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")]
            )

            if save_path:
                stego_img.save(save_path)
                self.decoded_image_path = save_path
                ModernModal(self.root, "Success",
                           f"Image saved successfully!\n\n{os.path.basename(save_path)}",
                           "success")
        except Exception as e:
            ModernModal(self.root, "Error", str(e), "error")

    def process_decode(self):
        """Process the decoding."""
        if not self.decoded_image_path:
            ModernModal(self.root, "Error", "Please select a stego image first.", "error")
            return

        password = self.entry_pass_dec.get()

        try:
            msg = StegoEngine.decode_lsb(self.decoded_image_path, password)
            self.txt_output.delete("1.0", tk.END)
            self.txt_output.insert("1.0", msg)
        except Exception as e:
            ModernModal(self.root, "Error", str(e), "error")

    def run_analysis(self):
        """Run the mathematical analysis."""
        if not self.src_image_path or not self.decoded_image_path:
            ModernModal(self.root, "Error",
                       "Need both original and stego images loaded.\n\nEncode an image first, or load both images separately.",
                       "error")
            return

        try:
            orig = Image.open(self.src_image_path)
            stego = Image.open(self.decoded_image_path)

            if orig.size != stego.size:
                ModernModal(self.root, "Error",
                           "Images must be the same size for comparison.",
                           "error")
                return

            mse, psnr = StegoEngine.calculate_metrics(orig, stego)

            self.lbl_mse.config(text=f"{mse:.6f}")
            psnr_text = f"{psnr:.2f} dB" if psnr != float('inf') else "∞ (Identical)"
            self.lbl_psnr.config(text=psnr_text)

            if mse < 0.1:
                ModernModal(self.root, "Analysis Complete",
                           "The images are mathematically almost identical!\n\nThis proves the steganography is virtually invisible to the human eye.",
                           "success")
            else:
                ModernModal(self.root, "Analysis Complete",
                           f"MSE: {mse:.6f}\nPSNR: {psnr_text}\n\nThe lower the MSE and higher the PSNR, the better the quality.",
                           "info")

        except Exception as e:
            ModernModal(self.root, "Error", str(e), "error")

    def _switch_tab(self, index):
        """Switch to a different tab."""
        self.current_tab = index

        # Update tab buttons
        for i, tab in enumerate(self.tabs):
            tab.set_active(i == index)

        # Show/hide tab frames
        for i, frame in enumerate(self.tab_frames):
            if i == index:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

    # -------------------------------------------------------------------------
    # RESET/CLEAR FUNCTIONS
    # -------------------------------------------------------------------------

    def _clear_message(self):
        """Clear the secret message text area."""
        self.txt_msg.delete("1.0", tk.END)

    def _clear_output(self):
        """Clear the output text area."""
        self.txt_output.delete("1.0", tk.END)


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()

    # Set app icon (if available)
    try:
        if platform.system() == "Windows":
            root.iconbitmap("icon.ico")
    except:
        pass

    app = StegoApp(root)
    root.state('zoomed')  # <--- MAXIMIZE WINDOW
    root.mainloop()