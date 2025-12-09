import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ExifTags
import numpy as np
import math
import os
import random

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
        
        # Mean Squared Error (Maths Concept: Statistics/Matrices)
        mse = np.mean((img1 - img2) ** 2)
        
        if mse == 0:
            return 0, float('inf') # Images are identical
        
        # Peak Signal-to-Noise Ratio (Maths Concept: Logarithms)
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
            # XOR operation
            encrypted.append(chr(ord(char) ^ ord(key[i % key_len])))
        return ''.join(encrypted)

    @staticmethod
    def encode_lsb(image_path, message, password=None):
        """
        Hides data in the Least Significant Bit of the image matrix.
        """
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # 1. Prepare Message
        if password:
            message = StegoEngine.xor_encrypt(message, password)
            
        # Add a delimiter to know when to stop reading
        message += "$$STOP$$"
        
        binary_message = StegoEngine.to_binary(message)
        data_len = len(binary_message)
        
        # 2. Check Capacity (Maths: Area/Volume)
        total_pixels = img_array.size # Rows * Cols * Channels
        if data_len > total_pixels:
            raise ValueError(f"Message too large. Need {data_len} bits, image has {total_pixels} pixels.")

        # 3. Flatten the matrix for easier iteration
        flat_img = img_array.flatten()
        
        # 4. Modify LSBs (The Core Maths)
        # We iterate through the flattened array and modify the last bit
        for i in range(data_len):
            # Clear LSB (Bitwise AND with 11111110)
            flat_img[i] = flat_img[i] & 254 
            # Set LSB (Bitwise OR with message bit)
            flat_img[i] = flat_img[i] | int(binary_message[i])
            
        # 5. Reshape back to image matrix
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
        
        # Extract LSBs
        # Note: In a real scenario, we wouldn't read the whole image, just until delimiter
        # But for this demo, we read chunks to be efficient
        
        chunk_size = 1000 # Read in chunks to avoid freezing
        decoded_string = ""
        
        for i in range(len(flat_img)):
            # Get the LSB (Bitwise AND 1)
            binary_data += str(flat_img[i] & 1)
            
            # Every 8 bits = 1 character
            if len(binary_data) >= 8:
                char_code = int(binary_data[:8], 2)
                char = chr(char_code)
                decoded_string += char
                binary_data = binary_data[8:] # Remove processed bits
                
                # Check for delimiter
                if decoded_string.endswith("$$STOP$$"):
                    final_msg = decoded_string[:-8] # Remove delimiter
                    if password:
                        return StegoEngine.xor_encrypt(final_msg, password) # XOR is symmetric
                    return final_msg
                    
        return "No hidden message found or delimiter missing."

    @staticmethod
    def get_exif_data(image_path):
        """Extracts metadata from the image."""
        img = Image.open(image_path)
        exif_data = img.getexif()
        info = []
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                info.append(f"{tag}: {value}")
        else:
            info.append("No EXIF data found.")
        return "\n".join(info[:10]) # Limit to top 10 to avoid clutter

# -------------------------------------------------------------------------
# FRONTEND LOGIC (The GUI)
# -------------------------------------------------------------------------

class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CyberMaths Steganography Tool")
        self.root.geometry("900x650")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- Variables ---
        self.src_image_path = None
        self.stego_image_object = None
        self.decoded_image_path = None
        
        # --- Main Layout ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # --- TABS ---
        self.tab_encode = ttk.Frame(self.notebook)
        self.tab_decode = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_encode, text="  Encode (Hide)  ")
        self.notebook.add(self.tab_decode, text="  Decode (Reveal)  ")
        self.notebook.add(self.tab_analysis, text="  Maths Analysis  ")
        
        self._setup_encode_tab()
        self._setup_decode_tab()
        self._setup_analysis_tab()

    # ---------------------------------------------------------------------
    # ENCODE TAB
    # ---------------------------------------------------------------------
    def _setup_encode_tab(self):
        # Left Panel: Image
        left_frame = ttk.Frame(self.tab_encode)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        lbl_instr = ttk.Label(left_frame, text="1. Select Cover Image", font=("Arial", 10, "bold"))
        lbl_instr.pack(anchor="w")
        
        self.btn_load_enc = ttk.Button(left_frame, text="Load Image", command=self.load_image_encode)
        self.btn_load_enc.pack(fill="x", pady=5)
        
        self.lbl_img_preview_enc = ttk.Label(left_frame, text="No Image Selected", relief="sunken")
        self.lbl_img_preview_enc.pack(fill="both", expand=True)

        # Right Panel: Controls
        right_frame = ttk.Frame(self.tab_encode)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(right_frame, text="2. Secret Message", font=("Arial", 10, "bold")).pack(anchor="w")
        self.txt_msg = tk.Text(right_frame, height=5)
        self.txt_msg.pack(fill="x", pady=5)
        
        ttk.Label(right_frame, text="3. Security (Optional)", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        ttk.Label(right_frame, text="Encryption Password:").pack(anchor="w")
        self.entry_pass_enc = ttk.Entry(right_frame, show="*")
        self.entry_pass_enc.pack(fill="x")
        
        ttk.Label(right_frame, text="4. Technique", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        self.algo_var = tk.StringVar(value="LSB")
        ttk.Radiobutton(right_frame, text="Standard LSB", variable=self.algo_var, value="LSB").pack(anchor="w")
        ttk.Radiobutton(right_frame, text="LSB + XOR Encryption", variable=self.algo_var, value="XOR").pack(anchor="w")
        
        self.btn_encode = ttk.Button(right_frame, text="ENCRYPT & SAVE IMAGE", command=self.process_encode)
        self.btn_encode.pack(fill="x", pady=20)
        
    # ---------------------------------------------------------------------
    # DECODE TAB
    # ---------------------------------------------------------------------
    def _setup_decode_tab(self):
        main_frame = ttk.Frame(self.tab_decode)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.btn_load_dec = ttk.Button(main_frame, text="Load Stego Image", command=self.load_image_decode)
        self.btn_load_dec.pack(fill="x")
        
        self.lbl_img_preview_dec = ttk.Label(main_frame, text="No Image Selected", relief="sunken")
        self.lbl_img_preview_dec.pack(fill="both", expand=True, pady=10)
        
        ttk.Label(main_frame, text="Decryption Password (if used):").pack(anchor="w")
        self.entry_pass_dec = ttk.Entry(main_frame, show="*")
        self.entry_pass_dec.pack(fill="x", pady=5)
        
        self.btn_decode = ttk.Button(main_frame, text="REVEAL HIDDEN MESSAGE", command=self.process_decode)
        self.btn_decode.pack(fill="x", pady=10)
        
        ttk.Label(main_frame, text="Hidden Message:").pack(anchor="w")
        self.txt_output = tk.Text(main_frame, height=5)
        self.txt_output.pack(fill="x")

    # ---------------------------------------------------------------------
    # ANALYSIS TAB
    # ---------------------------------------------------------------------
    def _setup_analysis_tab(self):
        main_frame = ttk.Frame(self.tab_analysis)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Mathematical Analysis", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Metrics Display
        self.lbl_mse = ttk.Label(main_frame, text="MSE (Mean Squared Error): N/A", font=("Courier", 12))
        self.lbl_mse.pack(anchor="w")
        
        self.lbl_psnr = ttk.Label(main_frame, text="PSNR (Signal-to-Noise Ratio): N/A", font=("Courier", 12))
        self.lbl_psnr.pack(anchor="w")
        
        ttk.Label(main_frame, text="\nEXIF Metadata:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        self.txt_exif = tk.Text(main_frame, height=10)
        self.txt_exif.pack(fill="x", pady=5)
        
        ttk.Button(main_frame, text="Calculate Metrics (Compare Original vs Stego)", command=self.run_analysis).pack(pady=20)

    # ---------------------------------------------------------------------
    # HELPER FUNCTIONS
    # ---------------------------------------------------------------------
    def load_image_encode(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.src_image_path = path
            self._show_image(path, self.lbl_img_preview_enc)

    def load_image_decode(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.bmp")]) # JPG destroys LSB
        if path:
            self.decoded_image_path = path
            self._show_image(path, self.lbl_img_preview_dec)
            
            # Show EXIF immediately
            exif_info = StegoEngine.get_exif_data(path)
            self.txt_exif.delete(1.0, tk.END)
            self.txt_exif.insert(tk.END, exif_info)

    def _show_image(self, path, label_widget):
        img = Image.open(path)
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        label_widget.config(image=photo, text="")
        label_widget.image = photo # Keep reference

    def process_encode(self):
        if not self.src_image_path:
            messagebox.showerror("Error", "Please load an image first.")
            return
            
        msg = self.txt_msg.get("1.0", tk.END).strip()
        if not msg:
            messagebox.showerror("Error", "Please enter a message.")
            return

        password = self.entry_pass_enc.get() if self.algo_var.get() == "XOR" else None
        
        try:
            # Run the engine
            stego_img = StegoEngine.encode_lsb(self.src_image_path, msg, password)
            self.stego_image_object = stego_img # Save in memory for analysis
            
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
            if save_path:
                stego_img.save(save_path)
                messagebox.showinfo("Success", f"Image saved to {save_path}")
                self.decoded_image_path = save_path # Auto-load for analysis
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_decode(self):
        if not self.decoded_image_path:
            messagebox.showerror("Error", "Please load a Stego image.")
            return
            
        password = self.entry_pass_dec.get()
        
        try:
            msg = StegoEngine.decode_lsb(self.decoded_image_path, password)
            self.txt_output.delete(1.0, tk.END)
            self.txt_output.insert(tk.END, msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_analysis(self):
        if not self.src_image_path or not self.decoded_image_path:
            messagebox.showerror("Error", "Need both Original and Stego images loaded to compare.")
            return
            
        try:
            orig = Image.open(self.src_image_path)
            stego = Image.open(self.decoded_image_path)
            
            # Ensure size matches (in case user loaded wrong images)
            if orig.size != stego.size:
                messagebox.showerror("Error", "Images must be same size for MSE/PSNR.")
                return

            mse, psnr = StegoEngine.calculate_metrics(orig, stego)
            
            self.lbl_mse.config(text=f"MSE: {mse:.4f} (Lower is better)")
            self.lbl_psnr.config(text=f"PSNR: {psnr:.2f} dB (Higher is better)")
            
            if mse < 0.1:
                messagebox.showinfo("Analysis", "The images are mathematically almost identical!\nThis proves the steganography is invisible.")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = StegoApp(root)
    root.mainloop()
