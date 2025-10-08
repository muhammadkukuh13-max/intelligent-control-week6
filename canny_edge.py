import cv2
import numpy as np
import os

def find_first_image(folder_path):
    """Cari file gambar pertama (jpg/png/jpeg) di folder dan subfolder"""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, file)
    return None

def canny_edge_detection(image_path):
    """Mendeteksi tepi menggunakan metode Canny Edge Detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {image_path}")

    # Tahap preprocessing dan deteksi tepi
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)

    # Tampilkan hasil
    cv2.imshow("Canny Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Simpan hasil
    result_path = os.path.join(os.path.dirname(image_path), "canny_result.jpg")
    cv2.imwrite(result_path, edges)
    print(f"Hasil disimpan di: {result_path}")

    return result_path

def process_rail_segmentation_images():
    """Proses gambar dari folder rail_segmentation"""
    folder_path = os.path.join("archive", "rail_segmentation")
    image_path = find_first_image(folder_path)

    if image_path is None:
        raise FileNotFoundError("Tidak ada file gambar (.jpg/.png) ditemukan di folder rail_segmentation!")

    print(f"Gambar ditemukan: {image_path}")
    return canny_edge_detection(image_path)

# === Jalankan program utama ===
if __name__ == "__main__":
    process_rail_segmentation_images()