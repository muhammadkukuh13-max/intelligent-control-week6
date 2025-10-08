from ultralytics import YOLO
import cv2
import os
import glob

# === Load model YOLOv8 Instance Segmentation ===
model = YOLO("yolov8n-seg.pt")

def detect_rail_lane(folder_path, delay_time=5000):
    """
    Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation.
    delay_time = waktu tampil (ms), default 5000 ms = 5 detik.
    """
    
    abs_folder_path = os.path.join(os.getcwd(), folder_path)
    print(f"Mengecek folder: {abs_folder_path}")

    # Cari semua file gambar (.jpg, .png, .jpeg) termasuk di subfolder
    image_files = glob.glob(os.path.join(abs_folder_path, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(abs_folder_path, "**", "*.png"), recursive=True) + \
                  glob.glob(os.path.join(abs_folder_path, "**", "*.jpeg"), recursive=True)

    if not image_files:
        raise FileNotFoundError(f"Tidak ada gambar ditemukan di folder: {abs_folder_path}")

    image_path = image_files[0]
    print(f"Memproses gambar: {image_path}")

    # Jalankan deteksi YOLOv8
    results = model(image_path)

    # Ambil hasil gambar yang sudah diberi bounding box
    result_img = results[0].plot()

    # Tampilkan hasil di jendela OpenCV
    cv2.imshow("YOLOv8 Rail Lane Detection", result_img)
    print(f"Tampilan hasil selama {delay_time/1000} detik...")
    cv2.waitKey(delay_time)  # tunggu beberapa detik (ms)
    cv2.destroyAllWindows()

    # Simpan hasil di folder yang sama dengan gambar
    output_path = os.path.join(os.path.dirname(image_path), "lane_detection_result.jpg")
    cv2.imwrite(output_path, result_img)
    print(f"Hasil deteksi disimpan di: {output_path}")


# === Jalankan program utama ===
if __name__ == "__main__":
    detect_rail_lane("archive/rail_segmentation/test", delay_time=8000)  # tampil 8 detik