# combined_detection.py
from ultralytics import YOLO
import cv2
import os
import glob
import sys
from canny_edge import canny_edge_detection

# Load model YOLOv8 Instance Segmentation
model = YOLO("yolov8n-seg.pt")


def find_first_image_in_folder(folder):
    """Cari gambar pertama di folder (rekursif)."""
    if not os.path.isdir(folder):
        return None
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.tif"]
    for p in patterns:
        files = glob.glob(os.path.join(folder, p), recursive=True)
        if files:
            files.sort()
            return files[0]
    return None


def resolve_image_path(image_path_or_folder):
    """
    Terima path file atau folder. 
    Jika file ada -> return file absolute.
    Jika folder -> cari gambar pertama di folder tersebut.
    Jika tidak ada -> coba beberapa lokasi umum (archive/...).
    """
    # 1) langsung file absolute
    cand = os.path.abspath(image_path_or_folder)
    if os.path.isfile(cand):
        return cand

    # 2) jika folder yang valid -> cari gambar di dalamnya
    if os.path.isdir(cand):
        found = find_first_image_in_folder(cand)
        if found:
            return os.path.abspath(found)

    # 3) coba beberapa lokasi umum relatif ke project
    common_folders = [
        os.path.join("archive", "rail_segmentation", "test", "images"),
        os.path.join("archive", "rail_segmentation", "test"),
        os.path.join("archive", "rail_segmentation"),
        os.path.join("archive")
    ]
    for f in common_folders:
        f_abs = os.path.abspath(f)
        found = find_first_image_in_folder(f_abs)
        if found:
            return os.path.abspath(found)

    # 4) cari seluruh folder project (expensive but fallback)
    found = find_first_image_in_folder(os.getcwd())
    if found:
        return os.path.abspath(found)

    return None


def combined_detection(image_input, display_time_ms=8000):
    """
    image_input: bisa berupa path file atau path folder (atau biarkan kosong untuk auto-search).
    display_time_ms: lama tampil jendela (ms). Gunakan 0 untuk tunggu key press.
    """
    print("Working dir:", os.getcwd())
    print("Mencari gambar untuk input:", image_input)

    image_path = resolve_image_path(image_input)
    if image_path is None:
        # tampilkan informasi debug agar mudah diperbaiki manual
        print("\n=== Debug info ===")
        print("CWD:", os.getcwd())
        print("Isi folder project (top-level):", os.listdir(os.getcwd()))
        archive_path = os.path.abspath("archive")
        print("Archive exists:", os.path.exists(archive_path), "->", archive_path)
        if os.path.exists(archive_path):
            try:
                print("Isi archive (first 20):", os.listdir(archive_path)[:20])
            except Exception:
                pass
        raise FileNotFoundError(f"Tidak menemukan gambar untuk input: {image_input}\nCoba pastikan folder/filename benar atau letakkan gambar di archive/rail_segmentation/test/images/")

    print("Menggunakan gambar:", image_path)

    # 1) Canny Edge Detection (mengembalikan path hasil)
    try:
        canny_result_path = canny_edge_detection(image_path)
    except Exception as e:
        raise RuntimeError(f"Gagal pada Canny Edge Detection: {e}")

    if not os.path.isfile(canny_result_path):
        raise FileNotFoundError(f"Hasil Canny tidak ditemukan di: {canny_result_path}")

    # 2) YOLOv8 inference
    print("Menjalankan YOLOv8 pada gambar sumber...")
    results = model(image_path)
    lane_img = results[0].plot()  # numpy array HxWx3 (BGR)

    # 3) gabungkan dengan hasil Canny
    edges = cv2.imread(canny_result_path, cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise FileNotFoundError(f"Gagal membaca hasil Canny di: {canny_result_path}")

    # resize edges jika ukuran beda (sesuaikan ukuran lane_img)
    if (edges.shape[0], edges.shape[1]) != (lane_img.shape[0], lane_img.shape[1]):
        edges = cv2.resize(edges, (lane_img.shape[1], lane_img.shape[0]))

    combined = cv2.addWeighted(lane_img, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # 4) tampilkan dan simpan
    cv2.imshow("Combined Detection", combined)
    print(f"Tampilkan selama {display_time_ms} ms (0 berarti tunggu tombol)...")
    key = cv2.waitKey(display_time_ms if display_time_ms != 0 else 0)
    cv2.destroyAllWindows()

    out_path = os.path.join(os.path.dirname(image_path), "combined_result.jpg")
    cv2.imwrite(out_path, combined)
    print("Hasil gabungan disimpan di:", out_path)
    return out_path


if __name__ == "__main__":
    # Jika pengguna memberi argumen, gunakan itu; jika tidak, beri folder default
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
    else:
        user_input = "archive/rail_segmentation/test/images"  # default try

    # contoh: python combined_detection.py archive/rail_segmentation/test
    combined_detection(user_input, display_time_ms=8000)