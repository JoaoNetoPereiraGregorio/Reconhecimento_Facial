import os
import time


def save_upload_file(file, upload_dir="./uploads/"):
    """Save an uploaded Flask file to disk without overwriting existing files."""
    os.makedirs(upload_dir, exist_ok=True)
    upload_filename = f"upload_{int(time.time())}_{os.path.basename(file.filename)}"
    upload_path = os.path.join(upload_dir, upload_filename)

    base, ext = os.path.splitext(upload_filename)
    counter = 1
    while os.path.exists(upload_path):
        upload_filename = f"{base}_{counter}{ext}"
        upload_path = os.path.join(upload_dir, upload_filename)
        counter += 1

    file.save(upload_path)
    renamed = counter > 1
    return upload_path, upload_filename, renamed
