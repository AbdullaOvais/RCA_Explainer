import os
import zipfile

INPUT_FOLDER = "./oran_specifications_latest"   # Folder containing ZIP files

# Create output folder if you want separate extraction area (optional)
# OUTPUT_FOLDER = "./unzipped"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(".zip"):
        zip_path = os.path.join(INPUT_FOLDER, filename)
        
        # Extract into folder with same name as zip (without .zip)
        extract_folder = os.path.join(INPUT_FOLDER, os.path.splitext(filename)[0])
        os.makedirs(extract_folder, exist_ok=True)

        print(f"🔓 Extracting: {filename}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            print(f"✅ Extracted to: {extract_folder}")
        except zipfile.BadZipFile:
            print(f"❌ Bad ZIP file: {filename}")

print("🎉 All ZIP files processed!")
