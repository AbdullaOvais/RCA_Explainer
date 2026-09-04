import os
import json
import base64

# Path to your MinerU JSON file
json_file = "./output/2019.09.30-ORAN-WG2.A1AP_v01.00.json; filename*=UTF-8''2019.09.30-ORAN-WG2.A1AP_v01.00.json"

# Folder to save extracted images
output_img_folder = "./output/images"
os.makedirs(output_img_folder, exist_ok=True)

# Load JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Get the first (or only) PDF key in "results"
pdf_key = list(data["results"].keys())[0]

# Extract images dictionary
images_dict = data["results"][pdf_key]["images"]

# Save each image
for img_name, b64_data in images_dict.items():
    # Remove "data:image/jpeg;base64," prefix if present
    if b64_data.startswith("data:image"):
        b64_data = b64_data.split(",")[1]
    img_path = os.path.join(output_img_folder, img_name)
    with open(img_path, "wb") as img_file:
        img_file.write(base64.b64decode(b64_data))
    print(f"Saved: {img_path}")

print(f"All images saved to {output_img_folder}")
