import os
import subprocess
import shutil

# ------------------------------
# Folders
# ------------------------------
INPUT_FOLDER = "./oran_specifications_latest"  # Root folder containing DOCX/DOC files

# ------------------------------
# Convert DOC/DOCX → PDF using AbiWord
# ------------------------------
def convert_doc_to_pdf(doc_path):
    folder = os.path.dirname(doc_path)
    filename = os.path.basename(doc_path)
    base_name = os.path.splitext(filename)[0]
    pdf_path = os.path.join(folder, base_name + ".pdf")

    if os.path.exists(pdf_path):
        print(f"⏩ Skipping (already converted): {pdf_path}")
        return pdf_path

    try:
        # Run AbiWord with timeout
        subprocess.run(
            ["abiword", "--to=pdf", doc_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20
        )

        if os.path.exists(pdf_path):
            print(f"✅ Converted: {doc_path} → {pdf_path}")
            os.remove(doc_path)
            print(f"🗑️ Removed original file: {doc_path}")
            return pdf_path
        else:
            print(f"❌ PDF not found after conversion for: {doc_path}")
            return None

    except subprocess.TimeoutExpired:
        print(f"⏳ Timeout! AbiWord got stuck converting: {doc_path}")
        return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed for {doc_path}: {e}")
        return None

# ------------------------------
# Recursively convert all DOC/DOCX files
# ------------------------------
for root, _, files in os.walk(INPUT_FOLDER):
    for filename in files:
        if filename.lower().endswith((".doc", ".docx")):
            input_path = os.path.join(root, filename)
            convert_doc_to_pdf(input_path)

print("🎉 All DOC/DOCX files processed (including subfolders)!")
