import os
import requests
import json
from docx2pdf import convert  # <-- new

API_URL = "http://10.9.64.22:8000/file_parse"  # MinerU endpoint
DOCS_FOLDER = "./oran_specifications_latest"
OUTPUT_FOLDER = "./output/final_parsed_oran_latest"
TEMP_PDF_FOLDER = "./temp_oran_pdfs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_PDF_FOLDER, exist_ok=True)





# ------------------------------
# Parse document via MinerU
# ------------------------------
def parse_document(file_path):
    filename = os.path.basename(file_path)
    ext = filename.lower().split(".")[-1]

    # Only parse PDFs (DOCX already converted)
    if ext != "pdf":
        print(f"Skipping unsupported file: {filename}")
        return None

    with open(file_path, "rb") as f:
        files = {"files": (filename, f, "application/pdf")}
        data = {
            "output_dir": OUTPUT_FOLDER,
            "return_md": True,
            "return_middle_json": False,
            "return_images": True
        }
        response = requests.post(API_URL, files=files, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed: {filename}, Status: {response.status_code}")
        print(response.text)
        return None


# ------------------------------
# Main loop
# ------------------------------
for filename in os.listdir(DOCS_FOLDER):
    if not filename.lower().endswith((".pdf")):
        continue

    input_path = os.path.join(DOCS_FOLDER, filename)
    output_file = os.path.join(OUTPUT_FOLDER, filename.rsplit(".", 1)[0] + ".md")

    # ✅ Skip if already processed
    if os.path.exists(output_file):
        print(f"⏩ Skipping {filename} (already processed)")
        continue

    # Convert DOCX → PDF if needed
    if filename.lower().endswith(".docx"):
        pdf_path = convert_docx_to_pdf(input_path, TEMP_PDF_FOLDER)
        if not pdf_path:
            continue
        file_to_parse = pdf_path
    else:
        file_to_parse = input_path

    print(f"📄 Parsing {file_to_parse}...")
    result = parse_document(file_to_parse)

    if result:
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2, ensure_ascii=False)
        print(f"✅ Saved output: {output_file}")

print("\n🎉 All documents processed successfully!")






#docker run --gpus all --shm-size 32g -p 8000:8000 --ipc=host --restart always mineru-vllm:latest mineru-api --host 0.0.0.0 --port 8000


# Json Format:
# {
#   "backend": "pipeline",
#   "version": "2.5.4",
#   "results": {
#     "<pdf_filename>": {
#       "middle_json": {
#         "pdf_info": [
#           {
#             "page_idx": 0,
#             "page_size": [595, 842],
#             "preproc_blocks": [
#               {
#                 "type": "title" | "text" | "table",
#                 "bbox": [x1, y1, x2, y2],
#                 "lines": [
#                   {
#                     "bbox": [x1, y1, x2, y2],
#                     "spans": [
#                       {
#                         "bbox": [x1, y1, x2, y2],
#                         "score": 1.0,
#                         "content": "Extracted text content",
#                         "type": "text" | "inline_equation" | "table"
#                       }
#                     ],
#                     "index": <int>
#                   }
#                 ],
#                 "index": <float>
#               }
#             ],
#             "discarded_blocks": [
#               {
#                 "type": "discarded",
#                 "bbox": [x1, y1, x2, y2],
#                 "lines": [
#                   {
#                     "bbox": [x1, y1, x2, y2],
#                     "spans": [
#                       {
#                         "bbox": [x1, y1, x2, y2],
#                         "score": <float>,
#                         "content": "Logo or unwanted text",
#                         "type": "text"
#                       }
#                     ]
#                   }
#                 ]
#               }
#             ],
#             "para_blocks": [
#               {
#                 "type": "title" | "text" | "table",
#                 "bbox": [x1, y1, x2, y2],
#                 "lines": [...],
#                 "blocks": [
#                   {
#                     "type": "table_body",
#                     "bbox": [x1, y1, x2, y2],
#                     "spans": [
#                       {
#                         "html": "<table>...</table>",
#                         "image_path": "<image_filename>.jpg"
#                       }
#                     ]
#                   }
#                 ]
#               }
#             ]
#           }
#         ]
#       },
#       "images": {
#         "<image_filename>.jpg": "<base64 or path>"
#       }
#     }
#   }
# }

# Md Format:
# {
#   "backend": "pipeline",
#   "version": "2.5.4",
#   "results": {
#     "<pdf_filename>": {
#       "md_content": "# <Document title>\n\nMarkdown version of the document text with headings, paragraphs, HTML tables, and image references like:\n\n![](images/<image_hash>.jpg)\n\nTables are stored as inline HTML, equations are preserved, and chapters/sections are represented with markdown headings.\n",
#       "images": {
#         "<image_hash>.jpg": "<base64_encoded_image_or_relative_path>",
#         "<image_hash>.jpg": "<base64_encoded_image_or_relative_path>"
#       }
#     }
#   }
# }
