import os
import re
import json
import requests
import pdfplumber
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ----------------------------
# CONFIG
# ----------------------------
OLLAMA_URL = "http://10.9.64.22:11435/api/generate"
MODEL_NAME = "deepseek-r1:32b"
DOCS_FOLDER = "./docs"

# ----------------------------
# Step 1: LLM query
# ----------------------------
def ask_deepseek(prompt: str):
    payload = {"model": MODEL_NAME, "prompt": prompt}
    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    full_output = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                full_output += data["response"]
    return full_output.strip()

# ----------------------------
# Step 2: Table Extraction
# ----------------------------
def extract_tables_from_pdf(pdf_path):
    tables_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                # Flatten table rows into text
                table_text = "\n".join([" | ".join([str(cell) for cell in row]) for row in table])
                tables_text.append(table_text)
    return tables_text

def summarize_table(table_text):
    prompt = f"""
You are an expert in summarizing tables for technical documents.
Summarize the following table into concise and clear text:

Table:
{table_text}

Summary:
"""
    return ask_deepseek(prompt)

# ----------------------------
# Step 3: Figure Extraction
# ----------------------------
def extract_figures_from_pdf(pdf_path, save_folder="./figures"):
    os.makedirs(save_folder, exist_ok=True)
    figure_images = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for img_index, img in enumerate(page.images):
                # Crop image from page
                bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                pil_img = page.to_image(resolution=300).original.crop(bbox)
                image_path = os.path.join(save_folder, f"{os.path.basename(pdf_path)}_page{i}_fig{img_index}.png")
                pil_img.save(image_path)
                figure_images.append(image_path)

    return figure_images

# ----------------------------
# Step 4: Figure Summarization using BLIP-2
# ----------------------------
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

def summarize_figure(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(images=raw_image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    description = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Optionally, refine with DeepSeek
    prompt = f"Summarize the insights from this figure:\n{description}"
    summary = ask_deepseek(prompt)
    return summary

# ----------------------------
# Step 5: Process All PDFs
# ----------------------------
def process_documents(folder=DOCS_FOLDER):
    all_table_summaries = []
    all_figure_summaries = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not filename.lower().endswith(".pdf"):
            continue

        print(f"📄 Processing {filename}...")

        # Tables
        tables = extract_tables_from_pdf(path)
        for t in tables:
            summary = summarize_table(t)
            all_table_summaries.append(summary)

        # Figures
        figures = extract_figures_from_pdf(path)
        for f in figures:
            summary = summarize_figure(f)
            all_figure_summaries.append(summary)

    return all_table_summaries, all_figure_summaries

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    table_summaries, figure_summaries = process_documents()

    print("\n📝 Table Summaries:")
    for s in table_summaries:
        print(s, "\n---")

    print("\n📝 Figure Summaries:")
    for s in figure_summaries:
        print(s, "\n---")
