import os
import re
import requests

# --------------------------
# CONFIG
# --------------------------
OLLAMA_URL = "http://10.9.64.22:11434/api/generate"
MODEL_NAME = "gemma3:12b"
MD_FOLDER = "/home/michael/Desktop/rca_explainer/output/final_parsed_with_image"
OUTPUT_FOLDER = "/home/michael/Desktop/rca_explainer/output/final_parsed_with_table"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------
# FUNCTION: Summarize an HTML table
# --------------------------
def summarize_table(table_html, table_index):
    prompt = (
        "You are analyzing a technical or analytical document that contains HTML tables. "
        "Read the table carefully and provide a concise, factual maximum 4–5 sentence summary "
        "explaining what the table describes, what kind of data or mapping it presents, "
        "and what its purpose is. Keep it objective and context-aware and don't miss the mappings and dependencies.\n\n"
        f"Here is the table:\n{table_html}"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip() or "(Table summary not available)"
    except Exception as e:
        print(f"[ERROR] Failed to summarize table {table_index}: {e}")
        return "(Table summary not available)"

# --------------------------
# FUNCTION: Replace tables in markdown
# --------------------------
def replace_tables_with_summaries(md_content):
    # Find all <table>...</table> blocks (non-greedy)
    tables = re.findall(r"<table.*?</table>", md_content, flags=re.DOTALL | re.IGNORECASE)
    print(f"🧾 Found {len(tables)} tables in document.")

    for i, table_html in enumerate(tables, start=1):
        summary = summarize_table(table_html, i)
        print(f"[Table {i}] -> {summary[:100]}...")

        # Replace table with summary text
        replacement = f"> **Table Summary:** {summary}\n"
        md_content = md_content.replace(table_html, replacement)

    return md_content

# --------------------------
# PROCESS ALL .md FILES
# --------------------------
for file_name in os.listdir(MD_FOLDER):
    if not file_name.endswith(".md"):
        continue

    input_path = os.path.join(MD_FOLDER, file_name)
    output_path = os.path.join(OUTPUT_FOLDER, file_name)

    # ✅ Skip if already processed
    if os.path.exists(output_path):
        print(f"⏩ Skipping '{file_name}' (already processed)")
        continue

    # Read markdown text
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            md_content = f.read()
    except Exception as e:
        print(f"[ERROR] Could not read {file_name}: {e}")
        continue

    # Replace tables with summaries
    updated_md = replace_tables_with_summaries(md_content)

    # Save result
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(updated_md)

    print(f"✅ Processed and saved: {output_path}")

print("🎉 All table summaries completed successfully!")
