import os
import json
import base64
import requests

# --------------------------
# CONFIG
# --------------------------
OLLAMA_URL = "http://192.168.50.140:11434/api/generate"
MODEL_NAME = "gemma3:12b"
MD_JSON_FOLDER = "/home/michael/Desktop/rca_explainer/output/final_parsed_oran_latest"
OUTPUT_FOLDER = "/home/michael/Desktop/rca_explainer/output/final_parsed_oran_latest_with_image"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------
# FUNCTION: Summarize an image
# --------------------------
def summarize_image(image_b64, image_name):
    prompt_text = (
        "You are an expert in 5G, 3GPP, and O-RAN. I will give you an image from a "
        "5G/O-RAN specification. Convert the image into a detailed, high-fidelity text "
        "description that preserves every entity, relationship, interface, direction, "
        "hierarchy, flow, and grouping exactly as shown. The result must be suitable "
        "for building a knowledge graph (GraphRAG). Follow the exact style shown in "
        "the examples below.\n\n"

        "EXAMPLE 1 (Architecture Image → Text):\n"
        "Entities:\n"
        "- O-RU – Radio Unit (RF, Low-PHY)\n"
        "- O-DU – Distributed Unit (High-PHY, MAC, RLC)\n"
        "- O-CU – Central Unit:\n"
        "  - CU-CP – Control Plane\n"
        "  - CU-UP – User Plane\n\n"

        "Relationships:\n"
        "- O-RU → O-DU: Fronthaul (Low-PHY ↔ High-PHY), bidirectional\n"
        "- O-DU → CU-CP: F1-C interface\n"
        "- O-DU → CU-UP: F1-U interface\n"
        "- CU-CP ↔ Near-RT RIC: E2 interface\n"
        "- Near-RT RIC ↔ O-DU/O-CU: A1 interface\n\n"

        "Hierarchy:\n"
        "- gNB = RU + DU + CU-CP + CU-UP\n\n"

        "Notes:\n"
        "- Include all labels in the image.\n\n"

        "EXAMPLE 2 (Call Flow Image → Text):\n"
        "Entities:\n"
        "- UE\n"
        "- gNB\n"
        "- AMF\n\n"

        "Flow:\n"
        "1. UE → gNB: RRCSetupRequest\n"
        "2. gNB → UE: RRCSetup\n"
        "3. UE → gNB: RRCSetupComplete\n"
        "4. gNB → AMF: InitialUEMessage\n\n"

        "Relationships:\n"
        "- Each arrow direction must match the diagram.\n\n"

        "WHEN YOU GET A NEW IMAGE, ALWAYS OUTPUT USING THIS EXACT TEMPLATE:\n\n"

        "Entities:\n"
        "- List every entity/block exactly as shown.\n\n"

        "Relationships:\n"
        "- Entity A → Entity B: interface/protocol name, purpose, direction.\n\n"

        "Hierarchy:\n"
        "- Describe all parent-child groupings and internal sub-blocks.\n\n"

        "Flows (if present):\n"
        "- Step 1 → Step 2 → Step 3 with message names exactly as shown.\n\n"

        "Notes:\n"
        "- Include every label, footnote, number, and caption visible in the image.\n\n"

        "Do NOT simplify, infer, or omit anything. Transcribe the structure exactly "
        "so it can be converted to a knowledge graph."
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "format": "json",
        "images": [image_b64],
        "stream": False
    }


    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"[ERROR] Failed to summarize {image_name}: {e}")
        return "(Summary not available)"

# --------------------------
# PROCESS ALL .json FILES
# --------------------------
for file_name in os.listdir(MD_JSON_FOLDER):
    # Only process .json files
    if not file_name.endswith(".md"):
        continue

    json_path = os.path.join(MD_JSON_FOLDER, file_name)

    # Parse JSON to get pdf_name before checking skip condition
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_name}: {e}")
        continue

    results = data.get("results", {})
    for pdf_name, pdf_data in results.items():
        output_md_path = os.path.join(OUTPUT_FOLDER, f"{pdf_name}.md")

        # ✅ Skip if output already exists
        if os.path.exists(output_md_path):
            print(f"⏩ Skipping '{pdf_name}' (already processed)")
            continue

        md_content = pdf_data.get("md_content", "")
        images = pdf_data.get("images", {})

        # Process and summarize each image
        for image_name, image_value in images.items():
            # Handle base64 data URIs, file paths, or direct base64
            if isinstance(image_value, str) and image_value.startswith("data:image"):
                # Strip prefix like 'data:image/jpeg;base64,'
                image_b64 = image_value.split(",", 1)[1]
            elif os.path.exists(image_value):
                with open(image_value, "rb") as img_file:
                    image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            else:
                # Assume already base64 without prefix
                image_b64 = image_value

            summary = summarize_image(image_b64, image_name)
            print(f"[{image_name}] -> {summary[:80]}...")

            # Inject summary after the image markdown
            image_ref = f"![](images/{image_name})"
            summary_text = f"{image_ref}\n\n> **Image Summary:** {summary}\n"
            md_content = md_content.replace(image_ref, summary_text)

        # Save new markdown
        with open(output_md_path, "w") as out_file:
            out_file.write(md_content)

        print(f"✅ Processed and saved: {output_md_path}")

print("🎉 All files processed successfully!")
