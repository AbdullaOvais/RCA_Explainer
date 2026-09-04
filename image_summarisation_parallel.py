import os
import json
import base64
import requests
import itertools
import threading
import queue
from typing import Dict, Tuple

# --------------------------
# MULTI-NODE OLLAMA CONFIG
# --------------------------
OLLAMA_NODES = [
    "http://127.0.0.1:11440/api/generate",
    "http://127.0.0.1:11441/api/generate",
    "http://127.0.0.1:11442/api/generate",
    "http://127.0.0.1:11443/api/generate",
    "http://127.0.0.1:11444/api/generate",
    "http://127.0.0.1:11445/api/generate",
    "http://127.0.0.1:11446/api/generate",
    "http://127.0.0.1:11447/api/generate",
]

MODEL_NAME = "gemma3:12b"
MD_JSON_FOLDER = "/home/michael/Desktop/rca_explainer/output/final_parsed_oran_latest"
OUTPUT_FOLDER = "/home/michael/Desktop/rca_explainer/output/final_parsed_oran_latest_with_image"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------
# PROMPT (unchanged)
# --------------------------
PROMPT_TEXT = (
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

# --------------------------
# Summarize helper (with retry)
# --------------------------
def call_ollama_endpoint(endpoint: str, image_b64: str, image_name: str, retries: int = 2, timeout: int = 300) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT_TEXT,
        "format": "json",
        "images": [image_b64],
        "stream": False
    }

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            last_exc = e
            print(f"[WARN] attempt {attempt} failed for {image_name} on {endpoint}: {e}")
    print(f"[ERROR] All attempts failed for {image_name} on {endpoint}: {last_exc}")
    return "(Summary not available)"

# --------------------------
# Worker function
# --------------------------
def worker_thread_fn(worker_id: int, endpoint: str, job_queue: "queue.Queue[Tuple[str, str, str]]",
                     result_dict: Dict[Tuple[str, str], str], result_lock: threading.Lock):
    """
    Each worker is pinned to a single endpoint (and therefore one GPU).
    job_queue items: (pdf_name, image_name, image_value_or_path_or_datauri)
    """
    thread_name = f"worker-{worker_id}"
    while True:
        job = job_queue.get()
        if job is None:
            # sentinel to stop
            job_queue.task_done()
            print(f"[{thread_name}] Received stop signal. Exiting.")
            break

        pdf_name, image_name, image_value = job
        try:
            # Normalize base64
            if isinstance(image_value, str) and image_value.startswith("data:image"):
                image_b64 = image_value.split(",", 1)[1]
            elif isinstance(image_value, str) and os.path.exists(image_value):
                with open(image_value, "rb") as img_file:
                    image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            else:
                image_b64 = image_value

            # Call the assigned endpoint
            summary = call_ollama_endpoint(endpoint, image_b64, image_name)

            # Store result in shared dict with lock
            with result_lock:
                result_dict[(pdf_name, image_name)] = summary

            print(f"[{thread_name}] Completed {image_name} for {pdf_name} on {endpoint}")

        except Exception as e:
            print(f"[{thread_name}] Exception processing {image_name}: {e}")
            with result_lock:
                result_dict[(pdf_name, image_name)] = "(Summary not available)"
        finally:
            job_queue.task_done()


# --------------------------
# PROCESS ALL .json FILES (MAIN)
# --------------------------
for file_name in os.listdir(MD_JSON_FOLDER):

    if not file_name.endswith(".md"):
        continue

    json_path = os.path.join(MD_JSON_FOLDER, file_name)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_name}: {e}")
        continue

    results = data.get("results", {})
    for pdf_name, pdf_data in results.items():
        output_md_path = os.path.join(OUTPUT_FOLDER, f"{pdf_name}.md")

        if os.path.exists(output_md_path):
            print(f"⏩ Skipping '{pdf_name}' (already processed)")
            continue

        md_content = pdf_data.get("md_content", "")
        images = pdf_data.get("images", {})

        # If no images, just write the md_content unchanged
        if not images:
            with open(output_md_path, "w") as out_file:
                out_file.write(md_content)
            print(f"✅ Processed and saved (no images): {output_md_path}")
            continue

        # Create job queue and shared result store for this PDF
        job_queue: "queue.Queue[Tuple[str, str, str]]" = queue.Queue()
        result_dict: Dict[Tuple[str, str], str] = {}
        result_lock = threading.Lock()

        # Start one worker per endpoint (pinning)
        workers = []
        for i, endpoint in enumerate(OLLAMA_NODES):
            t = threading.Thread(target=worker_thread_fn,
                                 args=(i, endpoint, job_queue, result_dict, result_lock),
                                 daemon=True)
            t.start()
            workers.append(t)
            print(f"[MAIN] Started worker-{i} -> {endpoint}")

        # enqueue all images for this pdf
        for image_name, image_value in images.items():
            job_queue.put((pdf_name, image_name, image_value))

        # Wait until all tasks are completed
        job_queue.join()
        print(f"[MAIN] All image jobs completed for PDF: {pdf_name}")

        # Send sentinel None to each worker to stop them cleanly
        for _ in workers:
            job_queue.put(None)
        # Wait for worker threads to exit
        for t in workers:
            t.join(timeout=5)

        # Replace image refs with corresponding summaries (original logic preserved)
        for image_name in images.keys():
            summary = result_dict.get((pdf_name, image_name), "(Summary not available)")
            print(f"[{image_name}] -> {summary[:80]}...")
            image_ref = f"![](images/{image_name})"
            summary_text = f"{image_ref}\n\n> **Image Summary:** {summary}\n"
            md_content = md_content.replace(image_ref, summary_text)

        # Save updated markdown
        with open(output_md_path, "w") as out_file:
            out_file.write(md_content)

        print(f"✅ Processed and saved: {output_md_path}")

print("🎉 All files processed successfully!")
