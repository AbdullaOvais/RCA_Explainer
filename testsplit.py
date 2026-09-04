import re
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------
# CONFIG
# --------------------------
BASE_CHUNK_SIZE = 2500
MAX_CHUNK_SIZE = 6000
SIMILARITY_THRESHOLD = 0.5
EMBED_MODEL = "all-MiniLM-L6-v2"
OVERLAP_SEGMENTS = 1

# MULTI-NODE OLLAMA CONFIG
OLLAMA_NODES = [
    "http://10.9.64.22:11434/api/generate",
     "http://127.0.0.1:11434/api/generate",
 ]
MODEL_NAME = "deepseek-r1:32b"

# --------------------------
# STRUCTURAL + CHUNK FUNCTIONS
# --------------------------
def structural_split(md_text):
    pattern = r'(?=^# |\n# |\n## |\n### |!\[\]\(images/|<table|> \*\*Image Summary:|> \*\*Table Summary:)'
    parts = re.split(pattern, md_text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]

def merge_short_segments(segments, min_len=400):
    merged, buffer = [], ""
    for seg in segments:
        if len(seg) < min_len or seg.startswith(("> **Image Summary:", "> **Table Summary:")):
            buffer += "\n\n" + seg
        else:
            if buffer:
                seg = buffer + "\n\n" + seg
                buffer = ""
            merged.append(seg)
    if buffer:
        merged.append(buffer)
    return merged

def dynamic_chunk_documents(md_text,
                            base_chunk_size=BASE_CHUNK_SIZE,
                            max_chunk_size=MAX_CHUNK_SIZE,
                            similarity_threshold=SIMILARITY_THRESHOLD,
                            model_name=EMBED_MODEL,
                            overlap_segments=OVERLAP_SEGMENTS):
    segments = structural_split(md_text)
    segments = merge_short_segments(segments)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(segments, normalize_embeddings=True)
    chunks, current_chunk_segments = [], [segments[0]]
    current_vec = embeddings[0]

    for i in range(1, len(segments)):
        sim = util.cos_sim(current_vec, embeddings[i]).item()
        combined_len = sum(len(s) for s in current_chunk_segments) + len(segments[i])
        if re.match(r"^#+\s+\d", segments[i]):
            chunks.append("\n\n".join(current_chunk_segments))
            overlap = current_chunk_segments[-overlap_segments:] if overlap_segments > 0 else []
            current_chunk_segments = overlap + [segments[i]]
            current_vec = embeddings[i]
            continue
        if sim > similarity_threshold and combined_len < max_chunk_size:
            current_chunk_segments.append(segments[i])
            current_vec = (current_vec + embeddings[i]) / 2
        else:
            current_len = sum(len(s) for s in current_chunk_segments)
            if current_len < base_chunk_size and chunks:
                prev_chunk = chunks.pop()
                merged_chunk = prev_chunk + "\n\n" + "\n\n".join(current_chunk_segments)
                chunks.append(merged_chunk)
            else:
                chunks.append("\n\n".join(current_chunk_segments))
            overlap = current_chunk_segments[-overlap_segments:] if overlap_segments > 0 else []
            current_chunk_segments = overlap + [segments[i]]
            current_vec = embeddings[i]
    if current_chunk_segments:
        chunks.append("\n\n".join(current_chunk_segments))
    return chunks

# --------------------------
# LLM EXTRACTION
# --------------------------
def extract_kg_from_chunk(chunk_text, model_name=MODEL_NAME, endpoint=None):
    if endpoint is None:
        raise ValueError("Ollama endpoint not specified.")

    prompt = """
You are an expert in 5G, O-RAN, 3GPP, fault management, and root cause analysis. 
You will receive a text chunk extracted from O-RAN or 5G specifications. 
Your job is to extract ONLY meaningful entities and relations required for 
knowledge graph construction (GraphRAG). Follow the strict rules below.

==================== STRICT RULES FOR ENTITY (NODE) EXTRACTION ====================
ALLOWED ENTITIES (keep these as nodes):
- 5G Network Functions (e.g., O-RU, O-DU, O-CU-CP, O-CU-UP, gNB, AMF, SMF, UPF).
- O-RAN Architecture Components (e.g., Near-RT RIC, Non-RT RIC, SMO, O1, O2).
- Functional Blocks (e.g., MAC, RLC, PDCP, PHY, scheduler, E2 Agent, E2 Termination).
- Interfaces (as entities only if described conceptually: F1, E1, E2, A1, O1, O2, Xn, NG).
- KPIs and Metrics (if meaningful: PRB Utilization, RSRP, RSRQ, SINR, Throughput).
- Faults, Events, Alarms, Anomalies (e.g., high latency, packet loss, RLF, HO failure).
- Procedures or Call Flow Steps (e.g., RRCSetup, HandoverRequest, Bearer Setup).

DISALLOWED ENTITIES (do NOT make these nodes):
- Filenames, section names, document numbers, spec IDs (e.g., TS 38.104, 3GPP 38.300).
- Generic words: 'table', 'figure', 'system', 'architecture', 'document', 'chapter'.
- Pure numbers unless representing KPIs/thresholds.
- Implementation-specific vendor examples.
- Full sentences or long phrases. Entities must be concise technical terms.

==================== STRICT RULES FOR RELATION (EDGE) EXTRACTION ====================
ALLOWED RELATION TYPES:
- Interface relations: 'connects_to', 'communicates_with', 'uses_interface'.
- Functional relations: 'contains', 'part_of', 'manages', 'controls', 'configured_by'.
- Data flow relations: 'sends', 'receives', 'reports_to', 'monitors'.
- RCA relations: 'causes', 'correlated_with', 'leads_to', 'results_in'.
- KPI relations: 'affects', 'degrades', 'improves', 'associated_with'.

DISALLOWED RELATIONS:
- Human-language descriptions that are not canonical relations.
- Relations involving disallowed entities.
- Duplicates and trivial restatements.

==================== SPECIAL RULES WHEN ENTITIES DO NOT EXIST ====================
- If no valid technical entities exist in the chunk, return empty arrays:
  "entities": [] and "relations": []
- Do NOT hallucinate entities.
- Do NOT infer functions or architecture unless explicitly present.

==================== OUTPUT FORMAT (STRICT JSON ONLY) ====================
Return ONLY valid JSON in this exact format:
{
  "entities": ["Entity1", "Entity2", ...],
  "relations": [
    {"subject": "Entity1", "predicate": "relation", "object": "Entity2"}
  ]
}

==================== IN-CONTEXT EXAMPLE ====================
INPUT TEXT EXAMPLE:
\"\"\"
The O-DU communicates with the O-RU over the Lower Layer Split (L1/L2). 
The O-DU also sends E2 reports to the Near-RT RIC via the E2 interface. 
High PRB Utilization can lead to increased latency.
\"\"\"

EXPECTED OUTPUT EXAMPLE:
{
  "entities": [
    "O-DU", "O-RU", "Near-RT RIC", "E2", "PRB Utilization", "latency"
  ],
  "relations": [
    {"subject": "O-DU", "predicate": "communicates_with", "object": "O-RU"},
    {"subject": "O-DU", "predicate": "uses_interface", "object": "E2"},
    {"subject": "O-DU", "predicate": "reports_to", "object": "Near-RT RIC"},
    {"subject": "PRB Utilization", "predicate": "causes", "object": "latency"}
  ]
}

==================== INPUT TEXT ====================
TEXT:
\"\"\"{chunk_text}\"\"\"
""".format(chunk_text=chunk_text)


    payload = {"model": model_name, "prompt": prompt, "stream": False}

    try:
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()

        # Parse Ollama output safely
        try:
            result = response.json()
            # If the Ollama API returned a single JSON object
            if "response" in result:
                text = result["response"]
            else:
                # Sometimes it streams multiple JSON lines – combine all responses
                text = ""
                for line in response.text.splitlines():
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            text += data["response"]
                    except json.JSONDecodeError:
                        continue
        except Exception:
            text = response.text.strip()

        # # 🧠 DEBUG: Print only the clean model output
        # print("\n" + "=" * 80)
        # print(f"🧩 Cleaned LLM reply from {endpoint}:")
        # print(text[:1000])  # limit to 1000 chars for readability
        # print("=" * 80 + "\n")

        # Try to extract JSON content from cleaned text
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            parsed_json = json.loads(match.group(0))
            print("✅ Parsed JSON:", json.dumps(parsed_json, indent=2))
            return parsed_json
        else:
            print("⚠️ No valid JSON detected in response.")
            return {"entities": [], "relations": []}

    except Exception as e:
        print(f"⚠️ Error from {endpoint}: {e}")
        return {"entities": [], "relations": []}


# --------------------------
# PARALLEL EXECUTION
# --------------------------
def process_chunk(idx, chunk, endpoint):
    print(f"🧩 [Node {endpoint.split('//')[1]}] Processing chunk {idx}")
    kg = extract_kg_from_chunk(chunk, endpoint=endpoint)
    return {
        "chunk_id": idx,
        "entities": kg.get("entities", []),
        "relations": kg.get("relations", []),
        "preview": chunk[:400] + ("..." if len(chunk) > 400 else "")
    }

# --------------------------
# MAIN PIPELINE
# --------------------------
if __name__ == "__main__":
    import itertools
    import os

    # --------------------------
    # INPUT / OUTPUT SETTINGS
    # --------------------------
    INPUT_DIR = "/home/michael/Desktop/rca_explainer/output/final_parsed_with_table"
    OUTPUT_DIR = os.path.join(INPUT_DIR, "kg_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all Markdown files in folder
    md_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".md")]

    print(f"📁 Found {len(md_files)} markdown files in {INPUT_DIR}\n")

    for md_filename in md_files:
        input_path = os.path.join(INPUT_DIR, md_filename)
        base_name = os.path.splitext(md_filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_kg.json")

        # --------------------------
        # SKIP IF ALREADY PROCESSED
        # --------------------------
        if os.path.exists(output_path):
            print(f"⏩ Skipping {md_filename} (already processed)\n")
            continue

        # --------------------------
        # READ FILE
        # --------------------------
        print(f"⚙️ Processing {md_filename}...")
        with open(input_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        # --------------------------
        # DYNAMIC CHUNKING
        # --------------------------
        print("🔧 Performing dynamic chunking...")
        chunks = dynamic_chunk_documents(md_text)
        print(f"✅ {len(chunks)} chunks generated for {md_filename}.\n")

        # --------------------------
        # PARALLEL LLM CALLS
        # --------------------------
        node_cycle = itertools.cycle(OLLAMA_NODES)
        all_results = []

        print(f"🚀 Distributing {len(chunks)} chunks across {len(OLLAMA_NODES)} Ollama nodes...")

        with ThreadPoolExecutor(max_workers=len(OLLAMA_NODES)) as executor:
            futures = []
            for i, chunk in enumerate(chunks, 1):
                endpoint = next(node_cycle)
                futures.append(executor.submit(process_chunk, i, chunk, endpoint))

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)

        # --------------------------
        # SAVE PER-FILE OUTPUT
        # --------------------------
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(sorted(all_results, key=lambda x: x["chunk_id"]),
                      out, indent=2, ensure_ascii=False)

        print(f"✅ Knowledge Graph extraction complete for {md_filename}")
        print(f"📄 Saved results to: {output_path}\n")

