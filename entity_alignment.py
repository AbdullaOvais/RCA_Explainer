import os
import json
import shutil
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import hdbscan

DATA_DIR = "imagegeneration/final_parsed_oran_latest_triplets"  # change to your path
BAD_DIR = "bad_files/"
OUT_DIR = "kg_oran_latest_output/"

os.makedirs(BAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

#####################################################################
# Helper: reject file with printed reason
#####################################################################
def reject_file(fpath, file, reason):
    print(f"❌ Rejected {file}: {reason}")
    shutil.move(fpath, os.path.join(BAD_DIR, file))


#####################################################################
# Convert string relations → structured triples
#####################################################################
def parse_relation_string(rel_str):
    if not isinstance(rel_str, str):
        return None
    parts = rel_str.split()
    if len(parts) < 3:
        return None

    verb_idx = None
    for i, token in enumerate(parts):
        if re.match(r".*(ed|es)$", token) or token.lower() in {
            "is", "are", "was", "were",
            "has", "have",
            "defines", "provides", "contains", "supports",
            "manages", "creates", "uses", "offers"
        }:
            verb_idx = i
            break

    if verb_idx is None or verb_idx == 0 or verb_idx >= len(parts) - 1:
        return None

    subject = " ".join(parts[:verb_idx])
    predicate = parts[verb_idx]
    obj = " ".join(parts[verb_idx + 1:])
    return {
        "subject": subject.strip(),
        "predicate": predicate.strip(),
        "object": obj.strip()
    }


#####################################################################
# Clean relation list
#####################################################################
def safe_load_relations(rel_list):
    cleaned = []
    for r in rel_list:

        # Already correct dict
        if isinstance(r, dict) and all(k in r for k in ["subject", "predicate", "object"]):
            # Must not contain Nones
            if r["subject"] and r["object"]:
                cleaned.append(r)
            continue

        # String relation → parse
        if isinstance(r, str):
            fixed = parse_relation_string(r)
            if fixed and fixed["subject"] and fixed["object"]:
                cleaned.append(fixed)
            continue

    return cleaned


#####################################################################
# 1. SCAN FILES, VALIDATE, EXTRACT ENTITIES + CLEAN RELATIONS
#####################################################################
all_entities = set()
all_relations = []
valid_files = 0
invalid_relations_count = 0

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

for file in files:
    fpath = os.path.join(DATA_DIR, file)

    # Load JSON
    try:
        obj = json.load(open(fpath))
    except:
        reject_file(fpath, file, "Invalid JSON format")
        continue

    # Must be a list of chunks
    if not isinstance(obj, list):
        reject_file(fpath, file, "Top-level JSON must be a LIST")
        continue

    file_has_valid_chunk = False

    for chunk in obj:
        if not isinstance(chunk, dict):
            continue

        if not all(k in chunk for k in ["chunk_id", "entities", "relations"]):
            continue

        repaired_rel = safe_load_relations(chunk["relations"])

        if len(chunk["relations"]) > 0 and len(repaired_rel) == 0:
            invalid_relations_count += len(chunk["relations"])
            continue

        # This chunk is valid
        file_has_valid_chunk = True

        # Add entities
        for e in chunk["entities"]:
            if isinstance(e, str):
                all_entities.add(e)

        # Add repaired relations
        all_relations.extend(repaired_rel)

    if not file_has_valid_chunk:
        reject_file(fpath, file, "No valid chunks after cleaning")
        continue

    valid_files += 1


print("\n====================== SUMMARY ======================")
print("Valid files:", valid_files)
print("Invalid files moved:", len(os.listdir(BAD_DIR)))
print("Entities collected:", len(all_entities))
print("Valid relations collected:", len(all_relations))
print("Malformed relations dropped:", invalid_relations_count)
print("=====================================================\n")


#####################################################################
# 2. EMBEDDINGS WITH RAM-SAFE BATCHING
#####################################################################
all_entities = list(all_entities)
initial_entity_count = len(all_entities)
batch_size = 2048

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print(f"Generating embeddings in batches of {batch_size} ...")

embeddings_list = []
for i in tqdm(range(0, initial_entity_count, batch_size)):
    batch = all_entities[i:i+batch_size]
    emb = model.encode(batch, convert_to_numpy=True, batch_size=len(batch), show_progress_bar=False)
    embeddings_list.append(emb)

import numpy as np
embeddings = np.vstack(embeddings_list)

# Normalize to make cosine=euclidean
embeddings = normalize(embeddings)


#####################################################################
# 3. HDBSCAN CLUSTERING (Memory Efficient)
#####################################################################
print("Running HDBSCAN clustering...")

clusterer = hdbscan.HDBSCAN(
    metric='euclidean',
    min_cluster_size=2,
    cluster_selection_epsilon=0.15,
    approx_min_span_tree=True
)

labels = clusterer.fit_predict(embeddings)

#####################################################################
# 4. BUILD CLUSTERS AND CANONICAL ENTITY MAP
#####################################################################
clusters = {}
for i, label in enumerate(labels):
    clusters.setdefault(label, []).append(all_entities[i])

canonical_map = {}
canonical_entities = []

for cid, ents in clusters.items():
    # label -1 are noise points (singletons)
    if cid == -1:
        for e in ents:
            canonical_map[e] = e
            canonical_entities.append(e)
        continue

    canonical = max(ents, key=len)
    canonical_entities.append(canonical)
    for e in ents:
        canonical_map[e] = canonical

final_entity_count = len(canonical_entities)

print("Entities after merging:", final_entity_count)
print("Reduction:", initial_entity_count - final_entity_count)


#####################################################################
# 5. REWRITE RELATIONS SAFELY
#####################################################################
def to_str(x):
    if isinstance(x, list):
        return " ".join(str(i) for i in x)
    if isinstance(x, dict):
        return None  # cannot parse dict
    if not isinstance(x, str):
        return None
    return x.strip()


canonical_relations = []

for r in all_relations:

    subj = to_str(r.get("subject"))
    pred = to_str(r.get("predicate"))
    obj = to_str(r.get("object"))

    # drop invalid ones
    if not subj or not obj or not pred:
        continue

    if subj not in canonical_map or obj not in canonical_map:
        continue

    canonical_relations.append({
        "subject": canonical_map[subj],
        "predicate": pred,
        "object": canonical_map[obj]
    })


#####################################################################
# 6. SAVE OUTPUT
#####################################################################
json.dump(canonical_entities, open(os.path.join(OUT_DIR,"canonical_entities.json"),"w"), indent=2)
json.dump(canonical_relations, open(os.path.join(OUT_DIR,"canonical_relations.json"),"w"), indent=2)
# Convert NumPy int64 keys → Python int keys
clusters_clean = { int(k): v for k, v in clusters.items() }

json.dump(clusters_clean, open(os.path.join(OUT_DIR,"entity_clusters.json"),"w"), indent=2)

stats = {
    "valid_files": valid_files,
    "invalid_files": len(os.listdir(BAD_DIR)),
    "initial_entities": initial_entity_count,
    "final_entities": final_entity_count,
    "entity_reduction": initial_entity_count - final_entity_count,
    "relations_final": len(canonical_relations),
    "malformed_relations_dropped": invalid_relations_count
}
json.dump(stats, open(os.path.join(OUT_DIR,"statistics.json"),"w"), indent=2)

print("\nProcessing complete 🎉")
