import os
import json

FOLDER = "./imagegeneration/final_parsed_oran_latest_triplets"  # change to your path

unique_entities = set()

for filename in os.listdir(FOLDER):
    if filename.endswith(".json"):
        filepath = os.path.join(FOLDER, filename)

        with open(filepath, "r") as f:
            data = json.load(f)

        for chunk in data:
            entities = chunk.get("entities", [])
            for e in entities:
                # Normalize to string
                if isinstance(e, dict):
                    e = json.dumps(e, sort_keys=True)

                unique_entities.add(e)

print("Total unique entities:", len(unique_entities))
