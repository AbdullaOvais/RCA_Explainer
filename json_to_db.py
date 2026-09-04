import os
import json
import re
from neo4j import GraphDatabase

# ----------------------------
# 1. Connect to Neo4j
# ----------------------------
uri = "neo4j://localhost:7687"
username = "neo4j"
password = "RagNeo@12"

driver = GraphDatabase.driver(uri, auth=(username, password))

# ----------------------------
# 2. Helper: sanitize predicate
# ----------------------------
def clean_predicate(pred):
    pred = pred.strip()
    pred = pred.replace(" ", "_")
    pred = re.sub(r"[^A-Za-z0-9_]", "", pred)
    return pred.upper()

# ----------------------------
# 3. Insert into graph
# ----------------------------
def insert_graph(json_data):

    with driver.session() as session:
        for chunk in json_data:

            # Insert entities
            for ent in chunk["entities"]:
                session.run(
                    "MERGE (:Entity {name: $name})",
                    name=ent
                )

            # ------------- Insert Relations -------------
            for rel in chunk.get("relations", []):
                subject = rel.get("subject")
                pred_raw = rel.get("predicate")
                obj = rel.get("object")

                # Skip bad relations
                if not subject or not pred_raw or not obj:
                    print("⚠️  Skipping invalid relation:", rel)
                    continue

                predicate = clean_predicate(pred_raw)

                cypher = f"""
                MATCH (s:Entity {{name: $subject}})
                MATCH (o:Entity {{name: $object}})
                MERGE (s)-[r:{predicate}]->(o)
                """

                session.run(cypher, subject=subject, object=obj)

    print("Graph ingestion completed for current file.")

# ----------------------------
# 4. Load ALL JSON files from folder
# ----------------------------
def load_folder(folder_path):
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            full_path = os.path.join(folder_path, filename)
            print(f"Loading {full_path}")

            with open(full_path, "r") as f:
                file_data = json.load(f)

                # if each file contains a list
                if isinstance(file_data, list):
                    all_data.extend(file_data)
                else:
                    # if it's a dict (single chunk)
                    all_data.append(file_data)

    return all_data

# ----------------------------
# 5. Run Everything
# ----------------------------
folder = "./kg_results"   # change to your folder
data = load_folder(folder)
insert_graph(data)
