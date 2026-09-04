import os
import json
from transformers import pipeline
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load REBEL model (relation extraction)
print("📥 Loading REBEL model (Babelscape/rebel-large)...")
rebel = pipeline("text2text-generation", model="Babelscape/rebel-large")
print("✅ REBEL model loaded.\n")

def extract_triples_from_text(text, max_new_tokens=256):
    """
    Uses REBEL model to extract triples from text.
    Returns a list of (head, relation, tail).
    """
    output = rebel(text, max_new_tokens=max_new_tokens)[0]['generated_text']
    triples = []
    current = {"head": None, "relation": None, "tail": None}
    
    for token in output.split():
        if token == "<triplet>":
            if current["head"] and current["relation"] and current["tail"]:
                triples.append((current["head"], current["relation"], current["tail"]))
            current = {"head": None, "relation": None, "tail": None}
        elif token == "<subj>":
            current["relation"] = None
        elif token == "<obj>":
            current["tail"] = None
        else:
            if current["head"] is None:
                current["head"] = token
            elif current["relation"] is None:
                current["relation"] = token if current["relation"] is None else current["relation"] + " " + token
            else:
                current["tail"] = token if current["tail"] is None else current["tail"] + " " + token

    if current["head"] and current["relation"] and current["tail"]:
        triples.append((current["head"], current["relation"], current["tail"]))

    return triples

def process_oran_specs(input_folder="./docs", output_file="oran_triples.json"):
    """
    Load all O-RAN spec files, extract triples, and save to JSON.
    """
    print(f"📂 Scanning folder: {input_folder}")
    loaders = []
    for filename in os.listdir(input_folder):
        path = os.path.join(input_folder, filename)
        if filename.endswith(".pdf"):
            loaders.append(UnstructuredPDFLoader(path))
            print(f"  ➡️ Found PDF: {filename}")
        elif filename.endswith(".docx"):
            loaders.append(UnstructuredWordDocumentLoader(path))
            print(f"  ➡️ Found Word DOCX: {filename}")
        elif filename.endswith(".md"):
            loaders.append(UnstructuredMarkdownLoader(path))
            print(f"  ➡️ Found Markdown: {filename}")

    # Load documents
    print("\n📖 Loading documents...")
    all_docs = []
    for loader in loaders:
        docs = loader.load()
        all_docs.extend(docs)
    print(f"✅ Loaded {len(all_docs)} documents.\n")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"✂️ Split into {len(chunks)} chunks (512 tokens each, 50 overlap).\n")

    # Extract triples
    all_triples = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"🔎 Processing chunk {i}/{len(chunks)}...")
        triples = extract_triples_from_text(chunk.page_content)
        if triples:
            print(f"   ✅ Extracted {len(triples)} triples.")
            # Print first 2 triples to verify content
            for t in triples[:2]:
                print(f"      - {t}")
            all_triples.extend(triples)
        else:
            print("   ⚠️ No triples found.")


    # Save extracted triples
    with open(output_file, "w") as f:
        json.dump(all_triples, f, indent=2)

    print(f"\n📦 Finished! Extracted total {len(all_triples)} triples.")
    print(f"💾 Saved to: {output_file}")

# Run pipeline
process_oran_specs()
