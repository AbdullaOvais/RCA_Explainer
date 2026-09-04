import os
import json
import requests
import networkx as nx
import re
from transformers import pipeline
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# CONFIG
# ----------------------------
OLLAMA_URL = "http://10.9.64.22:11435/api/generate"
MODEL_NAME = "deepseek-r1:32b"
DOCS_FOLDER = "./docs"

# ----------------------------
# Step 1. Load & Split Documents
# ----------------------------
def load_documents(input_folder=DOCS_FOLDER):
    loaders = []
    for filename in os.listdir(input_folder):
        path = os.path.join(input_folder, filename)
        if filename.endswith(".pdf"):
            loaders.append(UnstructuredPDFLoader(path))
        elif filename.endswith(".docx"):
            loaders.append(UnstructuredWordDocumentLoader(path))
        elif filename.endswith(".md"):
            loaders.append(UnstructuredMarkdownLoader(path))

    all_docs = []
    for loader in loaders:
        all_docs.extend(loader.load())
    return all_docs

def split_documents(docs, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

# ----------------------------
# Step 2. Use DeepSeek to Extract Triples
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
# Step 3. Build Knowledge Graph
# ----------------------------
def build_graph(chunks):
    G = nx.DiGraph()
    all_triples = []
    for i, chunk in enumerate(chunks, 1):
        print(f"🔎 Processing chunk {i}/{len(chunks)}...")

        # ✅ Print the input chunk
        print("   📝 Chunk text:")
        print(chunk.page_content[:500] + ("..." if len(chunk.page_content) > 500 else ""))  # truncate if very long

        # Extract triples
        output = ask_deepseek(f"""
Extract knowledge graph triples from the following text.
Rules:
"I'm trying to build this knowledge graph from 5G O-RAN specifications. i'M FURTHER USING IT FOR ROOT CAUSE ANALYSIS. Thus choose relations that are relevant for troubleshooting and RCA."                             
Format output strictly as JSON list of objects:
[{{"head": "...", "relation": "...", "tail": "..."}}]

Text:
{chunk.page_content}
""")

        # ✅ Print the raw LLM output
        print("   🤖 DeepSeek raw output:")
        print(output)

        # Parse JSON safely
        triples = []
        try:
            clean_output = output.strip()
            # Extract first JSON array in the text
            match = re.search(r'\[\s*{.*?}\s*\]', clean_output, re.DOTALL)
            if match:
                json_text = match.group(0)
                triples = json.loads(json_text)
                if not isinstance(triples, list):
                    triples = []
            else:
                print("⚠️ No JSON array found in LLM output.")
        except Exception as e:
            print(f"⚠️ Failed to parse LLM output as JSON: {e}")
            triples = []


        if triples:
            print("   📌 Triples extracted:")
            for t in triples:
                head = t.get("head", "N/A")
                rel = t.get("relation", "N/A")
                tail = t.get("tail", "N/A") 

                # ✅ Print each triple
                print(f"      ({head}) -[{rel}]-> ({tail})")

                # Add to graph
                G.add_node(head, type="entity")
                G.add_node(tail, type="entity")
                G.add_edge(head, tail, relation=rel)

            all_triples.extend(triples)
            print(f"   ✅ Found {len(triples)} triples.")
        else:
            print("   ⚠️ No triples found.")
    return G, all_triples


# ----------------------------
# Step 4. Graph Retrieval
# ----------------------------
def retrieve_context(G, query_entity, depth=2):
    """Traverse neighbors around a query entity."""
    if query_entity not in G:
        return "No info in graph."
    
    paths = []
    for target in nx.single_source_shortest_path_length(G, query_entity, cutoff=depth):
        if target != query_entity:
            rels = []
            try:
                path_nodes = nx.shortest_path(G, query_entity, target)
                for i in range(len(path_nodes)-1):
                    u, v = path_nodes[i], path_nodes[i+1]
                    rels.append(f"{u} -[{G[u][v]['relation']}]-> {v}")
                paths.append(" → ".join(rels))
            except nx.NetworkXNoPath:
                continue
    return "\n".join(paths) if paths else "No relations found."

# ----------------------------
# Step 5. RAG Answering
# ----------------------------
def rag_answer(G, query):
    # Step A: Extract main entity from query
    entity_prompt = f"""
Extract the main entity from the following question. 
Return only the entity name as plain text.

Question: {query}
"""
    entity = ask_deepseek(entity_prompt).split("\n")[0].strip()

    # Step B: Retrieve graph context
    context = retrieve_context(G, entity, depth=2)

    # Step C: Ask DeepSeek final answer
    rag_prompt = f"""
Answer the following question using the knowledge graph context.

Question:
{query}

Knowledge Graph Context:
{context}
"""
    return ask_deepseek(rag_prompt)

# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":
    print("📂 Loading documents...")
    docs = load_documents()
    chunks = split_documents(docs)

    print(f"✂️ Split into {len(chunks)} chunks.\n")

    print("🔧 Building Knowledge Graph...")
    G, triples = build_graph(chunks)
    print(f"\n📦 Built graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")

    # Example Query
    user_query = "What happens if CU fails?"
    answer = rag_answer(G, user_query)

    print("\n❓ Query:", user_query)
    print("💡 Answer:", answer)
