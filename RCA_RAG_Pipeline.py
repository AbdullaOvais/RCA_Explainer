import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
import pickle
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



USE_LOCAL_LLM = True  # Set to True to use localhost LLM like DeepSeek
LOCAL_LLM_URL = "http://10.9.66.233:11434/api/generate"
LOCAL_LLM_MODEL = "deepseek-r1:32b"
if USE_LOCAL_LLM:
    PROCESS_ALL_RCA_FILES = True  # Set to True to process all RCA json files
else:
    PROCESS_ALL_RCA_FILES = False

# PROCESS_ALL_RCA_FILES = False


load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

with open('topology.pkl', 'rb') as file:
    topology = pickle.load(file)


def query_llm(prompt):
    if USE_LOCAL_LLM:
        return query_local_llm(prompt)
    else:
        return query_gemini_api(prompt)


def query_gemini_api(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"[ERROR] Gemini API {response.status_code}: {response.text}"


def query_local_llm(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LOCAL_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 16384
        }
    }

    try:
        response = requests.post(LOCAL_LLM_URL, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "[ERROR] No response from local LLM.")
        else:
            return f"[ERROR] Local LLM {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"[ERROR] Local LLM request failed: {e}"



def get_feature_metadata(feature_name):
    _feature_metadata_cache = None

    if _feature_metadata_cache is None:
        try:
            with open("metrics_description.json", "r") as f:
                _feature_metadata_cache = json.load(f)
        except FileNotFoundError:
            print("[ERROR] metrics_description.json not found.")
            _feature_metadata_cache = {}
        except json.JSONDecodeError:
            print("[ERROR] Invalid JSON format in metrics_description.json.")
            _feature_metadata_cache = {}

    return _feature_metadata_cache.get(feature_name, {})



def dynamic_chunk_documents(documents, base_chunk_size=2500, max_chunk_size=8000, similarity_threshold=0.5):
    """
    Dynamically chunks documents based on semantic similarity between sentences.

    Args:
        documents (list[Document]): LangChain documents.
        base_chunk_size (int): Minimum chunk size in characters.
        max_chunk_size (int): Maximum chunk size allowed per chunk.
        similarity_threshold (float): Cosine similarity threshold for merging.

    Returns:
        list[Document]: List of dynamically chunked documents.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dynamic_chunks = []

    for doc in documents:
        text = doc.page_content
        sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 0]

        if not sentences:
            continue

        # Compute embeddings
        embeddings = model.encode(sentences, convert_to_tensor=True)
        current_chunk = sentences[0]
        current_len = len(current_chunk)

        for i in range(1, len(sentences)):
            sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()

            # merge if semantically similar and within max length
            if sim > similarity_threshold and current_len + len(sentences[i]) < max_chunk_size:
                current_chunk += ". " + sentences[i]
                current_len = len(current_chunk)
            else:
                # finalize chunk if too different or long
                if len(current_chunk) >= base_chunk_size:
                    new_doc = doc.copy()
                    new_doc.page_content = current_chunk.strip()
                    dynamic_chunks.append(new_doc)
                current_chunk = sentences[i]
                current_len = len(current_chunk)

        # Add final chunk
        if len(current_chunk.strip()) > 0:
            new_doc = doc.copy()
            new_doc.page_content = current_chunk.strip()
            dynamic_chunks.append(new_doc)

    return dynamic_chunks



def index_documents():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = "./chroma_rag_index"
    manifest_path = os.path.join(persist_dir, "index_manifest.json")

    # Load already indexed files
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            indexed_files = json.load(f)
    else:
        indexed_files = []

    new_loaders = []
    newly_indexed_files = []

    for filename in os.listdir("./docs"):
        path = os.path.join("./docs", filename)
        if filename in indexed_files:
            continue  # Skip already indexed

        try:
            if filename.endswith(".pdf"):
                new_loaders.append(UnstructuredPDFLoader(path))
            elif filename.endswith(".docx"):
                new_loaders.append(UnstructuredWordDocumentLoader(path))
            elif filename.endswith(".md"):
                new_loaders.append(UnstructuredMarkdownLoader(path))
            elif filename.endswith(".json") and "grafana" in filename:
                with open(path, 'r') as f:
                    json_data = json.load(f)
                    content = json.dumps(json_data, indent=2)
                    temp_path = f"./docs/{filename}.txt"
                    with open(temp_path, "w") as tf:
                        tf.write(content)
                    new_loaders.append(UnstructuredMarkdownLoader(temp_path))

            newly_indexed_files.append(filename)

        except Exception as e:
            print(f"[ERROR] Failed to prepare loader for {filename}: {e}")

    if not new_loaders:
        print("✅ No new documents to index.")
        return

    # Load and split documents
    all_docs = []
    for loader in new_loaders:
        try:
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load documents from {loader}: {e}")

    if not all_docs:
        print("⚠️ No valid content found in new files.")
        return

    # splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    # docs = splitter.split_documents(all_docs)
    docs = dynamic_chunk_documents(all_docs)
    print(f"✅ Created {len(docs)} dynamic chunks from {len(all_docs)} original docs.")


    # Load existing DB and add new docs
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    # vectordb.add_documents(docs)
    max_batch = 5000
    for i in range(0, len(docs), max_batch):
        batch = docs[i:i + max_batch]
        vectordb.add_documents(batch)
    # vectordb.persist()

    # Safe persist workaround using _client + _persist_path
    if hasattr(vectordb, '_persist_path') and hasattr(vectordb, '_client'):
        vectordb._client.persist()

    # Update manifest
    indexed_files += newly_indexed_files
    with open(manifest_path, "w") as f:
        json.dump(indexed_files, f, indent=2)

    print(f"✅ Indexed {len(docs)} chunks from {len(newly_indexed_files)} new files.")


def parse_methods_with_samples(file_path):
    """
    Loads a JSON file and returns it as-is without restructuring.
    
    Args:
        file_path (str): Path to JSON file.
    
    Returns:
        dict: Original JSON content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return {}


def load_rca_json_files_parsed():
    """
    Loads RCA files and parses them using parse_methods_with_samples.
    Respects PROCESS_ALL_RCA_FILES flag.
    Returns list of tuples: (parsed_data, filename)
    """
    rca_dir = "rf_shap_explanations"
    rca_files = [f for f in os.listdir(rca_dir) if os.path.isfile(os.path.join(rca_dir, f))]
    

    if not rca_files:
        raise FileNotFoundError("No _explanation.json files found in RCA_inferences directory.")

    def safe_parse(file_path):
        try:
            return parse_methods_with_samples(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to parse file {file_path}: {e}")
            return None

    if PROCESS_ALL_RCA_FILES:
        result = []
        for f in rca_files:
            full_path = os.path.join(rca_dir, f)
            parsed = safe_parse(full_path)
            if parsed:
                result.append((parsed, f))
        return result
    else:
        latest_file = max(rca_files, key=lambda f: os.path.getctime(os.path.join(rca_dir, f)))
        full_path = os.path.join(rca_dir, latest_file)
        print(f"Using RCA file: {latest_file}")
        parsed = safe_parse(full_path)
        if not parsed:
            raise ValueError(f"Latest RCA file is invalid or could not be parsed: {latest_file}")
        return [(parsed, latest_file)]


def query_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"[ERROR] Gemini API {response.status_code}: {response.text}"


def evaluate_response_with_llm(prompt, explanation):
    eval_prompt = f"""
You are an expert evaluator. Below is a task prompt followed by a generated explanation.

--- Prompt ---
{prompt}

--- Generated Explanation ---
{explanation}

Evaluate the explanation using the following criteria:
1. Correctness (out of 5)
2. Completeness (out of 5)
3. Clarity (out of 5)
4. Relevance to context (out of 5)
5. Mitigation quality (out of 5)

Justify each score briefly. End with a final score out of 10 and verdict on production-readiness.

Return in markdown format.
"""
    return query_llm(eval_prompt)


def match_explanation_to_rag_sources(explanation, retrieved_contexts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_matches = {}
    explanation_sentences = [s.strip() for s in explanation.split('\n') if len(s.strip()) > 0]

    for feature, docs in retrieved_contexts.items():
        context_chunks = [doc.page_content for doc in docs]
        context_embeddings = model.encode(context_chunks, convert_to_tensor=True)

        for sentence in explanation_sentences:
            sent_emb = model.encode(sentence, convert_to_tensor=True)
            scores = util.cos_sim(sent_emb, context_embeddings)[0]
            best_match_idx = scores.argmax().item()
            best_score = scores[best_match_idx].item()
            if best_score > 0.6:
                sentence_matches[sentence] = {
                    "feature": feature,
                    "source": context_chunks[best_match_idx],
                    "score": best_score
                }

    return sentence_matches



def run_pipeline_from_rca_file():
    rca_file_list = load_rca_json_files_parsed()

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./chroma_rag_index", embedding_function=embedding)
    retriever = vectordb.as_retriever(search_type="mmr", k=3)

    matches =0
    total = 0
    for parsed_data, filename in rca_file_list:
        prompt_lines = [
            f"You are an expert RCA assistant. The RCA system has flagged the following features as root causes.",
            f"File: {filename}\n"
        ]

        retrieved_contexts = {}
        all_edges = []
        print(type(parsed_data))                   # <class 'dict'>
        print(parsed_data.keys())  
         
        samples_data = parsed_data.get("samples", [])

        # parsed_data structure:
        # {
        #    "methods": {
        #       method_name: {"root_causes": [...], "edges": [...]},
        #       ...
        #    },
        #    "samples": [...]
        # }
        
        for method, result in parsed_data["methods"].items():
            prompt_lines.append(f"Method: {method}")

            # Loop over every key/value in the method's dictionary
            for key, value in result.items():
                # If it's a list, print each item on its own line
                if isinstance(value, list):
                    line = f"  {key}:"
                    prompt_lines.append(line)
                    
                    for item in value:
                        line = f"    - {item}"
                        prompt_lines.append(line)
                        
                # If it's a dictionary, pretty-print it
                elif isinstance(value, dict):
                    line = f"  {key}:"
                    prompt_lines.append(line)
                    
                    for sub_k, sub_v in value.items():
                        line = f"    {sub_k}: {sub_v}"
                        prompt_lines.append(line)
                        
                # Otherwise, just append the value directly
                else:
                    line = f"  {key}: {value}"
                    prompt_lines.append(line)
                    if(key == "predicted_label"):
                        print(line)
                    
                    
            root_cause_features = result.get("decision_path_features", [])

            for feature in root_cause_features:
                if isinstance(feature, dict):
                    feature_name = feature.get("name", "")
                else:
                    feature_name = feature

                if not feature_name:
                    continue

                meta = get_feature_metadata(feature_name)
                query_text = f"{feature_name}. {meta.get('description', '')}"
                docs = retriever.get_relevant_documents(query_text)
                retrieved_contexts[feature_name] = docs

            # edges = result.get("edges", [])
            # if edges:
            #     all_edges.extend(edges)

        # Add edges to prompt
        # if all_edges:
        #     prompt_lines.append("\nCausal Graph Edges (Directed):")
        #     for edge in all_edges:
        #         prompt_lines.append(f"- {edge[0]} ➝ {edge[1]}")

        # Add samples to prompt
        if samples_data:
            prompt_lines.append("\nSample Data Points:")
            for sample in samples_data:
                sample_str = ", ".join(f"{k}: {v}" for k, v in sample.items() if k != "index")
                prompt_lines.append(f"- Sample #{sample.get('index', '?')}: {sample_str}")
        
        if len(samples_data) >= 65:
            sample_65 = samples_data[64]  # 0-based
            sample_timestamp = sample_65.get("timestamp")  # Adjust key if different
        else:
            print(" Less than 65 samples available.")
            sample_timestamp = None

        # Add other prompt instructions and topology info
        prompt_lines.append("\nSetup is deployed in a docker network using docker containers with CPU isolation for each container.")
        prompt_lines.append("\nWe have induced CPU stress, Memory Stress using stress-ng tool and packet loss using TC tool. So, give response accordingly.")
        prompt_lines.append(f"\nTopology is as follows: {topology}")
        # prompt_lines.append("\nNOTE: PROVIDE A UNIFIED RESPONSE TO THE FOLLOWING QUESTIONS FOR ALL FEATURES, CONSIDERING THE INTERDEPENDENCY AMONG THEM.")
        # prompt_lines.append("\nFor each feature above, explain:")
        # prompt_lines.append("1. What the metric means")
        # prompt_lines.append("2. Why an anomaly is problematic")
        # prompt_lines.append("3. How it affects the system")
        # prompt_lines.append("4. Suggested mitigation strategies")

        
        # Add stress options for each container
        prompt_lines.append("\nBelow are the possible stress scenarios for each container:")
        containers = ["srscu0", "srscu1", "srsdu0", "srsdu1", "srsdu2"]
        stress_types = ["is under CPU stress", "is under memory stress", "is under network stress"]
        for container in containers:
            for stress in stress_types:
                prompt_lines.append(f"- {container} {stress}")

        prompt_lines.append(f"\nGiven the timestamp: {sample_timestamp}, choose only one as the correct stress scenario as root cause as the the same format from the options above.")
        prompt_lines.append("\n Give response in the below json template format strictly")
        prompt_lines.append('''
    'fault_explanation: <Explain what the predicted label means, what fault has occurred, and its context>',
    'root_cause_explanation: <Explain the underlying reason for the fault, using data from features, decision paths, and model outputs>',
    'selected_root_cause_option: <One of the given options, copied exactly as written>'
''')
        #*****************************************#

        full_prompt = "\n".join(prompt_lines)
        explanation = query_llm(full_prompt)
        # ---  Compare LLM answer with CSV ---#
        correct_answer = None
        if sample_timestamp:
            # Also format JSON timestamp to seconds only
            TS = datetime.strptime(sample_timestamp, "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")             
            df = pd.read_csv("prometheus_combined.csv")  # replace with actual CSV path
            # Convert CSV timestamps to datetime, truncate to seconds, and format as string till seconds only
            df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.floor("S").dt.strftime("%Y-%m-%d %H:%M:%S")
            row = df[df["Timestamp"] == TS]
            if not row.empty:
                correct_answer = row.iloc[0]["stress_sentence"]

        if correct_answer:
            llm_choice = explanation.strip()
            match = 1 if correct_answer.lower() in llm_choice.lower() else 0
            print(f"\nTimestamp: {sample_timestamp}")
            print(f"LLM answer: {llm_choice}")
            print(f"Correct answer: {correct_answer}")
            if match==1:
                matches += 1
            total += 1
            
        else:
            print("\n Correct answer not found for this timestamp.")
        evaluation = evaluate_response_with_llm(full_prompt, explanation)
        references = match_explanation_to_rag_sources(explanation, retrieved_contexts)

        references_md = "\n\n## 🔍 Explanation References\n"
        for sentence, info in references.items():
            references_md += f"- **Explanation**: {sentence}\n"
            references_md += f"  - **Feature**: `{info['feature']}`\n"
            references_md += f"  - **Matched Context** (Score: {info['score']:.2f}):\n    ```\n{info['source']}\n    ```\n\n"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"RCA_Explanations/rca_explanation_report_{filename.replace('.json','')}_{timestamp}.html"

        html_lines = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<title>RCA Explainability Report</title>",
            "<script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>",
            "<style>",
            "body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 2rem; background-color: #f4f4f9; color: #333; }",
            "h1, h3 { color: #2c3e50; }",
            ".container { max-width: 960px; margin: auto; }",
            "#markdown-content, #markdown-eval, #markdown-references { padding: 1rem; background: #f0f0f0; border-left: 4px solid #3498db; white-space: pre-wrap; margin-bottom: 2rem; }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
            f"<h1>RCA Explainability Report</h1>",
            f"<h3>Generated using: <code>{'Local LLM' if USE_LOCAL_LLM else GEMINI_MODEL}</code></h3>",
            f"<h4>Source File: {filename}</h4>",
            "<div id='markdown-content'></div>",
            "<div id='markdown-eval'></div>",
            "<div id='markdown-references'></div>",
            "<script type='text/markdown' id='md-source'>",
            explanation,
            "</script>",
            "<script type='text/markdown' id='md-eval'>",
            evaluation,
            "</script>",
            "<script type='text/markdown' id='md-references'>",
            references_md,
            "</script>",
            "<script>",
            "document.getElementById('markdown-content').innerHTML = marked.parse(document.getElementById('md-source').textContent);",
            "document.getElementById('markdown-eval').innerHTML = marked.parse(document.getElementById('md-eval').textContent);",
            "document.getElementById('markdown-references').innerHTML = marked.parse(document.getElementById('md-references').textContent);",
            "</script>",
            "</div>",
            "</body>",
            "</html>"
        ]

        with open(report_filename, "w") as f:
            f.write("\n".join(html_lines))

        print(f"\n✅ HTML report saved to: {report_filename}")

    if total > 0:
        accuracy = matches / total
        print(f"\nOverall Accuracy: {accuracy:.2%}")
    else:
        print("\nNo samples to evaluate.")



if __name__ == "__main__":
    index_documents()
    run_pipeline_from_rca_file()