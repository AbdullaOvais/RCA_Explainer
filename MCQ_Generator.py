# main.py
import os
import json
import re
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from dotenv import load_dotenv
import requests


import time

# --- Configuration ---
# IMPORTANT: Set your Google API key as an environment variable
# for this script to work.
# You can get a key from https://aistudio.google.com/app/apikey
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

load_dotenv()


# --- Mode Control ---
# Options: "generate_only", "validate_only", "generate_and_validate"
MODE = "generate_only"



# Toggle between Gemini API and local LLM (like DeepSeek)
USE_LOCAL_LLM = True  # Set to True to use local model via Ollama or similar
LOCAL_LLM_URL = "http://localhost:11434/api/generate"
LOCAL_LLM_MODEL = "deepseek-r1:32b"


def query_gemini_llm(prompt):
    """
    Query Gemini LLM using the Google Generative AI SDK.
    """
    response = model.generate_content(prompt)
    return response.text


def query_local_llm(prompt):
    """
    Query a local LLM endpoint (e.g., DeepSeek via Ollama at localhost:11434).
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LOCAL_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 8192}
    }

    response = requests.post(LOCAL_LLM_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "[ERROR] No valid response.")
    else:
        raise RuntimeError(f"[ERROR {response.status_code}] {response.text}")

def append_mcq_to_file(mcq, filename="mcq_database.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(mcq) + "\n")

if not USE_LOCAL_LLM:
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
    except (AttributeError, ValueError, Exception) as e:
        print(f"Error configuring Google AI: {e}")
        print("Please ensure the GOOGLE_API_KEY environment variable is set correctly.")
        exit()

# --- 1. Document Loading ---
def load_documents_from_directory(directory_path):
    """
    Loads documents from a specified directory, handling .pdf, .docx, and .md files.
    """
    print(f"Loading documents from '{directory_path}'...")
    all_text = ""
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        print("Please create it and add your documents.")
        return ""

    for filename in os.listdir(directory_path):
        path = os.path.join(directory_path, filename)
        loader = None
        try:
            if filename.endswith(".pdf"):
                loader = UnstructuredPDFLoader(path)
            elif filename.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            elif filename.endswith(".md"):
                loader = UnstructuredMarkdownLoader(path)
            # The user's JSON loading logic can be added here if needed.
            # For now, focusing on the primary document types.

            if loader:
                print(f"  - Loading {filename}...")
                documents = loader.load()
                for doc in documents:
                    all_text += doc.page_content + "\n\n"
        except Exception as e:
            print(f"    - Could not load or process {filename}. Error: {e}")

    print(f"Successfully loaded content from {len(os.listdir(directory_path))} file(s).")
    return all_text


# --- 2. Text Processing: Splitting the Document ---
def split_text_into_chunks(text, chunk_size=1536, chunk_overlap=256):
    """
    Splits a long text into smaller chunks using RecursiveCharacterTextSplitter.
    """
    if not text:
        return []
    print(f"Splitting text into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    print(f"Successfully created {len(chunks)} chunks.")
    return chunks

###### fault prediction as option to the LLM to select the correct response ######

# --- 3. LLM-based MCQ Generation Stages ---


def make_llm_call(prompt, retries=3, delay=5):
    """
    A unified LLM call function supporting Gemini or local LLM via a toggle.
    """
    for attempt in range(retries):
        try:
            if USE_LOCAL_LLM:
                return query_local_llm(prompt)
            else:
                return query_gemini_llm(prompt)
        except Exception as e:
            print(f"LLM call failed with error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Returning None.")
                return None

def generate_mcq(chunk):
    """
    Stage 1: Generator LLM. Takes a text chunk and prompts the LLM to create an MCQ.
    """
    print(f"\n----- GENERATOR STAGE -----")
    prompt = f"""
    Based on the following content, please generate exactly one multiple-choice question (MCQ).

    Content:
    "{chunk}"

    Instructions:
    1. Create a clear and concise question based on the content.
    2. Provide four distinct options (A, B, C, D).
    3. One option must be the correct answer derived directly from the text.
    4. The other three options should be plausible but incorrect distractors.
    5. Enclose the final output in a single JSON object with the keys "question", "options" (as a list of strings), and "correct_answer" (with the letter and text).
    """
    print("Prompting Generator LLM...")
    response_text = make_llm_call(prompt)



    if response_text:
        try:
            # Use regex to find the JSON object within the response text
            json_str_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if not json_str_match:
                 json_str_match = re.search(r'(\{.*?\})', response_text, re.DOTALL)

            if json_str_match:
                json_str = json_str_match.group(1)
                mcq_data = json.loads(json_str)
                print("Generator produced a valid MCQ.")
                print("Generated MCQ JSON:\n", json.dumps(mcq_data, indent=2))
                return mcq_data
            else:
                 print(f"Error: Could not find a JSON object in the response.")
                 return None
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error: Generator failed to produce valid JSON. Error: {e}")
            print(f"Raw response was:\n{response_text}")
            return None
    return None


def validate_mcq(chunk, mcq_data):
    """
    Stage 2: Validator LLM. Checks if the generated question is coherent and answerable.
    """
    print(f"\n----- VALIDATOR STAGE -----")
    if not mcq_data:
        print("Skipping validation due to no input.")
        return False

    correct_answer = mcq_data.get("correct_answer", "")

    # Normalize correct_answer to get the letter (A/B/C/D)
    if isinstance(correct_answer, str):
        # Expecting something like "C. It reflects CPU activity."
        match = re.match(r'^([A-Da-d])[\.\)]', correct_answer.strip())
        if match:
            generator_answer_letter = match.group(1).upper()
        else:
            print("Error: Couldn't parse answer letter from string format.")
            return False
    elif isinstance(correct_answer, dict):
        # Accept either "option" or "letter" as valid keys
        generator_answer_letter = (
            correct_answer.get("option") or
            correct_answer.get("letter") or
            ""
        ).strip().upper()
        if generator_answer_letter not in ['A', 'B', 'C', 'D']:
            print("Error: Missing or invalid 'letter'/'option' key in correct_answer dict.")
            return False
    else:
        print("Error: Unrecognized correct_answer format.")
        return False

    prompt = f"""
You are a validation agent. Verify if a multiple-choice question is valid and answerable based *only* on the provided context.

Context:
\"\"\"{chunk}\"\"\"

Question: {mcq_data['question']}
Options:
{chr(10).join(mcq_data['options'])}

Instructions:
Based *only* on the context, determine the correct option letter (A, B, C, or D). Your answer must be just the single capital letter of the correct option.
"""
    print("Prompting Validator LLM...")
    validator_answer_letter = make_llm_call(prompt)

    if validator_answer_letter:
        validator_answer_letter = validator_answer_letter.strip().upper()
        print(f"Generator's Answer: {generator_answer_letter}")
        print(f"Validator's Answer: {validator_answer_letter}")

        if generator_answer_letter == validator_answer_letter:
            print("Validation successful: Answers match.")
            return True
        else:
            print("Validation failed: Answers do not match.")
            return False
    return False




def categorize_mcq(mcq_data):
    """
    Stage 3: Categorizer LLM. Assigns a difficulty level.
    """
    print(f"\n----- CATEGORIZER STAGE -----")
    if not mcq_data:
        return "Uncategorized"

    prompt = f"""
    You are a difficulty assessment agent. Classify the following question's difficulty as 'Easy', 'Intermediate', or 'Difficult'.

    Question: {mcq_data['question']}
    Answer: {mcq_data['correct_answer']}

    Difficulty Criteria:
    - Easy: Basic concepts, definitions, or directly stated facts.
    - Intermediate: Requires comprehension, application, or combining info from a few sentences.
    - Difficult: Demands deep understanding, synthesis, or knowledge of subtle details.

    Your response must be only one word: 'Easy', 'Intermediate', or 'Difficult'.
    """
    print("Prompting Categorizer LLM...")
    category = make_llm_call(prompt)

    if category and category.strip() in ['Easy', 'Intermediate', 'Difficult']:
        final_category = category.strip()
        print(f"Categorization successful: {final_category}")
        return final_category
    else:
        print(f"Categorization failed or returned invalid category: {category}")
        return "Uncategorized"

# --- Main Execution Pipeline ---
if __name__ == "__main__":
    # Check if API key is configured before proceeding
    if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
        print("Execution stopped. Please configure your GOOGLE_API_KEY.")
    else:
        docs_directory = "./docs/"
        # 1. Load all content from the directory
        full_document_text = load_documents_from_directory(docs_directory)

        if full_document_text:
            # 2. Split the combined document text into chunks
            text_chunks = split_text_into_chunks(full_document_text)
            final_mcq_database = []

            if MODE == "generate_only":
                for i, chunk in enumerate(text_chunks):
                    print(f"\n===== GENERATING ONLY: Chunk {i+1}/{len(text_chunks)} =====")
                    generated_mcq = generate_mcq(chunk)
                    if generated_mcq:
                        difficulty = categorize_mcq(generated_mcq)
                        generated_mcq['difficulty'] = difficulty
                        generated_mcq['source_chunk'] = chunk
                        final_mcq_database.append(generated_mcq)
                        append_mcq_to_file(generated_mcq)  # <--- NEW LINE
                        print("✅ SUCCESS: MCQ generated and categorized.")
                    else:
                        print("❌ FAILED: MCQ generation failed.")

            elif MODE == "validate_only":
                print("\n===== VALIDATING ONLY: Reading from mcq_database.json =====")
                if not os.path.exists('mcq_database.json'):
                    print("❌ Error: 'mcq_database.json' not found.")
                else:
                    with open("mcq_database.json", "r") as f:
                        preloaded_mcqs = json.load(f)

                    for i, mcq in enumerate(preloaded_mcqs):
                        print(f"\n----- VALIDATING MCQ {i+1}/{len(preloaded_mcqs)} -----")
                        question = mcq.get("question")
                        chunk = mcq.get("source_chunk")
                        if not question or not chunk:
                            print("⚠️ Skipping due to missing question or source_chunk.")
                            continue

                        is_valid = validate_mcq(chunk, mcq)
                        if is_valid:
                            difficulty = categorize_mcq(mcq)
                            mcq["difficulty"] = difficulty
                            final_mcq_database.append(mcq)
                            append_mcq_to_file(mcq)  # <--- NEW LINE
                            print("✅ VALID: Added to final DB.")
                        else:
                            print("❌ INVALID: Rejected.")

            elif MODE == "generate_and_validate":
                for i, chunk in enumerate(text_chunks):
                    print(f"\n===== GENERATE + VALIDATE: Chunk {i+1}/{len(text_chunks)} =====")
                    generated_mcq = generate_mcq(chunk)
                    if generated_mcq:
                        is_valid = validate_mcq(chunk, generated_mcq)
                        if is_valid:
                            difficulty = categorize_mcq(generated_mcq)
                            generated_mcq['difficulty'] = difficulty
                            generated_mcq['source_chunk'] = chunk
                            final_mcq_database.append(generated_mcq)
                            append_mcq_to_file(generated_mcq)  # <--- NEW LINE
                            print("✅ SUCCESS: MCQ added to DB.")
                        else:
                            print("❌ FAILED: MCQ failed validation.")
                    else:
                        print("❌ FAILED: MCQ generation failed.")

            else:
                print(f"❌ Invalid MODE selected: {MODE}")


            # 4. Display and save the final results
            print("\n\n*****************************************")
            print("      MCQ Generation Pipeline Complete      ")
            print("*****************************************")
            print(f"Total MCQs generated: {len(final_mcq_database)}")

            if final_mcq_database:
                output_filename = "mcq_database.json"
                with open(output_filename, "w") as f:
                    json.dump(final_mcq_database, f, indent=2)
                print(f"\nResults saved to '{output_filename}'")
