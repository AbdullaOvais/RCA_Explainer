"""
Document Chunking and Vector Database Storage
Reads markdown files, chunks them dynamically, and stores embeddings in a vector database
"""

import re
import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any

# --------------------------
# CONFIG
# --------------------------
BASE_CHUNK_SIZE = 2500
MAX_CHUNK_SIZE = 6000
SIMILARITY_THRESHOLD = 0.5
EMBED_MODEL = "all-MiniLM-L6-v2"
OVERLAP_SEGMENTS = 1

# --------------------------
# CHUNKING FUNCTIONS (from your code)
# --------------------------
def structural_split(md_text):
    """Split markdown by structural elements"""
    pattern = r'(?=^# |\n# |\n## |\n### |!\[\]\(images/|<table|> \*\*Image Summary:|> \*\*Table Summary:)'
    parts = re.split(pattern, md_text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]

def merge_short_segments(segments, min_len=400):
    """Merge short segments to avoid tiny chunks"""
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
    """
    Dynamic chunking based on semantic similarity and structural boundaries
    """
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
# VECTOR DATABASE CLASS
# --------------------------
class VectorDatabase:
    """
    Simple vector database using FAISS-like storage
    Stores chunks with their embeddings and metadata
    """
    
    def __init__(self, embedding_model_name: str = EMBED_MODEL):
        """
        Initialize vector database
        
        Args:
            embedding_model_name: Name of sentence-transformers model
        """
        self.model = SentenceTransformer(embedding_model_name)
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        
    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]] = None):
        """
        Add document chunks to the vector database
        
        Args:
            chunks: List of text chunks
            metadata: Optional list of metadata dicts for each chunk
        """
        if metadata is None:
            metadata = [{"index": i} for i in range(len(chunks))]
        
        print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
        chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        self.chunks.extend(chunks)
        self.embeddings.extend(chunk_embeddings)
        self.metadata.extend(metadata)
        
        print(f"✅ Added {len(chunks)} chunks to vector database")
    
    def save(self, filepath: str):
        """
        Save vector database to disk
        
        Args:
            filepath: Path to save the database
        """
        data = {
            'chunks': self.chunks,
            'embeddings': np.array(self.embeddings),
            'metadata': self.metadata,
            'model_name': self.model.get_sentence_embedding_dimension()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Vector database saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load vector database from disk
        
        Args:
            filepath: Path to load the database from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.embeddings = data['embeddings'].tolist()
        self.metadata = data['metadata']
        
        print(f"📂 Vector database loaded from {filepath}")
        print(f"   - Total chunks: {len(self.chunks)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with chunks, scores, and metadata
        """
        if not self.chunks:
            return []
        
        query_embedding = self.model.encode(query)
        
        # Calculate cosine similarities
        similarities = []
        for idx, chunk_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append({
                'chunk': self.chunks[idx],
                'score': float(similarity),
                'metadata': self.metadata[idx]
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        return {
            'total_chunks': len(self.chunks),
            'avg_chunk_length': np.mean([len(c) for c in self.chunks]) if self.chunks else 0,
            'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0
        }


# --------------------------
# MAIN PROCESSING PIPELINE
# --------------------------
def process_markdown_folder(input_dir: str, 
                           output_db_path: str,
                           file_pattern: str = "*.md"):
    """
    Process all markdown files in a folder and create vector database
    
    Args:
        input_dir: Directory containing markdown files
        output_db_path: Path to save the vector database
        file_pattern: Pattern to match files (default: "*.md")
    """
    import glob
    
    # Find all markdown files
    search_pattern = os.path.join(input_dir, file_pattern)
    md_files = glob.glob(search_pattern)
    
    if not md_files:
        print(f"⚠️ No markdown files found in {input_dir}")
        return
    
    print(f"📁 Found {len(md_files)} markdown files")
    print("="*80)
    
    # Initialize vector database
    vector_db = VectorDatabase(embedding_model_name=EMBED_MODEL)
    
    # Process each file
    total_chunks = 0
    for idx, md_file in enumerate(md_files, 1):
        filename = os.path.basename(md_file)
        print(f"\n[{idx}/{len(md_files)}] Processing: {filename}")
        
        # Read file
        with open(md_file, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        # Chunk the document
        print(f"   🔧 Chunking document...")
        chunks = dynamic_chunk_documents(md_text)
        print(f"   ✅ Generated {len(chunks)} chunks")
        
        # Create metadata for each chunk
        metadata = [
            {
                'filename': filename,
                'filepath': md_file,
                'chunk_index': i,
                'chunk_length': len(chunk),
                'preview': chunk[:200] + "..." if len(chunk) > 200 else chunk
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Add to vector database
        vector_db.add_documents(chunks, metadata)
        total_chunks += len(chunks)
    
    print("\n" + "="*80)
    print(f"✅ Processed {len(md_files)} files, {total_chunks} total chunks")
    
    # Save vector database
    vector_db.save(output_db_path)
    
    # Print statistics
    stats = vector_db.get_stats()
    print("\n📊 Vector Database Statistics:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Average chunk length: {stats['avg_chunk_length']:.0f} characters")
    print(f"   - Embedding dimension: {stats['embedding_dimension']}")
    
    return vector_db


def test_search(vector_db: VectorDatabase, query: str, top_k: int = 3):
    """
    Test search functionality
    
    Args:
        vector_db: Vector database instance
        query: Search query
        top_k: Number of results
    """
    print("\n" + "="*80)
    print(f"🔍 Testing search with query: '{query}'")
    print("="*80)
    
    results = vector_db.search(query, top_k=top_k)
    
    for i, result in enumerate(results, 1):
        print(f"\n[Result {i}] Score: {result['score']:.4f}")
        print(f"File: {result['metadata']['filename']}")
        print(f"Chunk: {result['metadata']['chunk_index']}")
        print(f"Preview: {result['chunk'][:300]}...")
        print("-"*80)


# --------------------------
# EXAMPLE USAGE
# --------------------------
if __name__ == "__main__":
    
    # Configuration
    INPUT_DIR = "/home/michael/Desktop/imagegeneration/final_parsed_oran_latest_with_image"
    OUTPUT_DB_PATH = "/home/michael/Desktop/imagegeneration/vector_db/oran_chunks.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_DB_PATH), exist_ok=True)
    
    # Process all markdown files and create vector database
    vector_db = process_markdown_folder(
        input_dir=INPUT_DIR,
        output_db_path=OUTPUT_DB_PATH,
        file_pattern="*.md"
    )
    
    # Test search (optional)
    if vector_db:
        test_queries = [
            "O-RAN architecture components",
            "fault management procedures",
            "E2 interface specifications"
        ]
        
        for query in test_queries:
            test_search(vector_db, query, top_k=3)
    
    print("\n✅ Pipeline complete!")