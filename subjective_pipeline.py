"""
Complete Inference System for Subjective Questions
Uses Neo4j knowledge graph, vector database, and Ollama for LLM inference
"""

import json
import re
import pickle
import requests
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import itertools

# --------------------------
# CONFIG
# --------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"

# OLLAMA CONFIG
OLLAMA_NODES = [
    "http://10.9.64.22:11434/api/generate",
    # "http://127.0.0.1:11434/api/generate",
]
MODEL_NAME = "deepseek-r1:32b"


# --------------------------
# OLLAMA LLM CALLER
# --------------------------
def call_ollama_llm(prompt: str, model_name: str = MODEL_NAME, endpoint: str = None) -> str:
    """Call Ollama LLM with a prompt"""
    if endpoint is None:
        endpoint = OLLAMA_NODES[0]
    
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    
    try:
        response = requests.post(endpoint, json=payload, timeout=600)
        response.raise_for_status()
        
        try:
            result = response.json()
            if "response" in result:
                text = result["response"]
            else:
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
        
        return text.strip()
    
    except Exception as e:
        print(f"⚠️ Error calling Ollama at {endpoint}: {e}")
        return f"[Error: {str(e)}]"


class OllamaLLM:
    """Ollama LLM wrapper with load balancing"""
    
    def __init__(self, nodes: List[str] = OLLAMA_NODES, model_name: str = MODEL_NAME):
        self.nodes = nodes
        self.model_name = model_name
        self.node_cycle = itertools.cycle(nodes)
    
    def __call__(self, prompt: str) -> str:
        endpoint = next(self.node_cycle)
        return call_ollama_llm(prompt, self.model_name, endpoint)


# --------------------------
# KNOWLEDGE GRAPH RAG
# --------------------------
class KnowledgeGraphRAG:
    """KG-RAG: Extract keywords → Find KG context → Retrieve vectors → LLM inference"""
    
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str,
                 vector_db_path: str, llm: Optional[OllamaLLM] = None,
                 embedding_model_name: str = EMBED_MODEL):
        if neo4j_username and neo4j_password:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        else:
            self.driver = GraphDatabase.driver(neo4j_uri)
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm = llm or OllamaLLM()
        self.load_vector_db(vector_db_path)
        
    def close(self):
        self.driver.close()
    
    def load_vector_db(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_embeddings = np.array(data['embeddings'])
            self.metadata = data['metadata']
    
    def extract_keywords(self, query: str) -> List[str]:
        with self.driver.session() as session:
            result = session.run("MATCH (n:Node) RETURN n.name AS name LIMIT 10000")
            all_entities = [record['name'] for record in result]
        
        query_lower = query.lower()
        keywords = []
        # for entity in all_entities:
        #     entity_pattern = r'\b' + re.escape(entity.lower()) + r'\b'
        #     if re.search(entity_pattern, query_lower):
        #         keywords.append(entity)
        
        if not keywords:
            keywords = self.find_similar_entities_simple(query, top_k=3)
        
        return keywords
    
    def find_similar_entities_simple(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode(query).tolist()
        
        with self.driver.session() as session:
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes('node_embeddings', $top_k, $query_embedding)
                    YIELD node, score
                    RETURN node.name AS name, score
                """, query_embedding=query_embedding, top_k=top_k)
                return [record['name'] for record in result]
            except Exception:
                return self._fallback_similarity_search(query, top_k)
    
    def _fallback_similarity_search(self, query: str, top_k: int) -> List[str]:
        query_embedding = self.embedding_model.encode(query)
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Node)
                WHERE n.embedding IS NOT NULL
                RETURN n.name AS name, n.embedding AS embedding
            """)
            
            entities_with_scores = []
            for record in result:
                entity_embedding = np.array(record['embedding'])
                similarity = np.dot(query_embedding, entity_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entity_embedding)
                )
                entities_with_scores.append({'entity': record['name'], 'score': float(similarity)})
            
            entities_with_scores.sort(key=lambda x: x['score'], reverse=True)
            return [e['entity'] for e in entities_with_scores[:top_k]]
    
    def get_neighbors_and_relations(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH path = (n:Node {{name: $entity_name}})-[r*1..{depth}]-(connected:Node)
                WITH path, relationships(path) as rels
                UNWIND rels as rel
                RETURN 
                    startNode(rel).name as start_name,
                    type(rel) as rel_type,
                    endNode(rel).name as end_name,
                    [node in nodes(path) | node.name] AS path_nodes
                LIMIT 100
            """, entity_name=entity_name)
            
            relations = []
            neighbors = set()
            relation_set = set()
            
            for record in result:
                for node_name in record['path_nodes']:
                    if node_name != entity_name and node_name:
                        neighbors.add(node_name)
                
                start_name = record['start_name']
                end_name = record['end_name']
                rel_type = record['rel_type']
                
                if start_name and end_name and rel_type:
                    rel_key = (start_name, rel_type, end_name)
                    if rel_key not in relation_set:
                        relation_set.add(rel_key)
                        relations.append({
                            'subject': start_name,
                            'predicate': rel_type.lower().replace('_', ' '),
                            'object': end_name
                        })
            
            return {'entity': entity_name, 'neighbors': list(neighbors), 'relations': relations}
    
    def build_kg_context(self, keywords: List[str], depth: int = 1) -> str:
        all_neighbors = set()
        all_relations = []
        
        for keyword in keywords:
            kg_info = self.get_neighbors_and_relations(keyword, depth)
            all_neighbors.update(kg_info['neighbors'])
            all_relations.extend(kg_info['relations'])
        
        context_parts = [f"Keywords: {', '.join(keywords)}"]
        
        if all_neighbors:
            context_parts.append(f"\nRelated Entities: {', '.join(list(all_neighbors))}")
        
        seen = set()
        unique_relations = []
        for rel in all_relations:
            key = (rel['subject'], rel['predicate'], rel['object'])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        if unique_relations:
            context_parts.append("\nRelations:")
            for rel in unique_relations:
                context_parts.append(f"- {rel['subject']} → {rel['predicate']} → {rel['object']}")
        
        return "\n".join(context_parts)
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query)
        
        similarities = []
        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append({
                'chunk': self.chunks[idx],
                'score': float(similarity),
                'metadata': self.metadata[idx]
            })
        
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def build_vector_context(self, chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, chunk_info in enumerate(chunks, 1):
            context_parts.append(f"[Document {i}] {chunk_info['chunk']}")
        return "\n\n".join(context_parts)
    
    def inference(self, query: str, top_k_chunks: int = 3, depth: int = 1) -> Dict[str, Any]:
        keywords = self.extract_keywords(query)
        kg_context = self.build_kg_context(keywords, depth)
        expanded_query = f"{query} {' '.join(keywords)}"
        similar_chunks = self.retrieve_similar_chunks(expanded_query, top_k_chunks)
        vector_context = self.build_vector_context(similar_chunks)
        
        prompt = f"""You are an expert in 5G, O-RAN, and telecommunications.

CONTEXT PROVIDED (for reference only - may or may not be helpful):
=== Knowledge Graph ===
{kg_context}

=== Retrieved Documents ===
{vector_context}

=== QUESTION ===
{query}

INSTRUCTIONS:
Provide a comprehensive, detailed answer to the question above. Use the context where relevant, but rely primarily on your expertise. Your answer should be well-structured and thorough."""
        
        answer = self.llm(prompt)
        
        return {
            'query': query,
            'method': 'KG-RAG',
            'answer': answer,
            'context': f"{kg_context}\n\n{vector_context}"
        }


# --------------------------
# VECTOR RAG
# --------------------------
class VectorRAG:
    """Traditional RAG: Uses only vector similarity"""
    
    def __init__(self, vector_db_path: str, llm: Optional[OllamaLLM] = None,
                 embedding_model_name: str = EMBED_MODEL):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm = llm or OllamaLLM()
        self.load_vector_db(vector_db_path)
    
    def load_vector_db(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_embeddings = np.array(data['embeddings'])
            self.metadata = data['metadata']
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query)
        
        similarities = []
        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append({
                'chunk': self.chunks[idx],
                'score': float(similarity),
                'metadata': self.metadata[idx]
            })
        
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def inference(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        similar_chunks = self.retrieve_similar_chunks(query, top_k)
        
        context_parts = []
        for i, chunk_info in enumerate(similar_chunks, 1):
            context_parts.append(f"[Document {i}] {chunk_info['chunk']}")
        rag_context = "\n\n".join(context_parts)
        
        prompt = f"""You are an expert in 5G, O-RAN, and telecommunications.

CONTEXT PROVIDED (for reference only - may or may not be helpful):
=== Retrieved Documents ===
{rag_context}

=== QUESTION ===
{query}

INSTRUCTIONS:
Provide a comprehensive, detailed answer to the question above. Use the context where relevant, but rely primarily on your expertise. Your answer should be well-structured and thorough."""
        
        answer = self.llm(prompt)
        
        return {
            'query': query,
            'method': 'RAG',
            'answer': answer,
            'context': rag_context
        }


# --------------------------
# NO-RAG
# --------------------------
class NoRAG:
    """No-RAG: Direct LLM inference"""
    
    def __init__(self, llm: Optional[OllamaLLM] = None):
        self.llm = llm or OllamaLLM()
    
    def inference(self, query: str) -> Dict[str, Any]:
        prompt = f"""You are an expert in 5G, O-RAN, and telecommunications.

=== QUESTION ===
{query}

INSTRUCTIONS:
Provide a comprehensive, detailed answer based on your knowledge. Your answer should be well-structured and thorough."""
        
        answer = self.llm(prompt)
        
        return {
            'query': query,
            'method': 'No-RAG',
            'answer': answer,
            'context': None
        }