"""
Complete Inference System with KG-RAG, RAG, and No-RAG
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

# OLLAMA CONFIG (from your code)
OLLAMA_NODES = [
    "http://10.9.64.22:11434/api/generate",
    # "http://192.168.50.140:11434/api/generate",
]
MODEL_NAME = "deepseek-r1:32b"


# --------------------------
# OLLAMA LLM CALLER (from your code)
# --------------------------
def call_ollama_llm(prompt: str, model_name: str = MODEL_NAME, endpoint: str = None) -> str:
    """
    Call Ollama LLM with a prompt
    
    Args:
        prompt: Input prompt for the LLM
        model_name: Model name to use
        endpoint: Ollama endpoint URL
        
    Returns:
        LLM response text
    """
    if endpoint is None:
        endpoint = OLLAMA_NODES[0]  # Use first node by default
    
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    
    try:
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        
        # Parse Ollama output safely
        try:
            result = response.json()
            if "response" in result:
                text = result["response"]
            else:
                # Sometimes it streams multiple JSON lines
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
    """
    Ollama LLM wrapper with load balancing across multiple nodes
    """
    
    def __init__(self, nodes: List[str] = OLLAMA_NODES, model_name: str = MODEL_NAME):
        """
        Initialize Ollama LLM
        
        Args:
            nodes: List of Ollama node endpoints
            model_name: Model name to use
        """
        self.nodes = nodes
        self.model_name = model_name
        self.node_cycle = itertools.cycle(nodes)
    
    def __call__(self, prompt: str) -> str:
        """
        Call LLM with round-robin load balancing
        
        Args:
            prompt: Input prompt
            
        Returns:
            LLM response
        """
        endpoint = next(self.node_cycle)
        return call_ollama_llm(prompt, self.model_name, endpoint)


# --------------------------
# KNOWLEDGE GRAPH RAG
# --------------------------
class KnowledgeGraphRAG:
    """
    KG-RAG: Extract keywords → Find KG context → Retrieve vectors → LLM inference
    """
    
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str,
                 vector_db_path: str,
                 llm: Optional[OllamaLLM] = None,
                 embedding_model_name: str = EMBED_MODEL):
        """
        Initialize KG-RAG system
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            vector_db_path: Path to vector database pickle file
            llm: Ollama LLM instance
            embedding_model_name: Sentence transformer model name
        """
        if neo4j_username is not None and neo4j_password is not None:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        else:
            self.driver = GraphDatabase.driver(neo4j_uri)
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm = llm or OllamaLLM()
        
        # Load vector database
        self.load_vector_db(vector_db_path)
        
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def load_vector_db(self, path: str):
        """Load pre-computed vector database"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_embeddings = np.array(data['embeddings'])
            self.metadata = data['metadata']
        
        print(f"📂 Loaded vector database: {len(self.chunks)} chunks")
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query by finding matching entities in KG
        
        Args:
            query: User query
            
        Returns:
            List of keywords that exist as entities in KG
        """
        query_lower = query.lower()
        
        with self.driver.session() as session:
            # Get all entity names from KG
            result = session.run("MATCH (n:Node) RETURN n.name AS name LIMIT 10000")
            all_entities = [record['name'] for record in result]
        
        print(f"   🔍 Searching through {len(all_entities)} entities in KG...")
        
        # Find entities that appear in the query (case-insensitive matching)
        keywords = []
        for entity in all_entities:
            # Check if entity appears in query (whole word match)
            entity_pattern = r'\b' + re.escape(entity.lower()) + r'\b'
            if re.search(entity_pattern, query_lower):
                keywords.append(entity)
        
        print(f"   ✓ Found {len(keywords)} exact matches: {keywords[:5]}")
        
        # If no exact matches, find semantically similar entities
        if not keywords:
            print(f"   ⚠️ No exact matches, trying semantic search...")
            keywords = self.find_similar_entities_simple(query, top_k=3)
            print(f"   ✓ Semantic search returned: {keywords}")
        
        return keywords
    
    def find_similar_entities_simple(self, query: str, top_k: int = 3) -> List[str]:
        """
        Find semantically similar entities from knowledge graph using Neo4j vector search
        
        Args:
            query: User query
            top_k: Number of similar entities to retrieve
            
        Returns:
            List of entity names
        """
        query_embedding = self.embedding_model.encode(query).tolist()
        
        with self.driver.session() as session:
            try:
                # Use Neo4j native vector search (requires Neo4j 5.11+)
                result = session.run("""
                    CALL db.index.vector.queryNodes('node_embeddings', $top_k, $query_embedding)
                    YIELD node, score
                    RETURN node.name AS name, score
                """, query_embedding=query_embedding, top_k=top_k)
                
                return [record['name'] for record in result]
            
            except Exception as e:
                # Fallback to manual similarity if vector index doesn't exist
                print(f"⚠️ Vector search failed, using fallback: {e}")
                return self._fallback_similarity_search(query, top_k)
    
    def _fallback_similarity_search(self, query: str, top_k: int) -> List[str]:
        """Fallback method if native vector search is unavailable"""
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
                entities_with_scores.append({
                    'entity': record['name'],
                    'score': float(similarity)
                })
            
            entities_with_scores.sort(key=lambda x: x['score'], reverse=True)
            return [e['entity'] for e in entities_with_scores[:top_k]]
    
    def get_neighbors_and_relations(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get neighboring nodes and their relations for a given entity
        
        Args:
            entity_name: Name of the entity
            depth: Traversal depth (1 = direct neighbors, 2 = neighbors of neighbors)
            
        Returns:
            Dictionary with neighbors and relations
        """
        print(f"   🔎 Finding neighbors for: '{entity_name}' (depth={depth})")
        
        with self.driver.session() as session:
            # First check if entity exists
            check_result = session.run("""
                MATCH (n:Node {name: $entity_name})
                RETURN count(n) as count
            """, entity_name=entity_name)
            
            entity_exists = check_result.single()['count'] > 0
            
            if not entity_exists:
                print(f"   ⚠️ Entity '{entity_name}' not found in graph!")
                # Try fuzzy matching
                fuzzy_result = session.run("""
                    MATCH (n:Node)
                    WHERE toLower(n.name) CONTAINS toLower($entity_name)
                    RETURN n.name as name
                    LIMIT 5
                """, entity_name=entity_name)
                
                similar = [r['name'] for r in fuzzy_result]
                if similar:
                    print(f"   💡 Similar entities found: {similar[:3]}")
                
                return {
                    'entity': entity_name,
                    'neighbors': [],
                    'relations': []
                }
            
            # Get neighbors and relations - FIXED QUERY
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
            record_count = 0
            relation_set = set()  # To avoid duplicates
            
            for record in result:
                record_count += 1
                
                # Collect neighbors from path
                for node_name in record['path_nodes']:
                    if node_name != entity_name and node_name is not None:
                        neighbors.add(node_name)
                
                # Collect relation
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
            
            print(f"   ✓ Found {len(neighbors)} neighbors, {len(relations)} relations ({record_count} paths)")
            
            if len(relations) == 0 and record_count > 0:
                print(f"   ⚠️ Warning: Found paths but no valid relations extracted!")
            
            return {
                'entity': entity_name,
                'neighbors': list(neighbors),
                'relations': relations
            }
    
    def build_kg_context(self, keywords: List[str], depth: int = 1) -> str:
        """
        Build knowledge graph context from keywords
        
        Args:
            keywords: List of extracted keywords
            depth: Traversal depth
            
        Returns:
            Formatted KG context string
        """
        print(f"   📊 Building context for {len(keywords)} keywords: {keywords}")
        
        context_parts = ["Knowledge Graph Context:\n"]
        context_parts.append("Keywords identified: " + ", ".join(keywords) + "\n")
        
        all_neighbors = set()
        all_relations = []
        
        for keyword in keywords:
            kg_info = self.get_neighbors_and_relations(keyword, depth)
            all_neighbors.update(kg_info['neighbors'])
            all_relations.extend(kg_info['relations'])
        
        print(f"   📈 Total: {len(all_neighbors)} unique neighbors, {len(all_relations)} relations")
        
        # Add neighbors
        if all_neighbors:
            context_parts.append(f"\nRelated Entities ({len(all_neighbors)}):")
            for neighbor in list(all_neighbors):  # Limit to 20
                context_parts.append(f"- {neighbor}")
        else:
            context_parts.append("\n⚠️ No related entities found in knowledge graph")
        
        # Add relations (remove duplicates)
        seen = set()
        unique_relations = []
        for rel in all_relations:
            key = (rel['subject'], rel['predicate'], rel['object'])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        if unique_relations:
            context_parts.append(f"\nRelations ({len(unique_relations)}):")
            for rel in unique_relations:  # Limit to 30 relations
                context_parts.append(f"- {rel['subject']} --[{rel['predicate']}]--> {rel['object']}")
        else:
            context_parts.append("\n⚠️ No relations found in knowledge graph")
        
        return "\n".join(context_parts)
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar chunks from vector database
        
        Args:
            query: User query (potentially expanded with KG context)
            top_k: Number of chunks to retrieve
            
        Returns:
            List of similar chunks with scores
        """
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
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
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def build_vector_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context from retrieved vector chunks
        
        Args:
            chunks: Retrieved chunks with metadata
            
        Returns:
            Formatted vector context string
        """
        context_parts = ["\nRetrieved Document Context:\n"]
        
        for i, chunk_info in enumerate(chunks, 1):
            context_parts.append(f"\n[Document {i}] (relevance: {chunk_info['score']:.3f})")
            context_parts.append(f"Source: {chunk_info['metadata'].get('filename', 'unknown')}")
            context_parts.append(chunk_info['chunk'])  # Limit chunk size
            context_parts.append("-" * 40)
        
        return "\n".join(context_parts)
    
    def inference(self, query: str, top_k_entities: int = 5, 
                  top_k_chunks: int = 5, depth: int = 1, options: List[str] = None) -> Dict[str, Any]:
        """
        Perform KG-RAG inference
        Pipeline: Extract keywords → KG context → Vector retrieval → LLM
        
        Args:
            query: User query (question text)
            top_k_entities: Number of entities for KG expansion (unused if keywords found)
            top_k_chunks: Number of chunks to retrieve from vector DB
            depth: KG traversal depth
            options: List of MCQ options (optional)
            
        Returns:
            Response with context and answer
        """
        print(f"\n🔍 Processing query: {query}")
        
        # Step 1: Extract keywords from query
        print("\n📝 Step 1: Extracting keywords from query...")
        keywords = self.extract_keywords(query)
        
        # Step 2: Get KG context (neighbors and relations)
        print("\n🕸️ Step 2: Building knowledge graph context...")
        kg_context = self.build_kg_context(keywords, depth)
        
        # Step 3: Expand query with KG context for better vector retrieval
        expanded_query = f"{query} {' '.join(keywords)}"
        print(f"\n🔄 Step 3: Expanded query: {expanded_query}")
        
        # Step 4: Retrieve similar chunks from vector DB
        print(f"\n📚 Step 4: Retrieving top {top_k_chunks} similar chunks from vector DB...")
        similar_chunks = self.retrieve_similar_chunks(expanded_query, top_k_chunks)
        vector_context = self.build_vector_context(similar_chunks)
        
        # Step 5: Combine all context
        combined_context = kg_context + "\n\n" + vector_context
        
        print(f"\n📋 Combined Context Stats:")
        print(f"   - KG context length: {len(kg_context)} chars")
        print(f"   - Vector context length: {len(vector_context)} chars")
        print(f"   - Total context length: {len(combined_context)} chars")
        print(f"\n📋 Context Preview (first 800 chars):")
        print(combined_context[:800])
        print("..." if len(combined_context) > 800 else "")
        
        # Step 6: Create final prompt for LLM
        print("\n🤖 Step 5: Calling LLM...")
        
        # Format the question with options if provided
        formatted_query = query
        if options:
            formatted_query = f"{query}\n\nOptions:\n" + "\n".join(options)
        
        prompt = f"""You are an expert in 5G, O-RAN, and telecommunications. You will be provided with:
1. Knowledge graph context (related entities and their relationships)
2. Retrieved document excerpts

IMPORTANT NOTES ABOUT THE CONTEXT:
- The context provided below is NOT the question itself
- It contains related information that MAY help answer the question
- Some context may be helpful, some may not be directly relevant
- Use your judgment to identify what is useful

=== KNOWLEDGE GRAPH CONTEXT ===
{kg_context}

=== RETRIEVED DOCUMENT CONTEXT ===
{vector_context}

=== MAIN QUESTION ===
{formatted_query}

INSTRUCTIONS:
1. Carefully read the main question above
2. Use the provided context where relevant, but don't force connections
3. Answer based on your knowledge combined with useful context
4. Provide your answer in STRICT JSON format as shown below

REQUIRED OUTPUT FORMAT (strict JSON only):
{{
  "answer": "A",
  "explanation": "Brief explanation of why this answer is correct"
}}

Where "answer" must be ONLY the option letter (A, B, C, or D).
Do not include any text before or after the JSON."""
        
        # Get LLM response
        answer = self.llm(prompt)
        
        return {
            'query': query,
            'options': options,
            'formatted_query': formatted_query,
            'method': 'KG-RAG',
            'keywords': keywords,
            'kg_context': kg_context,
            'vector_context': vector_context,
            'combined_context': combined_context,
            'answer': answer
        }


# --------------------------
# VECTOR RAG
# --------------------------
class VectorRAG:
    """
    Traditional RAG: Uses only vector similarity on text chunks
    """
    
    def __init__(self, vector_db_path: str,
                 llm: Optional[OllamaLLM] = None,
                 embedding_model_name: str = EMBED_MODEL):
        """
        Initialize Vector RAG system
        
        Args:
            vector_db_path: Path to vector database pickle file
            llm: Ollama LLM instance
            embedding_model_name: Sentence transformer model name
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm = llm or OllamaLLM()
        
        # Load vector database
        self.load_vector_db(vector_db_path)
    
    def load_vector_db(self, path: str):
        """Load pre-computed vector database"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_embeddings = np.array(data['embeddings'])
            self.metadata = data['metadata']
        
        print(f"📂 Loaded vector database: {len(self.chunks)} chunks")
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar chunks using vector similarity
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of similar chunks with scores
        """
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
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
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def generate_context_prompt(self, query: str, top_k: int = 5) -> str:
        """
        Generate context-enriched prompt from retrieved chunks
        
        Args:
            query: User query
            top_k: Number of chunks
            
        Returns:
            Context string for LLM
        """
        similar_chunks = self.retrieve_similar_chunks(query, top_k)
        
        context_parts = ["Retrieved Context:\n"]
        
        for i, chunk_info in enumerate(similar_chunks, 1):
            context_parts.append(f"\n[Chunk {i}] (relevance: {chunk_info['score']:.3f})")
            context_parts.append(f"Source: {chunk_info['metadata'].get('filename', 'unknown')}")
            context_parts.append(chunk_info['chunk'])  # Limit chunk size
            context_parts.append("-" * 40)
        
        return "\n".join(context_parts)
    
    def inference(self, query: str, top_k: int = 5, options: List[str] = None) -> Dict[str, Any]:
        """
        Perform RAG inference
        
        Args:
            query: User query (question text)
            top_k: Number of chunks to retrieve
            options: List of MCQ options (optional)
            
        Returns:
            Response with context and answer
        """
        # Format the question with options if provided
        formatted_query = query
        if options:
            formatted_query = f"{query}\n\nOptions:\n" + "\n".join(options)
        
        # Generate context
        rag_context = self.generate_context_prompt(query, top_k)
        
        # Create prompt
        prompt = f"""You are an expert in 5G, O-RAN, and telecommunications. You will be provided with retrieved document excerpts.

IMPORTANT NOTES ABOUT THE CONTEXT:
- The context provided below is NOT the question itself
- It contains related information that MAY help answer the question
- Some context may be helpful, some may not be directly relevant
- Use your judgment to identify what is useful

=== RETRIEVED DOCUMENT CONTEXT ===
{rag_context}

=== MAIN QUESTION ===
{formatted_query}

INSTRUCTIONS:
1. Carefully read the main question above
2. Use the provided context where relevant, but don't force connections
3. Answer based on your knowledge combined with useful context
4. Provide your answer in STRICT JSON format as shown below

REQUIRED OUTPUT FORMAT (strict JSON only):
{{
  "answer": "A",
  "explanation": "Brief explanation of why this answer is correct"
}}

Where "answer" must be ONLY the option letter (A, B, C, or D).
Do not include any text before or after the JSON."""
        
        # Get LLM response
        answer = self.llm(prompt)
        
        return {
            'query': query,
            'options': options,
            'formatted_query': formatted_query,
            'method': 'RAG',
            'context': rag_context,
            'answer': answer
        }


# --------------------------
# NO-RAG
# --------------------------
class NoRAG:
    """
    No-RAG: Direct LLM inference without any retrieval
    """
    
    def __init__(self, llm: Optional[OllamaLLM] = None):
        """
        Initialize No-RAG system
        
        Args:
            llm: Ollama LLM instance
        """
        self.llm = llm or OllamaLLM()
    
    def inference(self, query: str, options: List[str] = None) -> Dict[str, Any]:
        """
        Perform direct LLM inference without retrieval
        
        Args:
            query: User query (question text)
            options: List of MCQ options (optional)
            
        Returns:
            Response with answer only
        """
        # Format the question with options if provided
        formatted_query = query
        if options:
            formatted_query = f"{query}\n\nOptions:\n" + "\n".join(options)
        
        # Direct prompt
        prompt = f"""You are an expert in 5G, O-RAN, and telecommunications.

=== MAIN QUESTION ===
{formatted_query}

INSTRUCTIONS:
1. Answer based on your knowledge
2. Provide your answer in STRICT JSON format as shown below

REQUIRED OUTPUT FORMAT (strict JSON only):
{{
  "answer": "A",
  "explanation": "Brief explanation of why this answer is correct"
}}

Where "answer" must be ONLY the option letter (A, B, C, or D).
Do not include any text before or after the JSON."""
        
        # Get LLM response
        answer = self.llm(prompt)
        
        return {
            'query': query,
            'options': options,
            'formatted_query': formatted_query,
            'method': 'No-RAG',
            'context': None,
            'answer': answer
        }


# --------------------------
# MAIN COMPARISON PIPELINE
# --------------------------
if __name__ == "__main__":
    
    # Configuration
    NEO4J_URI = "bolt://localhost:7688"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "mypassword123"
    VECTOR_DB_PATH = "/home/michael/Desktop/rca_explainer/vector_db/oran_chunks.pkl"
    
    # Test query
    query = "What is the connection between O-RAN and SMO in fault management?"
    
    print("="*80)
    print("INFERENCE COMPARISON")
    print("="*80)
    print(f"\nQuery: {query}\n")
    
    # Initialize shared LLM
    llm = OllamaLLM(nodes=OLLAMA_NODES, model_name=MODEL_NAME)
    
    # 1. KG-RAG Inference
    print("\n" + "="*80)
    print("1. KG-RAG (Keywords → KG Context → Vector Retrieval → LLM)")
    print("="*80)
    try:
        kg_rag = KnowledgeGraphRAG(
            NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
            VECTOR_DB_PATH, llm=llm
        )
        kg_result = kg_rag.inference(query, top_k_chunks=3, depth=1)
        
        print(f"\n🔑 Keywords Found: {kg_result['keywords']}")
        print(f"\n📊 KG Context Preview (first 400 chars):")
        print(kg_result['kg_context'][:400] + "...")
        print(f"\n📚 Vector Context Preview (first 400 chars):")
        print(kg_result['vector_context'][:400] + "...")
        print(f"\n💡 Answer:")
        print(kg_result['answer'])
        
        kg_rag.close()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Vector RAG Inference
    print("\n\n" + "="*80)
    print("2. Vector RAG (Traditional Retrieval)")
    print("="*80)
    try:
        vector_rag = VectorRAG(VECTOR_DB_PATH, llm=llm)
        rag_result = vector_rag.inference(query, top_k=3)
        
        print(f"\n📊 Context Preview (first 500 chars):")
        print(rag_result['context'][:500] + "...\n")
        print(f"💡 Answer:")
        print(rag_result['answer'])
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3. No-RAG Inference
    print("\n\n" + "="*80)
    print("3. No-RAG (Direct LLM)")
    print("="*80)
    try:
        no_rag = NoRAG(llm=llm)
        no_rag_result = no_rag.inference(query)
        
        print(f"\n💡 Answer:")
        print(no_rag_result['answer'])
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)