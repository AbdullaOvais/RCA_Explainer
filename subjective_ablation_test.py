"""
Ablation Test Suite for Subjective Questions
Uses LLM-as-a-Judge to evaluate answer quality
"""

import json
import re
import time
from typing import List, Dict, Any
from datetime import datetime
import os
import numpy as np


class SubjectiveAblationSuite:
    """Run ablation tests with LLM-as-a-Judge evaluation"""
    
    def __init__(self, kg_rag, vector_rag, no_rag, judge_llm, output_dir: str = "./ablation_results"):
        self.kg_rag = kg_rag
        self.vector_rag = vector_rag
        self.no_rag = no_rag
        self.judge_llm = judge_llm  # Separate LLM instance for judging
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Import sentence transformer for semantic similarity
        try:
            from sentence_transformers import SentenceTransformer, util
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.semantic_available = True
        except:
            self.semantic_available = False
            print("⚠️ sentence-transformers not available, semantic similarity will be skipped")
    
    def load_questions_from_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        questions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        print(f"📁 Loaded {len(questions)} questions from {filepath}")
        return questions
    
    def calculate_semantic_similarity(self, answer: str, ground_truth: str) -> float:
        """Calculate semantic similarity between answer and ground truth"""
        if not self.semantic_available:
            return None
        
        try:
            from sentence_transformers import util
            
            answer_emb = self.embedding_model.encode(answer, convert_to_tensor=True)
            gt_emb = self.embedding_model.encode(ground_truth, convert_to_tensor=True)
            
            similarity = util.cos_sim(answer_emb, gt_emb)
            return float(similarity[0][0])
        except Exception as e:
            print(f"   ⚠️ Error calculating semantic similarity: {e}")
            return None
    
    def run_all_methods(self, question: str) -> Dict[str, str]:
        """Run all three methods and collect their answers"""
        print(f"\n{'='*80}")
        print(f"Processing: {question[:100]}...")
        print(f"{'='*80}")
        
        results = {}
        
        # KG-RAG
        print("\n🔹 Running Method 1 (KG-RAG)...")
        start = time.time()
        try:
            kg_result = self.kg_rag.inference(question, top_k_chunks=3, depth=2)
            results['method1'] = {
                'answer': kg_result['answer'],
                'time': time.time() - start,
                'success': True,
                'actual_method': 'KG-RAG'
            }
            print(f"   ✅ Completed in {results['method1']['time']:.2f}s")
        except Exception as e:
            results['method1'] = {
                'answer': f"[Error: {str(e)}]", 
                'time': time.time() - start, 
                'success': False,
                'actual_method': 'KG-RAG'
            }
            print(f"   ❌ Error: {e}")
        
        # RAG
        print("\n🔹 Running Method 2 (RAG)...")
        start = time.time()
        try:
            rag_result = self.vector_rag.inference(question, top_k=3)
            results['method2'] = {
                'answer': rag_result['answer'],
                'time': time.time() - start,
                'success': True,
                'actual_method': 'RAG'
            }
            print(f"   ✅ Completed in {results['method2']['time']:.2f}s")
        except Exception as e:
            results['method2'] = {
                'answer': f"[Error: {str(e)}]", 
                'time': time.time() - start, 
                'success': False,
                'actual_method': 'RAG'
            }
            print(f"   ❌ Error: {e}")
        
        # No-RAG
        print("\n🔹 Running Method 3 (No-RAG)...")
        start = time.time()
        try:
            no_rag_result = self.no_rag.inference(question)
            results['method3'] = {
                'answer': no_rag_result['answer'],
                'time': time.time() - start,
                'success': True,
                'actual_method': 'No-RAG'
            }
            print(f"   ✅ Completed in {results['method3']['time']:.2f}s")
        except Exception as e:
            results['method3'] = {
                'answer': f"[Error: {str(e)}]", 
                'time': time.time() - start, 
                'success': False,
                'actual_method': 'No-RAG'
            }
            print(f"   ❌ Error: {e}")
        
        return results
    
    def evaluate_with_llm_judge(self, question: str, ground_truth: str, 
                                method1_answer: str, method2_answer: str, 
                                method3_answer: str) -> Dict[str, Any]:
        """Use LLM as judge to evaluate all three answers"""
        
        judge_prompt = f"""You are an expert evaluator for 5G, O-RAN, and telecommunications answers. You will evaluate three different answers to the same question against a reference answer.

=== QUESTION ===
{question}

=== REFERENCE ANSWER (Ground Truth) ===
{ground_truth}

=== ANSWER 1 ===
{method1_answer}

=== ANSWER 2 ===
{method2_answer}

=== ANSWER 3 ===
{method3_answer}

EVALUATION CRITERIA:
1. **Accuracy**: How factually correct is the answer compared to the reference?
2. **Completeness**: Does it cover all key points from the reference answer?
3. **Relevance**: Does it stay on topic and avoid irrelevant information?
4. **Clarity**: Is the answer well-structured and easy to understand?
5. **Technical Depth**: Does it demonstrate appropriate technical understanding?

INSTRUCTIONS:
Evaluate each answer independently and provide scores on a scale of 0-10 for each criterion.
Calculate an overall score as the average of all criteria scores.
IMPORTANT: Provide the final calculated numeric value for overall_score, not the arithmetic expression.

OUTPUT FORMAT (strict JSON only):
{{
  "answer1": {{
    "accuracy": <score>,
    "completeness": <score>,
    "relevance": <score>,
    "clarity": <score>,
    "technical_depth": <score>,
    "overall_score": <calculated_average_as_number>,
    "brief_justification": "<justification text>"
  }},
  "answer2": {{
    "accuracy": <score>,
    "completeness": <score>,
    "relevance": <score>,
    "clarity": <score>,
    "technical_depth": <score>,
    "overall_score": <calculated_average_as_number>,
    "brief_justification": "<justification text>"
  }},
  "answer3": {{
    "accuracy": <score>,
    "completeness": <score>,
    "relevance": <score>,
    "clarity": <score>,
    "technical_depth": <score>,
    "overall_score": <calculated_average_as_number>,
    "brief_justification": "<justification text>"
  }},
  "ranking": ["answer1", "answer2", "answer3"],
  "summary": "<comparative analysis text>"
}}

Provide ONLY the JSON output with actual numeric scores, no placeholder text, no arithmetic expressions."""
        
        print("\n⚖️ Evaluating with LLM Judge...")
        judge_response = self.judge_llm(judge_prompt)
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            judge_response = re.sub(r'```json\s*', '', judge_response)
            judge_response = re.sub(r'```\s*', '', judge_response)
            
            # Remove thinking tokens if present (for models like deepseek-r1)
            judge_response = re.sub(r'<think>.*?</think>', '', judge_response, flags=re.DOTALL)
            judge_response = re.sub(r'<thought>.*?</thought>', '', judge_response, flags=re.DOTALL)
            
            # Fix common JSON formatting issues
            # Remove arithmetic expressions in overall_score
            judge_response = re.sub(
                r'"overall_score":\s*\([^)]+\)\s*/\s*\d+\s*=\s*(\d+\.?\d*)',
                r'"overall_score": \1',
                judge_response
            )
            
            # Strip any remaining whitespace
            judge_response = judge_response.strip()
            
            # Try to find JSON object (more flexible pattern)
            json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', judge_response, re.DOTALL)
            
            if not json_match:
                # Fallback: look for the outermost braces
                start_idx = judge_response.find('{')
                if start_idx != -1:
                    # Find matching closing brace
                    brace_count = 0
                    for i in range(start_idx, len(judge_response)):
                        if judge_response[i] == '{':
                            brace_count += 1
                        elif judge_response[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = judge_response[start_idx:i+1]
                                evaluation = json.loads(json_str)
                                break
                    else:
                        raise ValueError("Could not find matching closing brace")
                else:
                    raise ValueError("No JSON object found in response")
            else:
                evaluation = json.loads(json_match.group(0))
            
            # Map answer1/2/3 back to method1/2/3 for internal consistency
            mapped_eval = {}
            for key in ['answer1', 'answer2', 'answer3']:
                if key in evaluation:
                    method_key = key.replace('answer', 'method')
                    mapped_eval[method_key] = evaluation[key]
            
            # If already using method keys, keep them
            for key in ['method1', 'method2', 'method3']:
                if key in evaluation:
                    mapped_eval[key] = evaluation[key]
            
            # Update the evaluation dict
            if mapped_eval:
                for k, v in mapped_eval.items():
                    evaluation[k] = v
            
            # Update ranking to use method naming
            if 'ranking' in evaluation:
                new_ranking = []
                for item in evaluation['ranking']:
                    if 'answer' in item:
                        new_ranking.append(item.replace('answer', 'method'))
                    else:
                        new_ranking.append(item)
                evaluation['ranking'] = new_ranking
            
            print("   ✅ Evaluation complete")
            return evaluation
            
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parsing error: {e}")
            print(f"   Response preview: {judge_response[:300]}...")
            return None
        except Exception as e:
            print(f"   ❌ Error parsing evaluation: {e}")
            print(f"   Response preview: {judge_response[:300]}...")
            return None
    
    def run_ablation_study(self, questions: List[Dict[str, Any]], 
                          max_questions: int = None) -> Dict[str, Any]:
        """Run complete ablation study"""
        
        if max_questions:
            questions = questions[:max_questions]
        
        print("\n" + "="*80)
        print("STARTING SUBJECTIVE ABLATION STUDY WITH LLM-AS-JUDGE")
        print("="*80)
        print(f"Questions: {len(questions)}")
        print(f"Semantic Similarity: {'Enabled' if self.semantic_available else 'Disabled'}")
        print("="*80)
        
        all_results = []
        
        for idx, question_data in enumerate(questions, 1):
            question = question_data.get('question', '')
            ground_truth = question_data.get('answer', '')
            
            print(f"\n[{idx}/{len(questions)}]")
            
            # Run all three methods
            method_results = self.run_all_methods(question)
            
            # Calculate semantic similarities
            semantic_scores = {}
            if self.semantic_available:
                print("\n📊 Calculating semantic similarities...")
                for method in ['method1', 'method2', 'method3']:
                    if method_results[method]['success']:
                        sim = self.calculate_semantic_similarity(
                            method_results[method]['answer'], 
                            ground_truth
                        )
                        semantic_scores[method] = sim
                        actual_name = method_results[method]['actual_method']
                        print(f"   {actual_name}: {sim:.4f}" if sim else f"   {actual_name}: N/A")
            
            # Evaluate with LLM judge
            evaluation = None
            if all(r['success'] for r in method_results.values()):
                evaluation = self.evaluate_with_llm_judge(
                    question,
                    ground_truth,
                    method_results['method1']['answer'],
                    method_results['method2']['answer'],
                    method_results['method3']['answer']
                )
            else:
                print("   ⚠️ Skipping evaluation due to method failures")
            
            # Store results
            result = {
                'question': question,
                'ground_truth': ground_truth,
                'method1_answer': method_results['method1']['answer'],
                'method2_answer': method_results['method2']['answer'],
                'method3_answer': method_results['method3']['answer'],
                'method1_time': method_results['method1']['time'],
                'method2_time': method_results['method2']['time'],
                'method3_time': method_results['method3']['time'],
                'method1_actual': method_results['method1']['actual_method'],
                'method2_actual': method_results['method2']['actual_method'],
                'method3_actual': method_results['method3']['actual_method'],
                'semantic_similarity': semantic_scores,
                'evaluation': evaluation
            }
            all_results.append(result)
            
            # Print scores if evaluation succeeded
            if evaluation:
                print(f"\n📊 LLM Judge Scores:")
                
                # Check which key format the evaluation uses
                method_keys = []
                if 'method1' in evaluation:
                    method_keys = ['method1', 'method2', 'method3']
                elif 'answer1' in evaluation:
                    method_keys = ['answer1', 'answer2', 'answer3']
                else:
                    print("   ⚠️ Unexpected evaluation format")
                    method_keys = []
                
                if method_keys:
                    for i, method_key in enumerate(method_keys, 1):
                        actual_method = f'method{i}'
                        if actual_method in method_results:
                            actual_name = method_results[actual_method]['actual_method']
                            if method_key in evaluation:
                                score = evaluation[method_key]['overall_score']
                                print(f"   {actual_name}: {score:.1f}/10")
                    
                    # Determine winner
                    if 'ranking' in evaluation and evaluation['ranking']:
                        winner_key = evaluation['ranking'][0]
                        # Convert answer/method key to method number
                        if 'answer' in winner_key:
                            winner_num = winner_key.replace('answer', '')
                        elif 'method' in winner_key:
                            winner_num = winner_key.replace('method', '')
                        else:
                            winner_num = '1'
                        
                        winner_method = f'method{winner_num}'
                        if winner_method in method_results:
                            winner_actual = method_results[winner_method]['actual_method']
                            print(f"   🏆 Winner: {winner_actual}")
        
        # Calculate summary
        summary = self.calculate_summary(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"subjective_results_{timestamp}.json")
        
        full_results = {
            'timestamp': timestamp,
            'num_questions': len(questions),
            'semantic_similarity_enabled': self.semantic_available,
            'summary': summary,
            'detailed_results': all_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return full_results
    
    def calculate_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics"""
        
        valid_results = [r for r in all_results if r['evaluation'] is not None]
        
        if not valid_results:
            return {'error': 'No valid evaluations'}
        
        # Map method names to actual implementations
        method_mapping = {
            'method1': 'kg_rag',
            'method2': 'rag',
            'method3': 'no_rag'
        }
        
        # Aggregate scores
        methods = ['method1', 'method2', 'method3']
        summary = {}
        
        for method, actual_name in method_mapping.items():
            scores = [r['evaluation'][method]['overall_score'] for r in valid_results]
            times = [r[f'{method}_time'] for r in all_results]
            
            # Semantic similarity scores
            semantic_scores = []
            for r in valid_results:
                if 'semantic_similarity' in r and method in r['semantic_similarity']:
                    sim = r['semantic_similarity'][method]
                    if sim is not None:
                        semantic_scores.append(sim)
            
            metrics = {
                'avg_llm_score': sum(scores) / len(scores),
                'min_llm_score': min(scores),
                'max_llm_score': max(scores),
                'avg_time': sum(times) / len(times),
                'total_evaluated': len(valid_results)
            }
            
            if semantic_scores:
                metrics['avg_semantic_similarity'] = sum(semantic_scores) / len(semantic_scores)
                metrics['min_semantic_similarity'] = min(semantic_scores)
                metrics['max_semantic_similarity'] = max(semantic_scores)
            
            summary[actual_name] = metrics
        
        # Count wins
        wins = {'kg_rag': 0, 'rag': 0, 'no_rag': 0}
        for result in valid_results:
            winner = result['evaluation']['ranking'][0]
            actual_winner = method_mapping[winner]
            wins[actual_winner] += 1
        
        summary['wins'] = wins
        summary['total_questions'] = len(all_results)
        summary['successful_evaluations'] = len(valid_results)
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary"""
        
        if 'error' in summary:
            print(f"\n❌ {summary['error']}")
            return
        
        print("\n" + "="*80)
        print("SUBJECTIVE ABLATION STUDY RESULTS")
        print("="*80)
        
        print(f"\nTotal Questions: {summary['total_questions']}")
        print(f"Successful Evaluations: {summary['successful_evaluations']}")
        
        print("\n" + "-"*80)
        print("DETAILED SCORES")
        print("-"*80)
        
        for method in ['kg_rag', 'rag', 'no_rag']:
            metrics = summary[method]
            print(f"\n{method.upper().replace('_', '-')}:")
            print(f"  LLM Judge Score: {metrics['avg_llm_score']:.2f}/10 (range: {metrics['min_llm_score']:.1f}-{metrics['max_llm_score']:.1f})")
            
            if 'avg_semantic_similarity' in metrics:
                print(f"  Semantic Similarity: {metrics['avg_semantic_similarity']:.4f} (range: {metrics['min_semantic_similarity']:.4f}-{metrics['max_semantic_similarity']:.4f})")
            
            print(f"  Avg Inference Time: {metrics['avg_time']:.2f}s")
            print(f"  Wins: {summary['wins'][method]}")
        
        print("\n" + "-"*80)
        print("COMPARISON TABLE")
        print("-"*80)
        print(f"{'Metric':<30} | {'KG-RAG':<12} | {'RAG':<12} | {'No-RAG':<12}")
        print("-"*80)
        
        print(f"{'LLM Judge Score':<30} | {summary['kg_rag']['avg_llm_score']:>12.2f} | {summary['rag']['avg_llm_score']:>12.2f} | {summary['no_rag']['avg_llm_score']:>12.2f}")
        
        if 'avg_semantic_similarity' in summary['kg_rag']:
            print(f"{'Semantic Similarity':<30} | {summary['kg_rag']['avg_semantic_similarity']:>12.4f} | {summary['rag']['avg_semantic_similarity']:>12.4f} | {summary['no_rag']['avg_semantic_similarity']:>12.4f}")
        
        print(f"{'Wins':<30} | {summary['wins']['kg_rag']:>12} | {summary['wins']['rag']:>12} | {summary['wins']['no_rag']:>12}")
        print(f"{'Avg Time (s)':<30} | {summary['kg_rag']['avg_time']:>12.2f} | {summary['rag']['avg_time']:>12.2f} | {summary['no_rag']['avg_time']:>12.2f}")
        
        # Determine best methods
        best_wins = max(summary['wins'], key=summary['wins'].get)
        best_llm_score = max(['kg_rag', 'rag', 'no_rag'], key=lambda m: summary[m]['avg_llm_score'])
        
        print("\n" + "="*80)
        print(f"🏆 Most Wins: {best_wins.upper().replace('_', '-')} ({summary['wins'][best_wins]} wins)")
        print(f"⭐ Highest LLM Score: {best_llm_score.upper().replace('_', '-')} ({summary[best_llm_score]['avg_llm_score']:.2f}/10)")
        
        if 'avg_semantic_similarity' in summary['kg_rag']:
            best_semantic = max(['kg_rag', 'rag', 'no_rag'], key=lambda m: summary[m]['avg_semantic_similarity'])
            print(f"🔗 Highest Semantic Similarity: {best_semantic.upper().replace('_', '-')} ({summary[best_semantic]['avg_semantic_similarity']:.4f})")
        
        print("="*80)


# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == "__main__":
    from subjective_pipeline import KnowledgeGraphRAG, VectorRAG, NoRAG, OllamaLLM
    
    # Configuration
    NEO4J_URI = "bolt://localhost:7688"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "mypassword123"
    VECTOR_DB_PATH = "/home/michael/Desktop/imagegeneration/vector_db/oran_chunks.pkl"
    QUESTIONS_FILE = "/home/michael/Desktop/qa_dataset.jsonl"
    OUTPUT_DIR = "./ablation_results"
    
    # Initialize LLMs
    inference_llm = OllamaLLM()  # For generating answers
    judge_llm = OllamaLLM()      # For evaluating answers
    
    # Initialize systems
    print("Initializing systems...")
    kg_rag = KnowledgeGraphRAG(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, 
                                VECTOR_DB_PATH, llm=inference_llm)
    vector_rag = VectorRAG(VECTOR_DB_PATH, llm=inference_llm)
    no_rag = NoRAG(llm=inference_llm)
    
    # Initialize test suite
    test_suite = SubjectiveAblationSuite(kg_rag, vector_rag, no_rag, 
                                         judge_llm, output_dir=OUTPUT_DIR)
    
    # Load questions
    questions = test_suite.load_questions_from_jsonl(QUESTIONS_FILE)
    
    # Run ablation study (limit to first 16 for testing)
    results = test_suite.run_ablation_study(questions, max_questions=16)
    
    # Print summary
    test_suite.print_summary(results['summary'])
    
    # Close connections
    kg_rag.close()
    
    print("\n✅ Subjective ablation study complete!")
    
    def load_questions_from_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        questions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        print(f"📁 Loaded {len(questions)} questions from {filepath}")
        return questions
    
    def run_all_methods(self, question: str) -> Dict[str, str]:
        """Run all three methods and collect their answers"""
        print(f"\n{'='*80}")
        print(f"Processing: {question[:100]}...")
        print(f"{'='*80}")
        
        results = {}
        
        # KG-RAG
        print("\n🔹 Running KG-RAG...")
        start = time.time()
        try:
            kg_result = self.kg_rag.inference(question, top_k_chunks=5, depth=2)
            results['kg_rag'] = {
                'answer': kg_result['answer'],
                'time': time.time() - start,
                'success': True
            }
            print(f"   ✅ Completed in {results['kg_rag']['time']:.2f}s")
        except Exception as e:
            results['kg_rag'] = {'answer': f"[Error: {str(e)}]", 'time': time.time() - start, 'success': False}
            print(f"   ❌ Error: {e}")
        
        # RAG
        print("\n🔹 Running RAG...")
        start = time.time()
        try:
            rag_result = self.vector_rag.inference(question, top_k=5)
            results['rag'] = {
                'answer': rag_result['answer'],
                'time': time.time() - start,
                'success': True
            }
            print(f"   ✅ Completed in {results['rag']['time']:.2f}s")
        except Exception as e:
            results['rag'] = {'answer': f"[Error: {str(e)}]", 'time': time.time() - start, 'success': False}
            print(f"   ❌ Error: {e}")
        
        # No-RAG
        print("\n🔹 Running No-RAG...")
        start = time.time()
        try:
            no_rag_result = self.no_rag.inference(question)
            results['no_rag'] = {
                'answer': no_rag_result['answer'],
                'time': time.time() - start,
                'success': True
            }
            print(f"   ✅ Completed in {results['no_rag']['time']:.2f}s")
        except Exception as e:
            results['no_rag'] = {'answer': f"[Error: {str(e)}]", 'time': time.time() - start, 'success': False}
            print(f"   ❌ Error: {e}")
        
        return results
    
    def evaluate_with_llm_judge(self, question: str, ground_truth: str, 
                                kg_rag_answer: str, rag_answer: str, 
                                no_rag_answer: str) -> Dict[str, Any]:
        """Use LLM as judge to evaluate all three answers"""
        
        judge_prompt = f"""You are an expert evaluator for 5G, O-RAN, and telecommunications answers. You will evaluate three different answers to the same question against a reference answer.

=== QUESTION ===
{question}

=== REFERENCE ANSWER (Ground Truth) ===
{ground_truth}

=== ANSWER 1 (KG-RAG Method) ===
{kg_rag_answer}

=== ANSWER 2 (RAG Method) ===
{rag_answer}

=== ANSWER 3 (No-RAG Method) ===
{no_rag_answer}

EVALUATION CRITERIA:
1. **Accuracy**: How factually correct is the answer compared to the reference?
2. **Completeness**: Does it cover all key points from the reference answer?
3. **Relevance**: Does it stay on topic and avoid irrelevant information?
4. **Clarity**: Is the answer well-structured and easy to understand?
5. **Technical Depth**: Does it demonstrate appropriate technical understanding?

INSTRUCTIONS:
Evaluate each answer independently and provide scores on a scale of 0-10 for each criterion.
Calculate an overall score as the average of all criteria scores.

OUTPUT FORMAT (strict JSON only):
{{
  "kg_rag": {{
    "accuracy": 8.5,
    "completeness": 7.0,
    "relevance": 9.0,
    "clarity": 8.0,
    "technical_depth": 7.5,
    "overall_score": 8.0,
    "brief_justification": "Answer is accurate and relevant but misses some key points about..."
  }},
  "rag": {{
    "accuracy": 7.0,
    "completeness": 6.5,
    "relevance": 8.0,
    "clarity": 7.5,
    "technical_depth": 6.0,
    "overall_score": 7.0,
    "brief_justification": "Good overview but lacks depth in..."
  }},
  "no_rag": {{
    "accuracy": 6.0,
    "completeness": 5.5,
    "relevance": 7.0,
    "clarity": 6.5,
    "technical_depth": 5.0,
    "overall_score": 6.0,
    "brief_justification": "Basic understanding but misses critical details about..."
  }},
  "ranking": ["kg_rag", "rag", "no_rag"],
  "summary": "KG-RAG provided the most comprehensive answer with better technical depth..."
}}

Provide ONLY the JSON output, no other text."""
        
        print("\n⚖️ Evaluating with LLM Judge...")
        judge_response = self.judge_llm(judge_prompt)
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            judge_response = re.sub(r'```json\s*', '', judge_response)
            judge_response = re.sub(r'```\s*', '', judge_response)
            
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', judge_response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group(0))
                print("   ✅ Evaluation complete")
                return evaluation
            else:
                print("   ⚠️ Could not parse evaluation JSON")
                return None
        except Exception as e:
            print(f"   ❌ Error parsing evaluation: {e}")
            return None
    
    def run_ablation_study(self, questions: List[Dict[str, Any]], 
                          max_questions: int = None) -> Dict[str, Any]:
        """Run complete ablation study"""
        
        if max_questions:
            questions = questions[:max_questions]
        
        print("\n" + "="*80)
        print("STARTING SUBJECTIVE ABLATION STUDY WITH LLM-AS-JUDGE")
        print("="*80)
        print(f"Questions: {len(questions)}")
        print("="*80)
        
        all_results = []
        
        for idx, question_data in enumerate(questions, 1):
            question = question_data.get('question', '')
            ground_truth = question_data.get('answer', '')
            
            print(f"\n[{idx}/{len(questions)}]")
            
            # Run all three methods
            method_results = self.run_all_methods(question)
            
            # Evaluate with LLM judge
            evaluation = None
            if all(r['success'] for r in method_results.values()):
                evaluation = self.evaluate_with_llm_judge(
                    question,
                    ground_truth,
                    method_results['kg_rag']['answer'],
                    method_results['rag']['answer'],
                    method_results['no_rag']['answer']
                )
            else:
                print("   ⚠️ Skipping evaluation due to method failures")
            
            # Store results
            result = {
                'question': question,
                'ground_truth': ground_truth,
                'kg_rag_answer': method_results['kg_rag']['answer'],
                'rag_answer': method_results['rag']['answer'],
                'no_rag_answer': method_results['no_rag']['answer'],
                'kg_rag_time': method_results['kg_rag']['time'],
                'rag_time': method_results['rag']['time'],
                'no_rag_time': method_results['no_rag']['time'],
                'evaluation': evaluation
            }
            all_results.append(result)
            
            # Print scores if evaluation succeeded
            if evaluation:
                print(f"\n📊 Scores:")
                print(f"   KG-RAG: {evaluation['kg_rag']['overall_score']:.1f}/10")
                print(f"   RAG:    {evaluation['rag']['overall_score']:.1f}/10")
                print(f"   No-RAG: {evaluation['no_rag']['overall_score']:.1f}/10")
                print(f"   Winner: {evaluation['ranking'][0].upper()}")
        
        # Calculate summary
        summary = self.calculate_summary(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"subjective_results_{timestamp}.json")
        
        full_results = {
            'timestamp': timestamp,
            'num_questions': len(questions),
            'summary': summary,
            'detailed_results': all_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return full_results
    
    def calculate_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics"""
        
        valid_results = [r for r in all_results if r['evaluation'] is not None]
        
        if not valid_results:
            return {'error': 'No valid evaluations'}
        
        # Aggregate scores
        methods = ['kg_rag', 'rag', 'no_rag']
        summary = {}
        
        for method in methods:
            scores = [r['evaluation'][method]['overall_score'] for r in valid_results]
            times = [r[f'{method}_time'] for r in all_results]
            
            summary[method] = {
                'avg_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'avg_time': sum(times) / len(times),
                'total_evaluated': len(valid_results)
            }
        
        # Count wins
        wins = {method: 0 for method in methods}
        for result in valid_results:
            winner = result['evaluation']['ranking'][0]
            wins[winner] += 1
        
        summary['wins'] = wins
        summary['total_questions'] = len(all_results)
        summary['successful_evaluations'] = len(valid_results)
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary"""
        
        if 'error' in summary:
            print(f"\n❌ {summary['error']}")
            return
        
        print("\n" + "="*80)
        print("SUBJECTIVE ABLATION STUDY RESULTS")
        print("="*80)
        
        print(f"\nTotal Questions: {summary['total_questions']}")
        print(f"Successful Evaluations: {summary['successful_evaluations']}")
        
        print("\n" + "-"*80)
        print("AVERAGE SCORES (0-10 scale)")
        print("-"*80)
        
        for method in ['kg_rag', 'rag', 'no_rag']:
            metrics = summary[method]
            print(f"\n{method.upper().replace('_', '-')}:")
            print(f"  Average Score: {metrics['avg_score']:.2f}/10")
            print(f"  Range: {metrics['min_score']:.1f} - {metrics['max_score']:.1f}")
            print(f"  Avg Time: {metrics['avg_time']:.2f}s")
            print(f"  Wins: {summary['wins'][method]}")
        
        print("\n" + "-"*80)
        print("COMPARISON TABLE")
        print("-"*80)
        print(f"{'Metric':<25} | {'KG-RAG':<10} | {'RAG':<10} | {'No-RAG':<10}")
        print("-"*80)
        
        print(f"{'Average Score':<25} | {summary['kg_rag']['avg_score']:>10.2f} | {summary['rag']['avg_score']:>10.2f} | {summary['no_rag']['avg_score']:>10.2f}")
        print(f"{'Wins':<25} | {summary['wins']['kg_rag']:>10} | {summary['wins']['rag']:>10} | {summary['wins']['no_rag']:>10}")
        print(f"{'Avg Time (s)':<25} | {summary['kg_rag']['avg_time']:>10.2f} | {summary['rag']['avg_time']:>10.2f} | {summary['no_rag']['avg_time']:>10.2f}")
        
        # Determine best method
        best_method = max(summary['wins'], key=summary['wins'].get)
        best_score = max([summary[m]['avg_score'] for m in ['kg_rag', 'rag', 'no_rag']])
        best_score_method = [m for m in ['kg_rag', 'rag', 'no_rag'] if summary[m]['avg_score'] == best_score][0]
        
        print("\n" + "="*80)
        print(f"🏆 Most Wins: {best_method.upper().replace('_', '-')} ({summary['wins'][best_method]} wins)")
        print(f"⭐ Highest Average Score: {best_score_method.upper().replace('_', '-')} ({best_score:.2f}/10)")
        print("="*80)


# --------------------------
# MAIN EXECUTION
# --------------------------
# if __name__ == "__main__":
#     from subjective_inference import KnowledgeGraphRAG, VectorRAG, NoRAG, OllamaLLM
    
#     # Configuration
#     NEO4J_URI = "bolt://localhost:7688"
#     NEO4J_USERNAME = "neo4j"
#     NEO4J_PASSWORD = "mypassword123"
#     VECTOR_DB_PATH = "/path/to/vector_db/oran_chunks.pkl"
#     QUESTIONS_FILE = "/path/to/subjective_questions.jsonl"
#     OUTPUT_DIR = "./ablation_results"
    
#     # Initialize LLMs
#     inference_llm = OllamaLLM()  # For generating answers
#     judge_llm = OllamaLLM()      # For evaluating answers
    
#     # Initialize systems
#     print("Initializing systems...")
#     kg_rag = KnowledgeGraphRAG(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, 
#                                 VECTOR_DB_PATH, llm=inference_llm)
#     vector_rag = VectorRAG(VECTOR_DB_PATH, llm=inference_llm)
#     no_rag = NoRAG(llm=inference_llm)
    
#     # Initialize test suite
#     test_suite = SubjectiveAblationSuite(kg_rag, vector_rag, no_rag, 
#                                          judge_llm, output_dir=OUTPUT_DIR)
    
#     # Load questions
#     questions = test_suite.load_questions_from_jsonl(QUESTIONS_FILE)
    
#     # Run ablation study (limit to first 10 for testing)
#     results = test_suite.run_ablation_study(questions, max_questions=10)
    
#     # Print summary
#     test_suite.print_summary(results['summary'])
    
#     # Close connections
#     kg_rag.close()
    
#     print("\n✅ Subjective ablation study complete!")