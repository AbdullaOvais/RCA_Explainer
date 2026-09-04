"""
Ablation Test Suite for KG-RAG, RAG, and No-RAG
Loads questions from JSONL, runs all three methods, and calculates accuracy
"""

import json
import re
import time
from typing import List, Dict, Any
from datetime import datetime
import os


class AblationTestSuite:
    """
    Run ablation tests comparing KG-RAG, RAG, and No-RAG
    """
    
    def __init__(self, kg_rag, vector_rag, no_rag, output_dir: str = "./ablation_results"):
        """
        Initialize test suite
        
        Args:
            kg_rag: KnowledgeGraphRAG instance
            vector_rag: VectorRAG instance
            no_rag: NoRAG instance
            output_dir: Directory to save results
        """
        self.kg_rag = kg_rag
        self.vector_rag = vector_rag
        self.no_rag = no_rag
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_questions_from_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load questions from JSONL file
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            List of question dictionaries
        """
        questions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        
        print(f"📁 Loaded {len(questions)} questions from {filepath}")
        return questions
    
    def parse_llm_response(self, response: str) -> str:
        """
        Parse LLM response to extract the answer option
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted option (A, B, C, or D) or None if parsing fails
        """
        try:
            # Try to find JSON in the response
            import re
            
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            
            # Try to find JSON object
            json_match = re.search(r'\{[^}]*"answer"\s*:\s*"([A-D])"[^}]*\}', response, re.IGNORECASE | re.DOTALL)
            
            if json_match:
                option = json_match.group(1).upper()
                return option
            
            # Fallback: try to parse as JSON
            try:
                parsed = json.loads(response.strip())
                if 'answer' in parsed:
                    return str(parsed['answer']).upper().strip()
            except:
                pass
            
            # Last resort: look for single letter A-D
            single_letter = re.search(r'\b([A-D])\b', response)
            if single_letter:
                return single_letter.group(1)
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Error parsing response: {e}")
            return None
    
    def evaluate_answer(self, predicted: str, ground_truth: str, question: str) -> Dict[str, Any]:
        """
        Evaluate predicted answer against ground truth
        For MCQ: extracts options and compares them
        
        Args:
            predicted: Model's predicted answer (raw LLM output)
            ground_truth: Ground truth answer (option letter)
            question: Original question
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Parse the predicted answer to get option
        predicted_option = self.parse_llm_response(predicted)
        
        # Ground truth should be just the option letter (A, B, C, or D)
        # Handle different formats
        if isinstance(ground_truth, dict) and 'answer' in ground_truth:
            gt_option = str(ground_truth['answer']).upper().strip()
        else:
            gt_option = str(ground_truth).upper().strip()
        
        # Ensure ground truth is single letter
        if len(gt_option) > 1:
            # Extract first letter if it's like "A)" or "A."
            match = re.search(r'^([A-D])', gt_option)
            if match:
                gt_option = match.group(1)
        
        # Calculate metrics
        exact_match = (predicted_option == gt_option) if predicted_option else False
        
        return {
            'predicted_option': predicted_option,
            'ground_truth_option': gt_option,
            'exact_match': exact_match,
            'parse_success': predicted_option is not None,
            'raw_predicted': predicted[:200] if predicted else None  # Store first 200 chars for debugging
        }
    
    def run_single_test(self, question_data: Dict[str, Any], method: str) -> Dict[str, Any]:
        """
        Run a single test for a specific method, now including options for multiple-choice questions.
        """
        # 1. Extraction: Get the question and options from the data
        question = question_data.get('question', question_data.get('input', ''))
        ground_truth = question_data.get('correct_answer', question_data.get('output', ''))
        options = question_data.get('options', []) # ✅ Extracted options list
        
        start_time = time.time()
        
        try:
            # 2. Execution: Pass options to the respective inference methods
            if method == 'kg_rag':
                # Assuming kg_rag.inference accepts 'options' keyword argument
                result = self.kg_rag.inference(question, options=options, top_k_chunks=3, depth=2)
                predicted = result['answer']
                context_length = len(result.get('combined_context', ''))
            
            elif method == 'rag':
                # Assuming vector_rag.inference accepts 'options' keyword argument
                result = self.vector_rag.inference(question, options=options, top_k=3)
                predicted = result['answer']
                context_length = len(result.get('context', ''))
            
            elif method == 'no_rag':
                # Assuming no_rag.inference accepts 'options' keyword argument
                result = self.no_rag.inference(question, options=options)
                predicted = result['answer']
                context_length = 0
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            inference_time = time.time() - start_time
            
            # 3. Evaluation: Using ground_truth and predicted answer
            metrics = self.evaluate_answer(predicted, ground_truth, question)
            
            return {
                'question': question,
                'options': options,      # Included in return for better traceability
                'ground_truth': ground_truth,
                'predicted': predicted,
                'method': method,
                'inference_time': inference_time,
                'context_length': context_length,
                'metrics': metrics,
                'success': True,
                'error': None
            }
        
        except Exception as e:
            inference_time = time.time() - start_time
            return {
                'question': question,
                'ground_truth': ground_truth,
                'predicted': None,
                'method': method,
                'inference_time': inference_time,
                'context_length': 0,
                'metrics': None,
                'success': False,
                'error': str(e)
            }
    
    def run_ablation_study(self, questions: List[Dict[str, Any]], 
                          methods: List[str] = ['kg_rag', 'rag', 'no_rag']) -> Dict[str, Any]:
        """
        Run complete ablation study
        
        Args:
            questions: List of question dictionaries
            methods: List of methods to test
            
        Returns:
            Complete results dictionary
        """
        print("\n" + "="*80)
        print("STARTING ABLATION STUDY")
        print("="*80)
        print(f"Questions: {len(questions)}")
        print(f"Methods: {', '.join(methods)}")
        print("="*80 + "\n")
        
        all_results = {method: [] for method in methods}
        
        for idx, question_data in enumerate(questions, 1):
            print(f"\n[{idx}/{len(questions)}] Processing question...")
            question = question_data.get('question', question_data.get('input', ''))
            print(f"Q: {question[:100]}...")
            
            for method in methods:
                print(f"  → Running {method.upper()}...", end=' ')
                result = self.run_single_test(question_data, method)
                all_results[method].append(result)
                
                if result['success']:
                    print(f"✅ ({result['inference_time']:.2f}s)")
                else:
                    print(f"❌ Error: {result['error']}")
        
        # Calculate aggregate metrics
        summary = self.calculate_summary_metrics(all_results, methods)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"ablation_results_{timestamp}.json")
        
        full_results = {
            'timestamp': timestamp,
            'num_questions': len(questions),
            'methods': methods,
            'summary': summary,
            'detailed_results': all_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return full_results
    
    def calculate_summary_metrics(self, all_results: Dict[str, List], 
                                  methods: List[str]) -> Dict[str, Any]:
        """
        Calculate summary metrics across all methods
        
        Args:
            all_results: Dictionary mapping methods to their results
            methods: List of method names
            
        Returns:
            Summary metrics dictionary
        """
        summary = {}
        
        for method in methods:
            results = all_results[method]
            successful = [r for r in results if r['success']]
            
            if not successful:
                summary[method] = {
                    'total_questions': len(results),
                    'successful': 0,
                    'failed': len(results),
                    'error': 'All tests failed'
                }
                continue
            
            # Calculate MCQ accuracy
            parseable = [r for r in successful if r['metrics']['parse_success']]
            correct = [r for r in parseable if r['metrics']['exact_match']]
            
            # Aggregate metrics
            metrics = {
                'total_questions': len(results),
                'successful': len(successful),
                'failed': len(results) - len(successful),
                'parseable_responses': len(parseable),
                'parse_failure_rate': 1 - (len(parseable) / len(successful)) if successful else 1.0,
                'accuracy': len(correct) / len(parseable) if parseable else 0.0,
                'raw_accuracy': len(correct) / len(successful) if successful else 0.0,
                'avg_inference_time': sum(r['inference_time'] for r in successful) / len(successful),
                'avg_context_length': sum(r['context_length'] for r in successful) / len(successful),
            }
            
            summary[method] = metrics
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """
        Print formatted summary of results
        
        Args:
            summary: Summary metrics dictionary
        """
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS - MCQ ACCURACY")
        print("="*80)
        
        for method, metrics in summary.items():
            print(f"\n{method.upper().replace('_', '-')}")
            print("-" * 40)
            
            if 'error' in metrics:
                print(f"❌ {metrics['error']}")
                continue
            
            print(f"Questions: {metrics['successful']}/{metrics['total_questions']} successful")
            print(f"Parseable Responses: {metrics['parseable_responses']}/{metrics['successful']}")
            print(f"Parse Failure Rate: {metrics['parse_failure_rate']:.2%}")
            print(f"\n🎯 Accuracy Metrics:")
            print(f"  - MCQ Accuracy: {metrics['accuracy']:.2%} ({int(metrics['accuracy'] * metrics['parseable_responses'])}/{metrics['parseable_responses']} correct)")
            print(f"  - Raw Accuracy: {metrics['raw_accuracy']:.2%} (including parse failures)")
            print(f"\n⏱️ Performance:")
            print(f"  - Avg Inference Time: {metrics['avg_inference_time']:.2f}s")
            print(f"  - Avg Context Length: {metrics['avg_context_length']:.0f} chars")
        
        print("\n" + "="*80)
        
        # Comparison table
        print("\nCOMPARISON TABLE")
        print("-" * 80)
        print(f"{'Metric':<30} | {'KG-RAG':<12} | {'RAG':<12} | {'No-RAG':<12}")
        print("-" * 80)
        
        if all(method in summary for method in ['kg_rag', 'rag', 'no_rag']):
            metrics_to_compare = [
                ('MCQ Accuracy', 'accuracy', '.2%'),
                ('Parse Success Rate', lambda m: 1 - m['parse_failure_rate'], '.2%'),
                ('Inference Time (s)', 'avg_inference_time', '.2f'),
                ('Context Length (chars)', 'avg_context_length', '.0f'),
            ]
            
            for label, key, fmt in metrics_to_compare:
                if callable(key):
                    kg_val = key(summary['kg_rag'])
                    rag_val = key(summary['rag'])
                    no_rag_val = key(summary['no_rag'])
                else:
                    kg_val = summary['kg_rag'].get(key, 0)
                    rag_val = summary['rag'].get(key, 0)
                    no_rag_val = summary['no_rag'].get(key, 0)
                
                print(f"{label:<30} | {kg_val:{fmt}} | {rag_val:{fmt}} | {no_rag_val:{fmt}}")
        
        print("=" * 80)
        
        # Highlight best performing method
        if all(method in summary for method in ['kg_rag', 'rag', 'no_rag']):
            accuracies = {
                'KG-RAG': summary['kg_rag']['accuracy'],
                'RAG': summary['rag']['accuracy'],
                'No-RAG': summary['no_rag']['accuracy']
            }
            best_method = max(accuracies, key=accuracies.get)
            print(f"\n🏆 Best Performing Method: {best_method} ({accuracies[best_method]:.2%} accuracy)")
        
        print("=" * 80)


# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == "__main__":
    # Import the inference classes (assuming they're in the same file or imported)
    from pipeline import KnowledgeGraphRAG, VectorRAG, NoRAG, OllamaLLM
    
    # Configuration
    NEO4J_URI = "bolt://localhost:7688"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "mypassword123"
    VECTOR_DB_PATH = "/home/abdulla-ovais/Desktop/rca_explainer/imagegeneration/vector_db/oran_chunks.pkl"
    QUESTIONS_FILE = "/home/abdulla-ovais/Desktop/rca_explainer/mcq_database2.json"  # UPDATE THIS
    OUTPUT_DIR = "./ablation_results"
    
    # Initialize LLM
    llm = OllamaLLM()
    
    # Initialize all three methods
    print("Initializing systems...")
    kg_rag = KnowledgeGraphRAG(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, 
                                VECTOR_DB_PATH, llm=llm)
    vector_rag = VectorRAG(VECTOR_DB_PATH, llm=llm)
    no_rag = NoRAG(llm=llm)
    
    # Initialize test suite
    test_suite = AblationTestSuite(kg_rag, vector_rag, no_rag, output_dir=OUTPUT_DIR)
    
    # Load questions
    questions = test_suite.load_questions_from_jsonl(QUESTIONS_FILE)
    
    # Run ablation study
    results = test_suite.run_ablation_study(questions)
    
    # Print summary
    test_suite.print_summary(results['summary'])
    
    # Close connections
    kg_rag.close()
    
    print("\n✅ Ablation study complete!")