from typing import List, Dict
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
from config.config import get_config

try:
    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available, using basic evaluation only")

class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self):
        self.config = get_config()
        self.metrics_dir = Path(self.config.metrics_output_path)
        self.metrics_dir.mkdir(exist_ok=True)
        logger.info("RAG Evaluator initialized")
    
    def evaluate(self, predictions: List[Dict], ground_truths: List[str] = None) -> Dict:
        """Evaluate RAG predictions"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(predictions),
            "basic_metrics": self._compute_basic_metrics(predictions),
        }
        
        if RAGAS_AVAILABLE and ground_truths:
            metrics["ragas_metrics"] = self._compute_ragas_metrics(predictions, ground_truths)
        
        # Save metrics if configured
        if self.config.save_eval_metrics:
            self._save_metrics(metrics)
        
        logger.info(f"Evaluation complete: {metrics['total_samples']} samples")
        return metrics
    
    def _compute_basic_metrics(self, predictions: List[Dict]) -> Dict:
        """Compute basic evaluation metrics"""
        metrics = {
            "avg_retrieval_score": 0,
            "avg_documents_retrieved": 0,
            "avg_answer_length": 0,
            "retrieval_score_distribution": {"low": 0, "medium": 0, "high": 0}
        }
        
        total_score = 0
        total_docs = 0
        total_length = 0
        
        for pred in predictions:
            # Average retrieval score
            if pred.get("retrieved_documents"):
                scores = [doc.get("score", 0) for doc in pred["retrieved_documents"]]
                avg_score = sum(scores) / len(scores) if scores else 0
                total_score += avg_score
                
                # Distribution
                if avg_score < 0.3:
                    metrics["retrieval_score_distribution"]["low"] += 1
                elif avg_score < 0.7:
                    metrics["retrieval_score_distribution"]["medium"] += 1
                else:
                    metrics["retrieval_score_distribution"]["high"] += 1
            
            # Average documents retrieved
            total_docs += pred.get("document_count", 0)
            
            # Average answer length
            total_length += len(pred.get("answer", ""))
        
        if predictions:
            metrics["avg_retrieval_score"] = total_score / len(predictions)
            metrics["avg_documents_retrieved"] = total_docs / len(predictions)
            metrics["avg_answer_length"] = total_length / len(predictions)
        
        return metrics
    
    def _compute_ragas_metrics(self, predictions: List[Dict], ground_truths: List[str]) -> Dict:
        """Compute RAGAS metrics"""
        try:
            # Prepare data for RAGAS
            eval_data = {
                "questions": [p["question"] for p in predictions],
                "contexts": [[doc["document"] for doc in p.get("retrieved_documents", [])] for p in predictions],
                "answers": [p["answer"] for p in predictions],
                "ground_truth": ground_truths
            }
            
            # Compute metrics (simplified)
            ragas_metrics = {
                "answer_relevancy": self._estimate_answer_relevancy(predictions),
                "faithfulness": self._estimate_faithfulness(predictions)
            }
            return ragas_metrics
        except Exception as e:
            logger.error(f"Error computing RAGAS metrics: {e}")
            return {}
    
    def _estimate_answer_relevancy(self, predictions: List[Dict]) -> float:
        """Estimate answer relevancy (simple heuristic)"""
        total_score = 0
        for pred in predictions:
            answer = pred.get("answer", "")
            question = pred.get("question", "")
            # Simple word overlap as proxy
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(question_words & answer_words)
            score = overlap / (len(question_words) + 1)  # +1 to avoid division by zero
            total_score += score
        
        return total_score / len(predictions) if predictions else 0
    
    def _estimate_faithfulness(self, predictions: List[Dict]) -> float:
        """Estimate faithfulness to retrieved context"""
        total_score = 0
        for pred in predictions:
            answer = pred.get("answer", "")
            context = pred.get("context_used", "")
            # Simple word overlap with context
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            overlap = len(answer_words & context_words)
            score = overlap / (len(answer_words) + 1) if answer_words else 0
            total_score += score
        
        return total_score / len(predictions) if predictions else 0
    
    def _save_metrics(self, metrics: Dict) -> None:
        """Save metrics to file"""
        filename = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {filename}")
