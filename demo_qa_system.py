"""
Demo script to test the fine-tuned QA model
This script shows how to use your fine-tuned EmbeddingGemma model for question answering
"""

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import json

class QASystem:
    """Simple QA system using fine-tuned EmbeddingGemma"""
    
    def __init__(self, model_path: str, knowledge_base_path: str = None):
        """
        Initialize QA system
        
        Args:
            model_path: Path to fine-tuned model
            knowledge_base_path: Path to JSON file with QA pairs
        """
        print(f"Loading model from: {model_path}")
        self.model = SentenceTransformer(model_path)
        
        # Load knowledge base
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
        else:
            self.create_sample_knowledge_base()
    
    def load_knowledge_base(self, path: str):
        """Load knowledge base from JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"Loaded {len(self.knowledge_base)} QA pairs from knowledge base")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            print("Using sample knowledge base instead")
            self.create_sample_knowledge_base()
    
    def create_sample_knowledge_base(self):
        """Create sample knowledge base for demonstration"""
        self.knowledge_base = [
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
            },
            {
                "question": "How does deep learning work?",
                "answer": "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data, similar to how the human brain processes information."
            },
            {
                "question": "What is natural language processing?",
                "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language in a valuable way."
            },
            {
                "question": "What are the applications of AI?",
                "answer": "AI applications include autonomous vehicles, medical diagnosis, recommendation systems, fraud detection, virtual assistants, and computer vision systems."
            },
            {
                "question": "What is the difference between AI and ML?",
                "answer": "AI is the broader concept of machines being able to carry out tasks intelligently, while ML is a subset of AI that focuses on machines learning from data."
            }
        ]
        print(f"Created sample knowledge base with {len(self.knowledge_base)} QA pairs")
    
    def encode_knowledge_base(self):
        """Pre-encode all answers in the knowledge base"""
        print("Encoding knowledge base...")
        
        # Extract answers and encode them
        self.answers = [f"text: {qa['answer']}" for qa in self.knowledge_base]
        self.answer_embeddings = self.model.encode(self.answers, show_progress_bar=True)
        
        print(f"Encoded {len(self.answers)} answers")
    
    def ask_question(self, question: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Answer a question using the knowledge base
        
        Args:
            question: User's question
            top_k: Number of top answers to return
            
        Returns:
            List of (question, answer, similarity_score) tuples
        """
        # Encode the question
        question_embedding = self.model.encode([f"query: {question}"])
        
        # Calculate similarities with all answers
        similarities = np.dot(question_embedding, self.answer_embeddings.T)[0]
        
        # Get top-k most similar answers
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similar_qa = self.knowledge_base[idx]
            similarity_score = similarities[idx]
            results.append((
                similar_qa['question'],
                similar_qa['answer'],
                float(similarity_score)
            ))
        
        return results
    
    def interactive_qa(self):
        """Interactive QA session"""
        print("\n🤖 Interactive QA System")
        print("Ask me anything! Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            user_question = input("\n❓ Your question: ").strip()
            
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_question:
                continue
            
            # Get answers
            results = self.ask_question(user_question, top_k=3)
            
            print(f"\n🎯 Top answers for: '{user_question}'")
            print("-" * 50)
            
            for i, (similar_q, answer, score) in enumerate(results, 1):
                print(f"\n{i}. Similarity: {score:.3f}")
                print(f"   Similar Q: {similar_q}")
                print(f"   Answer: {answer}")
    
    def evaluate_sample_questions(self):
        """Evaluate model on sample questions"""
        
        test_questions = [
            "What is AI?",
            "How do neural networks learn?",
            "What can artificial intelligence do?",
            "Explain machine learning vs AI",
            "How does NLP work?"
        ]
        
        print("\n📊 Sample Question Evaluation")
        print("=" * 50)
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            results = self.ask_question(question, top_k=1)
            
            if results:
                similar_q, answer, score = results[0]
                print(f"🎯 Best Answer (similarity: {score:.3f}):")
                print(f"   {answer[:150]}{'...' if len(answer) > 150 else ''}")
            else:
                print("❌ No answers found")

def main():
    """Main demo function"""
    
    print("🚀 QA System Demo")
    print("=" * 30)
    
    # Configuration
    MODEL_PATH = "./models/finetuned_qa_embeddinggemma"  # Path to fine-tuned model
    KNOWLEDGE_BASE_PATH = None  # Set to your QA data file, or None for sample data
    
    # Check if fine-tuned model exists
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Fine-tuned model not found at: {MODEL_PATH}")
        print("Please run finetune_qa.py first to create the fine-tuned model")
        print("Using original EmbeddingGemma model instead...")
        MODEL_PATH = "./models/embeddinggemma-300m"
        
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Original model not found at: {MODEL_PATH}")
            print("Please ensure your EmbeddingGemma model is in the correct location")
            return
    
    try:
        # Initialize QA system
        qa_system = QASystem(MODEL_PATH, KNOWLEDGE_BASE_PATH)
        
        # Encode knowledge base
        qa_system.encode_knowledge_base()
        
        # Evaluate sample questions
        qa_system.evaluate_sample_questions()
        
        # Start interactive session
        qa_system.interactive_qa()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your model path and dependencies")

if __name__ == "__main__":
    main()
