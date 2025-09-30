"""
Fine-tune EmbeddingGemma for Question Answering tasks
This script fine-tunes the embedding model to better match questions with relevant answers
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pandas as pd
import json
import numpy as np
from typing import List, Dict, Tuple
import os
from sklearn.model_selection import train_test_split

class QADataset(Dataset):
    """Dataset class for Question-Answer pairs"""
    
    def __init__(self, qa_pairs: List[Dict[str, str]]):
        self.qa_pairs = qa_pairs
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        return self.qa_pairs[idx]

class QAFineTuner:
    """Fine-tuner for EmbeddingGemma on QA tasks"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the EmbeddingGemma model
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f"Loaded model from: {model_path}")
        
        # Print model info
        print(f"Model max sequence length: {self.model.max_seq_length}")
        print(f"Model embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def prepare_training_data(self, qa_data: List[Dict[str, str]], negative_sampling: bool = True):
        """
        Prepare training examples for contrastive learning
        
        Args:
            qa_data: List of {"question": str, "answer": str} dictionaries
            negative_sampling: Whether to add negative examples
        """
        train_examples = []
        
        for i, qa_pair in enumerate(qa_data):
            question = qa_pair["question"]
            correct_answer = qa_pair["answer"]
            
            # Add positive example (question, correct answer, high similarity)
            train_examples.append(InputExample(
                texts=[f"query: {question}", f"text: {correct_answer}"], 
                label=1.0
            ))
            
            if negative_sampling and len(qa_data) > 1:
                # Add negative examples (question, wrong answer, low similarity)
                for j, other_qa in enumerate(qa_data):
                    if i != j:  # Different QA pair
                        wrong_answer = other_qa["answer"]
                        train_examples.append(InputExample(
                            texts=[f"query: {question}", f"text: {wrong_answer}"], 
                            label=0.0
                        ))
                        break  # Just add one negative example per question
        
        return train_examples
    
    def create_sample_data(self) -> List[Dict[str, str]]:
        """Create sample QA data for demonstration"""
        sample_data = [
            {
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris, which is also the country's largest city and political center."
            },
            {
                "question": "How does photosynthesis work?",
                "answer": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll."
            },
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "Romeo and Juliet was written by William Shakespeare, the famous English playwright, in the early 1590s."
            },
            {
                "question": "What is the largest planet in our solar system?",
                "answer": "Jupiter is the largest planet in our solar system, with a mass greater than all other planets combined."
            },
            {
                "question": "How do vaccines work?",
                "answer": "Vaccines work by training the immune system to recognize and fight specific pathogens by introducing weakened or inactive forms of the disease-causing organism."
            },
            {
                "question": "What is the speed of light?",
                "answer": "The speed of light in a vacuum is approximately 299,792,458 meters per second, which is considered a fundamental constant in physics."
            },
            {
                "question": "What causes earthquakes?",
                "answer": "Earthquakes are caused by the sudden release of energy along fault lines in the Earth's crust, often due to tectonic plate movement."
            }
        ]
        return sample_data
    
    def load_data_from_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load QA data from various file formats
        
        Supported formats:
        - JSON: [{"question": "...", "answer": "..."}, ...]
        - CSV: columns "question" and "answer"
        - JSONL: {"question": "...", "answer": "..."} per line
        """
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Using sample data.")
            return self.create_sample_data()
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
                data = df[['question', 'answer']].to_dict('records')
            elif file_ext == '.jsonl':
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            print(f"Loaded {len(data)} QA pairs from {file_path}")
            return data
            
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            print("Using sample data instead.")
            return self.create_sample_data()
    
    def fine_tune(self, 
                  data_path: str = None, 
                  output_path: str = "./models/finetuned_qa_model",
                  epochs: int = 3,
                  batch_size: int = 16,
                  learning_rate: float = 2e-5,
                  warmup_steps: int = 100):
        """
        Fine-tune the model on QA data
        """
        # Load or create training data
        if data_path:
            qa_data = self.load_data_from_file(data_path)
        else:
            print("No data path provided. Using sample data for demonstration.")
            qa_data = self.create_sample_data()
        
        # Split data into train/validation
        train_data, val_data = train_test_split(qa_data, test_size=0.2, random_state=42)
        print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")
        
        # Prepare training examples
        train_examples = self.prepare_training_data(train_data)
        val_examples = self.prepare_training_data(val_data, negative_sampling=False)
        
        print(f"Created {len(train_examples)} training examples")
        
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function - Cosine Similarity Loss for embedding similarity
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Create evaluator
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples, 
            name='qa_validation'
        )
        
        # Training arguments
        total_steps = len(train_dataloader) * epochs
        print(f"Training for {epochs} epochs, {total_steps} total steps")
        
        # Fine-tune the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=len(train_dataloader) // 2,  # Evaluate twice per epoch
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=True,
            optimizer_params={'lr': learning_rate},
        )
        
        print(f"Fine-tuning completed! Model saved to: {output_path}")
        return output_path
    
    def evaluate_qa_performance(self, test_data: List[Dict[str, str]]):
        """Evaluate the model's QA performance"""
        print("\nEvaluating QA Performance...")
        
        questions = [f"query: {qa['question']}" for qa in test_data]
        answers = [f"text: {qa['answer']}" for qa in test_data]
        
        # Encode questions and answers
        question_embeddings = self.model.encode(questions)
        answer_embeddings = self.model.encode(answers)
        
        # Calculate similarities
        similarities = np.dot(question_embeddings, answer_embeddings.T)
        
        # Check if correct answers have highest similarity
        correct_predictions = 0
        for i in range(len(test_data)):
            predicted_idx = np.argmax(similarities[i])
            if predicted_idx == i:  # Correct answer should have highest similarity
                correct_predictions += 1
        
        accuracy = correct_predictions / len(test_data)
        print(f"QA Accuracy: {accuracy:.2%}")
        
        # Show some examples
        print("\nExample predictions:")
        for i in range(min(3, len(test_data))):
            question = test_data[i]["question"]
            correct_answer = test_data[i]["answer"]
            predicted_idx = np.argmax(similarities[i])
            predicted_answer = test_data[predicted_idx]["answer"]
            similarity_score = similarities[i][predicted_idx]
            
            print(f"\nQ: {question}")
            print(f"Correct A: {correct_answer[:100]}...")
            print(f"Predicted A: {predicted_answer[:100]}...")
            print(f"Similarity: {similarity_score:.3f}")
            print(f"Correct: {'✓' if predicted_idx == i else '✗'}")

def main():
    """Main function to run the fine-tuning process"""
    
    # Configuration
    MODEL_PATH = "./models/embeddinggemma-300m"  # Path to your EmbeddingGemma model
    OUTPUT_PATH = "./models/finetuned_qa_embeddinggemma"
    DATA_PATH = None  # Set to your QA data file path, or None to use sample data
    
    # Training parameters
    EPOCHS = 3
    BATCH_SIZE = 8  # Smaller batch size for 300M model
    LEARNING_RATE = 2e-5
    
    print("🚀 Starting EmbeddingGemma QA Fine-tuning")
    print("=" * 50)
    
    # Initialize fine-tuner
    fine_tuner = QAFineTuner(MODEL_PATH)
    
    # Fine-tune the model
    output_path = fine_tuner.fine_tune(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Load the fine-tuned model for evaluation
    fine_tuned_model = QAFineTuner(output_path)
    
    # Evaluate on sample data
    sample_data = fine_tuner.create_sample_data()
    fine_tuned_model.evaluate_qa_performance(sample_data)
    
    print("\n✅ Fine-tuning completed successfully!")
    print(f"📁 Fine-tuned model saved at: {output_path}")
    print("\n📊 Next steps:")
    print("1. Test the model with your own QA data")
    print("2. Integrate into your QA system")
    print("3. Further fine-tune with domain-specific data")

if __name__ == "__main__":
    main()
