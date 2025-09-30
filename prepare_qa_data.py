"""
Data Preparation Script for QA Fine-tuning
This script helps you prepare your QA dataset in the correct format
"""

import json
import pandas as pd
import csv
from typing import List, Dict, Any
import os

class QADataPreparator:
    """Helper class to prepare QA datasets"""
    
    def __init__(self):
        self.supported_formats = ['.json', '.csv', '.jsonl', '.txt']
    
    def create_sample_dataset(self, output_path: str = "sample_qa_dataset.json"):
        """Create a larger sample dataset for training"""
        
        sample_qa_data = [
            {
                "question": "What is artificial intelligence?",
                "answer": "Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. It includes machine learning, natural language processing, and computer vision."
            },
            {
                "question": "How does machine learning work?",
                "answer": "Machine learning works by training algorithms on large datasets to identify patterns and make predictions. The system learns from examples without being explicitly programmed for each specific task."
            },
            {
                "question": "What is deep learning?",
                "answer": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data, similar to how the human brain processes information."
            },
            {
                "question": "What is natural language processing?",
                "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language in a valuable way. It combines computational linguistics with machine learning."
            },
            {
                "question": "What is computer vision?",
                "answer": "Computer vision is a field of AI that trains computers to interpret and understand visual information from the world, such as images and videos, enabling machines to identify objects, faces, and scenes."
            },
            {
                "question": "What is the difference between AI and ML?",
                "answer": "AI is the broader concept of machines being able to carry out tasks in a smart way, while ML is a subset of AI that focuses on the idea that machines can learn from data without being explicitly programmed."
            },
            {
                "question": "What is supervised learning?",
                "answer": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data to make predictions or decisions on new, unseen data."
            },
            {
                "question": "What is unsupervised learning?",
                "answer": "Unsupervised learning is a type of machine learning that finds hidden patterns or structures in data without labeled examples, such as clustering and dimensionality reduction."
            },
            {
                "question": "What is reinforcement learning?",
                "answer": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward through trial and error."
            },
            {
                "question": "What are neural networks?",
                "answer": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and can learn complex patterns from data."
            },
            {
                "question": "What is a transformer model?",
                "answer": "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data. It's the foundation for many modern NLP models like BERT and GPT."
            },
            {
                "question": "What is BERT?",
                "answer": "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that understands context from both directions in a sentence, making it highly effective for many NLP tasks."
            },
            {
                "question": "What is GPT?",
                "answer": "GPT (Generative Pre-trained Transformer) is a type of transformer model that generates human-like text by predicting the next word in a sequence based on the context of previous words."
            },
            {
                "question": "What is data preprocessing?",
                "answer": "Data preprocessing is the process of cleaning, transforming, and organizing raw data into a format suitable for machine learning algorithms, including handling missing values and feature scaling."
            },
            {
                "question": "What is feature engineering?",
                "answer": "Feature engineering is the process of selecting, modifying, or creating new features from raw data to improve the performance of machine learning models."
            },
            {
                "question": "What is overfitting?",
                "answer": "Overfitting occurs when a machine learning model learns the training data too well, including noise and random fluctuations, leading to poor performance on new, unseen data."
            },
            {
                "question": "What is underfitting?",
                "answer": "Underfitting occurs when a machine learning model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data."
            },
            {
                "question": "What is cross-validation?",
                "answer": "Cross-validation is a statistical method used to evaluate machine learning models by dividing data into multiple subsets and training/testing the model on different combinations to assess performance."
            },
            {
                "question": "What is a confusion matrix?",
                "answer": "A confusion matrix is a table used to evaluate the performance of a classification model, showing the number of correct and incorrect predictions for each class."
            },
            {
                "question": "What is precision and recall?",
                "answer": "Precision is the ratio of true positive predictions to all positive predictions, while recall is the ratio of true positive predictions to all actual positive instances in the data."
            }
        ]
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_qa_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created sample dataset with {len(sample_qa_data)} QA pairs")
        print(f"📁 Saved to: {output_path}")
        return sample_qa_data
    
    def convert_csv_to_json(self, csv_path: str, output_path: str = None):
        """Convert CSV file to JSON format"""
        
        if not output_path:
            output_path = csv_path.replace('.csv', '_converted.json')
        
        try:
            df = pd.read_csv(csv_path)
            
            # Check if required columns exist
            if 'question' not in df.columns or 'answer' not in df.columns:
                print("❌ CSV must have 'question' and 'answer' columns")
                return None
            
            qa_data = df[['question', 'answer']].to_dict('records')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Converted {len(qa_data)} QA pairs from CSV to JSON")
            print(f"📁 Saved to: {output_path}")
            return qa_data
            
        except Exception as e:
            print(f"❌ Error converting CSV: {e}")
            return None
    
    def convert_txt_to_json(self, txt_path: str, output_path: str = None, separator: str = "---"):
        """
        Convert text file to JSON format
        Expected format:
        Q: Question 1
        A: Answer 1
        ---
        Q: Question 2
        A: Answer 2
        ---
        """
        
        if not output_path:
            output_path = txt_path.replace('.txt', '_converted.json')
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by separator
            qa_blocks = content.split(separator)
            qa_data = []
            
            for block in qa_blocks:
                lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
                
                question = None
                answer = None
                
                for line in lines:
                    if line.startswith('Q:'):
                        question = line[2:].strip()
                    elif line.startswith('A:'):
                        answer = line[2:].strip()
                
                if question and answer:
                    qa_data.append({
                        "question": question,
                        "answer": answer
                    })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Converted {len(qa_data)} QA pairs from TXT to JSON")
            print(f"📁 Saved to: {output_path}")
            return qa_data
            
        except Exception as e:
            print(f"❌ Error converting TXT: {e}")
            return None
    
    def validate_qa_data(self, data_path: str):
        """Validate QA dataset format"""
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print("❌ Data should be a list of QA pairs")
                return False
            
            required_keys = {'question', 'answer'}
            valid_count = 0
            
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    print(f"❌ Item {i} is not a dictionary")
                    continue
                
                if not required_keys.issubset(item.keys()):
                    print(f"❌ Item {i} missing required keys: {required_keys - set(item.keys())}")
                    continue
                
                if not item['question'].strip() or not item['answer'].strip():
                    print(f"❌ Item {i} has empty question or answer")
                    continue
                
                valid_count += 1
            
            print(f"✅ Validated {valid_count}/{len(data)} QA pairs")
            print(f"📊 Success rate: {valid_count/len(data)*100:.1f}%")
            
            return valid_count > 0
            
        except Exception as e:
            print(f"❌ Error validating data: {e}")
            return False
    
    def split_dataset(self, data_path: str, train_ratio: float = 0.8):
        """Split dataset into train/validation sets"""
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Shuffle data
            import random
            random.shuffle(data)
            
            # Split
            split_idx = int(len(data) * train_ratio)
            train_data = data[:split_idx]
            val_data = data[split_idx:]
            
            # Save splits
            base_name = os.path.splitext(data_path)[0]
            train_path = f"{base_name}_train.json"
            val_path = f"{base_name}_val.json"
            
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Split dataset:")
            print(f"📁 Training: {len(train_data)} pairs → {train_path}")
            print(f"📁 Validation: {len(val_data)} pairs → {val_path}")
            
            return train_path, val_path
            
        except Exception as e:
            print(f"❌ Error splitting dataset: {e}")
            return None, None
    
    def analyze_dataset(self, data_path: str):
        """Analyze QA dataset statistics"""
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("📊 Dataset Analysis:")
            print(f"Total QA pairs: {len(data)}")
            
            # Calculate lengths
            question_lengths = [len(item['question'].split()) for item in data]
            answer_lengths = [len(item['answer'].split()) for item in data]
            
            print(f"\nQuestion Statistics:")
            print(f"  Average length: {sum(question_lengths)/len(question_lengths):.1f} words")
            print(f"  Min length: {min(question_lengths)} words")
            print(f"  Max length: {max(question_lengths)} words")
            
            print(f"\nAnswer Statistics:")
            print(f"  Average length: {sum(answer_lengths)/len(answer_lengths):.1f} words")
            print(f"  Min length: {min(answer_lengths)} words")
            print(f"  Max length: {max(answer_lengths)} words")
            
            # Sample QA pairs
            print(f"\nSample QA pairs:")
            for i in range(min(3, len(data))):
                print(f"\n{i+1}. Q: {data[i]['question']}")
                print(f"   A: {data[i]['answer'][:100]}{'...' if len(data[i]['answer']) > 100 else ''}")
            
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")

def main():
    """Main function to prepare QA data"""
    
    print("🛠️  QA Data Preparation Tool")
    print("=" * 40)
    
    preparator = QADataPreparator()
    
    print("\nChoose an option:")
    print("1. Create sample dataset")
    print("2. Convert CSV to JSON")
    print("3. Convert TXT to JSON")
    print("4. Validate dataset")
    print("5. Split dataset")
    print("6. Analyze dataset")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        output_path = input("Output path (default: sample_qa_dataset.json): ").strip()
        if not output_path:
            output_path = "sample_qa_dataset.json"
        preparator.create_sample_dataset(output_path)
    
    elif choice == "2":
        csv_path = input("Enter CSV file path: ").strip()
        if os.path.exists(csv_path):
            preparator.convert_csv_to_json(csv_path)
        else:
            print("❌ CSV file not found")
    
    elif choice == "3":
        txt_path = input("Enter TXT file path: ").strip()
        if os.path.exists(txt_path):
            preparator.convert_txt_to_json(txt_path)
        else:
            print("❌ TXT file not found")
    
    elif choice == "4":
        data_path = input("Enter JSON data file path: ").strip()
        if os.path.exists(data_path):
            preparator.validate_qa_data(data_path)
        else:
            print("❌ Data file not found")
    
    elif choice == "5":
        data_path = input("Enter JSON data file path: ").strip()
        if os.path.exists(data_path):
            train_ratio = float(input("Train ratio (default: 0.8): ") or "0.8")
            preparator.split_dataset(data_path, train_ratio)
        else:
            print("❌ Data file not found")
    
    elif choice == "6":
        data_path = input("Enter JSON data file path: ").strip()
        if os.path.exists(data_path):
            preparator.analyze_dataset(data_path)
        else:
            print("❌ Data file not found")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
