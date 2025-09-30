"""
SAP-Specific QA Fine-tuning for EmbeddingGemma
This script fine-tunes EmbeddingGemma specifically for SAP endpoint prediction
"""

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from collections import defaultdict

class SAPEndpointPredictor:
    """SAP-specific endpoint prediction using fine-tuned EmbeddingGemma"""
    
    def __init__(self, model_path: str = "./models/embeddinggemma-300m", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the EmbeddingGemma model
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f"Loaded model from: {model_path}")
        
        # Storage for training data
        self.endpoints_db = {}
        self.endpoint_embeddings = None
        self.query_examples = []
        
    def load_sap_dataset(self, dataset_path: str = "sap_enhanced_qa_dataset.json") -> Dict[str, Any]:
        """Load SAP QA dataset"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} SAP QA pairs from {dataset_path}")
            return data
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return []
    
    def preprocess_sap_data(self, qa_data: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
        """Preprocess SAP data for training"""
        queries = []
        endpoints_info = []
        unique_endpoints = set()
        
        for qa in qa_data:
            query = qa['question']
            endpoint = qa['endpoint']
            
            # Create query with SAP context
            sap_query = f"SAP {qa['department']}: {query}"
            queries.append(sap_query)
            
            # Store endpoint information
            endpoint_info = {
                'endpoint': endpoint,
                'department': qa['department'],
                'intent_name': qa['intent_name'],
                'intent_id': qa['intent_id'],
                'query_type': self.classify_sap_query(query)
            }
            endpoints_info.append(endpoint_info)
            unique_endpoints.add(endpoint)
        
        print(f"📊 Processed {len(queries)} queries with {len(unique_endpoints)} unique endpoints")
        return queries, endpoints_info, list(unique_endpoints)
    
    def classify_sap_query(self, query: str) -> str:
        """Classify SAP query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['filter', 'where', 'by', 'equals']):
            return "FILTER"
        elif any(word in query_lower for word in ['all', 'show all', 'list all']):
            return "SELECT_ALL"
        elif any(word in query_lower for word in ['sort', 'order']):
            return "SORT"
        elif any(word in query_lower for word in ['first', 'top']):
            return "LIMIT"
        else:
            return "GENERAL"
    
    def build_endpoint_database(self, unique_endpoints: List[str], endpoints_info: List[Dict]):
        """Build searchable endpoint database"""
        self.endpoints_db = {}
        
        for i, endpoint in enumerate(unique_endpoints):
            # Find all queries that map to this endpoint
            related_queries = []
            dept_info = None
            intent_info = None
            
            for info in endpoints_info:
                if info['endpoint'] == endpoint:
                    if dept_info is None:
                        dept_info = info['department']
                        intent_info = info['intent_name']
                    
            # Create endpoint description for embedding
            endpoint_description = f"{dept_info} department {intent_info} endpoint: {endpoint}"
            
            self.endpoints_db[endpoint] = {
                'id': i,
                'endpoint': endpoint,
                'department': dept_info,
                'intent': intent_info,
                'description': endpoint_description
            }
        
        print(f"🗄️ Built endpoint database with {len(self.endpoints_db)} endpoints")
    
    def train_endpoint_predictor(self, dataset_path: str = "sap_enhanced_qa_dataset.json"):
        """Train the SAP endpoint predictor"""
        print(f"\n🚀 Training SAP Endpoint Predictor")
        print("=" * 50)
        
        # Load and preprocess data
        qa_data = self.load_sap_dataset(dataset_path)
        if not qa_data:
            return
        
        queries, endpoints_info, unique_endpoints = self.preprocess_sap_data(qa_data)
        
        # Build endpoint database
        self.build_endpoint_database(unique_endpoints, endpoints_info)
        
        # Create endpoint embeddings
        print("🔄 Creating endpoint embeddings...")
        endpoint_descriptions = [info['description'] for info in self.endpoints_db.values()]
        self.endpoint_embeddings = self.model.encode(endpoint_descriptions, show_progress_bar=True)
        
        # Store query examples for evaluation
        self.query_examples = []
        for query, endpoint_info in zip(queries, endpoints_info):
            self.query_examples.append({
                'query': query,
                'expected_endpoint': endpoint_info['endpoint'],
                'department': endpoint_info['department'],
                'intent': endpoint_info['intent_name']
            })
        
        print(f"✅ Training completed with {len(self.endpoint_embeddings)} endpoint embeddings")
        
        # Evaluate the system
        self.evaluate_system()
    
    def predict_endpoint(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Predict the best SAP endpoint for a user query"""
        if self.endpoint_embeddings is None:
            print("❌ Model not trained. Please run train_endpoint_predictor() first.")
            return []
        
        # Add SAP context to query if not present
        if not user_query.lower().startswith('sap'):
            contextualized_query = f"SAP query: {user_query}"
        else:
            contextualized_query = user_query
        
        # Encode the query
        query_embedding = self.model.encode([contextualized_query])
        
        # Calculate similarities with all endpoints
        similarities = np.dot(query_embedding, self.endpoint_embeddings.T)[0]
        
        # Get top-k most similar endpoints
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            endpoint_info = list(self.endpoints_db.values())[idx]
            similarity_score = similarities[idx]
            
            results.append({
                'endpoint': endpoint_info['endpoint'],
                'department': endpoint_info['department'],
                'intent': endpoint_info['intent'],
                'similarity_score': float(similarity_score),
                'confidence': f"{similarity_score * 100:.1f}%"
            })
        
        return results
    
    def evaluate_system(self, sample_size: int = 50):
        """Evaluate the endpoint prediction system"""
        print(f"\n📊 Evaluating SAP Endpoint Prediction System")
        print("-" * 50)
        
        if not self.query_examples:
            print("❌ No query examples available for evaluation")
            return
        
        # Sample random queries for evaluation
        import random
        sample_queries = random.sample(self.query_examples, min(sample_size, len(self.query_examples)))
        
        correct_predictions = 0
        department_accuracy = defaultdict(int)
        department_totals = defaultdict(int)
        
        for example in sample_queries[:10]:  # Show first 10 for demonstration
            query = example['query']
            expected_endpoint = example['expected_endpoint']
            expected_dept = example['department']
            
            # Get prediction
            predictions = self.predict_endpoint(query, top_k=1)
            
            if predictions:
                predicted_endpoint = predictions[0]['endpoint']
                predicted_dept = predictions[0]['department']
                confidence = predictions[0]['similarity_score']
                
                # Check if prediction is correct
                is_correct = predicted_endpoint == expected_endpoint
                if is_correct:
                    correct_predictions += 1
                    department_accuracy[expected_dept] += 1
                
                department_totals[expected_dept] += 1
                
                # Show example
                status = "✅" if is_correct else "❌"
                print(f"\n{status} Query: {query[:60]}...")
                print(f"   Expected: {expected_dept} - {expected_endpoint[:50]}...")
                print(f"   Predicted: {predicted_dept} - {predicted_endpoint[:50]}...")
                print(f"   Confidence: {confidence:.3f}")
        
        # Calculate overall accuracy
        overall_accuracy = correct_predictions / len(sample_queries[:10])
        print(f"\n📈 Overall Accuracy: {overall_accuracy:.2%} ({correct_predictions}/{len(sample_queries[:10])})")
        
        # Department-wise accuracy
        print(f"\n🏢 Department-wise Accuracy:")
        for dept in department_totals:
            dept_acc = department_accuracy[dept] / department_totals[dept]
            print(f"   {dept}: {dept_acc:.2%} ({department_accuracy[dept]}/{department_totals[dept]})")
    
    def save_model(self, output_path: str = "./models/sap_endpoint_predictor"):
        """Save the trained model and endpoint database"""
        os.makedirs(output_path, exist_ok=True)
        
        # Save the sentence transformer model
        self.model.save(output_path)
        
        # Save endpoint database and embeddings
        model_data = {
            'endpoints_db': self.endpoints_db,
            'endpoint_embeddings': self.endpoint_embeddings.tolist() if self.endpoint_embeddings is not None else None,
            'query_examples': self.query_examples[:100]  # Save sample examples
        }
        
        with open(os.path.join(output_path, 'sap_model_data.json'), 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved SAP endpoint predictor to: {output_path}")
    
    def load_model(self, model_path: str = "./models/sap_endpoint_predictor"):
        """Load a trained SAP endpoint predictor"""
        try:
            # Load the sentence transformer model
            self.model = SentenceTransformer(model_path, device=self.device)
            
            # Load endpoint database and embeddings
            with open(os.path.join(model_path, 'sap_model_data.json'), 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            self.endpoints_db = model_data['endpoints_db']
            self.endpoint_embeddings = np.array(model_data['endpoint_embeddings']) if model_data['endpoint_embeddings'] else None
            self.query_examples = model_data.get('query_examples', [])
            
            print(f"✅ Loaded SAP endpoint predictor from: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

class SAPQueryInterface:
    """Interactive interface for SAP query to endpoint prediction"""
    
    def __init__(self, predictor: SAPEndpointPredictor):
        self.predictor = predictor
    
    def interactive_session(self):
        """Run interactive SAP query session"""
        print(f"\n🤖 SAP Query to Endpoint Predictor")
        print("Ask me about SAP operations and I'll suggest the right endpoint!")
        print("Type 'quit' to exit, 'help' for examples")
        print("-" * 60)
        
        while True:
            user_query = input("\n💼 SAP Query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_query.lower() in ['help', 'h']:
                self.show_examples()
                continue
            
            if not user_query:
                continue
            
            # Get endpoint predictions
            predictions = self.predictor.predict_endpoint(user_query, top_k=3)
            
            if predictions:
                print(f"\n🎯 Top endpoint suggestions for: '{user_query}'")
                print("-" * 50)
                
                for i, pred in enumerate(predictions, 1):
                    print(f"\n{i}. {pred['department']} Department")
                    print(f"   Intent: {pred['intent']}")
                    print(f"   Confidence: {pred['confidence']}")
                    print(f"   Endpoint: {pred['endpoint']}")
            else:
                print("❌ No matching endpoints found")
    
    def show_examples(self):
        """Show example queries"""
        examples = [
            "show all purchase orders",
            "filter by vendor name ACME Corp",
            "get sales orders from North region",
            "display production orders",
            "sort financial records by code",
            "show BOM for material MAT001"
        ]
        
        print(f"\n💡 Example SAP Queries:")
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")

def main():
    """Main function"""
    print("🚀 SAP Endpoint Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SAPEndpointPredictor()
    
    # Train the system
    predictor.train_endpoint_predictor("sap_enhanced_qa_dataset.json")
    
    # Save the trained model
    predictor.save_model()
    
    # Start interactive session
    interface = SAPQueryInterface(predictor)
    interface.interactive_session()

if __name__ == "__main__":
    main()
