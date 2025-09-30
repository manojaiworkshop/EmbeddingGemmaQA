"""
Improved SAP Endpoint Predictor with Better Training
This version addresses the low accuracy issues with better feature engineering and training
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict, Counter
import re

class ImprovedSAPPredictor:
    """Improved SAP endpoint predictor with better feature engineering"""
    
    def __init__(self, model_path: str = "./models/embeddinggemma-300m", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the EmbeddingGemma model
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f"Loaded model from: {model_path}")
        
        # SAP-specific feature extraction
        self.department_keywords = {
            'MM': ['purchase', 'po', 'vendor', 'material', 'procurement', 'buying', 'supplier', 'order', 'purchaseorder', 'purchasingorg', 'purchasinggroup'],
            'FI': ['financial', 'finance', 'codid', 'codva', 'accounting', 'money', 'payment', 'invoice', 'cost', 'budget'],
            'SD': ['sales', 'customer', 'region', 'selling', 'distribution', 'order', 'salesorder', 'client', 'revenue'],
            'PP': ['production', 'manufacturing', 'bom', 'material', 'production', 'plant', 'factory', 'produce', 'manufacture']
        }
        
        # Enhanced endpoint database
        self.endpoints_db = {}
        self.department_embeddings = {}
        self.keyword_weights = {}
        
    def extract_sap_features(self, query: str) -> Dict[str, float]:
        """Extract SAP-specific features from query"""
        query_lower = query.lower()
        features = {}
        
        # Department keyword matching with weights
        for dept, keywords in self.department_keywords.items():
            dept_score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Weight based on keyword importance
                    weight = 2.0 if keyword in ['purchase', 'sales', 'production', 'financial'] else 1.0
                    dept_score += weight
            features[f'dept_{dept}_score'] = dept_score
        
        # Query type features
        features['is_filter'] = 1.0 if any(word in query_lower for word in ['filter', 'where', 'by', 'equals']) else 0.0
        features['is_select_all'] = 1.0 if any(word in query_lower for word in ['all', 'show all', 'list all']) else 0.0
        features['is_sort'] = 1.0 if any(word in query_lower for word in ['sort', 'order by']) else 0.0
        features['is_limit'] = 1.0 if any(word in query_lower for word in ['first', 'top', 'limit']) else 0.0
        
        # Specific entity mentions
        features['mentions_po'] = 1.0 if 'po' in query_lower or 'purchase order' in query_lower else 0.0
        features['mentions_vendor'] = 1.0 if 'vendor' in query_lower else 0.0
        features['mentions_material'] = 1.0 if 'material' in query_lower else 0.0
        features['mentions_customer'] = 1.0 if 'customer' in query_lower else 0.0
        features['mentions_region'] = 1.0 if 'region' in query_lower else 0.0
        features['mentions_bom'] = 1.0 if 'bom' in query_lower else 0.0
        
        return features
    
    def create_enhanced_embeddings(self, query: str) -> np.ndarray:
        """Create enhanced embeddings combining semantic and feature-based"""
        
        # Get base semantic embedding
        base_embedding = self.model.encode([query])[0]
        
        # Extract SAP-specific features
        sap_features = self.extract_sap_features(query)
        feature_vector = np.array(list(sap_features.values()))
        
        # Combine embeddings (concatenate or weighted sum)
        # Method 1: Concatenate
        enhanced_embedding = np.concatenate([base_embedding, feature_vector])
        
        return enhanced_embedding
    
    def load_and_analyze_dataset(self, dataset_path: str = "sap_enhanced_qa_dataset.json"):
        """Load dataset and analyze patterns"""
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Dataset Analysis:")
        
        # Analyze department distribution
        dept_counts = Counter([qa['department'] for qa in data])
        print(f"Department distribution: {dict(dept_counts)}")
        
        # Analyze query patterns
        query_patterns = defaultdict(list)
        for qa in data:
            dept = qa['department']
            query = qa['question'].lower()
            query_patterns[dept].append(query)
        
        # Find common patterns per department
        for dept, queries in query_patterns.items():
            common_words = Counter()
            for query in queries:
                words = re.findall(r'\b\w+\b', query)
                common_words.update(words)
            
            print(f"{dept} top words: {dict(common_words.most_common(5))}")
        
        return data
    
    def create_rule_based_classifier(self, data: List[Dict]) -> Dict[str, Any]:
        """Create rule-based classifier as baseline"""
        
        rules = {
            'MM': {
                'keywords': ['purchase', 'po', 'vendor', 'material', 'procurement', 'buying'],
                'endpoints': set()
            },
            'FI': {
                'keywords': ['financial', 'codid', 'codva', 'accounting', 'finance'],
                'endpoints': set()
            },
            'SD': {
                'keywords': ['sales', 'customer', 'region', 'distribution'],
                'endpoints': set()
            },
            'PP': {
                'keywords': ['production', 'bom', 'manufacturing', 'plant'],
                'endpoints': set()
            }
        }
        
        # Collect endpoints for each department
        for qa in data:
            dept = qa['department']
            if dept in rules:
                rules[dept]['endpoints'].add(qa['endpoint'])
        
        return rules
    
    def rule_based_predict(self, query: str, rules: Dict[str, Any]) -> Tuple[str, float]:
        """Rule-based prediction as fallback"""
        
        query_lower = query.lower()
        dept_scores = {}
        
        for dept, rule_data in rules.items():
            score = 0
            for keyword in rule_data['keywords']:
                if keyword in query_lower:
                    score += 1
            dept_scores[dept] = score
        
        if dept_scores:
            best_dept = max(dept_scores.items(), key=lambda x: x[1])
            if best_dept[1] > 0:
                return best_dept[0], best_dept[1] / len(rules[best_dept[0]]['keywords'])
        
        return 'MM', 0.1  # Default fallback
    
    def train_improved_predictor(self, dataset_path: str = "sap_enhanced_qa_dataset.json"):
        """Train improved predictor with multiple strategies"""
        
        print(f"\n🚀 Training Improved SAP Predictor")
        print("=" * 50)
        
        # Load and analyze dataset
        data = self.load_and_analyze_dataset(dataset_path)
        
        # Create rule-based classifier
        self.rules = self.create_rule_based_classifier(data)
        
        # Create enhanced embeddings for all unique endpoints
        unique_endpoints = {}
        department_queries = defaultdict(list)
        
        for qa in data:
            endpoint = qa['endpoint']
            dept = qa['department']
            query = qa['question']
            
            if endpoint not in unique_endpoints:
                # Create enhanced description for endpoint
                endpoint_desc = f"{dept} department {qa['intent_name']}: {query}"
                unique_endpoints[endpoint] = {
                    'department': dept,
                    'intent': qa['intent_name'],
                    'description': endpoint_desc,
                    'sample_queries': []
                }
            
            unique_endpoints[endpoint]['sample_queries'].append(query)
            department_queries[dept].append(query)
        
        # Create department-specific embeddings
        print("🔄 Creating department-specific embeddings...")
        
        for dept, queries in department_queries.items():
            # Sample representative queries for each department
            sample_queries = queries[:10]  # Take first 10 as representatives
            dept_embeddings = []
            
            for query in sample_queries:
                enhanced_emb = self.create_enhanced_embeddings(query)
                dept_embeddings.append(enhanced_emb)
            
            # Average embeddings for department prototype
            self.department_embeddings[dept] = np.mean(dept_embeddings, axis=0)
        
        # Create endpoint embeddings
        self.endpoints_db = {}
        endpoint_embeddings = []
        
        for i, (endpoint, info) in enumerate(unique_endpoints.items()):
            # Use the best sample query for this endpoint
            best_query = info['sample_queries'][0]  # Take first as representative
            endpoint_embedding = self.create_enhanced_embeddings(best_query)
            
            self.endpoints_db[endpoint] = {
                'id': i,
                'endpoint': endpoint,
                'department': info['department'],
                'intent': info['intent'],
                'description': info['description'],
                'embedding': endpoint_embedding
            }
            
            endpoint_embeddings.append(endpoint_embedding)
        
        self.endpoint_embeddings = np.array(endpoint_embeddings)
        
        print(f"✅ Created {len(self.endpoints_db)} enhanced endpoint embeddings")
        
        # Evaluate the improved system
        self.evaluate_improved_system(data[:50])  # Test on first 50 samples
    
    def predict_endpoint_improved(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Improved endpoint prediction with multiple strategies"""
        
        if not hasattr(self, 'endpoint_embeddings') or self.endpoint_embeddings is None:
            print("❌ Model not trained. Please run train_improved_predictor() first.")
            return []
        
        # Strategy 1: Rule-based prediction
        rule_dept, rule_confidence = self.rule_based_predict(user_query, self.rules)
        
        # Strategy 2: Enhanced embedding similarity
        query_embedding = self.create_enhanced_embeddings(user_query)
        
        # Calculate similarities with all endpoints
        similarities = []
        for endpoint_info in self.endpoints_db.values():
            endpoint_emb = endpoint_info['embedding']
            # Cosine similarity
            similarity = np.dot(query_embedding, endpoint_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(endpoint_emb)
            )
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Strategy 3: Department prototype matching
        dept_similarities = {}
        for dept, dept_emb in self.department_embeddings.items():
            dept_sim = np.dot(query_embedding, dept_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(dept_emb)
            )
            dept_similarities[dept] = dept_sim
        
        # Combine strategies with weights
        combined_scores = similarities.copy()
        
        # Boost scores for endpoints matching rule-based department
        for i, (endpoint, info) in enumerate(self.endpoints_db.items()):
            if info['department'] == rule_dept:
                combined_scores[i] += 0.2  # Boost rule-based matches
            
            # Boost based on department prototype similarity
            dept_sim = dept_similarities.get(info['department'], 0)
            combined_scores[i] += dept_sim * 0.1
        
        # Get top-k predictions
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            endpoint_info = list(self.endpoints_db.values())[idx]
            similarity_score = float(combined_scores[idx])
            
            results.append({
                'endpoint': endpoint_info['endpoint'],
                'department': endpoint_info['department'],
                'intent': endpoint_info['intent'],
                'similarity_score': similarity_score,
                'confidence': f"{similarity_score * 100:.1f}%",
                'rule_match': endpoint_info['department'] == rule_dept
            })
        
        return results
    
    def evaluate_improved_system(self, test_data: List[Dict]):
        """Evaluate the improved system"""
        
        print(f"\n📊 Evaluating Improved SAP Predictor")
        print("-" * 50)
        
        correct_predictions = 0
        department_accuracy = defaultdict(int)
        department_totals = defaultdict(int)
        
        for i, example in enumerate(test_data[:20]):  # Test first 20
            query = example['question']
            expected_endpoint = example['endpoint']
            expected_dept = example['department']
            
            # Get improved prediction
            predictions = self.predict_endpoint_improved(query, top_k=1)
            
            if predictions:
                predicted_endpoint = predictions[0]['endpoint']
                predicted_dept = predictions[0]['department']
                confidence = predictions[0]['similarity_score']
                rule_match = predictions[0]['rule_match']
                
                # Check if prediction is correct
                is_correct = predicted_endpoint == expected_endpoint
                dept_correct = predicted_dept == expected_dept
                
                if is_correct:
                    correct_predictions += 1
                    department_accuracy[expected_dept] += 1
                elif dept_correct:
                    # Partial credit for correct department
                    correct_predictions += 0.5
                    department_accuracy[expected_dept] += 0.5
                
                department_totals[expected_dept] += 1
                
                # Show example
                status = "✅" if is_correct else "🟡" if dept_correct else "❌"
                rule_indicator = "🎯" if rule_match else ""
                
                print(f"\n{status} {rule_indicator} Query: {query[:50]}...")
                print(f"   Expected: {expected_dept} - {expected_endpoint[:40]}...")
                print(f"   Predicted: {predicted_dept} - {predicted_endpoint[:40]}...")
                print(f"   Confidence: {confidence:.3f}")
        
        # Calculate overall accuracy
        overall_accuracy = correct_predictions / len(test_data[:20])
        print(f"\n📈 Improved Overall Accuracy: {overall_accuracy:.1%}")
        
        # Department-wise accuracy
        print(f"\n🏢 Department-wise Accuracy:")
        for dept in department_totals:
            if department_totals[dept] > 0:
                dept_acc = department_accuracy[dept] / department_totals[dept]
                print(f"   {dept}: {dept_acc:.1%} ({department_accuracy[dept]:.1f}/{department_totals[dept]})")

def main():
    """Main function to test improved predictor"""
    
    print("🚀 Improved SAP Endpoint Predictor")
    print("=" * 50)
    
    # Initialize improved predictor
    predictor = ImprovedSAPPredictor()
    
    # Train with improved methods
    predictor.train_improved_predictor("sap_enhanced_qa_dataset.json")
    
    # Test with sample queries
    test_queries = [
        "show all purchase orders",
        "get financial records",
        "display sales orders", 
        "show production data",
        "filter by vendor name",
        "get customer information"
    ]
    
    print(f"\n🎯 Testing Improved Predictions:")
    print("-" * 40)
    
    for query in test_queries:
        predictions = predictor.predict_endpoint_improved(query, top_k=2)
        
        print(f"\nQuery: '{query}'")
        for i, pred in enumerate(predictions, 1):
            rule_match = "🎯" if pred['rule_match'] else ""
            confidence_color = "🟢" if pred['similarity_score'] > 0.7 else "🟡" if pred['similarity_score'] > 0.5 else "🔴"
            
            print(f"  {i}. {confidence_color} {rule_match} {pred['department']} - {pred['confidence']}")
            print(f"     {pred['endpoint'][:60]}...")

if __name__ == "__main__":
    main()
