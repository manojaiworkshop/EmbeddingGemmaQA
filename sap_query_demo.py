"""
SAP Query Demo - Complete End-to-End Solution
This script demonstrates the complete SAP query to endpoint prediction system
"""

import json
import os
from sap_endpoint_predictor import SAPEndpointPredictor

class SAPQueryDemo:
    """Demo class for SAP query to endpoint prediction"""
    
    def __init__(self):
        self.predictor = SAPEndpointPredictor()
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        model_path = "./models/sap_endpoint_predictor"
        
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'sap_model_data.json')):
            print("📂 Loading existing SAP model...")
            if self.predictor.load_model(model_path):
                return
        
        print("🔄 Training new SAP model...")
        self.predictor.train_endpoint_predictor("sap_enhanced_qa_dataset.json")
        self.predictor.save_model(model_path)
    
    def demo_queries(self):
        """Demonstrate various SAP queries"""
        
        demo_queries = [
            # MM Department queries
            "show all purchase orders",
            "get PO data for vendor ACME",
            "filter purchase orders by material",
            "display all purchase order materials",
            
            # FI Department queries  
            "get financial records",
            "filter by code C001 and value VALUE_01",
            "sort financial data descending",
            
            # SD Department queries
            "show sales orders",
            "get customers from North region",
            "display sales data",
            
            # PP Department queries
            "get production orders",
            "show BOM for material MAT001",
            "display production data",
            
            # Mixed/unclear queries
            "get all orders",
            "show customer data", 
            "filter by region",
            "display materials info"
        ]
        
        print(f"\n🎯 SAP Query Demonstration")
        print("=" * 60)
        print(f"Testing {len(demo_queries)} different SAP queries...")
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n--- Query {i}: '{query}' ---")
            
            predictions = self.predictor.predict_endpoint(query, top_k=2)
            
            if predictions:
                for j, pred in enumerate(predictions, 1):
                    confidence_emoji = "🟢" if pred['similarity_score'] > 0.7 else "🟡" if pred['similarity_score'] > 0.5 else "🔴"
                    print(f"  {j}. {confidence_emoji} {pred['department']} Department")
                    print(f"     Intent: {pred['intent']}")
                    print(f"     Confidence: {pred['confidence']}")
                    print(f"     Endpoint: {pred['endpoint'][:70]}{'...' if len(pred['endpoint']) > 70 else ''}")
            else:
                print("  ❌ No matching endpoints found")
    
    def analyze_department_coverage(self):
        """Analyze how well each department is covered"""
        
        print(f"\n📊 Department Coverage Analysis")
        print("=" * 50)
        
        # Test queries for each department
        department_tests = {
            'MM': [
                "show purchase orders",
                "get PO materials", 
                "filter by vendor",
                "purchase order data"
            ],
            'FI': [
                "financial records",
                "filter by codid",
                "sort financial data",
                "accounting entries"
            ],
            'SD': [
                "sales orders",
                "customer data",
                "filter by region",
                "sales information"
            ],
            'PP': [
                "production orders",
                "BOM data",
                "material BOM",
                "production information"
            ]
        }
        
        department_accuracy = {}
        
        for dept, test_queries in department_tests.items():
            correct_predictions = 0
            
            print(f"\n🏢 Testing {dept} Department:")
            
            for query in test_queries:
                predictions = self.predictor.predict_endpoint(query, top_k=1)
                
                if predictions:
                    predicted_dept = predictions[0]['department']
                    confidence = predictions[0]['similarity_score']
                    is_correct = predicted_dept == dept
                    
                    if is_correct:
                        correct_predictions += 1
                    
                    status = "✅" if is_correct else "❌"
                    print(f"  {status} '{query}' → {predicted_dept} ({confidence:.2f})")
                else:
                    print(f"  ❌ '{query}' → No prediction")
            
            accuracy = correct_predictions / len(test_queries)
            department_accuracy[dept] = accuracy
            print(f"  📈 {dept} Accuracy: {accuracy:.1%}")
        
        # Overall summary
        overall_accuracy = sum(department_accuracy.values()) / len(department_accuracy)
        print(f"\n🎯 Overall Department Accuracy: {overall_accuracy:.1%}")
    
    def interactive_mode(self):
        """Interactive mode for testing queries"""
        print(f"\n🤖 Interactive SAP Query Mode")
        print("Enter your SAP queries and get endpoint suggestions!")
        print("Commands: 'demo' for examples, 'analyze' for coverage, 'quit' to exit")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n💼 Enter SAP query (or command): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'demo':
                    self.demo_queries()
                    continue
                
                elif user_input.lower() == 'analyze':
                    self.analyze_department_coverage()
                    continue
                
                elif user_input.lower() in ['help', 'h']:
                    self.show_help()
                    continue
                
                elif not user_input:
                    continue
                
                # Process the SAP query
                predictions = self.predictor.predict_endpoint(user_input, top_k=3)
                
                if predictions:
                    print(f"\n🎯 Endpoint suggestions for: '{user_input}'")
                    print("-" * 50)
                    
                    for i, pred in enumerate(predictions, 1):
                        confidence_color = "🟢" if pred['similarity_score'] > 0.7 else "🟡" if pred['similarity_score'] > 0.5 else "🔴"
                        
                        print(f"\n{i}. {confidence_color} {pred['department']} Department")
                        print(f"   Intent: {pred['intent']}")
                        print(f"   Confidence: {pred['confidence']}")
                        print(f"   Endpoint: {pred['endpoint']}")
                        
                        # Add usage tip
                        if i == 1:  # Best match
                            if pred['similarity_score'] > 0.7:
                                print(f"   💡 Recommendation: High confidence - use this endpoint")
                            elif pred['similarity_score'] > 0.5:
                                print(f"   ⚠️  Recommendation: Medium confidence - verify parameters")
                            else:
                                print(f"   🔍 Recommendation: Low confidence - check query specificity")
                else:
                    print("❌ No matching endpoints found")
                    print("💡 Try being more specific about the SAP module or operation")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help information"""
        print(f"\n💡 SAP Query Help")
        print("-" * 30)
        print("Example queries by department:")
        print("\n📦 MM (Materials Management):")
        print("  • 'show all purchase orders'")
        print("  • 'filter PO by vendor name'")
        print("  • 'get materials data'")
        
        print("\n💰 FI (Financial Accounting):")
        print("  • 'get financial records'")
        print("  • 'filter by code C001'")
        print("  • 'sort financial data'")
        
        print("\n🛒 SD (Sales & Distribution):")
        print("  • 'show sales orders'")
        print("  • 'get customers from North region'")
        print("  • 'display customer data'")
        
        print("\n🏭 PP (Production Planning):")
        print("  • 'get production orders'")
        print("  • 'show BOM for material MAT001'")
        print("  • 'display production data'")
        
        print("\n📝 Commands:")
        print("  • 'demo' - Run demonstration queries")
        print("  • 'analyze' - Show department coverage analysis")
        print("  • 'help' - Show this help")
        print("  • 'quit' - Exit the program")

def main():
    """Main function"""
    print("🚀 SAP Query to Endpoint Prediction - Complete Demo")
    print("=" * 60)
    
    try:
        # Initialize demo
        demo = SAPQueryDemo()
        
        # Show initial stats
        if hasattr(demo.predictor, 'endpoints_db') and demo.predictor.endpoints_db:
            print(f"✅ Loaded {len(demo.predictor.endpoints_db)} SAP endpoints")
            
            # Show available departments
            departments = set()
            for endpoint_info in demo.predictor.endpoints_db.values():
                departments.add(endpoint_info['department'])
            print(f"📋 Available departments: {', '.join(sorted(departments))}")
        
        # Start interactive mode
        demo.interactive_mode()
        
    except Exception as e:
        print(f"❌ Error initializing demo: {e}")
        print("💡 Make sure you have run the SAP intent analyzer first!")

if __name__ == "__main__":
    main()
