"""
SAP Department Intent Analysis and QA Dataset Generator
This script extracts SAP department intents from config.yml and creates specialized QA datasets
"""

import yaml
import json
import os
from typing import Dict, List, Any, Tuple
import random

class SAPIntentAnalyzer:
    """Analyze SAP department intents from config file"""
    
    def __init__(self, config_path: str = "config.yml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.departments_data = self.extract_departments_data()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            print(f"✅ Loaded config from: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return {}
    
    def extract_departments_data(self) -> Dict[str, List[Dict]]:
        """Extract all department intents with their data"""
        departments = {}
        
        if 'sap_tools' not in self.config:
            print("❌ No 'sap_tools' section found in config")
            return departments
        
        dept_intents = self.config['sap_tools'].get('departments_intents', {})
        
        for dept_name, intents in dept_intents.items():
            if isinstance(intents, list):
                departments[dept_name] = intents
                print(f"📋 Found {len(intents)} intents in {dept_name}")
        
        return departments
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze the structure of department intents"""
        analysis = {
            'total_departments': len(self.departments_data),
            'departments': {},
            'total_intents': 0,
            'total_phrases': 0
        }
        
        for dept_name, intents in self.departments_data.items():
            dept_analysis = {
                'intent_count': len(intents),
                'phrase_count': 0,
                'intents': []
            }
            
            for intent in intents:
                intent_info = {
                    'id': intent.get('id'),
                    'intent': intent.get('intent'),
                    'phrase_count': len(intent.get('phrases', [])),
                    'has_endpoint': 'endpoint_template' in intent
                }
                dept_analysis['intents'].append(intent_info)
                dept_analysis['phrase_count'] += intent_info['phrase_count']
            
            analysis['departments'][dept_name] = dept_analysis
            analysis['total_intents'] += dept_analysis['intent_count']
            analysis['total_phrases'] += dept_analysis['phrase_count']
        
        return analysis
    
    def print_analysis_report(self):
        """Print detailed analysis report"""
        analysis = self.analyze_structure()
        
        print("\n🔍 SAP Department Intent Analysis Report")
        print("=" * 60)
        print(f"Total Departments: {analysis['total_departments']}")
        print(f"Total Intents: {analysis['total_intents']}")
        print(f"Total Phrases: {analysis['total_phrases']}")
        
        print("\n📊 Department Breakdown:")
        for dept_name, dept_data in analysis['departments'].items():
            dept_display = dept_name.upper().replace('_INTENTS', '')
            print(f"\n  {dept_display} Department:")
            print(f"    - Intents: {dept_data['intent_count']}")
            print(f"    - Phrases: {dept_data['phrase_count']}")
            
            for intent in dept_data['intents']:
                endpoint_status = "✅" if intent['has_endpoint'] else "❌"
                print(f"      • {intent['intent']} (ID: {intent['id']}) - {intent['phrase_count']} phrases {endpoint_status}")

class SAPQADatasetGenerator:
    """Generate QA dataset from SAP department intents"""
    
    def __init__(self, analyzer: SAPIntentAnalyzer):
        self.analyzer = analyzer
        
    def create_qa_pairs(self) -> List[Dict[str, str]]:
        """Create QA pairs from department intents"""
        qa_pairs = []
        
        for dept_name, intents in self.analyzer.departments_data.items():
            dept_display = dept_name.upper().replace('_INTENTS', '')
            
            for intent_data in intents:
                intent_id = intent_data.get('id')
                intent_name = intent_data.get('intent')
                phrases = intent_data.get('phrases', [])
                endpoint = intent_data.get('endpoint_template', '')
                
                # Create QA pairs for each phrase
                for phrase in phrases:
                    # Question is the user's natural language query
                    question = phrase.strip()
                    
                    # Answer contains the endpoint and metadata
                    answer = self.create_structured_answer(
                        department=dept_display,
                        intent_id=intent_id,
                        intent_name=intent_name,
                        endpoint=endpoint,
                        query=phrase
                    )
                    
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "department": dept_display,
                        "intent_id": intent_id,
                        "intent_name": intent_name,
                        "endpoint": endpoint
                    })
        
        return qa_pairs
    
    def create_structured_answer(self, department: str, intent_id: int, intent_name: str, 
                                endpoint: str, query: str) -> str:
        """Create structured answer for training"""
        return f"""Department: {department}
Intent: {intent_name} (ID: {intent_id})
Endpoint: {endpoint}
Query Type: {self.classify_query_type(query)}
Description: This query requests {self.describe_query_action(query)} from the {department} department using the {intent_name} operation."""
    
    def classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['filter', 'where', 'by', 'equals', 'is']):
            return "FILTER"
        elif any(word in query_lower for word in ['get all', 'show all', 'list all', 'display all']):
            return "SELECT_ALL"
        elif any(word in query_lower for word in ['sort', 'order by', 'descending', 'ascending']):
            return "SORT"
        elif any(word in query_lower for word in ['first', 'top', 'limit']):
            return "LIMIT"
        else:
            return "GENERAL"
    
    def describe_query_action(self, query: str) -> str:
        """Describe what the query is trying to do"""
        query_lower = query.lower()
        
        if 'purchase order' in query_lower or 'po' in query_lower:
            return "purchase order data"
        elif 'sales order' in query_lower:
            return "sales order information"
        elif 'production order' in query_lower:
            return "production order details"
        elif 'financial' in query_lower or 'codid' in query_lower:
            return "financial records"
        elif 'customer' in query_lower:
            return "customer information"
        elif 'material' in query_lower:
            return "material data"
        elif 'vendor' in query_lower:
            return "vendor information"
        elif 'bom' in query_lower:
            return "bill of materials data"
        else:
            return "relevant data"
    
    def create_enhanced_qa_pairs(self) -> List[Dict[str, str]]:
        """Create enhanced QA pairs with variations and context"""
        base_qa_pairs = self.create_qa_pairs()
        enhanced_pairs = []
        
        for qa in base_qa_pairs:
            # Add original pair
            enhanced_pairs.append(qa)
            
            # Add variations
            variations = self.generate_query_variations(qa['question'], qa['department'])
            for variation in variations:
                enhanced_qa = qa.copy()
                enhanced_qa['question'] = variation
                enhanced_pairs.append(enhanced_qa)
        
        return enhanced_pairs
    
    def generate_query_variations(self, original_query: str, department: str) -> List[str]:
        """Generate variations of queries for better training"""
        variations = []
        
        # Common SAP prefixes/suffixes
        prefixes = [
            "I need to",
            "Can you help me",
            "Please show me",
            "I want to see",
            "How do I get",
            "Display",
            "Retrieve"
        ]
        
        suffixes = [
            "from SAP",
            "in the system",
            "from the database",
            f"from {department} module",
            "please",
            "now"
        ]
        
        # Add prefix variations
        for prefix in prefixes[:2]:  # Limit to avoid too many variations
            variations.append(f"{prefix} {original_query.lower()}")
        
        # Add suffix variations
        for suffix in suffixes[:2]:
            variations.append(f"{original_query} {suffix}")
        
        return variations

def main():
    """Main function to analyze config and generate QA dataset"""
    
    print("🚀 SAP Intent Analysis and QA Dataset Generation")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SAPIntentAnalyzer("config.yml")
    
    if not analyzer.departments_data:
        print("❌ No department data found. Please check your config.yml file.")
        return
    
    # Print analysis report
    analyzer.print_analysis_report()
    
    # Generate QA dataset
    print(f"\n📝 Generating QA Dataset...")
    generator = SAPQADatasetGenerator(analyzer)
    
    # Create basic QA pairs
    basic_qa_pairs = generator.create_qa_pairs()
    print(f"✅ Created {len(basic_qa_pairs)} basic QA pairs")
    
    # Create enhanced QA pairs with variations
    enhanced_qa_pairs = generator.create_enhanced_qa_pairs()
    print(f"✅ Created {len(enhanced_qa_pairs)} enhanced QA pairs (with variations)")
    
    # Save datasets
    datasets = {
        'sap_basic_qa_dataset.json': basic_qa_pairs,
        'sap_enhanced_qa_dataset.json': enhanced_qa_pairs
    }
    
    for filename, data in datasets.items():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(data)} QA pairs to: {filename}")
    
    # Show sample QA pairs
    print(f"\n📋 Sample QA Pairs:")
    print("-" * 40)
    for i, qa in enumerate(basic_qa_pairs[:3]):
        print(f"\n{i+1}. Question: {qa['question']}")
        print(f"   Department: {qa['department']}")
        print(f"   Intent: {qa['intent_name']}")
        print(f"   Endpoint: {qa['endpoint'][:80]}{'...' if len(qa['endpoint']) > 80 else ''}")
    
    print(f"\n🎯 Next Steps:")
    print("1. Review the generated QA datasets")
    print("2. Run the SAP fine-tuning script")
    print("3. Test the endpoint prediction system")
    
    return basic_qa_pairs, enhanced_qa_pairs

if __name__ == "__main__":
    basic_qa, enhanced_qa = main()
