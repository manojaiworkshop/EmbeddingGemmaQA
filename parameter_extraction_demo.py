"""
SAP Parameter Extraction Demo - Standalone Version
This demonstrates parameter extraction and URL generation for SAP queries
"""

import re
import json
from typing import Dict, List, Any, Optional

class SAPParameterExtractor:
    """Extract parameters from SAP queries and generate complete OData URLs"""
    
    def __init__(self):
        # Load the enhanced QA dataset to understand patterns
        self.load_sap_patterns()
        
    def load_sap_patterns(self):
        """Load SAP patterns from dataset"""
        try:
            with open('sap_enhanced_qa_dataset.json', 'r') as f:
                self.sap_data = json.load(f)
            
            # Extract unique endpoints by department
            self.endpoints_by_dept = {}
            for item in self.sap_data:
                dept = item['department']
                if dept not in self.endpoints_by_dept:
                    self.endpoints_by_dept[dept] = []
                if item['endpoint'] not in [e['endpoint'] for e in self.endpoints_by_dept[dept]]:
                    self.endpoints_by_dept[dept].append({
                        'endpoint': item['endpoint'],
                        'intent': item['intent_name']
                    })
            
            print(f"✅ Loaded {len(self.sap_data)} SAP patterns")
            
        except Exception as e:
            print(f"❌ Error loading patterns: {e}")
            self.sap_data = []
            self.endpoints_by_dept = {}
    
    def extract_parameters_from_query(self, query: str) -> Dict[str, Any]:
        """Extract parameters from user query"""
        
        query_lower = query.lower()
        extracted_params = {}
        
        # Purchase Order patterns
        po_patterns = [
            r'purchase\s*order\s*([A-Z0-9]+)',
            r'\bpo\s*([A-Z0-9]+)',
            r'order\s*([A-Z0-9]+)'
        ]
        
        for pattern in po_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted_params['PurchaseOrder'] = match.group(1).upper()
                break
        
        # Vendor patterns
        vendor_patterns = [
            r'vendor\s*(?:id\s*)?([A-Z0-9]+)',
            r'supplier\s*([A-Z0-9]+)',
            r'vendor\s*name\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in vendor_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if 'name' in pattern:
                    extracted_params['VendorName'] = match.group(1)
                else:
                    extracted_params['VendorId'] = match.group(1).upper()
                break
        
        # Material patterns
        material_match = re.search(r'material\s*([A-Z0-9]+)', query, re.IGNORECASE)
        if material_match:
            extracted_params['Material'] = material_match.group(1).upper()
        
        # Plant patterns
        plant_match = re.search(r'plant\s*([A-Z0-9]+)', query, re.IGNORECASE)
        if plant_match:
            extracted_params['Plant'] = plant_match.group(1).upper()
        
        # Financial codes
        codid_match = re.search(r'(?:codid|code)\s*([A-Z0-9]+)', query, re.IGNORECASE)
        if codid_match:
            extracted_params['codid'] = codid_match.group(1).upper()
        
        codva_match = re.search(r'(?:codva|value)\s*([A-Z0-9_]+)', query, re.IGNORECASE)
        if codva_match:
            extracted_params['codva'] = codva_match.group(1).upper()
        
        # Region patterns
        region_patterns = ['NORTH', 'SOUTH', 'EAST', 'WEST']
        for region in region_patterns:
            if region.lower() in query_lower:
                extracted_params['region'] = region
                break
        
        # Customer patterns
        customer_match = re.search(r'customer\s*([A-Z0-9]+)', query, re.IGNORECASE)
        if customer_match:
            extracted_params['customer'] = customer_match.group(1).upper()
        
        # Known specific values
        specific_values = {
            'c001': 'codid',
            'c002': 'codid', 
            'c003': 'codid',
            'value_01': 'codva',
            'value_02': 'codva',
            'mat001': 'Material',
            'mat002': 'Material',
            'ven001': 'VendorId',
            'ven002': 'VendorId'
        }
        
        for value, param in specific_values.items():
            if value in query_lower:
                extracted_params[param] = value.upper()
        
        return extracted_params
    
    def predict_department_and_endpoint(self, query: str) -> Dict[str, Any]:
        """Simple rule-based department prediction"""
        
        query_lower = query.lower()
        
        # Department keywords
        dept_keywords = {
            'MM': ['purchase', 'po', 'vendor', 'material', 'procurement', 'buying', 'supplier'],
            'FI': ['financial', 'codid', 'codva', 'accounting', 'finance', 'code'],
            'SD': ['sales', 'customer', 'region', 'distribution'],
            'PP': ['production', 'bom', 'manufacturing', 'plant']
        }
        
        # Score each department
        dept_scores = {}
        for dept, keywords in dept_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                dept_scores[dept] = score
        
        if not dept_scores:
            return {'department': 'MM', 'confidence': 0.1, 'endpoint': None}
        
        # Get best department
        best_dept = max(dept_scores.items(), key=lambda x: x[1])
        department = best_dept[0]
        confidence = best_dept[1] / len(dept_keywords[department])
        
        # Get endpoint for department
        endpoint = None
        if department in self.endpoints_by_dept and self.endpoints_by_dept[department]:
            # Simple logic to pick endpoint based on query type
            if any(word in query_lower for word in ['filter', 'by', 'where']):
                # Look for conditional endpoint
                for ep in self.endpoints_by_dept[department]:
                    if 'filter' in ep['intent'] or 'conditional' in ep['intent']:
                        endpoint = ep['endpoint']
                        break
            
            if not endpoint:
                # Default to first endpoint
                endpoint = self.endpoints_by_dept[department][0]['endpoint']
        
        return {
            'department': department,
            'confidence': confidence,
            'endpoint': endpoint
        }
    
    def generate_complete_url(self, base_endpoint: str, parameters: Dict[str, Any], department: str) -> str:
        """Generate complete OData URL with parameters"""
        
        if not parameters:
            return base_endpoint
        
        # Handle template endpoints with placeholders
        if '{condition}' in base_endpoint:
            return self._fill_condition_template(base_endpoint, parameters)
        elif '{' in base_endpoint and '}' in base_endpoint:
            return self._fill_parameter_template(base_endpoint, parameters)
        else:
            return self._add_filter_parameters(base_endpoint, parameters)
    
    def _fill_condition_template(self, template: str, parameters: Dict[str, Any]) -> str:
        """Fill condition template with parameters"""
        
        conditions = []
        
        for param, value in parameters.items():
            if param == 'PurchaseOrder':
                conditions.append(f"PurchaseOrder eq '{value}'")
            elif param == 'VendorId':
                conditions.append(f"VendorId eq '{value}'")
            elif param == 'VendorName':
                conditions.append(f"VendorName eq '{value}'")
            elif param == 'Material':
                conditions.append(f"Material eq '{value}'")
            elif param == 'Plant':
                conditions.append(f"Plant eq '{value}'")
            elif param == 'codid':
                conditions.append(f"codid eq '{value}'")
            elif param == 'codva':
                conditions.append(f"codva eq '{value}'")
            elif param == 'region':
                conditions.append(f"region eq '{value}'")
            elif param == 'customer':
                conditions.append(f"customer eq '{value}'")
        
        if conditions:
            condition_string = ' and '.join(conditions)
            return template.replace('{condition}', condition_string)
        
        return template
    
    def _fill_parameter_template(self, template: str, parameters: Dict[str, Any]) -> str:
        """Fill parameter template placeholders"""
        
        filled_template = template
        
        for param, value in parameters.items():
            placeholder = '{' + param + '}'
            if placeholder in filled_template:
                filled_template = filled_template.replace(placeholder, str(value))
        
        return filled_template
    
    def _add_filter_parameters(self, base_url: str, parameters: Dict[str, Any]) -> str:
        """Add filter parameters to base URL"""
        
        filters = []
        
        for param, value in parameters.items():
            filters.append(f"{param} eq '{value}'")
        
        if filters:
            filter_string = ' and '.join(filters)
            
            if '$format=json' in base_url:
                base_url = base_url.replace('?$format=json', f'?$filter={filter_string}&$format=json')
            elif '?' in base_url:
                base_url += f'&$filter={filter_string}'
            else:
                base_url += f'?$filter={filter_string}'
        
        return base_url
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process complete query: predict + extract + generate"""
        
        print(f"\n🔍 Processing: '{query}'")
        print("-" * 60)
        
        # Predict department and endpoint
        prediction = self.predict_department_and_endpoint(query)
        department = prediction['department']
        base_endpoint = prediction['endpoint']
        confidence = prediction['confidence']
        
        print(f"📊 Department: {department} ({confidence:.1%} confidence)")
        
        if not base_endpoint:
            print("❌ No endpoint found")
            return {'success': False, 'error': 'No endpoint found'}
        
        print(f"🔗 Base endpoint: {base_endpoint[:80]}{'...' if len(base_endpoint) > 80 else ''}")
        
        # Extract parameters
        parameters = self.extract_parameters_from_query(query)
        
        if parameters:
            print(f"📋 Parameters found: {parameters}")
        else:
            print("📋 No parameters extracted")
        
        # Generate complete URL
        complete_url = self.generate_complete_url(base_endpoint, parameters, department)
        
        print(f"🎯 Complete URL:")
        print(f"   {complete_url}")
        
        # Check if ready to use
        has_placeholders = '{' in complete_url and '}' in complete_url
        status = "✅ Ready to use" if not has_placeholders else "🟡 Needs manual substitution"
        print(f"{status}")
        
        return {
            'success': True,
            'query': query,
            'department': department,
            'confidence': confidence,
            'base_endpoint': base_endpoint,
            'parameters': parameters,
            'complete_url': complete_url,
            'ready_to_use': not has_placeholders
        }

def demo_parameter_extraction():
    """Demo the parameter extraction system"""
    
    print("🚀 SAP Parameter Extraction Demo")
    print("=" * 50)
    
    extractor = SAPParameterExtractor()
    
    # Test queries with parameters
    test_queries = [
        "show purchase order PO12345",
        "get materials for vendor VEN001",
        "filter by vendor name 'ACME Corporation'",
        "show purchase orders for plant P001",
        "display financial records for code C001 and value VALUE_01",
        "get customers from NORTH region",
        "show BOM for material MAT001",
        "filter sales by customer CUST123",
        "show all purchase orders",  # No parameters
        "get production data for material MAT002"
    ]
    
    print(f"Testing {len(test_queries)} queries with parameter extraction:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*70}")
        print(f"TEST {i}")
        result = extractor.process_query(query)
        
        if i < len(test_queries):
            print()
    
    # Interactive mode
    print(f"\n{'='*70}")
    print("🤖 Interactive Mode")
    print("Enter your SAP queries to see parameter extraction in action!")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            user_query = input("\n💼 SAP Query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_query:
                continue
            
            extractor.process_query(user_query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_parameter_extraction()
