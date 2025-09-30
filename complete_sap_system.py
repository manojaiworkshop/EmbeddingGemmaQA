"""
SAP Parameter Extraction and URL Generation System
This system extracts parameters from user queries and generates complete OData URLs
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from improved_sap_predictor import ImprovedSAPPredictor

class SAPParameterExtractor:
    """Extract parameters from SAP queries and generate complete OData URLs"""
    
    def __init__(self):
        # Parameter patterns for different SAP entities
        self.parameter_patterns = {
            # MM Department parameters
            'PurchaseOrder': {
                'patterns': [r'purchase\s*order\s*(\w+)', r'po\s*(\w+)', r'order\s*(\w+)'],
                'field': 'PurchaseOrder'
            },
            'VendorId': {
                'patterns': [r'vendor\s*(?:id\s*)?(\w+)', r'supplier\s*(\w+)'],
                'field': 'VendorId'
            },
            'VendorName': {
                'patterns': [r'vendor\s*name\s*["\']([^"\']+)["\']', r'supplier\s*name\s*["\']([^"\']+)["\']'],
                'field': 'VendorName'
            },
            'Material': {
                'patterns': [r'material\s*(\w+)', r'mat\s*(\w+)', r'item\s*(\w+)'],
                'field': 'Material'
            },
            'Plant': {
                'patterns': [r'plant\s*(\w+)', r'location\s*(\w+)'],
                'field': 'Plant'
            },
            'DocType': {
                'patterns': [r'doc\s*type\s*(\w+)', r'document\s*type\s*(\w+)'],
                'field': 'DocType'
            },
            'PurchasingOrg': {
                'patterns': [r'purchasing\s*org\s*(\w+)', r'porg\s*(\w+)'],
                'field': 'PurchasingOrg'
            },
            'PurchasingGroup': {
                'patterns': [r'purchasing\s*group\s*(\w+)', r'pgroup\s*(\w+)'],
                'field': 'PurchasingGroup'
            },
            
            # FI Department parameters
            'codid': {
                'patterns': [r'codid\s*(\w+)', r'code\s*id\s*(\w+)', r'code\s*(\w+)'],
                'field': 'codid'
            },
            'codva': {
                'patterns': [r'codva\s*(\w+)', r'code\s*value\s*(\w+)', r'value\s*(\w+)'],
                'field': 'codva'
            },
            
            # SD Department parameters
            'region': {
                'patterns': [r'region\s*(\w+)', r'area\s*(\w+)', r'zone\s*(\w+)'],
                'field': 'region'
            },
            'customer': {
                'patterns': [r'customer\s*(\w+)', r'client\s*(\w+)'],
                'field': 'customer'
            },
            
            # PP Department parameters
            'material_bom': {
                'patterns': [r'material\s*(\w+)', r'mat\s*(\w+)', r'part\s*(\w+)'],
                'field': 'material'
            }
        }
        
        # Value patterns (quoted strings, specific values)
        self.value_patterns = {
            'quoted_string': r'["\']([^"\']+)["\']',
            'equals_value': r'(?:equals?|is|=)\s*([^\s,]+)',
            'specific_values': r'(?:NORTH|SOUTH|EAST|WEST|C001|C002|C003|VALUE_01|VALUE_02|MAT001|MAT002)',
            'alphanumeric': r'([A-Z0-9_]+)',
            'date': r'(\d{4}-\d{2}-\d{2})'
        }
        
    def extract_parameters_from_query(self, query: str, department: str) -> Dict[str, Any]:
        """Extract parameters from user query based on department context"""
        
        query_lower = query.lower()
        extracted_params = {}
        
        # Extract based on department context
        if department == 'MM':
            extracted_params.update(self._extract_mm_parameters(query, query_lower))
        elif department == 'FI':
            extracted_params.update(self._extract_fi_parameters(query, query_lower))
        elif department == 'SD':
            extracted_params.update(self._extract_sd_parameters(query, query_lower))
        elif department == 'PP':
            extracted_params.update(self._extract_pp_parameters(query, query_lower))
        
        # Extract general filter conditions
        extracted_params.update(self._extract_filter_conditions(query, query_lower))
        
        return extracted_params
    
    def _extract_mm_parameters(self, query: str, query_lower: str) -> Dict[str, str]:
        """Extract MM-specific parameters"""
        params = {}
        
        # Purchase Order
        for pattern in self.parameter_patterns['PurchaseOrder']['patterns']:
            match = re.search(pattern, query_lower)
            if match:
                params['PurchaseOrder'] = match.group(1).upper()
                break
        
        # Vendor ID
        for pattern in self.parameter_patterns['VendorId']['patterns']:
            match = re.search(pattern, query_lower)
            if match:
                params['VendorId'] = match.group(1).upper()
                break
        
        # Vendor Name (with quotes)
        vendor_name_match = re.search(r'vendor\s*name\s*["\']([^"\']+)["\']', query_lower)
        if vendor_name_match:
            params['VendorName'] = vendor_name_match.group(1)
        
        # Material
        material_match = re.search(r'material\s*([A-Z0-9_]+)', query, re.IGNORECASE)
        if material_match:
            params['Material'] = material_match.group(1).upper()
        
        # Plant
        plant_match = re.search(r'plant\s*([A-Z0-9_]+)', query, re.IGNORECASE)
        if plant_match:
            params['Plant'] = plant_match.group(1).upper()
        
        # Purchasing Organization
        porg_match = re.search(r'purchasing\s*org(?:anization)?\s*([A-Z0-9_]+)', query, re.IGNORECASE)
        if porg_match:
            params['PurchasingOrg'] = porg_match.group(1).upper()
        
        # Purchasing Group
        pgroup_match = re.search(r'purchasing\s*group\s*([A-Z0-9_]+)', query, re.IGNORECASE)
        if pgroup_match:
            params['PurchasingGroup'] = pgroup_match.group(1).upper()
        
        return params
    
    def _extract_fi_parameters(self, query: str, query_lower: str) -> Dict[str, str]:
        """Extract FI-specific parameters"""
        params = {}
        
        # CODID
        codid_match = re.search(r'codid\s*([A-Z0-9_]+)', query, re.IGNORECASE)
        if codid_match:
            params['codid'] = codid_match.group(1).upper()
        
        # CODVA
        codva_match = re.search(r'codva\s*([A-Z0-9_]+)', query, re.IGNORECASE)
        if codva_match:
            params['codva'] = codva_match.group(1).upper()
        
        # Look for specific known values
        if 'c001' in query_lower:
            params['codid'] = 'C001'
        if 'value_01' in query_lower:
            params['codva'] = 'VALUE_01'
        if 'value_02' in query_lower:
            params['codva'] = 'VALUE_02'
        
        return params
    
    def _extract_sd_parameters(self, query: str, query_lower: str) -> Dict[str, str]:
        """Extract SD-specific parameters"""
        params = {}
        
        # Region
        region_match = re.search(r'region\s*([A-Z]+)', query, re.IGNORECASE)
        if region_match:
            params['region'] = region_match.group(1).upper()
        
        # Look for specific regions
        if 'north' in query_lower:
            params['region'] = 'NORTH'
        elif 'south' in query_lower:
            params['region'] = 'SOUTH'
        elif 'east' in query_lower:
            params['region'] = 'EAST'
        elif 'west' in query_lower:
            params['region'] = 'WEST'
        
        # Customer
        customer_match = re.search(r'customer\s*([A-Z0-9_]+)', query, re.IGNORECASE)
        if customer_match:
            params['customer'] = customer_match.group(1).upper()
        
        return params
    
    def _extract_pp_parameters(self, query: str, query_lower: str) -> Dict[str, str]:
        """Extract PP-specific parameters"""
        params = {}
        
        # Material for BOM
        material_match = re.search(r'material\s*([A-Z0-9_]+)', query, re.IGNORECASE)
        if material_match:
            params['material'] = material_match.group(1).upper()
        
        # Look for specific materials
        if 'mat001' in query_lower:
            params['material'] = 'MAT001'
        elif 'mat002' in query_lower:
            params['material'] = 'MAT002'
        
        return params
    
    def _extract_filter_conditions(self, query: str, query_lower: str) -> Dict[str, Any]:
        """Extract general filter conditions"""
        params = {}
        
        # Date ranges
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
        if date_match:
            params['date'] = date_match.group(1)
        
        # Quoted values
        quoted_match = re.search(r'["\']([^"\']+)["\']', query)
        if quoted_match:
            params['quoted_value'] = quoted_match.group(1)
        
        # Numeric values
        numeric_match = re.search(r'(?:quantity|price|amount)\s*(\d+(?:\.\d+)?)', query_lower)
        if numeric_match:
            params['numeric_value'] = float(numeric_match.group(1))
        
        return params
    
    def generate_odata_url(self, base_endpoint: str, parameters: Dict[str, Any], department: str) -> str:
        """Generate complete OData URL with parameters"""
        
        # If no parameters extracted, return base endpoint
        if not parameters:
            return base_endpoint
        
        # Handle different endpoint types
        if '?$filter=' in base_endpoint:
            # Template endpoint with filter placeholder
            return self._fill_filter_template(base_endpoint, parameters, department)
        elif '?$format=json' in base_endpoint:
            # Base endpoint, add filter
            return self._add_filter_to_base(base_endpoint, parameters, department)
        else:
            # Other endpoint types
            return self._handle_special_endpoints(base_endpoint, parameters, department)
    
    def _fill_filter_template(self, template: str, parameters: Dict[str, Any], department: str) -> str:
        """Fill template endpoint with extracted parameters"""
        
        filled_url = template
        
        if department == 'MM':
            # Handle MM filter templates
            if 'PurchaseOrder' in parameters:
                condition = f"PurchaseOrder eq '{parameters['PurchaseOrder']}'"
                filled_url = template.replace('{condition}', condition)
            elif 'VendorId' in parameters:
                condition = f"VendorId eq '{parameters['VendorId']}'"
                filled_url = template.replace('{condition}', condition)
            elif 'Material' in parameters:
                condition = f"Material eq '{parameters['Material']}'"
                filled_url = template.replace('{condition}', condition)
            elif 'VendorName' in parameters:
                condition = f"VendorName eq '{parameters['VendorName']}'"
                filled_url = template.replace('{condition}', condition)
        
        elif department == 'FI':
            # Handle FI filter templates
            if 'codid' in parameters and 'codva' in parameters:
                condition = f"codid eq '{parameters['codid']}' and codva eq '{parameters['codva']}'"
                filled_url = template.replace("codid eq '{codid}' and codva eq '{codva}'", condition)
            elif 'codid' in parameters:
                condition = f"codid eq '{parameters['codid']}'"
                filled_url = template.replace('{codid}', parameters['codid'])
        
        elif department == 'SD':
            # Handle SD filter templates
            if 'region' in parameters:
                filled_url = template.replace('{region}', parameters['region'])
        
        elif department == 'PP':
            # Handle PP filter templates
            if 'material' in parameters:
                filled_url = template.replace('{material}', parameters['material'])
        
        return filled_url
    
    def _add_filter_to_base(self, base_url: str, parameters: Dict[str, Any], department: str) -> str:
        """Add filter parameters to base URL"""
        
        filters = []
        
        if department == 'MM':
            if 'PurchaseOrder' in parameters:
                filters.append(f"PurchaseOrder eq '{parameters['PurchaseOrder']}'")
            if 'VendorId' in parameters:
                filters.append(f"VendorId eq '{parameters['VendorId']}'")
            if 'Material' in parameters:
                filters.append(f"Material eq '{parameters['Material']}'")
            if 'Plant' in parameters:
                filters.append(f"Plant eq '{parameters['Plant']}'")
        
        elif department == 'FI':
            if 'codid' in parameters:
                filters.append(f"codid eq '{parameters['codid']}'")
            if 'codva' in parameters:
                filters.append(f"codva eq '{parameters['codva']}'")
        
        elif department == 'SD':
            if 'region' in parameters:
                filters.append(f"region eq '{parameters['region']}'")
            if 'customer' in parameters:
                filters.append(f"customer eq '{parameters['customer']}'")
        
        elif department == 'PP':
            if 'material' in parameters:
                filters.append(f"material eq '{parameters['material']}'")
        
        if filters:
            filter_string = ' and '.join(filters)
            # Insert filter before $format=json
            if '$format=json' in base_url:
                base_url = base_url.replace('?$format=json', f'?$filter={filter_string}&$format=json')
            elif '?' in base_url:
                base_url += f'&$filter={filter_string}'
            else:
                base_url += f'?$filter={filter_string}'
        
        return base_url
    
    def _handle_special_endpoints(self, endpoint: str, parameters: Dict[str, Any], department: str) -> str:
        """Handle special endpoint cases"""
        
        # For endpoints that already have specific parameters
        filled_endpoint = endpoint
        
        # Replace placeholders in the endpoint
        for param_name, param_value in parameters.items():
            placeholder = f'{{{param_name}}}'
            if placeholder in filled_endpoint:
                filled_endpoint = filled_endpoint.replace(placeholder, str(param_value))
        
        return filled_endpoint

class CompleteSAPSystem:
    """Complete SAP system with parameter extraction and URL generation"""
    
    def __init__(self):
        self.predictor = ImprovedSAPPredictor()
        self.parameter_extractor = SAPParameterExtractor()
        self.setup_system()
    
    def setup_system(self):
        """Setup the complete system"""
        print("🔧 Setting up Complete SAP System...")
        
        # Train the predictor if not already trained
        try:
            model_path = "./models/sap_endpoint_predictor"
            if not self.predictor.load_model(model_path):
                print("🔄 Training SAP predictor...")
                self.predictor.train_improved_predictor("sap_enhanced_qa_dataset.json")
        except:
            print("🔄 Training SAP predictor...")
            self.predictor.train_improved_predictor("sap_enhanced_qa_dataset.json")
        
        print("✅ Complete SAP System ready!")
    
    def process_sap_query(self, user_query: str) -> Dict[str, Any]:
        """Process complete SAP query: predict endpoint + extract parameters + generate URL"""
        
        print(f"\n🔍 Processing query: '{user_query}'")
        print("-" * 60)
        
        # Step 1: Predict endpoint
        predictions = self.predictor.predict_endpoint_improved(user_query, top_k=1)
        
        if not predictions:
            return {
                'success': False,
                'error': 'No matching endpoint found',
                'query': user_query
            }
        
        best_prediction = predictions[0]
        department = best_prediction['department']
        base_endpoint = best_prediction['endpoint']
        confidence = best_prediction['similarity_score']
        
        print(f"📊 Predicted: {department} Department ({confidence:.1%} confidence)")
        print(f"🔗 Base endpoint: {base_endpoint}")
        
        # Step 2: Extract parameters
        parameters = self.parameter_extractor.extract_parameters_from_query(user_query, department)
        
        if parameters:
            print(f"📋 Extracted parameters: {parameters}")
        else:
            print("📋 No parameters extracted")
        
        # Step 3: Generate complete URL
        complete_url = self.parameter_extractor.generate_odata_url(base_endpoint, parameters, department)
        
        print(f"🎯 Complete URL: {complete_url}")
        
        # Determine if URL is ready to use
        has_placeholders = '{' in complete_url and '}' in complete_url
        url_status = "Ready to use" if not has_placeholders else "Needs manual parameter substitution"
        
        print(f"✅ Status: {url_status}")
        
        return {
            'success': True,
            'query': user_query,
            'department': department,
            'intent': best_prediction['intent'],
            'confidence': confidence,
            'base_endpoint': base_endpoint,
            'extracted_parameters': parameters,
            'complete_url': complete_url,
            'ready_to_use': not has_placeholders,
            'status': url_status
        }
    
    def interactive_demo(self):
        """Interactive demo of the complete system"""
        
        print(f"\n🤖 Complete SAP Query Processing System")
        print("Enter your SAP queries and get complete OData URLs!")
        print("Type 'examples' for sample queries, 'quit' to exit")
        print("=" * 70)
        
        while True:
            try:
                user_input = input("\n💼 SAP Query: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'examples':
                    self.show_examples()
                    continue
                
                elif not user_input:
                    continue
                
                # Process the query
                result = self.process_sap_query(user_input)
                
                if result['success']:
                    # Show formatted result
                    self.display_result(result)
                else:
                    print(f"❌ {result['error']}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def display_result(self, result: Dict[str, Any]):
        """Display formatted result"""
        
        print(f"\n📋 RESULT SUMMARY")
        print("-" * 30)
        print(f"Department: {result['department']}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Parameters: {len(result['extracted_parameters'])} found")
        
        if result['extracted_parameters']:
            for param, value in result['extracted_parameters'].items():
                print(f"  • {param}: {value}")
        
        print(f"\n🔗 COMPLETE URL:")
        print(result['complete_url'])
        
        status_emoji = "🟢" if result['ready_to_use'] else "🟡"
        print(f"\n{status_emoji} Status: {result['status']}")
    
    def show_examples(self):
        """Show example queries with parameters"""
        
        examples = [
            "show purchase order PO12345",
            "get materials for vendor VEN001", 
            "filter by vendor name 'ACME Corp'",
            "show financial records for code C001 and value VALUE_01",
            "get customers from NORTH region",
            "display BOM for material MAT001",
            "show purchase orders for plant P001",
            "filter sales by customer CUST123"
        ]
        
        print(f"\n💡 Example SAP Queries with Parameters:")
        print("-" * 50)
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")

def main():
    """Main function"""
    
    print("🚀 Complete SAP Query Processing System")
    print("Endpoint Prediction + Parameter Extraction + URL Generation")
    print("=" * 70)
    
    # Initialize system
    system = CompleteSAPSystem()
    
    # Run interactive demo
    system.interactive_demo()

if __name__ == "__main__":
    main()
