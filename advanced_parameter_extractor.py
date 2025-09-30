"""
Advanced SAP Parameter Extraction using Parameter Config
This system uses the parameter_config.yml to properly extract SAP field parameters
"""

import yaml
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class AdvancedSAPParameterExtractor:
    """Advanced parameter extraction using parameter configuration"""
    
    def __init__(self, config_path: str = "parameter_config.yml"):
        self.config_path = config_path
        self.parameter_config = self.load_parameter_config()
        self.field_mappings = self.build_field_mappings()
        
    def load_parameter_config(self) -> Dict[str, Any]:
        """Load parameter configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded parameter config from: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading parameter config: {e}")
            return {}
    
    def build_field_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Build field mappings from config"""
        mappings = {}
        
        if 'parameter_mapping' not in self.parameter_config:
            return mappings
        
        po_params = self.parameter_config['parameter_mapping'].get('po_materials_parameters', {})
        
        for field_name, field_config in po_params.items():
            mappings[field_name] = {
                'odata_field': field_config.get('odata_field', field_name),
                'synonyms': field_config.get('synonyms', []),
                'patterns': field_config.get('patterns', []),
                'data_type': field_config.get('data_type', 'string'),
                'example_values': field_config.get('example_values', [])
            }
        
        print(f"✅ Built mappings for {len(mappings)} SAP fields")
        return mappings
    
    def extract_parameters_from_query(self, query: str) -> Dict[str, Any]:
        """Extract parameters using the configuration"""
        
        query_lower = query.lower()
        extracted_params = {}
        extraction_details = {}
        
        print(f"🔍 Analyzing query: '{query}'")
        
        # Process each configured field
        for field_name, field_config in self.field_mappings.items():
            param_result = self.extract_field_parameter(query, query_lower, field_name, field_config)
            
            if param_result:
                extracted_params[field_config['odata_field']] = param_result['value']
                extraction_details[field_name] = param_result
                print(f"  ✓ Found {field_name}: {param_result['value']} (via {param_result['method']})")
        
        return {
            'parameters': extracted_params,
            'details': extraction_details
        }
    
    def extract_field_parameter(self, query: str, query_lower: str, field_name: str, field_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a specific field parameter"""
        
        # Method 1: Try configured patterns
        for pattern in field_config['patterns']:
            result = self.try_pattern(query, query_lower, pattern, field_config['data_type'])
            if result:
                return {
                    'value': result,
                    'method': f'pattern: {pattern}',
                    'confidence': 0.9
                }
        
        # Method 2: Try synonyms with common operators
        for synonym in field_config['synonyms']:
            result = self.try_synonym_extraction(query, query_lower, synonym, field_config['data_type'])
            if result:
                return {
                    'value': result,
                    'method': f'synonym: {synonym}',
                    'confidence': 0.8
                }
        
        # Method 3: Try contextual extraction
        result = self.try_contextual_extraction(query, query_lower, field_name, field_config)
        if result:
            return {
                'value': result,
                'method': 'contextual',
                'confidence': 0.7
            }
        
        return None
    
    def try_pattern(self, query: str, query_lower: str, pattern: str, data_type: str) -> Optional[str]:
        """Try to match a specific pattern"""
        
        # Convert pattern to regex
        # Replace {value} with appropriate regex based on data type
        if data_type == 'string':
            value_regex = r'([A-Za-z0-9_\-\.]+|"[^"]+"|\'[^\']+\')'
        elif data_type == 'decimal':
            value_regex = r'([0-9]+\.?[0-9]*)'
        elif data_type == 'date':
            value_regex = r'(\d{4}-\d{2}-\d{2}|\d{2}\/\d{2}\/\d{4})'
        else:
            value_regex = r'([A-Za-z0-9_\-\.]+)'
        
        regex_pattern = pattern.replace('{value}', value_regex)
        regex_pattern = regex_pattern.replace('(', r'\(').replace(')', r'\)')
        regex_pattern = regex_pattern.replace('|', '|')
        
        # Try case-insensitive match
        match = re.search(regex_pattern, query, re.IGNORECASE)
        if match:
            value = match.group(-1)  # Last group should be the value
            return self.clean_extracted_value(value, data_type)
        
        return None
    
    def try_synonym_extraction(self, query: str, query_lower: str, synonym: str, data_type: str) -> Optional[str]:
        """Try to extract using synonyms with common operators"""
        
        common_ops = self.parameter_config.get('common_patterns', {}).get('comparison_operators', {})
        
        # Build patterns for each operator type
        for op_type, operators in common_ops.items():
            for operator in operators:
                if data_type == 'decimal' and op_type in ['greater_than', 'less_than']:
                    # For numeric comparisons
                    pattern = f"{re.escape(synonym)}\\s+{re.escape(operator)}\\s+([0-9]+\\.?[0-9]*)"
                elif op_type in ['equals', 'contains']:
                    # For exact matches
                    if data_type == 'string':
                        pattern = f"{re.escape(synonym)}\\s+{re.escape(operator)}\\s+([A-Za-z0-9_\\-\\.]+|\"[^\"]+\"|'[^']+')"
                    elif data_type == 'decimal':
                        pattern = f"{re.escape(synonym)}\\s+{re.escape(operator)}\\s+([0-9]+\\.?[0-9]*)"
                    elif data_type == 'date':
                        pattern = f"{re.escape(synonym)}\\s+{re.escape(operator)}\\s+(\\d{{4}}-\\d{{2}}-\\d{{2}})"
                    else:
                        pattern = f"{re.escape(synonym)}\\s+{re.escape(operator)}\\s+([A-Za-z0-9_\\-\\.]+)"
                else:
                    continue
                
                match = re.search(pattern, query_lower, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    return self.clean_extracted_value(value, data_type)
        
        return None
    
    def try_contextual_extraction(self, query: str, query_lower: str, field_name: str, field_config: Dict[str, Any]) -> Optional[str]:
        """Try contextual extraction based on field type and examples"""
        
        # Look for example values in the query
        for example in field_config.get('example_values', []):
            if str(example).lower() in query_lower:
                return str(example)
        
        # Field-specific contextual extraction
        if field_name == 'Salary' and field_config['data_type'] == 'decimal':
            # Look for salary patterns
            salary_patterns = [
                r'salary.*?([0-9]{4,6})',
                r'([0-9]{4,6}).*?salary',
                r'pay.*?([0-9]{4,6})',
                r'([0-9]{4,6}).*?pay',
                r'earn.*?([0-9]{4,6})',
                r'([0-9]{4,6}).*?earn'
            ]
            
            for pattern in salary_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    return match.group(1)
        
        elif field_name == 'Department':
            # Look for common department names
            dept_keywords = ['IT', 'HR', 'Finance', 'Sales', 'Marketing', 'Operations', 'Engineering', 'Support']
            for dept in dept_keywords:
                if dept.lower() in query_lower:
                    return dept
        
        return None
    
    def clean_extracted_value(self, value: str, data_type: str) -> str:
        """Clean and format extracted value"""
        
        # Remove quotes
        value = value.strip('\'"')
        
        if data_type == 'decimal':
            # Ensure it's a valid number
            try:
                float(value)
                return value
            except ValueError:
                return None
        elif data_type == 'date':
            # Validate date format
            if re.match(r'\d{4}-\d{2}-\d{2}', value):
                return value
            elif re.match(r'\d{2}/\d{2}/\d{4}', value):
                # Convert MM/DD/YYYY to YYYY-MM-DD
                parts = value.split('/')
                return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
        
        return value
    
    def build_odata_filter(self, parameters: Dict[str, str], query: str) -> str:
        """Build OData filter string from extracted parameters"""
        
        if not parameters:
            return ""
        
        filter_conditions = []
        query_lower = query.lower()
        
        for field, value in parameters.items():
            # Determine the operator based on query context
            operator = "eq"  # Default
            
            if any(op in query_lower for op in ['greater than', '>', 'more than', 'above']):
                operator = "gt"
            elif any(op in query_lower for op in ['less than', '<', 'below', 'under']):
                operator = "lt"
            elif any(op in query_lower for op in ['contains', 'includes', 'like']):
                operator = "contains"
            
            # Format condition based on data type
            field_config = None
            for fname, fconfig in self.field_mappings.items():
                if fconfig['odata_field'] == field:
                    field_config = fconfig
                    break
            
            if field_config and field_config['data_type'] == 'string':
                if operator == "contains":
                    filter_conditions.append(f"contains({field}, '{value}')")
                else:
                    filter_conditions.append(f"{field} {operator} '{value}'")
            else:
                filter_conditions.append(f"{field} {operator} {value}")
        
        return " and ".join(filter_conditions)
    
    def process_complete_query(self, query: str, base_endpoint: str = None) -> Dict[str, Any]:
        """Process complete query with parameter extraction and URL generation"""
        
        print(f"\n🔍 Processing Complete Query")
        print("=" * 50)
        print(f"Query: '{query}'")
        
        # Extract parameters
        extraction_result = self.extract_parameters_from_query(query)
        parameters = extraction_result['parameters']
        details = extraction_result['details']
        
        if not parameters:
            print("📋 No parameters extracted")
            return {
                'success': True,
                'query': query,
                'parameters': {},
                'filter_string': "",
                'complete_url': base_endpoint or "",
                'message': "No parameters found - returning base endpoint"
            }
        
        print(f"📋 Extracted {len(parameters)} parameters:")
        for field, value in parameters.items():
            print(f"  • {field}: {value}")
        
        # Build OData filter
        filter_string = self.build_odata_filter(parameters, query)
        print(f"🔧 OData Filter: {filter_string}")
        
        # Generate complete URL
        complete_url = base_endpoint or ""
        if filter_string and base_endpoint:
            if '?' in base_endpoint:
                if '$filter=' in base_endpoint:
                    # Replace existing filter
                    complete_url = re.sub(r'\$filter=[^&]*', f'$filter={filter_string}', base_endpoint)
                else:
                    complete_url = f"{base_endpoint}&$filter={filter_string}"
            else:
                complete_url = f"{base_endpoint}?$filter={filter_string}"
        
        print(f"🎯 Complete URL: {complete_url}")
        
        return {
            'success': True,
            'query': query,
            'parameters': parameters,
            'extraction_details': details,
            'filter_string': filter_string,
            'complete_url': complete_url,
            'parameter_count': len(parameters)
        }

def demo_advanced_extraction():
    """Demo the advanced parameter extraction system"""
    
    print("🚀 Advanced SAP Parameter Extraction Demo")
    print("Using parameter_config.yml for intelligent extraction")
    print("=" * 70)
    
    # Initialize extractor
    extractor = AdvancedSAPParameterExtractor()
    
    if not extractor.parameter_config:
        print("❌ Could not load parameter config. Please check parameter_config.yml")
        return
    
    # Test queries covering different scenarios
    test_queries = [
        # Salary queries (the problematic one)
        "i want po where salary is greater than 90000",
        "show purchase orders where salary equals 75000",
        "get records where pay is more than 50000",
        
        # Purchase Order queries
        "show purchase order 4500000001",
        "get po number 4500000002",
        "display purchase order equals PO12345",
        
        # Vendor queries
        "filter by vendor id 100001",
        "show records for supplier VEN001",
        "get data where vendor name is 'ABC Corp'",
        
        # Material queries
        "show material MAT001",
        "get records for part number PART123",
        "filter by material number MAT002",
        
        # Complex queries
        "show purchase orders where vendor id is 100001 and material equals MAT001",
        "get records where salary is greater than 60000 and department is IT",
        "filter by plant 1000 and quantity greater than 100",
        
        # Department queries
        "show employees in IT department",
        "get records where dept equals Finance",
        "filter by division Sales"
    ]
    
    base_endpoint = "/sap/opu/odata/sap/Z_CDS_VIEW_ALL_CDS/Z_Cds_View_All?$format=json&$top=5"
    
    print(f"Testing {len(test_queries)} queries:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*80}")
        print(f"TEST {i}: {query}")
        print(f"{'='*80}")
        
        result = extractor.process_complete_query(query, base_endpoint)
        
        if result['success']:
            print(f"\n✅ SUCCESS:")
            print(f"   Parameters: {result['parameter_count']} found")
            print(f"   Filter: {result['filter_string']}")
            print(f"   URL Ready: {'Yes' if result['complete_url'] else 'No'}")
        else:
            print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
        
        print()  # Add spacing between tests
    
    # Interactive mode
    print(f"{'='*80}")
    print("🤖 Interactive Mode - Test Your Own Queries!")
    print("Enter SAP queries to see advanced parameter extraction")
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
            
            result = extractor.process_complete_query(user_query, base_endpoint)
            
            if result['success'] and result['parameters']:
                print(f"\n🎯 EXTRACTED SUCCESSFULLY!")
                print(f"Parameters: {result['parameters']}")
                print(f"Ready-to-use URL: {result['complete_url']}")
            elif result['success']:
                print(f"\n📋 No parameters found in query")
            else:
                print(f"\n❌ Error: {result.get('error')}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_advanced_extraction()
