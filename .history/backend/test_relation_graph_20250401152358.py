"""
Debug script to test the relations_graph endpoint.
"""

import sys
import requests
import json

def test_relation_graph(word_id):
    """Test the relations graph endpoint for a given word ID."""
    url = f"http://localhost:10000/api/v2/words/{word_id}/relations/graph"
    
    try:
        response = requests.get(url)
        
        # Print status code
        print(f"Status code: {response.status_code}")
        
        # Print response
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Graph contains {len(result.get('nodes', []))} nodes and {len(result.get('edges', []))} edges")
            
            # Print summary of nodes and edges
            if 'nodes' in result and result['nodes']:
                print("\nSample nodes:")
                for node in result['nodes'][:3]:  # Print first 3 nodes
                    print(f"  - {node.get('label')} (ID: {node.get('id')})")
                    
            if 'edges' in result and result['edges']:
                print("\nSample edges:")
                for edge in result['edges'][:3]:  # Print first 3 edges
                    print(f"  - {edge.get('source')} → {edge.get('target')} ({edge.get('type')})")
                    
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_relation_graph.py <word_id>")
        sys.exit(1)
        
    word_id = sys.argv[1]
    test_relation_graph(word_id) 