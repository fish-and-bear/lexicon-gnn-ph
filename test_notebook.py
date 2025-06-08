import json

try:
    with open('Active_Learning_+_Explainability.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print(f"Successfully loaded notebook")
    print(f"Total cells: {len(notebook['cells'])}")
    
    # Check first few cells
    for i, cell in enumerate(notebook['cells'][:5]):
        print(f"Cell {i}: {cell['cell_type']}")
        
except Exception as e:
    print(f"Error: {e}") 