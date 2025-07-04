#!/usr/bin/env python3
"""
Script to inspect saved model parameters and print all edge types, hidden_dim, out_dim, and heads.
"""

import torch
import re

def inspect_model(model_path):
    """Inspect a saved model to understand its parameters."""
    print(f"\n{'='*60}")
    print(f"Inspecting model: {model_path}")
    print(f"{'='*60}")
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        
        print("Checkpoint keys:", list(checkpoint.keys()))
        
        if "model" in checkpoint:
            model_state = checkpoint["model"]
            print(f"\nModel state dict keys: {len(model_state.keys())}")
            
            # Analyze the first few keys to understand the architecture
            keys = list(model_state.keys())
            print("\nFirst 10 model keys:")
            for i, key in enumerate(keys[:10]):
                shape = model_state[key].shape
                print(f"  {key}: {shape}")
            
            # Look for patterns in the architecture
            conv_keys = [k for k in keys if "convs." in k]
            print(f"\nFound {len(conv_keys)} conv-related keys")
            
            # Analyze layer dimensions
            layer_dims = {}
            for key in conv_keys:
                if "lin_l.weight" in key or "lin_r.weight" in key:
                    shape = model_state[key].shape
                    if "convs.0" in key:
                        layer_dims["layer0"] = shape
                    elif "convs.1" in key:
                        layer_dims["layer1"] = shape
            
            print("\nLayer dimensions:")
            for layer, shape in layer_dims.items():
                print(f"  {layer}: {shape}")
            
            # Try to infer hidden dimensions
            if "layer0" in layer_dims:
                layer0_shape = layer_dims["layer0"]
                if len(layer0_shape) == 2:
                    print(f"\nInferred hidden_dim: {layer0_shape[0]}")
                    print(f"Inferred input_dim: {layer0_shape[1]}")
            
            if "layer1" in layer_dims:
                layer1_shape = layer_dims["layer1"]
                if len(layer1_shape) == 2:
                    print(f"Inferred output_dim: {layer1_shape[0]}")
        
        if "link_predictor" in checkpoint:
            link_state = checkpoint["link_predictor"]
            print(f"\nLink predictor keys: {list(link_state.keys())}")
            for key, tensor in link_state.items():
                print(f"  {key}: {tensor.shape}")
        
        # Extract edge types
        edge_types = set()
        for k in keys:
            m = re.match(r"convs\.\d+\.convs\.<(.+?)>", k)
            if m:
                edge = m.group(1)
                # Format: SRC___REL___DST
                parts = edge.split("___")
                if len(parts) == 3:
                    edge_types.add((parts[0], parts[1], parts[2]))
        print(f"\nEdge types in checkpoint ({len(edge_types)}):")
        for et in sorted(edge_types):
            print(f"  {et}")
        
        # Extract hidden_dim, out_dim, heads
        hidden_dim = None
        out_dim = None
        heads = None
        for k, v in model_state.items():
            if ".att" in k and len(v.shape) == 3:
                # att: [1, heads, hidden_dim//heads]
                heads = v.shape[1]
                hidden_dim = v.shape[1] * v.shape[2]
            if ".lin1.weight" in k and len(v.shape) == 2:
                out_dim = v.shape[0]
        
        print(f"\nInferred hidden_dim: {hidden_dim}")
        print(f"Inferred out_dim: {out_dim}")
        print(f"Inferred heads: {heads}")
        
        return {
            "edge_types": list(edge_types),
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "heads": heads
        }
        
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def main():
    models_to_inspect = [
        "gatv2_model.pt",
        "sage_model.pt",
        "gatv2_enhanced.pt",
        "sage_enhanced.pt"
    ]
    
    for model_path in models_to_inspect:
        inspect_model(model_path)

if __name__ == "__main__":
    main() 