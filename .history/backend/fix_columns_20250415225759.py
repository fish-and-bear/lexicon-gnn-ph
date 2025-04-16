#!/usr/bin/env python
"""
Fix database schema issue in routes.py.
Removes references to the completeness_score column which doesn't exist in the database.
"""

import re
import os
import shutil

def fix_routes_file():
    print("Fixing database schema issue in routes.py...")
    
    # Create a backup first
    if os.path.exists('routes.py'):
        shutil.copy('routes.py', 'routes.py.bak')
        print("Created backup: routes.py.bak")
    else:
        print("Error: routes.py not found")
        return
    
    with open('routes.py', 'r') as f:
        content = f.read()
    
    # Fix 1: Remove completeness_score from SELECT statement
    fixed = content.replace('w.has_baybayin, w.baybayin_form, w.completeness_score', 
                          'w.has_baybayin, w.baybayin_form')
    
    # Fix 2: Replace ORDER BY completeness_score with ORDER BY lemma
    fixed = fixed.replace('ORDER BY w.completeness_score', 'ORDER BY w.lemma')
    
    # Fix 3: Fix references to row[6] since completeness_score is not in results anymore
    fixed = fixed.replace("'completeness_score': row[6] if len(row) > 6 else None", 
                        "'completeness_score': None")
    
    # Write the fixed file
    with open('routes.fixed.py', 'w') as f:
        f.write(fixed)
    
    print("Fixed file saved as routes.fixed.py")
    print("To apply the fix, run: cp routes.fixed.py routes.py")

if __name__ == "__main__":
    fix_routes_file() 