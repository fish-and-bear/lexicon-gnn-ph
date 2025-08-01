#!/usr/bin/env python3
"""
Verification script for EMNLP 2025 System Demonstrations submission.
This script checks that the codebase is completely clean and ready for open source release.
"""

import os
import re
import json
from pathlib import Path

def check_for_sensitive_data():
    """Check for any remaining sensitive data in the codebase."""
    print("🔍 Checking for sensitive data...")
    
    sensitive_patterns = [
        r'password["\']?\s*[:=]\s*["\'][^"\']*["\']',
        r'api_key["\']?\s*[:=]\s*["\'][^"\']*["\']',
        r'secret["\']?\s*[:=]\s*["\'][^"\']*["\']',
        r'token["\']?\s*[:=]\s*["\'][^"\']*["\']',
        r'postgresql://[^@]+@[^/\s]+',
        r'DATABASE_URL=postgresql://[^@]+@[^/\s]+',
        r'postgres',
        r'localhost:5432',
        r'127\.0\.0\.1',
        r'192\.168\.',
        r'10\.0\.',
        r'172\.16\.',
        r'172\.17\.',
        r'172\.18\.',
        r'172\.19\.',
        r'172\.2[0-9]\.',
        r'172\.3[0-1]\.'
    ]
    
    found_sensitive = []
    
    for root, dirs, files in os.walk('.'):
        # Skip git directory
        if '.git' in root:
            continue
            
        # Skip node_modules and other excluded directories
        if any(excluded in root for excluded in ['.git', 'node_modules', '__pycache__', '.venv']):
            continue
            
        for file in files:
            if file.endswith(('.py', '.js', '.ts', '.tsx', '.jsx', '.json', '.yml', '.yaml', '.md', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for pattern in sensitive_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            found_sensitive.append({
                                'file': file_path,
                                'pattern': pattern,
                                'matches': matches[:3]  # Show first 3 matches
                            })
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
    
    if found_sensitive:
        print("❌ Found potentially sensitive data:")
        for item in found_sensitive:
            print(f"  - {item['file']}: {item['pattern']}")
            for match in item['matches']:
                print(f"    Found: {match}")
        return False
    else:
        print("✅ No sensitive data found")
        return True

def check_for_ai_assistance_hints():
    """Check for any hints of AI assistance in the codebase."""
    print("\n🤖 Checking for AI assistance hints...")
    
    ai_patterns = [
        r'generated by ai',
        r'created by ai',
        r'assisted by ai',
        r'with help from ai',
        r'using ai',
        r'ai helped',
        r'ai assisted',
        r'ai generated',
        r'ai created',
        r'ai wrote',
        r'ai developed',
        r'ai built',
        r'ai designed',
        r'ai implemented',
        r'ai suggested',
        r'ai recommended',
        r'ai provided',
        r'ai gave',
        r'ai offered',
        r'ai contributed',
        r'chatgpt',
        r'gpt-',
        r'claude',
        r'bard',
        r'copilot',
        r'cursor',
        r'github copilot',
        r'openai',
        r'anthropic',
        r'google ai',
        r'microsoft ai',
        r'ai assistant',
        r'ai tool',
        r'language model',
        r'llm',
        r'large language model'
    ]
    
    found_ai_hints = []
    
    for root, dirs, files in os.walk('.'):
        if '.git' in root:
            continue
            
        if any(excluded in root for excluded in ['.git', 'node_modules', '__pycache__', '.venv']):
            continue
            
        for file in files:
            if file.endswith(('.py', '.js', '.ts', '.tsx', '.jsx', '.json', '.yml', '.yaml', '.md', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for pattern in ai_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            found_ai_hints.append({
                                'file': file_path,
                                'pattern': pattern,
                                'matches': matches[:3]
                            })
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
    
    if found_ai_hints:
        print("❌ Found potential AI assistance hints:")
        for item in found_ai_hints:
            print(f"  - {item['file']}: {item['pattern']}")
            for match in item['matches']:
                print(f"    Found: {match}")
        return False
    else:
        print("✅ No AI assistance hints found")
        return True

def check_file_structure():
    """Check that essential files are present and unwanted files are absent."""
    print("\n📁 Checking file structure...")
    
    required_files = [
        'README.md',
        'LICENSE',
        'CONTRIBUTING.md',
        'requirements.txt',
        'package.json',
        'emnlp2025_demo_submission.tex',
        'backend/app.py',
        'src/App.tsx',
        'ml/README.md'
    ]
    
    unwanted_files = [
        'main.tex',
        'references.bib',
        'Angelica Anne Araneta Naguio - CV (EN-FIL) (2025).pdf',
        'submission_materials.txt',
        'my_db_config.json',
        'db_config.json',
        '.env',
        '*.pt',
        '*.pth',
        '*.ckpt',
        '*.safetensors'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ All required files present")
    
    # Check for unwanted files
    found_unwanted = []
    for pattern in unwanted_files:
        if '*' in pattern:
            # Handle wildcard patterns
            for root, dirs, files in os.walk('.'):
                if '.git' in root:
                    continue
                for file in files:
                    if file.endswith(pattern[1:]):  # Remove *
                        found_unwanted.append(os.path.join(root, file))
        else:
            if os.path.exists(pattern):
                found_unwanted.append(pattern)
    
    if found_unwanted:
        print("❌ Found unwanted files:")
        for file_path in found_unwanted:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ No unwanted files found")
    
    return True

def check_license_compliance():
    """Check that the codebase has proper licensing."""
    print("\n📜 Checking license compliance...")
    
    if not os.path.exists('LICENSE'):
        print("❌ LICENSE file not found")
        return False
    
    with open('LICENSE', 'r', encoding='utf-8') as f:
        license_content = f.read()
    
    if 'MIT License' in license_content:
        print("✅ MIT License found")
        return True
    else:
        print("❌ MIT License not found in LICENSE file")
        return False

def check_readme_quality():
    """Check that README.md is comprehensive and professional."""
    print("\n📖 Checking README quality...")
    
    if not os.path.exists('README.md'):
        print("❌ README.md not found")
        return False
    
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()
    
    required_sections = [
        'Philippine Lexicon GNN Toolkit',
        'Live Demo',
        'Installation',
        'Usage',
        'Contributing',
        'License'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in readme_content:
            missing_sections.append(section)
    
    if missing_sections:
        print("❌ Missing sections in README:")
        for section in missing_sections:
            print(f"  - {section}")
        return False
    else:
        print("✅ README.md is comprehensive")
        return True

def check_emnlp_submission():
    """Check that the EMNLP submission file is present and properly formatted."""
    print("\n📄 Checking EMNLP submission...")
    
    if not os.path.exists('emnlp2025_demo_submission.tex'):
        print("❌ EMNLP submission file not found")
        return False
    
    with open('emnlp2025_demo_submission.tex', 'r', encoding='utf-8') as f:
        submission_content = f.read()
    
    required_elements = [
        '\\title{Philippine Lexicon GNN Toolkit',
        '\\section{Introduction}',
        '\\section{Novelty and Technical Approach}',
        '\\section{System Evaluation and Licensing}',
        '\\section{Conclusion}',
        'https://explorer.hapinas.net/'
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in submission_content:
            missing_elements.append(element)
    
    if missing_elements:
        print("❌ Missing elements in EMNLP submission:")
        for element in missing_elements:
            print(f"  - {element}")
        return False
    else:
        print("✅ EMNLP submission is properly formatted")
        return True

def main():
    """Main verification function."""
    print("🔍 EMNLP 2025 System Demonstrations Verification")
    print("=" * 50)
    
    checks = [
        check_for_sensitive_data,
        check_for_ai_assistance_hints,
        check_file_structure,
        check_license_compliance,
        check_readme_quality,
        check_emnlp_submission
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Codebase is ready for EMNLP 2025 System Demonstrations submission")
        print("\n📋 Summary:")
        print("  - No sensitive data found")
        print("  - No AI assistance hints detected")
        print("  - All required files present")
        print("  - Proper licensing in place")
        print("  - Comprehensive documentation")
        print("  - EMNLP submission properly formatted")
        print("\n🚀 Ready for open source release!")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please address the issues above before submission")
    
    return all_passed

if __name__ == "__main__":
    main() 