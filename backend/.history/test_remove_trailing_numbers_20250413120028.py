#!/usr/bin/env python3
"""
Test script for remove_trailing_numbers function.
"""

from dictionary_manager import remove_trailing_numbers

# Test cases
test_cases = [
    ('suyo1', 'suyo'),
    ('alam23', 'alam'),
    ('bahay', 'bahay'),
    ('word123abc', 'word123abc'),  # Only trailing numbers should be removed
    ('', ''),  # Empty string
    ('123', ''),  # Just numbers
    ('aaa111bbb222', 'aaa111bbb'),  # Only trailing numbers
    ('aaa-111', 'aaa-'),  # With hyphen
]

# Run tests
print("Testing remove_trailing_numbers function:")
print("-" * 40)
for input_text, expected_output in test_cases:
    actual_output = remove_trailing_numbers(input_text)
    result = "PASS" if actual_output == expected_output else "FAIL"
    print(f"Input: '{input_text}'")
    print(f"Expected: '{expected_output}'")
    print(f"Actual: '{actual_output}'")
    print(f"Result: {result}")
    print("-" * 40) 