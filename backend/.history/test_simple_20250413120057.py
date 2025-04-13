#!/usr/bin/env python3
import re

def remove_trailing_numbers(text):
    if not text:
        return ""
    return re.sub(r'\d+$', '', text)

# Test cases
test_cases = [
    ('suyo1', 'suyo'),
    ('alam23', 'alam'),
    ('bahay', 'bahay'),
    ('word123abc', 'word123abc'),
    ('', ''),
    ('123', ''),
    ('aaa111bbb222', 'aaa111bbb'),
    ('aaa-111', 'aaa-'),
]

for input_text, expected in test_cases:
    result = remove_trailing_numbers(input_text)
    print(f"'{input_text}' -> '{result}' (Expected: '{expected}')")