#!/usr/bin/env python

def remove_trailing_numbers(text):
    """Remove trailing numbers from a string."""
    if not text:
        return ""
    import re
    return re.sub(r'\d+$', '', text)

# Test cases
test_cases = [
    'suyo1',
    'alam23',
    'bahay',
    'word123abc',
    '',
    '123',
    'aaa111bbb222',
    'aaa-111'
]

print("Testing remove_trailing_numbers function:")
print("-" * 40)
for input_text in test_cases:
    output = remove_trailing_numbers(input_text)
    print(f"Input: '{input_text}' → Output: '{output}'")