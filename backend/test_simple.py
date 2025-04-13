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

# Write results to a file
with open('test_results.txt', 'w', encoding='utf-8') as f:
    f.write("Testing remove_trailing_numbers function:\n")
    f.write("-" * 40 + "\n")
    for input_text in test_cases:
        output = remove_trailing_numbers(input_text)
        result_line = f"Input: '{input_text}' -> Output: '{output}'\n"
        f.write(result_line)

print("Results written to test_results.txt")