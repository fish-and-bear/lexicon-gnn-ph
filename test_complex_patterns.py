#!/usr/bin/env python3
"""
Test script to verify that the clean_word_in_text function correctly handles complex patterns.
"""

from clean_trailing_numbers import clean_word_in_text

def test_complex_patterns():
    """Test complex patterns with trailing numbers."""
    test_cases = [
        'likód3 o likuran',
        'túlog1-3 o pagtúlog',
        'súhol1,2 o panunuhol.',
        'súhol1-2 o panunuhol.',
        'labò3',
        'úngol1-',
        'úngol,2',
        'bábaw2-3',
        'gintô1'
    ]

    print("Testing complex patterns with trailing numbers:")
    print("=" * 50)
    
    for test in test_cases:
        cleaned, changes = clean_word_in_text(test)
        print(f'Original: "{test}"')
        print(f'Cleaned:  "{cleaned}"')
        print(f'Changes:  {changes}')
        print('-' * 50)
        
def test_middle_of_phrase():
    """Test words with trailing numbers in the middle of phrases."""
    test_cases = [
        'Ang túlog1-3 ay mahimbing',
        'Isang salitang iklî2 lamang',
        'May kakayahang úngol1, subalit hindi makapagsalita.',
        'Ang likód3 ng bahay',
        'Mga kursong ESP101, MATH202, at FIL301 ngayong semester',
        'Ito ang may sagupà1 sa kabilang bayan',
        'Ang kasagutan ng pagsubok ay a1, b3, c2, at d4.',
        'Ipaliwanag ang súhol1,2 sa kanyang talumpati.',
        'Ang KUSLAB1 ay kasangkapan sa laboratoryo.'
    ]
    
    print("\nTesting words with trailing numbers in the middle of phrases:")
    print("=" * 50)
    
    for test in test_cases:
        cleaned, changes = clean_word_in_text(test)
        print(f'Original: "{test}"')
        print(f'Cleaned:  "{cleaned}"')
        print(f'Changes:  {changes}')
        print('-' * 50)

if __name__ == "__main__":
    test_complex_patterns()
    test_middle_of_phrase() 