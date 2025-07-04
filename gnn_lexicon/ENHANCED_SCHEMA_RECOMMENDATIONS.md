# Enhanced GNN Schema Recommendations for Philippine Lexicon

## Current vs. Optimal Schema Analysis

### Current Implementation (Limited)
- **Nodes**: Word, Form, Sense, Language
- **Edges**: HAS_FORM, OF_WORD, HAS_SENSE, SHARES_PHONOLOGY, DERIVED_FROM
- **Issues**: Missing rich linguistic relationships, simulated edges, limited node types

### Recommended Enhanced Schema

## Node Types (Based on Database Tables)

### 1. **Word** (Core Entity)
- **Source**: `words` table
- **Features**: lemma, normalized_lemma, language_code, is_proper_noun, is_abbreviation, is_initialism, has_baybayin, baybayin_form, romanized_form, root_word_id, preferred_spelling, tags, idioms, pronunciation_data, source_info, word_metadata, badlit_form, hyphenation
- **Key**: Central entity connecting all linguistic information

### 2. **Definition** (Semantic Content)
- **Source**: `definitions` table  
- **Features**: definition_text, original_pos, standardized_pos_id, examples, usage_notes, tags, sources, definition_metadata
- **Key**: Rich semantic information for words

### 3. **PartOfSpeech** (Grammatical Category)
- **Source**: `parts_of_speech` table
- **Features**: code, name_en, name_tl, description
- **Key**: Standardized grammatical categories

### 4. **Relation** (Semantic Relationships)
- **Source**: `relations` table
- **Features**: relation_type, sources, metadata
- **Key**: Explicit semantic relationships between words

### 5. **Etymology** (Historical Information)
- **Source**: `etymologies` table
- **Features**: etymology_text, normalized_components, etymology_structure, language_codes, sources
- **Key**: Historical word origins and language connections

### 6. **Pronunciation** (Phonetic Information)
- **Source**: `pronunciations` table
- **Features**: type, value, tags, pronunciation_metadata
- **Key**: Phonetic representations (IPA, respelling, etc.)

### 7. **WordForm** (Morphological Variants)
- **Source**: `word_forms` table
- **Features**: form, is_canonical, is_primary, tags, sources
- **Key**: Inflected forms and spelling variants

### 8. **WordTemplate** (Morphological Patterns)
- **Source**: `word_templates` table
- **Features**: template_name, args, expansion, sources
- **Key**: Morphological derivation patterns

### 9. **DefinitionExample** (Usage Context)
- **Source**: `definition_examples` table
- **Features**: example_text, translation, example_type, reference, metadata, sources
- **Key**: Usage examples and quotations

### 10. **DefinitionCategory** (Semantic Categories)
- **Source**: `definition_categories` table
- **Features**: category_name, category_kind, parents, sources, category_metadata
- **Key**: Hierarchical semantic categorization

### 11. **DefinitionLink** (External References)
- **Source**: `definition_links` table
- **Features**: link_text, link_url, tags, link_metadata, sources
- **Key**: External references and cross-references

### 12. **Language** (Language Information)
- **Source**: `languages` table
- **Features**: code, name_en, name_tl, region, family, status
- **Key**: Language metadata

## Edge Types (Based on Database Relationships)

### Core Word Relationships
1. **`(Word, HAS_DEFINITION, Definition)`** - Word to its definitions
2. **`(Word, HAS_PRONUNCIATION, Pronunciation)`** - Word to its pronunciations
3. **`(Word, HAS_FORM, WordForm)`** - Word to its morphological forms
4. **`(Word, HAS_TEMPLATE, WordTemplate)`** - Word to its morphological templates
5. **`(Word, HAS_ETYMOLOGY, Etymology)`** - Word to its etymological information
6. **`(Word, HAS_PART_OF_SPEECH, PartOfSpeech)`** - Word to its grammatical category
7. **`(Word, ROOT_OF, Word)`** - Root word relationships (self-referential)
8. **`(Word, DERIVED_FROM, Word)`** - Derived word relationships (self-referential)

### Semantic Relationships
9. **`(Word, RELATED_TO, Word)`** - General semantic relationships
10. **`(Word, SYNONYM_OF, Word)`** - Synonym relationships
11. **`(Word, ANTONYM_OF, Word)`** - Antonym relationships
12. **`(Word, COGNATE_OF, Word)`** - Cognate relationships
13. **`(Word, TRANSLATION_OF, Word)`** - Translation relationships
14. **`(Word, SEE_ALSO, Word)`** - See-also relationships
15. **`(Word, VARIANT_OF, Word)`** - Variant relationships
16. **`(Word, DOUBLET_OF, Word)`** - Doublet relationships

### Definition Relationships
17. **`(Definition, HAS_EXAMPLE, DefinitionExample)`** - Definition to examples
18. **`(Definition, HAS_CATEGORY, DefinitionCategory)`** - Definition to categories
19. **`(Definition, HAS_LINK, DefinitionLink)`** - Definition to external links
20. **`(Definition, RELATED_TO, Definition)`** - Definition relationships

### Language Relationships
21. **`(Word, IN_LANGUAGE, Language)`** - Word to its language
22. **`(Etymology, INVOLVES_LANGUAGE, Language)`** - Etymology to involved languages

### Form Relationships
23. **`(WordForm, OF_WORD, Word)`** - Form back to its word
24. **`(WordTemplate, FOR_WORD, Word)`** - Template to its word

## Implementation Priority

### Phase 1: Core Linguistic Graph
- **Nodes**: Word, Definition, PartOfSpeech, WordForm
- **Edges**: HAS_DEFINITION, HAS_FORM, HAS_PART_OF_SPEECH, OF_WORD
- **Rationale**: Foundation for basic word relationships

### Phase 2: Semantic Relationships
- **Nodes**: Add Relation (as edge features)
- **Edges**: All semantic relationship types from `relations` table
- **Rationale**: Rich semantic understanding

### Phase 3: Historical & Phonetic
- **Nodes**: Add Etymology, Pronunciation
- **Edges**: HAS_ETYMOLOGY, HAS_PRONUNCIATION, INVOLVES_LANGUAGE
- **Rationale**: Historical and phonetic connections

### Phase 4: Morphological Patterns
- **Nodes**: Add WordTemplate
- **Edges**: HAS_TEMPLATE, FOR_WORD
- **Rationale**: Morphological derivation patterns

### Phase 5: Context & Categories
- **Nodes**: Add DefinitionExample, DefinitionCategory, DefinitionLink
- **Edges**: HAS_EXAMPLE, HAS_CATEGORY, HAS_LINK
- **Rationale**: Usage context and semantic categorization

## Key Advantages of Enhanced Schema

1. **Rich Semantic Information**: Leverages all 12 relation types from `relations` table
2. **Historical Connections**: Etymology relationships for diachronic analysis
3. **Morphological Patterns**: Word templates for derivation analysis
4. **Usage Context**: Examples and quotations for pragmatic understanding
5. **Semantic Categories**: Hierarchical categorization for domain-specific analysis
6. **Phonetic Information**: Multiple pronunciation types for phonetic analysis
7. **External References**: Links to external resources for comprehensive coverage

## Implementation Notes

- **Edge Features**: Use `relation_type`, `metadata`, `sources` as edge features
- **Node Features**: Use JSONB fields (`word_metadata`, `definition_metadata`, etc.) as rich node features
- **Filtering**: Focus on Tagalog (`language_code = 'tl'`) but maintain cross-lingual relationships
- **Scaling**: Implement progressive loading based on relationship density

This enhanced schema would create a much richer linguistic graph that captures the full complexity of your Philippine lexicon database. 