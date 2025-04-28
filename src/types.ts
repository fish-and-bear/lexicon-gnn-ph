// src/types.ts

// Define LangCode directly to avoid potential import issues across frontend/backend contexts
export type LangCode = string;

// --- Base Interfaces aligned with Backend Schemas (schemas.py) ---

/**
 * Represents a Part of Speech.
 * Aligned with PartOfSpeechSchema and parts_of_speech table.
 */
export interface PartOfSpeech {
  id: number; // dump_only
  code: string;
  name_en: string;
  name_tl: string;
  description?: string | null;
}

/**
 * Represents a usage example for a definition.
 * Aligned with ExampleSchema and definition_examples table.
 */
export interface Example {
  id?: number | null; // dump_only
  definition_id?: number | null; // dump_only (usually set server-side)
  example_text: string;
  translation?: string | null;
  reference?: string | null;
  example_type?: string | null; // dump_default="example"
  metadata?: Record<string, any> | null; // From MetadataField
  sources?: string | null; // TEXT column
  // Derived fields from post_dump
  romanization?: string | null; // Extracted from metadata
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents a category applied to a definition.
 * Aligned with DefinitionCategorySchema and definition_categories table.
 */
export interface DefinitionCategory {
  id?: number | null; // dump_only
  definition_id?: number | null; // Required in schema (set server-side?)
  category_name: string;
  category_kind?: string | null; // TEXT column
  parents?: string[] | null; // JSONB storing list of strings
  sources?: string | null; // TEXT column
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents a link associated with a definition (e.g., external reference, internal word link).
 * Aligned with DefinitionLinkSchema and definition_links table.
 */
export interface DefinitionLink {
  id?: number | null; // dump_only
  definition_id?: number | null; // Required in schema (set server-side?)
  link_text: string;
  link_metadata?: Record<string, any> | null; // From MetadataField (stores target_url, etc.)
  tags?: string | null; // TEXT column
  sources?: string | null; // TEXT column
  // Derived fields from post_dump
  target_url?: string | null;
  is_external?: boolean | null;
  is_wikipedia?: boolean | null;
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents a relationship between a definition and another word.
 * Aligned with DefinitionRelationSchema and definition_relations table.
 */
export interface DefinitionRelation {
  id?: number | null; // dump_only
  definition_id?: number | null; // Required in schema (set server-side?)
  word_id?: number | null; // Required in schema (ID of the related word)
  relation_type: string;
  sources?: string | null; // TEXT column
  // Nested data (dump_only)
  definition?: BasicDefinition | null; // Simplified DefinitionSchema
  word?: BasicWord | null; // Simplified WordSchema
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents a single definition of a word.
 * Aligned with DefinitionSchema and definitions table.
 * Note: Some fields in DefinitionSchema might be excluded by db_helpers insert function but could still be present in API reads.
 */
export interface RawDefinition {
  id: number; // dump_only
  word_id?: number | null; // dump_only
  definition_text: string;
  original_pos?: string | null; // Raw POS string
  standardized_pos_id?: number | null; // FK to parts_of_speech
  standardized_pos?: PartOfSpeech | null; // Nested PartOfSpeechSchema (dump_only)
  usage_notes?: string | null; // TEXT column
  // notes?: string | null; // Present in schema but potentially unused/removed in inserts
  // cultural_notes?: string | null; // Present in schema but potentially unused/removed in inserts
  // etymology_notes?: string | null; // Present in schema but potentially unused/removed in inserts
  // scientific_name?: string | null; // Present in schema but potentially unused/removed in inserts
  // verified?: boolean | null; // Present in schema but potentially unused/removed in inserts
  // verification_notes?: string | null; // Present in schema but potentially unused/removed in inserts
  tags?: string | null; // TEXT column
  metadata?: Record<string, any> | null; // From MetadataField
  // popularity_score?: number | null; // Present in schema but potentially unused/removed in inserts
  sources?: string | null; // TEXT column

  // Nested lists (dump_only, populated based on includes/query)
  examples?: Example[] | null;
  links?: DefinitionLink[] | null;
  categories?: DefinitionCategory[] | null;
  definition_relations?: DefinitionRelation[] | null; // Added, was missing

  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Minimal definition structure for use in nested contexts to avoid circular dependencies.
 */
export interface BasicDefinition {
  id: number;
  definition_text?: string | null;
  original_pos?: string | null;
}

/**
 * Represents a pronunciation entry (IPA, audio link).
 * Aligned with PronunciationType schema and pronunciations table.
 */
export interface Pronunciation {
  id?: number | null; // dump_only
  word_id?: number | null; // dump_only
  type: string; // e.g., 'ipa', 'audio'
  value: string; // IPA text or URL
  tags?: Record<string, any> | null; // From MetadataField
  pronunciation_metadata?: Record<string, any> | null; // From MetadataField (can include source info)
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents a relationship between two words (e.g., synonym, antonym).
 * Aligned with RelationSchema and relations table.
 */
export interface Relation {
  id?: number | null; // dump_only
  from_word_id?: number | null; // Required in schema
  to_word_id?: number | null; // Required in schema
  relation_type: string;
  sources?: string | null; // TEXT column
  metadata?: Record<string, any> | null; // From MetadataField
  // Nested data (dump_only)
  source_word?: BasicWord | null; // Simplified WordSchema
  target_word?: BasicWord | null; // Simplified WordSchema
  // Derived fields from post_dump
  target_gloss?: string | null; // Extracted from metadata or target_word
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents the etymology information for a word.
 * Aligned with EtymologySchema and etymologies table.
 */
export interface Etymology {
  id?: number | null; // dump_only
  word_id?: number | null; // Required in schema
  etymology_text?: string | null; // Required in schema
  normalized_components?: string | null; // TEXT column
  etymology_structure?: string | null; // TEXT column (might contain JSON string)
  language_codes?: string | null; // TEXT column (comma-separated?)
  sources?: string | null; // TEXT column
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents credit information associated with a word.
 * Aligned with CreditSchema and credits table.
 */
export interface Credit {
  id?: number | null; // dump_only
  word_id?: number | null; // Required in schema
  // role: string; // Missing in CreditSchema? Check model/DB. Assuming 'credit' field covers role/text.
  credit: string; // TEXT column (holds the credit text/role)
  sources?: string | null; // TEXT column
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents an affixation relationship between two words (root and affixed).
 * Aligned with AffixationSchema and affixations table.
 */
export interface Affixation {
  id?: number | null; // dump_only
  root_word_id?: number | null; // Required in schema
  affixed_word_id?: number | null; // Required in schema
  affix_type: string;
  sources?: string | null; // TEXT column
  // Nested data (dump_only)
  root_word?: BasicWord | null; // Simplified WordSchema
  affixed_word?: BasicWord | null; // Simplified WordSchema
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents an alternative form or spelling of a word.
 * Aligned with WordFormSchema and word_forms table.
 */
export interface WordForm {
  id?: number | null; // dump_only
  word_id?: number | null; // Required in schema
  form: string; // TEXT column
  is_canonical?: boolean | null; // dump_default=False
  is_primary?: boolean | null; // dump_default=False
  tags?: Record<string, any> | null; // From MetadataField
  sources?: string | null; // TEXT column
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Represents a template associated with a word (e.g., conjugation template).
 * Aligned with WordTemplateSchema and word_templates table.
 */
export interface WordTemplate {
  id?: number | null; // dump_only
  word_id?: number | null; // Required in schema
  template_name: string; // TEXT column
  args?: Record<string, any> | null; // From MetadataField
  expansion?: string | null; // TEXT column
  sources?: string | null; // TEXT column
  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Minimal word structure for use in nested contexts (like relations, affixations).
 * Aligned with WordSimpleSchema.
 */
export interface BasicWord {
  id: number; // dump_only
  lemma: string;
  normalized_lemma?: string | null; // dump_only
  language_code?: LangCode | null; // dump_default='tl'
  has_baybayin?: boolean | null; // dump_default=False
  baybayin_form?: string | null;
  romanized_form?: string | null; // dump_only
  root_word_id?: number | null;
  // Derived fields from post_dump
  is_root?: boolean | null;
  // Added for potential context display (might come from related definitions)
  gloss?: string | null;
  pos?: string | null;
}

/**
 * Comprehensive data structure for a single word, as returned by main word endpoints.
 * Aligned with WordSchema output, including nested lists and post-dump fields.
 */
export interface RawWordComprehensiveData {
  id: number; // dump_only
  lemma: string;
  normalized_lemma?: string | null; // dump_only
  language_code?: LangCode | null; // dump_default='tl'
  has_baybayin?: boolean | null; // dump_default=False
  baybayin_form?: string | null;
  romanized_form?: string | null; // dump_only
  root_word_id?: number | null;
  preferred_spelling?: string | null;
  tags?: string | null; // TEXT column
  idioms?: Record<string, any> | null; // From MetadataField
  // pronunciation_data?: Record<string, any> | null; // Old MetadataField, replaced by 'pronunciations' list
  source_info?: Record<string, any> | null; // From MetadataField
  word_metadata?: Record<string, any> | null; // From MetadataField
  data_hash?: string | null;
  badlit_form?: string | null; // TEXT column
  hyphenation?: Record<string, any> | null; // From MetadataField
  is_proper_noun?: boolean | null; // dump_default=False
  is_abbreviation?: boolean | null; // dump_default=False
  is_initialism?: boolean | null; // dump_default=False
  completeness_score?: number | null; // dump_default=0.0

  // Primary nested lists (populated based on includes/query)
  definitions?: RawDefinition[] | null;
  pronunciations?: Pronunciation[] | null; // List of PronunciationType schema
  etymologies?: Etymology[] | null; // List of EtymologySchema
  credits?: Credit[] | null; // List of CreditSchema
  forms?: WordForm[] | null; // List of WordFormSchema
  templates?: WordTemplate[] | null; // List of WordTemplateSchema
  definition_relations?: DefinitionRelation[] | null; // List of DefinitionRelationSchema

  // Relations and Affixations (separate and combined)
  outgoing_relations?: Relation[] | null; // List of RelationSchema
  incoming_relations?: Relation[] | null; // List of RelationSchema
  root_affixations?: Affixation[] | null; // List of AffixationSchema
  affixed_affixations?: Affixation[] | null; // List of AffixationSchema

  // Nested Objects (populated based on includes/query)
  root_word?: BasicWord | null; // Nested WordSimpleSchema
  derived_words?: BasicWord[] | null; // List of WordSimpleSchema

  // Fields added or combined in post_dump
  relations?: Relation[] | null; // Combined outgoing + incoming
  affixations?: Affixation[] | null; // Combined root + affixed
  is_root?: boolean | null; // Derived from root_word_id

  // Convenience fields often added in backend logic before sending
  lang_name?: string | null; // e.g., "Tagalog" derived from language_code
  gloss?: string | null; // Often the first definition text
  pos?: string | null; // Often the first definition's original_pos

  // Timestamps (dump_only)
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Alias for RawWordComprehensiveData, potentially used after client-side processing.
 * Ensures 'id' is always present.
 */
export type WordInfo = RawWordComprehensiveData & { id: number };

// --- Search Result Types ---

/**
 * Represents a single item in the search results list.
 * Aligned with the structure returned by the /search endpoint.
 */
export interface SearchResultItem {
  word_id: number; // Use word_id consistently
  lemma: string;
  lang_code: LangCode;
  lang_name: string; // Added in backend route
  gloss: string; // Added in backend route
  pos: string; // Added in backend route
  score: number; // Relevance score from search
  // Optional full word data if include_full=true
  word?: RawWordComprehensiveData | null;
}

/**
 * Represents the overall structure of the search results response.
 * Aligned with the structure returned by the /search endpoint.
 */
export interface SearchResults {
  results: SearchResultItem[]; // Use 'results' consistently
  total: number;
  page: number;
  per_page: number; // Corresponds to 'limit' in request
  query_details?: any; // Optional metadata about the query execution
  error?: string | null; // Optional error message
}

// --- Word Suggestion Type ---

/**
 * Represents a word suggestion item, typically for autocomplete.
 * Aligned with the structure returned by the /suggestions endpoint.
 */
export interface WordSuggestion {
  id: number; // Word ID
  lemma: string;
  language_code: LangCode;
  gloss?: string | null; // Often the first definition text
}

// --- Statistics Type ---

/**
 * Represents the structure of dictionary statistics.
 * Aligned with StatisticsSchema.
 */
export interface Statistics {
  total_words?: number | null;
  total_definitions?: number | null;
  total_etymologies?: number | null;
  total_relations?: number | null;
  total_affixations?: number | null;
  words_with_examples?: number | null;
  words_with_etymology?: number | null;
  words_with_relations?: number | null;
  words_with_baybayin?: number | null;
  words_by_language?: Record<string, number> | null;
  words_by_pos?: Record<string, number> | null;
  verification_stats?: Record<string, number> | null; // Might not be implemented yet
  quality_distribution?: Record<string, number> | null; // Might not be implemented yet
  update_frequency?: Record<string, number> | null; // Might not be implemented yet
  timestamp?: string | null; // Added for potential response inclusion
}

// --- Network and Tree Types (for Graph Visualizations) ---

/**
 * Represents a node in a word network graph (semantic or etymology).
 * Based on the structure generated by /semantic_network and /etymology/tree routes.
 */
export interface NetworkNode {
  id: number; // Word ID
  label: string; // Lemma (for display)
  word?: string | null; // Often lemma again
  language?: LangCode | null;
  type?: string | null; // e.g., relation type from source edge
  depth?: number | null; // Depth in the traversal
  has_baybayin?: boolean | null;
  baybayin_form?: string | null;
  normalized_lemma?: string | null; // Added from backend route
  main?: boolean | null; // Added from backend route (indicates the central node)
  [key: string]: any; // Allow extra properties from graph libraries (e.g., position)
}

/**
 * Represents a link (edge) in a word network graph.
 * Based on the structure generated by /semantic_network and /etymology/tree routes.
 */
export interface NetworkLink {
  id?: string; // Added from backend route (e.g., "sourceId-targetId")
  source: number | NetworkNode; // Source node ID (or object in d3)
  target: number | NetworkNode; // Target node ID (or object in d3)
  type: string; // Relation type
  weight?: number | null; // Renamed from value to match backend 'weight'
  directed?: boolean | null; // Added from backend route
  [key: string]: any; // Allow extra properties from graph libraries
}

/**
 * Represents the structure of the response from the /semantic_network endpoint.
 */
export interface WordNetwork {
  nodes: NetworkNode[];
  links: NetworkLink[]; // Use 'links' consistently (not 'edges')
  metadata: {
    root_word?: string | null;
    normalized_lemma?: string | null;
    language_code?: LangCode | null;
    depth?: number | null;
    total_nodes?: number | null;
    total_edges?: number | null; // Use total_edges for consistency if backend uses it
    include_affixes?: boolean | null;
    include_etymology?: boolean | null;
    cluster_threshold?: number | null;
  };
}

/**
 * Represents the structure of the response from the /etymology/tree endpoint.
 */
export interface EtymologyTree {
  word?: string | null; // Root word lemma for the tree
  etymology_tree?: any | null; // The raw nested tree structure from backend (if needed)
  complete?: boolean | null; // Indicates if the tree construction was complete
  // Processed nodes/links for visualization
  nodes?: NetworkNode[] | null;
  links?: NetworkLink[] | null;
  metadata?: {
    word?: string | null;
    max_depth?: number | null;
  } | null;
}

// --- API Option Types ---

/**
 * Represents the available query parameters for the /search endpoint.
 * Based on SearchQuerySchema and SearchFilterSchema.
 */
export interface SearchOptions {
  q?: string | null; // Query string (required by schema, but optional here for flexibility)
  page?: number | null; // Corresponds to 'offset' calculation: offset = (page - 1) * per_page
  per_page?: number | null; // Corresponds to 'limit' in schema
  mode?: "all" | "exact" | "prefix" | "suffix" | "fuzzy" | null; // 'fuzzy' might not be in schema
  sort?: "relevance" | "alphabetical" | "created" | "updated" | "completeness" | null;
  order?: "asc" | "desc" | null;
  language?: LangCode | null;
  pos?: string | null; // Part of speech code

  // Feature filters
  has_etymology?: boolean | null;
  has_pronunciation?: boolean | null;
  has_baybayin?: boolean | null;
  exclude_baybayin?: boolean | null; // Added based on schema
  has_forms?: boolean | null;
  has_templates?: boolean | null;

  // Direct boolean filters (may require specific backend query logic)
  is_root?: boolean | null;
  is_proper_noun?: boolean | null;
  is_abbreviation?: boolean | null;
  is_initialism?: boolean | null;

  // Date range filters (ISO Date strings)
  date_added_from?: string | null;
  date_added_to?: string | null;
  date_modified_from?: string | null;
  date_modified_to?: string | null;

  // Count filters
  min_definition_count?: number | null;
  max_definition_count?: number | null;
  min_relation_count?: number | null;
  max_relation_count?: number | null;

  // Score filters
  min_completeness?: number | null;
  max_completeness?: number | null;

  // Include options (control nested data in response)
  include_full?: boolean | null; // Load full RawWordComprehensiveData in results
  include_definitions?: boolean | null;
  include_pronunciations?: boolean | null;
  include_etymologies?: boolean | null;
  include_relations?: boolean | null; // Combined relations
  include_forms?: boolean | null;
  include_templates?: boolean | null;
  include_metadata?: boolean | null; // Include word_metadata, source_info etc.?
  include_related_words?: boolean | null; // Load nested source/target words in relations?
  include_definition_relations?: boolean | null;
}

/**
 * Represents the available query parameters for the /semantic_network endpoint.
 */
export interface WordNetworkOptions {
  depth?: number | null;
  include_affixes?: boolean | null;
  include_etymology?: boolean | null;
  cluster_threshold?: number | null;
}

// --- Structures for Editing ---
// These add temporary client-side IDs (`temp_id`) for managing lists in UI forms
// and adapt the structure for form binding.

/** Helper type for items that can have a temporary client-side ID */
interface PossiblyEditable {
  temp_id?: string;
}

/** Base type for editable nested items, ensuring temp_id exists */
interface BaseEditable extends PossiblyEditable {
  temp_id: string; // Ensure temp_id is always added
}

export interface EditableExample extends Omit<Example, 'id' | 'definition_id' | 'created_at' | 'updated_at' | 'romanization'>, BaseEditable {}
export interface EditableDefinitionCategory extends Omit<DefinitionCategory, 'id' | 'definition_id' | 'created_at' | 'updated_at'>, BaseEditable {}
export interface EditableDefinitionLink extends Omit<DefinitionLink, 'id' | 'definition_id' | 'created_at' | 'updated_at' | 'target_url' | 'is_external' | 'is_wikipedia'>, BaseEditable {}
export interface EditableDefinitionRelation extends Omit<DefinitionRelation, 'id' | 'definition_id' | 'created_at' | 'updated_at' | 'definition' | 'word'>, BaseEditable {
    word_id: number; // Expect word_id based on backend schema (might need UI lookup for lemma->id)
    related_lemma?: string; // For display/lookup in UI
}

export interface EditableDefinition extends Omit<RawDefinition, 'id' | 'word_id' | 'standardized_pos' | 'created_at' | 'updated_at' | 'examples' | 'links' | 'categories' | 'definition_relations'>, BaseEditable {
  // Use simple ID for POS during editing
  standardized_pos_id?: number | null;
  // Use editable versions of nested items
  examples: EditableExample[];
  links: EditableDefinitionLink[];
  categories: EditableDefinitionCategory[];
  definition_relations: EditableDefinitionRelation[];
}

export interface EditablePronunciation extends Omit<Pronunciation, 'id' | 'word_id' | 'created_at' | 'updated_at'>, BaseEditable {}

export interface EditableRelation extends Omit<Relation, 'id' | 'from_word_id' | 'to_word_id' | 'source_word' | 'target_word' | 'created_at' | 'updated_at' | 'target_gloss'>, BaseEditable {
  // Need to handle target_word lookup/selection in UI
  target_lemma: string; // For UI lookup/display
}

export interface EditableAffixation extends Omit<Affixation, 'id' | 'root_word_id' | 'affixed_word_id' | 'root_word' | 'affixed_word' | 'created_at' | 'updated_at'>, BaseEditable {
  // Need UI lookup for lemma -> ID mapping
  root_lemma?: string;
  affixed_lemma?: string;
}

export interface EditableCredit extends Omit<Credit, 'id' | 'word_id' | 'created_at' | 'updated_at'>, BaseEditable {}
export interface EditableWordForm extends Omit<WordForm, 'id' | 'word_id' | 'created_at' | 'updated_at'>, BaseEditable {}
export interface EditableWordTemplate extends Omit<WordTemplate, 'id' | 'word_id' | 'created_at' | 'updated_at'>, BaseEditable {}

// Removed EditableEtymology as etymology is currently singular/less complex in RawWordComprehensiveData

/**
 * Structure for the main word editor form state.
 */
export interface EditableWordData {
  // Core identifying fields (usually read-only in edit mode, required for create)
  id?: number; // Keep track of ID if editing existing word
  lemma: string;
  language_code: LangCode;

  // Editable simple fields/flags (map directly from RawWordComprehensiveData)
  gloss?: string | null; // Convenience field, might derive from first definition
  pos?: string | null; // Convenience field, might derive from first definition
  preferred_spelling?: string | null;
  raw_tags?: string | null; // Use raw_tags for TEXT tags field editing
  idioms_json?: string | null; // Edit JSONB as string
  source_info_json?: string | null; // Edit JSONB as string
  word_metadata_json?: string | null; // Edit JSONB as string
  badlit_form?: string | null;
  hyphenation_json?: string | null; // Edit JSONB as string
  is_proper_noun?: boolean | null;
  is_abbreviation?: boolean | null;
  is_initialism?: boolean | null;

  // Editable complex nested data (singular)
  etymology_data?: Etymology | null; // For now, treat as potentially editable object

  // Editable nested arrays
  definitions: EditableDefinition[];
  pronunciations: EditablePronunciation[]; // Name matches nested list in RawWordComprehensiveData
  relations: EditableRelation[]; // Name matches combined relations in RawWordComprehensiveData
  affixations: EditableAffixation[]; // Name matches combined affixations in RawWordComprehensiveData
  credits: EditableCredit[];
  forms: EditableWordForm[];
  templates: EditableWordTemplate[];

  // Add lang_name for display convenience if needed
  lang_name?: string | null;
}

// --- Conversion Functions (for Editing Forms) ---

// Helper to generate temporary IDs for UI list management
const generateTempId = (): string => `temp_${Date.now()}_${Math.random()}`;

// Helper to safely stringify potential JSON objects for form fields
const safeStringify = (data: any): string | null => {
    if (data === null || data === undefined) return null;
    if (typeof data === 'string') return data; // Already a string
    try {
        return JSON.stringify(data, null, 2); // Pretty print
    } catch (e) {
        console.error("Failed to stringify data:", data, e);
        return null;
    }
};

// Helper to safely parse JSON strings from form fields
const safeParse = (jsonString: string | null | undefined): Record<string, any> | null => {
    if (jsonString === null || jsonString === undefined || jsonString.trim() === '') return null;
    try {
        const parsed = JSON.parse(jsonString);
        // Ensure it's an object, not an array or primitive
        return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed) ? parsed : null;
    } catch (e) {
        console.error("Failed to parse JSON string:", jsonString, e);
        return null; // Return null on parse error
    }
};


// Function to convert RawWordComprehensiveData to EditableWordData for form binding
export function convertToEditable(rawData: RawWordComprehensiveData): EditableWordData {
  const tempId = generateTempId(); // Generate one ID for the root if needed later

  // Helper to add temp_id to array items
  const addTempIds = <T extends object>(items: T[] | null | undefined): (T & BaseEditable)[] => {
    return (items ?? []).map((item) => ({ ...item, temp_id: generateTempId() }));
  };

  return {
    // Core fields
    id: rawData.id, // Keep ID for updates
    lemma: rawData.lemma,
    language_code: rawData.language_code ?? 'tl', // Default if missing
    gloss: rawData.gloss ?? rawData.definitions?.[0]?.definition_text ?? null, // Use gloss or first definition
    pos: rawData.pos ?? rawData.definitions?.[0]?.original_pos ?? null, // Use pos or first definition's pos
    lang_name: rawData.lang_name ?? null,

    // Simple nested data / flags converted for form editing
    preferred_spelling: rawData.preferred_spelling ?? null,
    raw_tags: rawData.tags ?? null, // Map 'tags' (TEXT) to 'raw_tags'
    idioms_json: safeStringify(rawData.idioms), // Stringify MetadataField
    source_info_json: safeStringify(rawData.source_info), // Stringify MetadataField
    word_metadata_json: safeStringify(rawData.word_metadata), // Stringify MetadataField
    badlit_form: rawData.badlit_form ?? null,
    hyphenation_json: safeStringify(rawData.hyphenation), // Stringify MetadataField
    is_proper_noun: rawData.is_proper_noun ?? false, // Default boolean
    is_abbreviation: rawData.is_abbreviation ?? false, // Default boolean
    is_initialism: rawData.is_initialism ?? false, // Default boolean

    // Singular complex data
    etymology_data: rawData.etymologies?.[0] ?? null, // Use first etymology if available (schema has list)

    // Convert arrays to Editable* arrays, adding temp_ids
    definitions: (rawData.definitions ?? []).map((d): EditableDefinition => ({
      ...d, // Spread existing fields
      temp_id: generateTempId(),
      standardized_pos_id: d.standardized_pos_id ?? d.standardized_pos?.id ?? null,
      // Convert nested arrays within definition
      examples: addTempIds(d.examples),
      categories: addTempIds(d.categories),
      links: addTempIds(d.links),
      definition_relations: addTempIds(d.definition_relations).map(dr => ({
          ...dr,
          word_id: dr.word_id ?? 0, // Ensure word_id is number
          related_lemma: dr.word?.lemma // Add lemma for display
      })),
    })),
    pronunciations: addTempIds(rawData.pronunciations),
    // Use combined fields from post_dump for editing base
    relations: addTempIds(rawData.relations).map(r => ({
        ...r,
        target_lemma: r.target_word?.lemma ?? '' // Ensure target_lemma exists
    })),
    affixations: addTempIds(rawData.affixations).map(a => ({
        ...a,
        root_lemma: a.root_word?.lemma,
        affixed_lemma: a.affixed_word?.lemma,
    })),
    credits: addTempIds(rawData.credits),
    forms: addTempIds(rawData.forms),
    templates: addTempIds(rawData.templates),
  };
}

// --- Types for API Submission ---

// Define the structure the API expects for word creation/update payload
// This should generally align with the fields the backend *accepts* for a word,
// excluding dump_only fields, derived fields, and combined lists if the backend expects specifics.
type WordCoreSubmissionPayload = Omit<RawWordComprehensiveData,
    // Exclude dump_only and strictly derived fields
    'id' | 'normalized_lemma' | 'romanized_form' | 'data_hash' |
    'created_at' | 'updated_at' | 'root_word' | 'derived_words' | 'is_root' |
    'completeness_score' | 'lang_name' | 'gloss' | 'pos' |
    // Exclude combined fields (API likely handles combination or uses specifics)
    'relations' | 'affixations' |
    // Exclude fields that are usually handled via separate relationships/tables if not directly on Word model
    'definitions' | 'pronunciations' | 'etymologies' | 'credits' |
    'outgoing_relations' | 'incoming_relations' | 'root_affixations' | 'affixed_affixations' |
    'forms' | 'templates' | 'definition_relations'
    // Keep direct fields like lemma, language_code, metadata fields etc.
>;

// Define submission types for nested items, excluding read-only/derived fields and temp_id

// Define submission types for nested items WITHIN a definition
type ExampleSubmissionPayload = Omit<EditableExample, 'temp_id'>;
type DefinitionCategorySubmissionPayload = Omit<EditableDefinitionCategory, 'temp_id'>;
type DefinitionLinkSubmissionPayload = Omit<EditableDefinitionLink, 'temp_id'>;
type DefinitionRelationSubmissionPayload = Omit<EditableDefinitionRelation, 'temp_id' | 'related_lemma'>; // Also remove UI-only related_lemma

// Redefine DefinitionSubmissionPayload using the explicit nested types
type DefinitionSubmissionPayload = Omit<EditableDefinition,
  'temp_id' | 'standardized_pos' | 'examples' | 'categories' | 'links' | 'definition_relations'
> & {
  examples?: ExampleSubmissionPayload[] | null;
  categories?: DefinitionCategorySubmissionPayload[] | null;
  links?: DefinitionLinkSubmissionPayload[] | null;
  definition_relations?: DefinitionRelationSubmissionPayload[] | null;
};

type PronunciationSubmissionPayload = Omit<EditablePronunciation, 'temp_id'>;
type RelationSubmissionPayload = Omit<EditableRelation, 'temp_id' | 'target_lemma'> & { target_word_id?: number }; // Backend likely needs target ID
type AffixationSubmissionPayload = Omit<EditableAffixation, 'temp_id' | 'root_lemma' | 'affixed_lemma'> & { root_word_id?: number; affixed_word_id?: number }; // Backend needs IDs
type CreditSubmissionPayload = Omit<EditableCredit, 'temp_id'>;
type WordFormSubmissionPayload = Omit<EditableWordForm, 'temp_id'>;
type WordTemplateSubmissionPayload = Omit<EditableWordTemplate, 'temp_id'>;
type EtymologySubmissionPayload = Omit<Etymology, 'id' | 'word_id' | 'created_at' | 'updated_at'>; // Based on singular etymology in editable form

// Type for the complete submission payload, combining core fields and nested arrays
export type WordSubmissionPayload = WordCoreSubmissionPayload & {
    // Add back nested arrays with their submission types
    // Names should match what the backend endpoint expects (e.g., might expect 'pronunciation_data' if updating that JSONB directly)
    definitions?: DefinitionSubmissionPayload[] | null;
    pronunciations?: PronunciationSubmissionPayload[] | null;
    relations?: RelationSubmissionPayload[] | null; // Or maybe outgoing/incoming separately? Check API route.
    affixations?: AffixationSubmissionPayload[] | null; // Or maybe root/affixed separately? Check API route.
    credits?: CreditSubmissionPayload[] | null;
    forms?: WordFormSubmissionPayload[] | null;
    templates?: WordTemplateSubmissionPayload[] | null;
    etymologies?: EtymologySubmissionPayload[] | null; // API likely expects list based on schema
};

// Function to convert EditableWordData back into a structure suitable for API submission
export function convertFromEditable(editableData: EditableWordData): WordSubmissionPayload {

    // Helper to strip temp_id and potentially other UI-only fields before submission
    const stripTempId = <T extends { temp_id?: string }, S>(
        items: T[] | undefined | null
    ): S[] | null => {
        if (!items) return null;
        return items.map(({ temp_id, ...rest }) => rest as S); // Basic stripping
    };

    // Helper to strip temp_ids from definitions and their nested arrays
    const prepareDefinitionsForSubmission = (
        defs: EditableDefinition[] | undefined | null
    ): DefinitionSubmissionPayload[] | null => {
        if (!defs) return null;
        return defs.map(({ temp_id, examples, categories, links, definition_relations, ...rest }) => ({
            ...rest, // Spread the remaining fields of EditableDefinition
            // Strip temp_ids from nested arrays
            examples: stripTempId<EditableExample, ExampleSubmissionPayload>(examples), // Use specific payload type
            categories: stripTempId<EditableDefinitionCategory, DefinitionCategorySubmissionPayload>(categories), // Use specific payload type
            links: stripTempId<EditableDefinitionLink, DefinitionLinkSubmissionPayload>(links), // Use specific payload type
            definition_relations: stripTempId<EditableDefinitionRelation, DefinitionRelationSubmissionPayload>(definition_relations), // Use specific payload type
        }));
    };

    // NOTE: Need logic here to convert target_lemma/root_lemma/affixed_lemma back to IDs for relations/affixations
    // This typically involves lookups against the API or a local cache before submission.
    // For now, we assume the IDs are somehow populated before calling this function or handled by the API.

    const submissionPayload: WordSubmissionPayload = {
        // Core fields
        lemma: editableData.lemma,
        language_code: editableData.language_code,
        preferred_spelling: editableData.preferred_spelling,
        tags: editableData.raw_tags, // Map raw_tags back to tags
        // Parse JSON string fields back to objects/null
        idioms: safeParse(editableData.idioms_json),
        source_info: safeParse(editableData.source_info_json),
        word_metadata: safeParse(editableData.word_metadata_json),
        badlit_form: editableData.badlit_form,
        hyphenation: safeParse(editableData.hyphenation_json),
        is_proper_noun: editableData.is_proper_noun,
        is_abbreviation: editableData.is_abbreviation,
        is_initialism: editableData.is_initialism,

        // Convert Editable* arrays back, stripping temp_ids
        definitions: prepareDefinitionsForSubmission(editableData.definitions),
        pronunciations: stripTempId<EditablePronunciation, PronunciationSubmissionPayload>(editableData.pronunciations),
        relations: stripTempId<EditableRelation, RelationSubmissionPayload>(editableData.relations), // Assumes API accepts combined list
        affixations: stripTempId<EditableAffixation, AffixationSubmissionPayload>(editableData.affixations), // Assumes API accepts combined list
        credits: stripTempId<EditableCredit, CreditSubmissionPayload>(editableData.credits),
        forms: stripTempId<EditableWordForm, WordFormSubmissionPayload>(editableData.forms),
        templates: stripTempId<EditableWordTemplate, WordTemplateSubmissionPayload>(editableData.templates),
        // Handle etymology (assuming API expects a list, even if form handles singular)
        etymologies: editableData.etymology_data
            ? [ (({ id, word_id, created_at, updated_at, ...rest }) => rest)(editableData.etymology_data) ] // Create list from single item
            : null,
    };

    // Clean null/undefined fields before submission? Optional, depends on API.
    // Object.keys(submissionPayload).forEach(key => {
    //     if (submissionPayload[key] === null || submissionPayload[key] === undefined) {
    //         delete submissionPayload[key];
    //     }
    // });

    return submissionPayload;
}