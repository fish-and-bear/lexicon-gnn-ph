export interface PartOfSpeech {
  id: number;
  code: string;
  name_en: string;
  name_tl: string;
  description?: string;
}

// Type for the RAW definition data possibly coming from the API
export interface RawDefinition {
  id: number;
  definition_text?: string;
  text?: string; // Alias for definition_text
  original_pos?: string | null;
  standardized_pos_id?: number | null;
  standardized_pos?: PartOfSpeech | null;
  examples?: string | undefined; // CHANGED: Expect string from backend
  usage_notes?: string | undefined; // CHANGED: Expect string from backend
  tags?: string | undefined; // CHANGED: Expect string from backend
  sources?: string | undefined; // CHANGED: Expect string from backend
  relations?: Array<{ id: number; type: string; word: string; sources?: string | string[] }>;
  confidence_score?: number;
  is_verified?: boolean;
  verified_by?: string;
  verified_at?: string;
  created_at?: string;
  updated_at?: string;
}

// Type for the CLEANED definition data used in the frontend
export interface Definition extends Omit<RawDefinition, 'examples' | 'usage_notes' | 'tags' | 'sources' | 'relations' | 'standardized_pos_id'> {
  text: string; // Ensure text is always present
  examples: string[]; // Always an array
  usage_notes: string[]; // Always an array
  tags: string[]; // Always an array
  sources: string[]; // Always an array
  part_of_speech: PartOfSpeech | null; // Use the nested object
  relations: Array<{ id: number; type: string; word: string; sources: string[] }>; // Ensure sources is array

  // ADD: Definition Categories, Links, and Relations
  categories?: DefinitionCategory[];
  links?: DefinitionLink[];
  definition_relations?: DefinitionRelation[];
}

// Type for the RAW etymology data
export interface RawEtymology {
  id: number;
  etymology_text?: string;
  text?: string; // Alias
  normalized_components?: string; // Often a string
  components?: string | undefined; // CHANGED: Expect string from backend
  etymology_structure?: string;
  language_codes?: string | undefined; // CHANGED: Expect string from backend
  sources?: string | undefined; // CHANGED: Expect string from backend
  confidence_level?: 'low' | 'medium' | 'high';
  verification_status?: 'unverified' | 'pending' | 'verified' | 'rejected';
  verification_notes?: string;
  created_at?: string;
  updated_at?: string;
}

// Type for the CLEANED etymology data
export interface Etymology extends Omit<RawEtymology, 'components' | 'language_codes' | 'sources' | 'normalized_components'> {
  text: string; // Ensure text is always present
  components: string[]; // Always an array
  languages: string[]; // Always an array (renamed from language_codes)
  sources: string[]; // Always an array
}

export interface Pronunciation {
  id: number;
  type: string;
  value: string;
  tags?: Record<string, any> | null;
  sources?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface CleanPronunciation extends Omit<Pronunciation, 'sources' | 'value'> {
  text: string; // Use 'text' consistently in the frontend
  ipa?: string; // Extract from value if type is 'ipa'
  audio_url?: string; // Extract from value if type is 'audio'
  sources: string[]; // Always an array
}

export interface Credit {
  id: number;
  credit: string;
  created_at?: string;
  updated_at?: string;
}

export interface BasicWord {
  id: number;
  lemma: string;
  normalized_lemma?: string;
  language_code?: string;
  has_baybayin?: boolean;
  baybayin_form?: string | null;
  romanized_form?: string | null;
  completeness_score?: number;
}

export interface Relation {
  id: number;
  relation_type: string;
  metadata?: Record<string, any>;
  sources?: string | string[]; // Can be string or array
  source_word?: BasicWord;
  target_word?: BasicWord;
  // Allow string indexing for dynamic property access
  [key: string]: any;
}

export interface CleanRelation extends Omit<Relation, 'sources'> {
  sources: string[]; // Always an array
}

export interface Affixation {
  id: number;
  affix_type: string;
  sources?: string | string[]; // Can be string or array
  created_at?: string;
  updated_at?: string;
  root_word?: BasicWord;
  affixed_word?: BasicWord;
}

export interface CleanAffixation extends Omit<Affixation, 'sources'> {
  sources: string[]; // Always an array
}

export interface RelatedWord {
  id: number;
  lemma: string;
  normalized_lemma?: string | null;
  language_code?: string | null;
  has_baybayin?: boolean | null;
  baybayin_form?: string | null;
}

// Relations structure used in WordInfo
export interface Relations {
  synonyms?: RelatedWord[];
  antonyms?: RelatedWord[];
  variants?: RelatedWord[];
  related?: RelatedWord[];
  kaugnay?: RelatedWord[];
  derived?: RelatedWord[];
  derived_from?: RelatedWord[];
  root?: RelatedWord | null;
  root_of?: RelatedWord[];
  component_of?: RelatedWord[];
  cognate?: RelatedWord[];
  main?: RelatedWord[];
  derivative?: RelatedWord[];
  etymology?: RelatedWord[];
  associated?: RelatedWord[];
  other?: RelatedWord[];
  // Allow flexible keys, but specific types above are preferred
  [key: string]: RelatedWord[] | RelatedWord | null | undefined;
}

export interface Idiom {
  phrase?: string;
  text?: string;
  meaning?: string;
  example?: string;
  source?: string;
  // Allow other potential fields from JSON
  [key: string]: any;
}

// Type for the raw comprehensive API response
export interface RawWordComprehensiveData {
  id: number;
  lemma: string;
  normalized_lemma?: string;
  language_code?: string;
  has_baybayin?: boolean;
  baybayin_form?: string | null;
  romanized_form?: string | null;
  root_word_id?: number | null;
  preferred_spelling?: string | null;
  tags?: string | undefined; // CORRECTED: Expect string or undefined from backend
  data_hash?: string;
  search_text?: string;
  created_at?: string;
  updated_at?: string;
  definitions?: RawDefinition[];
  etymologies?: RawEtymology[];
  pronunciations?: Pronunciation[];
  credits?: Credit[];
  root_word?: BasicWord | null;
  derived_words?: BasicWord[];
  outgoing_relations?: Relation[];
  incoming_relations?: Relation[];
  root_affixations?: Affixation[];
  affixed_affixations?: Affixation[];
  data_completeness?: Record<string, boolean>;
  relation_summary?: Record<string, number>;
  // Allow other fields that might exist
  [key: string]: any;
}

// Create a new WordInfo interface with all required fields
export interface WordInfo {
  id: number;
  lemma: string;
  normalized_lemma?: string | null;
  language_code?: string | null;
  has_baybayin?: boolean | null;
  baybayin_form?: string | null;
  romanized_form?: string | null;
  root_word_id?: number | null;
  preferred_spelling?: string | null;
  tags?: string | null; // Backend returns string
  data_hash?: string | null;
  search_text?: string | null;
  created_at?: string | null; // ISO string
  updated_at?: string | null; // ISO string

  // Related data - Ensure these match backend structure and names
  definitions?: Definition[] | null;
  etymologies?: Etymology[] | null;
  pronunciations?: Pronunciation[] | null; // Array from backend
  credits?: Credit[] | null;
  root_word?: RelatedWord | null;
  derived_words?: RelatedWord[] | null;
  outgoing_relations?: Relation[] | null; // Array from backend
  incoming_relations?: Relation[] | null; // Array from backend
  root_affixations?: Affixation[] | null;
  affixed_affixations?: Affixation[] | null;

  // ADDED Missing Fields
  forms?: WordForm[] | null;
  templates?: WordTemplate[] | null;
  idioms?: Idiom[] | Record<string, any> | null; // Backend might send array or object
  badlit_form?: string | null;
  hyphenation?: Record<string, any> | null;
  is_proper_noun?: boolean | null;
  is_abbreviation?: boolean | null;
  is_initialism?: boolean | null;
  word_metadata?: Record<string, any> | null; // Add word_metadata if backend provides it
  source_info?: Record<string, any> | null; // Add source_info if backend provides it
  
  // NEW fields from API improvements
  pronunciation_data?: Record<string, any> | null; // Added for API improvements
  is_root?: boolean | null; // Computed property based on root_word_id

  // Added backend fields
  data_completeness?: Record<string, boolean> | null;
  relation_summary?: Record<string, number> | null;
  
  // Added semantic network data from graph
  semantic_network?: {
    nodes: NetworkNode[];
    links: NetworkLink[];
  } | null;
  
  // Added server error field for error handling
  server_error?: string | null;
}

// Search Result Types
export interface SearchWordResult extends BasicWord {
  definitions?: Array<{ id: number; definition_text: string; part_of_speech: string | null }>;
  completeness_score?: number;
}

export interface SearchResult {
  words: SearchWordResult[];
  total: number;
  page: number;
  perPage: number;
  query: string;
  error?: string; // Optional error message
}

// Other existing types remain the same...
export interface NetworkNode {
  id: number;
  word: string;
  label: string;
  language?: string;
  type: string;
  depth?: number;
  definitions?: string[];
  path?: Array<{ type: string; word: string }>;
  has_baybayin?: boolean;
  baybayin_form?: string | null;
}

export interface NetworkWordInfo {
  id: number;
  word: string;
  definitions: string[];
  relations: {
    synonyms: Array<{ word: string; sources: string[] }>;
    antonyms: Array<{ word: string; sources: string[] }>;
    derived: Array<{ word: string; sources: string[] }>;
    related: Array<{ word: string; sources: string[] }>;
    root: { word: string; sources: string[] } | null;
  };
  depth: number;
  cluster: string | null;
}

export interface WordNetwork {
  nodes: Array<{
    id: number;           // Numeric ID from API
    label: string;        // The word string, used as label (from API)
    word: string;         // Keep for now, maybe used elsewhere?
    language?: string;    // Language code from API
    type: string;         // Added: Relationship type (e.g., 'synonym', 'root')
    depth?: number;       // Added: Calculated depth in graph traversal
    definitions?: string[]; // Optional definitions
    path?: Array<{ type: string; word: string }>; // Optional path info
    has_baybayin?: boolean; // ADDED: Optional Baybayin flag
    baybayin_form?: string | null; // ADDED: Optional Baybayin form
    // Add other useful properties if needed, e.g., normalized_lemma?
  }>;
  edges: Array<{
    source: number; // Keep as number to match node ID
    target: number; // Keep as number to match node ID
    type: string;
    // Add other properties if needed
  }>;
  metadata: {
    root_word: string;
    normalized_lemma: string;
    language_code: string;
    depth: number;
    total_nodes: number;
    total_edges: number;
    include_affixes?: boolean;
    include_etymology?: boolean;
    cluster_threshold?: number;
  };
}

export interface SearchOptions {
  page?: number;
  per_page?: number;
  mode?: 'all' | 'exact' | 'prefix' | 'suffix' | 'fuzzy';
  sort?: 'relevance' | 'alphabetical' | 'created' | 'updated' | 'completeness';
  order?: 'asc' | 'desc';
  
  // Filter parameters
  language?: string;
  pos?: string;
  
  // Feature filters
  has_etymology?: boolean;
  has_pronunciation?: boolean;
  has_baybayin?: boolean;
  exclude_baybayin?: boolean;
  has_forms?: boolean;
  has_templates?: boolean;
  
  // Advanced boolean filters
  is_root?: boolean;
  is_proper_noun?: boolean;
  is_abbreviation?: boolean;
  is_initialism?: boolean;
  
  // Date range filters
  date_added_from?: string;
  date_added_to?: string;
  date_modified_from?: string;
  date_modified_to?: string;
  
  // Definition and relation count filters
  min_definition_count?: number;
  max_definition_count?: number;
  min_relation_count?: number;
  max_relation_count?: number;
  
  // Completeness score range
  min_completeness?: number;
  max_completeness?: number;
  
  // Include options
  include_full?: boolean;
  include_definitions?: boolean;
  include_pronunciations?: boolean;
  include_etymologies?: boolean;
  include_relations?: boolean;
  include_forms?: boolean;
  include_templates?: boolean;
  include_metadata?: boolean;
  include_related_words?: boolean;
  include_definition_relations?: boolean;
}

// Etymology Tree Types (assuming backend structure is relatively stable)
export interface ComponentWord {
  id: number;
  word: string;
  normalized_lemma: string;
  language: string;
  has_baybayin: boolean;
  baybayin_form: string | null;
  romanized_form: string | null;
  etymologies: Array<{
    id: number;
    text: string;
    etymology_text?: string;
    languages: string[];
    sources: string[];
  }>;
  components: string[];
  component_words: ComponentWord[];
}

export interface EtymologyTree {
  word: string; // From backend 'word'
  etymology_tree: any; // The actual tree structure from backend
  complete: boolean; // Flag indicating if generation finished
  // Added nodes and edges for frontend display
  nodes: Array<{
    id: number;
    label: string;
    language?: string;
    [key: string]: any;
  }>;
  edges: Array<{
    source: number;
    target: number;
    [key: string]: any;
  }>;
  // Simplified structure for frontend use if needed - transformation required
  // id?: number;
  // language?: string;
  // has_baybayin?: boolean;
  // baybayin_form?: string | null;
  // romanized_form?: string | null;
  // etymologies?: Array<{ ... }>;
  // components?: string[];
  // component_words?: ComponentWord[];
  metadata?: { // If we decide to add metadata back
    word: string;
    max_depth: number;
  };
}

// Statistics Types (align with /test endpoint)
export interface Statistics {
  total_words: number;
  total_definitions: number;
  total_etymologies: number;
  total_relations: number;
  words_with_examples?: number; // Optional based on schema
  words_with_etymology?: number; // Optional based on schema
  words_with_relations?: number; // Optional based on schema
  words_with_baybayin?: number; // Optional based on schema
  words_by_language: Record<string, number>;
  words_by_pos: Record<string, number>;
  verification_stats?: Record<string, number>; // Optional based on schema
  quality_distribution?: Record<string, number>; // Optional based on schema
  update_frequency?: Record<string, number>; // Optional based on schema
  timestamp: string; // ISO timestamp string
}

export interface WordNetworkOptions {
  depth?: number;
  include_affixes?: boolean;
  include_etymology?: boolean;
  cluster_threshold?: number;
}

// Add this new interface to fix compilation errors
export interface WordNetworkGraph extends WordNetwork {
  // This inherits all properties from WordNetwork and can be extended if needed
}

// Add this after the NetworkNode interface
export interface NetworkLink {
  source: number | { id: number };
  target: number | { id: number };
  type: string;
  value?: number;
  [key: string]: any;
}

// ADD: DefinitionCategory type
export interface DefinitionCategory {
  id: number;
  definition_id: number;
  category_name: string;
  description?: string | null;
  category_kind?: string | null;
  tags?: Record<string, any> | null;
  category_metadata?: Record<string, any> | null;
  parents?: string[];
}

// ADD: DefinitionLink type
export interface DefinitionLink {
  id: number;
  definition_id: number;
  link_text: string;
  target_url: string;
  display_text?: string | null;
  is_external?: boolean;
  tags?: Record<string, any> | null;
  link_metadata?: Record<string, any> | null;
}

// ADD: DefinitionRelation type
export interface DefinitionRelation {
  id: number;
  definition_id: number;
  word_id: number; // ID of the related word
  relation_type: string;
  relation_data?: Record<string, any> | null;
  related_word?: BasicWord; // Include basic info of the related word
}

// ADD: WordForm Type
export interface WordForm {
  id: number;
  form: string;
  tags?: Record<string, any> | null;
  is_canonical?: boolean;
  is_primary?: boolean;
}

// ADD: WordTemplate Type
export interface WordTemplate {
  id: number;
  template_name: string;
  args?: Record<string, any> | null;
  expansion?: string | null;
}