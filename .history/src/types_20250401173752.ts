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
}

export interface Relation {
  id: number;
  relation_type: string;
  metadata?: Record<string, any>;
  sources?: string | string[]; // Can be string or array
  source_word?: BasicWord;
  target_word?: BasicWord;
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

// CLEAN WordInfo type used throughout the frontend application
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

  // Added backend fields
  data_completeness?: Record<string, boolean> | null;
  relation_summary?: Record<string, number> | null;

  // REMOVED Fields not in comprehensive response:
  // pronunciation?: Pronunciation | null;
  // relations?: Record<string, string[]>;
  // idioms?: (string | IdiomObject)[];
  // source_info?: Record<string, any>;
  // badlit_form?: string | null;
  // hyphenation?: Record<string, any>;
  // is_proper_noun?: boolean | null;
  // is_abbreviation?: boolean | null;
  // is_initialism?: boolean | null;
  // is_root?: boolean | null;
}

// Search Result Types
export interface SearchWordResult extends BasicWord {
  definitions?: Array<{ id: number; definition_text: string; part_of_speech: string | null }>;
}

export interface SearchResult {
  words: SearchWordResult[];
  total: number;
  page: number;
  perPage: number;
  query: string;
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
  word: string;
  nodes: NetworkNode[];
  edges: Array<{
    source: number;
    target: number;
    type: string;
    sources: string[];
  }>;
  clusters: Record<string, number[]>;
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
  page?: number; // Optional
  per_page?: number; // Optional
  exclude_baybayin?: boolean;
  pos?: string;
  source?: string;
  language?: string;
  mode?: 'all' | 'exact' | 'phonetic' | 'baybayin' | 'fuzzy' | 'etymology' | 'semantic' | 'root' | 'affixed';
  sort?: 'relevance' | 'alphabetical' | 'created' | 'updated' | 'quality' | 'frequency' | 'complexity';
  order?: 'asc' | 'desc';
  is_real_word?: boolean;
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
  api_version: string;
  database: {
    connected: boolean;
    language_count?: number;
    word_count?: number;
    stats_error?: string;
  };
  message: string;
  status: string;
  timestamp: string;
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