export interface PartOfSpeech {
  id: number;
  code: string;
  name_en: string;
  name_tl: string;
  description?: string | null;
}

// Type for the RAW definition data possibly coming from the API
export interface RawDefinition {
  id: number;
  definition_text?: string;
  original_pos?: string | null;
  standardized_pos_id?: number | null;
  standardized_pos?: PartOfSpeech | null;
  examples?: string | null;
  usage_notes?: string | null;
  tags?: string | null;
  sources?: string | null;
  related_definitions?: RawDefinition[];
  created_at?: string;
  updated_at?: string;
}

// Type for the CLEANED definition data used in the frontend
export interface Definition extends Omit<RawDefinition, 'examples' | 'usage_notes' | 'tags' | 'sources' | 'standardized_pos_id' | 'related_definitions'> {
  definition_text: string;
  examples: string[];
  usage_notes: string[];
  tags: string[];
  sources: string[];
  part_of_speech: PartOfSpeech | null;
  related_definitions?: Definition[];
}

// Type for the RAW etymology data
export interface RawEtymology {
  id: number;
  etymology_text?: string;
  normalized_components?: string | null;
  etymology_structure?: string | null;
  language_codes?: string | null;
  sources?: string | null;
  created_at?: string;
  updated_at?: string;
}

// Type for the CLEANED etymology data
export interface Etymology extends Omit<RawEtymology, 'language_codes' | 'sources' | 'normalized_components'> {
  etymology_text: string;
  components: string[];
  languages: string[];
  sources: string[];
}

// Type for CLEANED pronunciation data
// export interface Pronunciation extends Omit<RawPronunciation, 'sources' | 'tags' | 'pronunciation_metadata'> {
//   sources: string[]; // Always an array
//   tags: Record<string, any>; // Processed JSON
//   pronunciation_metadata: Record<string, any>; // Processed JSON
//   // Add extracted fields if needed by components
//   ipa?: string; // Extracted from value if type is 'ipa'
//   audio_url?: string; // Extracted from value if type is 'audio'
// }

// Redefined CLEANED pronunciation data without Omit to avoid linter issues
export interface Pronunciation {
  // Fields directly from RawPronunciation
  id: number;
  type: string;
  value: string; // Keep original value for reference if needed
  created_at?: string | null;
  updated_at?: string | null;

  // Cleaned/Processed fields
  sources?: string[]; // Optional array (processed from raw string/null)
  tags?: Record<string, any>; // Optional object (processed from raw string/null/object)
  pronunciation_metadata?: Record<string, any>; // Optional object (processed from raw string/null/object)

  // Add extracted fields if needed by components
  ipa?: string; // Extracted from value if type is 'ipa'
  audio_url?: string; // Extracted from value if type is 'audio'
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
  metadata?: Record<string, any> | string | null;
  sources?: string | null;
  source_word?: BasicWord;
  target_word?: BasicWord;
  created_at?: string;
  updated_at?: string;
}

export interface Affixation {
  id: number;
  affix_type: string;
  sources?: string | null;
  created_at?: string;
  updated_at?: string;
  root_word?: BasicWord;
  affixed_word?: BasicWord;
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
  tags?: string | null;
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
  tags?: string | null;
  data_hash?: string | null;
  search_text?: string | null;
  created_at?: string | null;
  updated_at?: string | null;

  definitions?: Definition[] | null;
  etymologies?: Etymology[] | null;
  pronunciations?: Pronunciation[] | null;
  credits?: Credit[] | null;
  root_word?: RelatedWord | null;
  derived_words?: RelatedWord[] | null;
  outgoing_relations?: Relation[] | null;
  incoming_relations?: Relation[] | null;
  root_affixations?: Affixation[] | null;
  affixed_affixations?: Affixation[] | null;

  data_completeness?: Record<string, boolean> | null;
  relation_summary?: Record<string, number> | null;
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
  lemma: string;
  language_code?: string | null;
}

export interface NetworkEdge {
  id: string;
  source: number;
  target: number;
  type: string;
  metadata?: Record<string, any> | string | null;
}

export interface WordNetworkResponse {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  stats: {
    node_count: number;
    edge_count: number;
    depth: number;
  };
}

export interface SearchOptions {
  q: string;
  mode?: 'all' | 'exact' | 'prefix' | 'baybayin' | 'fuzzy' | 'etymology' | 'semantic' | 'root' | 'affixed';
  language?: string | null;
  pos?: 'n' | 'v' | 'adj' | 'adv' | 'pron' | 'prep' | 'conj' | 'intj' | 'det' | 'affix' | null;
  include_full?: boolean;
  sort?: 'relevance' | 'alphabetical' | 'created' | 'updated' | 'quality' | 'frequency' | 'complexity';
  order?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
  page?: number;
  per_page?: number;
  exclude_baybayin?: boolean;
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
  word: string;
  etymology_tree: any;
  complete: boolean;
  metadata?: {
    word: string;
    max_depth: number;
  };
}

// Statistics Types (align with /test endpoint)
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
}

export interface WordNetworkOptions {
  depth?: number;
  include_affixes?: boolean;
  include_etymology?: boolean;
  cluster_threshold?: number;
}

// --- Language ---
export interface Language {
  code: string;
  name_en?: string | null;
  name_tl?: string | null;
  description?: string | null;
}

// --- Word Form --- Added based on backend schema
export interface WordForm {
  id: number;
  form: string;
  tags?: Record<string, any> | string | null; // Backend JSONB, might be string or object
  is_canonical?: boolean | null;
  is_primary?: boolean | null;
  created_at?: string;
  updated_at?: string;
}

// --- Word Template --- Added based on backend schema
export interface WordTemplate {
  id: number;
  template_name: string;
  args?: Record<string, any> | string | null; // Backend JSONB, might be string or object
  expansion?: string | null;
  created_at?: string;
  updated_at?: string;
}

// Type for the raw comprehensive API response
// Renamed from RawWordComprehensiveData for clarity
export interface RawWordData {
  id: number;
  lemma: string;
  normalized_lemma?: string;
  language_code?: string;
  has_baybayin?: boolean;
  baybayin_form?: string | null;
  romanized_form?: string | null;
  root_word_id?: number | null;
  preferred_spelling?: string | null;
  tags?: string | null;
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
  forms?: WordForm[] | null;
  templates?: WordTemplate[] | null;
  language?: Language | null;
  data_completeness?: Record<string, boolean | number | null> | null;
  relation_summary?: Record<string, number>;
  [key: string]: any;
}

// CLEAN WordInfo type used throughout the frontend application after processing RawWordData
export interface WordInfo extends Omit<RawWordData,
  'definitions' | 'etymologies' | 'pronunciations' | 'outgoing_relations' | 'incoming_relations' |
  'root_affixations' | 'affixed_affixations' | 'forms' | 'templates' | 'language' |
  'tags' | 'idioms' | 'source_info' | 'word_metadata' | 'hyphenation'
> {
  root_affixations?: Affixation[] | null;
  affixed_affixations?: Affixation[] | null;
  forms?: WordForm[] | null;
  templates?: WordTemplate[] | null;
  language?: Language | null;
  completeness_score?: number | null;
}

// Search Result Types - Align with backend /search endpoint
// Type for individual result item when include_full=false
export interface SearchResultItem {
  id: number;
  lemma: string;
  normalized_lemma?: string | null;
  language_code?: string | null;
  has_baybayin?: boolean | null;
  baybayin_form?: string | null;
  romanized_form?: string | null;
  root_word_id?: number | null;
  is_root?: boolean | null;
  // Add other fields if backend sends more in simple mode
}

// Type for the overall search response
export interface SearchResponse {
  count: number;
  offset: number;
  pos?: string | null;
  results: SearchResultItem[] | RawWordData[];
}