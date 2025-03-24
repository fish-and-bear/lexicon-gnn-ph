export interface PartOfSpeech {
  id: number;
  code: string;
  name_en: string;
  name_tl: string;
  description?: string;
}

export interface Definition {
  id: number;
  text: string;
  definition_text?: string;
  original_pos: string | null;
  part_of_speech: PartOfSpeech | null;
  examples: string[];
  usage_notes: string[];
  sources: string[];
  relations: Array<{
    id: number;
    type: string;
    word: string;
    sources: string[];
  }>;
  confidence_score: number;
  is_verified: boolean;
  verified_by?: string;
  verified_at?: string;
}

export interface Etymology {
  id: number;
  text: string;
  etymology_text?: string;
  components: string[];
  languages: string[];
  sources: string[];
  confidence_level: 'low' | 'medium' | 'high';
  verification_status: 'unverified' | 'pending' | 'verified' | 'rejected';
  verification_notes?: string;
}

export interface Affixation {
  id: number;
  type: string;
  root_word: string;
  affixed_word: string;
  sources: string[];
  created_at: string;
}

export interface Relation {
  id: number;
  type: string;
  from_word: string;
  to_word: string;
  sources: string[];
  created_at: string;
}

export interface RelatedWord {
  word: string;
  sources: string[];
}

// Extended to support more relation types
export interface Relations {
  // Core Filipino dictionary relation types
  synonyms?: RelatedWord[];            // Words with the same or similar meaning
  antonyms?: RelatedWord[];            // Words with opposite meanings
  variants?: RelatedWord[];            // Alternative forms or spellings of this word
  related?: RelatedWord[];             // Words semantically related to this word
  kaugnay?: RelatedWord[];             // Filipino for "related" - culturally specific relationships
  
  // Etymological and derivational relations
  derived?: RelatedWord[];             // Words derived from this word
  derived_from?: RelatedWord[];        // Words from which this word is derived
  root?: RelatedWord | null;           // The root word of this word
  root_of?: RelatedWord[];             // Words for which this word serves as a root
  
  // Structural relations
  component_of?: RelatedWord[];        // Words that use this word as a component
  cognate?: RelatedWord[];             // Words sharing the same linguistic origin
  
  // Semantic categorizations
  main?: RelatedWord[];                // Primary words related to this term
  derivative?: RelatedWord[];          // Derivative relationships
  etymology?: RelatedWord[];           // Words related through etymology
  associated?: RelatedWord[];          // Words associated with this term  
  other?: RelatedWord[];               // Other uncategorized relationships
  
  [key: string]: any;                  // Allow for additional relation types
}

export interface BaybayinInfo {
  has_baybayin: boolean;
  baybayin_form: string | null;
  romanized_form: string | null;
}

export interface Idiom {
  phrase?: string;
  text?: string;
  meaning?: string;
  example?: string;
  source?: string;
}

export interface WordInfo {
  id: number;
  lemma: string;
  normalized_lemma: string;
  language_code: string;
  preferred_spelling: string | null;
  tags: string[];
  has_baybayin: boolean;
  baybayin_form: string | null;
  romanized_form: string | null;
  pronunciation: {
    text: string;
    ipa?: string;
    audio_url?: string;
  } | null;
  source_info: Record<string, any>;
  data_hash: string;
  complexity_score: number;
  usage_frequency: number;
  created_at: string;
  updated_at: string;
  last_lookup_at: string | null;
  view_count: number;
  last_viewed_at: string | null;
  is_verified: boolean;
  verification_notes: string | null;
  data_quality_score: number;
  definitions: Definition[];
  etymologies: Etymology[];
  relations: Relations;
  idioms?: Idiom[] | string[]; // Support both object and string arrays
}

export interface NetworkNode {
  id: number;
  word: string;
  normalized_lemma: string;
  language: string;
  has_baybayin: boolean;
  baybayin_form: string | null;
  type: string;
  path: Array<{ type: string; word: string }>;
  definitions?: string[];
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
  main_words: string[];
  root_words: string[];
  antonyms: string[];
  derived_words: string[];
  related_words: string[];
  synonyms: string[];
  kaugnay: string[];
  other: string[];
}

export interface SearchOptions {
  page: number;
  per_page: number;
  exclude_baybayin: boolean;
  pos?: string;
  source?: string;
  language?: string;
  mode?: 'all' | 'exact' | 'phonetic' | 'baybayin';
  sort?: 'relevance' | 'alphabetical' | 'created' | 'updated';
  order?: 'asc' | 'desc';
  is_real_word?: boolean;
}

export interface SearchResult {
  words: Array<{ id: number; word: string }>;
  page: number;
  perPage: number;
  total: number;
}

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
    languages: string[];
    sources: string[];
  }>;
  components: string[];
  component_words: ComponentWord[];
  metadata: {
    word: string;
    normalized_lemma: string;
    language: string;
    max_depth: number;
    group_by_language: boolean;
  };
}

export interface Statistics {
  words: {
    total: number;
    with_definitions: number;
    with_etymology: number;
    with_baybayin: number;
    by_language: Record<string, number>;
  };
  definitions: {
    total: number;
    verified: number;
    by_pos: Record<string, number>;
  };
  relations: {
    total: number;
    by_type: Record<string, number>;
  };
  sources: {
    total: number;
    by_name: Record<string, number>;
  };
  last_updated: string;
}

export interface WordNetworkOptions {
  depth?: number;
  include_affixes?: boolean;
  include_etymology?: boolean;
  cluster_threshold?: number;
}