export interface PartOfSpeech {
  code: string;
  name_en: string;
  name_tl: string;
  description?: string;
}

export interface Definition {
  text: string;
  part_of_speech: PartOfSpeech | null;
  examples: string[];
  usage_notes: string[];
  sources: string;
  relations: Array<{
    word: string;
    type: string;
    sources: string;
  }>;
}

export interface Etymology {
  text: string;
  normalized_components: string[];
  language_codes: string[];
  sources: string;
}

export interface Affixation {
  affixed_word?: string;
  root_word?: string;
  type: string;
  sources: string;
}

export interface Relations {
  synonyms: Array<{ word: string; sources: string }>;
  antonyms: Array<{ word: string; sources: string }>;
  related: Array<{ word: string; sources: string }>;
  derived: Array<{ word: string; sources: string }>;
  root: { word: string; sources: string } | null;
}

export interface BaybayinInfo {
  has_baybayin: boolean;
  form: string | null;
  romanized: string | null;
}

export interface WordInfo {
  meta: {
    version: string;
    word: string;
    timestamp: string;
  };
  data: {
    word: string;
    normalized_lemma: string;
    language_code: string;
    baybayin: BaybayinInfo | null;
    pronunciation: any | null;
    preferred_spelling: string | null;
    tags: string[];
    idioms: any[];
    etymologies: Etymology[];
    definitions: Definition[];
    relations: Relations;
    affixations: {
      as_root: Affixation[];
      as_affixed: Affixation[];
    };
    source_info: Record<string, any>;
  };
}

export interface NetworkNode {
  word: string;
  definition: string;
  synonyms: string[];
  antonyms: string[];
  derived: string[];
  related: string[];
  root: string | null;
}

export interface WordNetwork {
  [key: string]: NetworkNode;
}

export interface SearchOptions {
  page: number;
  per_page: number;
  exclude_baybayin: boolean;
  pos?: string;
  source?: string;
}

export interface SearchResult {
  words: Array<{ id: number; word: string }>;
  page: number;
  perPage: number;
  total: number;
}