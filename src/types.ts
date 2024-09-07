export interface Meaning {
    definition: string;
    source?: string;
  }
  
  export interface Definition {
    partOfSpeech: string;
    meanings: Meaning[];
    usageNotes?: string[];
    examples?: string[];
    tags?: string[];
  }
  
  export interface Pronunciation {
    text?: string;
    ipa?: string;
    audio?: string[];
  }
  
  export interface Etymology {
    kaikki?: string;
    components?: { component: string; order: number }[];
    text?: string[];
    parsed?: string[];
  }
  
  export interface Relationships {
    rootWord?: string;
    derivatives?: string[];
    synonyms?: string[];
    antonyms?: string[];
    associatedWords?: string[];
    relatedTerms?: string[];
    hypernyms?: string[];
    hyponyms?: string[];
    meronyms?: string[];
    holonyms?: string[];
  }
  
  export interface Form {
    form: string;
    tags?: string[];
  }
  
  export interface HeadTemplate {
    name: string;
    args?: Record<string, any>;
    expansion?: string;
  }
  
  export interface Inflection {
    name: string;
    args?: Record<string, any>;
  }
  
  export interface WordInfo {
    meta: {
      version: string;
      word: string;
      timestamp: string;
    };
    data: {
      word: string;
      pronunciation?: Pronunciation;
      etymology?: Etymology;
      definitions: Definition[];
      relationships: Relationships;
      forms?: Form[];
      languages?: string[];
      tags?: string[];
      headTemplates?: HeadTemplate[];
      inflections?: Inflection[];
      alternateForms?: string[];
    };
  }
  
  export interface NetworkWordInfo {
    word: string;
    definition?: string;  // Add this line
    definitions?: Array<{
      partOfSpeech: string;
      meanings: Array<{
        definition: string;
        source?: string;
      }>;
    }>;
    derivatives?: string[];
    root_word?: string;
    synonyms?: string[];
    antonyms?: string[];
    associated_words?: string[];
    related_terms?: string[];
    hypernyms?: string[];
    hyponyms?: string[];
    meronyms?: string[];
    holonyms?: string[];
    etymology?: {
      parsed?: string[];
      text?: string[];
      components?: { component: string; order: number }[];
    };
  }
  
  export interface WordNetwork {
    [key: string]: NetworkWordInfo;
  }
  
  // Add these types to your types.ts file
  export interface SearchOptions {
    page?: number;
    per_page?: number;
    exclude_baybayin?: boolean;
  }

  export interface SearchResult {
    words: Array<{ id: number; word: string }>;
    page: number;
    perPage: number;
    total: number;
  }