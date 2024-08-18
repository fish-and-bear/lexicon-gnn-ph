// Define the structure for a single meaning and its source
export interface Meaning {
  definition: string;
  source?: string;
}

// Define the structure for a single definition, including part of speech, meanings, and examples
export interface Definition {
  partOfSpeech: string;
  meanings: Meaning[];
  usageNotes?: string[];
  examples?: string[];
  tags?: string[];
}

// Define the structure for pronunciation
export interface Pronunciation {
  text?: string;
  ipa?: string;
  audio?: string[];
}

// Define the structure for etymology
export interface Etymology {
  kaikki?: string;
  components?: { component: string; order: number }[];
  text?: string[];
}

// Define the structure for relationships
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

// Define the structure for forms
export interface Form {
  form: string;
  tags?: string[];
}

// Define the structure for head templates
export interface HeadTemplate {
  name: string;
  args?: Record<string, any>;
  expansion?: string;
}

// Define the structure for inflections
export interface Inflection {
  name: string;
  args?: Record<string, any>;
}

// Define the structure for the simplified word information used in the network
export interface NetworkWordInfo {
  word: string;
  pronunciation?: string;
  languages?: string[];
  definitions?: {
    part_of_speech: string;
    meanings: {
      definition: string;
      source?: string;
    }[];
  }[];
  related_words: string[];
}
// Define the structure for the word network, mapping words to their simplified info
export interface WordNetwork {
  [key: string]: NetworkWordInfo;
}

export interface WordInfo {
  meta: {
    version: string;
    word: string;
    timestamp: string;
  };
  data: {
    word: string;
    pronunciation?: {
      text?: string;
      ipa?: string;
      audio?: string[];
    };
    etymology?: {
      kaikki?: string;
      components?: { component: string; order: number }[];
      text?: string[];
      parsed?: string[];
    };
    definitions: {
      partOfSpeech: string;
      meanings: {
        definition: string;
        source?: string;
      }[];
      usageNotes?: string[];
      examples?: string[];
      tags?: string[];
    }[];
    relationships: {
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
    };
    forms?: { form: string; tags: string[] }[];
    languages?: string[];
    tags?: string[];
    headTemplates?: {
      name: string;
      args?: Record<string, any>;
      expansion?: string;
    }[];
    inflections?: {
      name: string;
      args?: Record<string, any>;
    }[];
    alternateForms?: string[];
  };
}
