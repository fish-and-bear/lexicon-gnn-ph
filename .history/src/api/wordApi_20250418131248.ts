import { 
  WordInfo, 
  SearchOptions, 
  SearchResult, 
  PartOfSpeech,
  Statistics,
  EtymologyTree,
  Relation,
  WordNetwork as ImportedWordNetwork
} from "../types";
// Note: axios and other imports are removed for this test

// --- Dummy Exports ---

export async function fetchWordNetwork(word: string, options: any = {}): Promise<ImportedWordNetwork> {
  console.warn('DUMMY fetchWordNetwork called for:', word);
  return Promise.resolve({ nodes: [], edges: [], metadata: { root_word: word, normalized_lemma: word, language_code: 'tl', depth: 0, total_nodes: 0, total_edges: 0 } }); 
}

export async function fetchWordDetails(word: string): Promise<WordInfo> {
  console.warn('DUMMY fetchWordDetails called for:', word);
  // Return a structure that matches WordInfo but is minimal
  return Promise.resolve({ 
    id: 0, 
    lemma: 'dummy', 
    normalized_lemma: 'dummy', 
    language_code: 'tl', 
    definitions: [], 
    etymologies: [],
    pronunciations: [],
    credits: [],
    outgoing_relations: [],
    incoming_relations: [],
    root_affixations: [],
    affixed_affixations: [],
    // Add other required fields from WordInfo with default values if needed
    has_baybayin: false,
    root_word_id: null,
    tags: null,
  } as WordInfo); // Type assertion might be needed depending on WordInfo definition
}

export async function searchWords(query: string, options: SearchOptions): Promise<SearchResult> {
  console.warn('DUMMY searchWords called with query:', query);
  return Promise.resolve({ words: [], page: 1, perPage: 0, total: 0, query });
}

export async function getRandomWord(): Promise<WordInfo> {
  console.warn('DUMMY getRandomWord called');
  return Promise.resolve({ 
    id: 0, 
    lemma: 'dummy_random', 
    normalized_lemma: 'dummy_random', 
    language_code: 'tl', 
    definitions: [], 
    etymologies: [],
    pronunciations: [],
    credits: [],
    outgoing_relations: [],
    incoming_relations: [],
    root_affixations: [],
    affixed_affixations: [],
    has_baybayin: false,
    root_word_id: null,
    tags: null,
  } as WordInfo); // Type assertion might be needed
}

export const testApiConnection = async (): Promise<boolean> => {
  console.warn('DUMMY testApiConnection called');
  return Promise.resolve(true);
};

export function resetCircuitBreaker() {
  console.warn('DUMMY resetCircuitBreaker called');
  // No actual logic needed
}

export async function getPartsOfSpeech(): Promise<PartOfSpeech[]> {
  console.warn('DUMMY getPartsOfSpeech called');
  return Promise.resolve([]);
}

export async function getStatistics(): Promise<Statistics> {
  console.warn('DUMMY getStatistics called');
  // Return a minimal object matching Statistics type
  return Promise.resolve({ 
    total_words: 0, 
    total_definitions: 0, 
    total_relations: 0, 
    total_etymologies: 0, // Added missing field
    words_by_language: {}, 
    words_by_pos: {}, 
    timestamp: new Date().toISOString(),
    // Add other optional fields from Statistics with default values if needed
    words_with_baybayin: 0,
    words_with_etymology: 0,
    words_with_examples: 0,
    words_with_pronunciation: 0,
  });
}

export async function getBaybayinWords(page: number = 1, limit: number = 20, language: string = 'tl'): Promise<any[]> {
  console.warn('DUMMY getBaybayinWords called');
  return Promise.resolve([]);
}

export async function getAffixes(language: string = 'tl', type?: string): Promise<any[]> {
  console.warn('DUMMY getAffixes called');
  return Promise.resolve([]);
}

export async function getRelations(language: string = 'tl', type?: string): Promise<any[]> {
  console.warn('DUMMY getRelations called');
  return Promise.resolve([]);
}

export async function getAllWords(page: number = 1, perPage: number = 20, language: string = 'tl'): Promise<any[]> {
  console.warn('DUMMY getAllWords called');
  return Promise.resolve([]);
}

export async function getEtymologyTree(wordId: number, maxDepth: number = 2 ): Promise<EtymologyTree> {
  console.warn('DUMMY getEtymologyTree called for ID:', wordId);
   // Return a minimal object matching EtymologyTree type
   return Promise.resolve({ nodes: [], edges: [], word: 'dummy', etymology_tree: {}, complete: false });
}

export async function fetchWordRelations(word: string): Promise<{outgoing_relations: Relation[], incoming_relations: Relation[]}> {
   console.warn('DUMMY fetchWordRelations called for:', word);
   return Promise.resolve({ outgoing_relations: [], incoming_relations: [] });
}

// Add any other functions imported by WordExplorer as needed, with dummy implementations

// Original code is completely removed/commented out for this test