import axios, { AxiosError, AxiosResponse, AxiosRequestConfig } from 'axios';
import { 
  WordInfo, 
  SearchOptions, 
  SearchResult, 
  Etymology,
  PartOfSpeech,
  Statistics,
  EtymologyTree,
  Definition,
  Credit,
  SearchWordResult,
  RawWordComprehensiveData,
  Pronunciation,
  Relation,
  Affixation,
  RelatedWord,
  WordNetwork as ImportedWordNetwork
} from "../types";

// --- Dummy Exports ---

export async function fetchWordNetwork(word: string, options: any = {}): Promise<any> {
  console.warn('DUMMY fetchWordNetwork called');
  return Promise.resolve({ nodes: [], edges: [], metadata: {} }); 
}

export async function fetchWordDetails(word: string): Promise<any> {
  console.warn('DUMMY fetchWordDetails called');
  return Promise.resolve({ id: 0, lemma: 'dummy', definitions: [], etymologies: [] });
}

export async function searchWords(query: string, options: SearchOptions): Promise<SearchResult> {
  console.warn('DUMMY searchWords called');
  return Promise.resolve({ words: [], page: 1, perPage: 0, total: 0, query });
}

export async function getRandomWord(): Promise<any> {
  console.warn('DUMMY getRandomWord called');
  return Promise.resolve({ id: 0, lemma: 'dummy_random', definitions: [], etymologies: [] });
}

export const testApiConnection = async (): Promise<boolean> => {
  console.warn('DUMMY testApiConnection called');
  return Promise.resolve(true);
};

export function resetCircuitBreaker() {
  console.warn('DUMMY resetCircuitBreaker called');
}

export async function getPartsOfSpeech(): Promise<any[]> {
  console.warn('DUMMY getPartsOfSpeech called');
  return Promise.resolve([]);
}

export async function getStatistics(): Promise<any> {
  console.warn('DUMMY getStatistics called');
  return Promise.resolve({});
}

export async function getBaybayinWords(page: number = 1, limit: number = 20, language: string = 'tl'): Promise<any> {
  console.warn('DUMMY getBaybayinWords called');
  return Promise.resolve([]);
}

export async function getAffixes(language: string = 'tl', type?: string): Promise<any> {
  console.warn('DUMMY getAffixes called');
  return Promise.resolve([]);
}

export async function getRelations(language: string = 'tl', type?: string): Promise<any> {
  console.warn('DUMMY getRelations called');
  return Promise.resolve([]);
}

export async function getAllWords(page: number = 1, perPage: number = 20, language: string = 'tl'): Promise<any> {
  console.warn('DUMMY getAllWords called');
  return Promise.resolve([]);
}

export async function getEtymologyTree(wordId: number, maxDepth: number = 2 ): Promise<any> {
  console.warn('DUMMY getEtymologyTree called');
   return Promise.resolve({ nodes: [], edges: [], word: 'dummy', etymology_tree: {}, complete: false });
}

export async function fetchWordRelations(word: string): Promise<any> {
   console.warn('DUMMY fetchWordRelations called');
   return Promise.resolve({ outgoing_relations: [], incoming_relations: [] });
}

// --- Commented out Original Code ---
/*
// Environment and configuration constants
const ENV = process.env.NODE_ENV || 'development';
// ... (rest of the original code commented out) ...
*/

// Add any other necessary dummy exports if WordExplorer imports more
// ... (Example: export const someOtherFunction = () => {};)