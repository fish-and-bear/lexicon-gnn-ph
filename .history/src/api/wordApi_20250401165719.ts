import axios, { AxiosError, AxiosResponse, AxiosRequestConfig } from 'axios';
import { 
  WordNetwork, 
  WordInfo, 
  SearchOptions, 
  SearchResult, 
  Etymology,
  PartOfSpeech,
  Statistics,
  EtymologyTree,
  Definition,
  CleanPronunciation,
  CleanRelation,
  CleanAffixation,
  Relations,
  Idiom,
  Credit,
  BasicWord,
  SearchWordResult,
  RawWordComprehensiveData,
  RawDefinition,
  RawEtymology,
  Pronunciation,
  Relation,
  Affixation,
  RelatedWord
} from "../types";
import { sanitizeInput } from '../utils/sanitizer';
import { getCachedData, setCachedData, clearOldCache } from '../utils/caching';

// Environment and configuration constants
const ENV = process.env.NODE_ENV || 'development';
const CONFIG = {
  development: {
    baseURL: 'http://localhost:10000/api/v2',
    timeout: 10000,
    retries: 3,
    failureThreshold: 5,
    resetTimeout: 60000,
    retryDelay: 1000,
    maxRetryDelay: 10000
  },
  production: {
    baseURL: process.env.REACT_APP_API_BASE_URL?.replace('/api/v1', '/api/v2') || 'https://fil-relex.onrender.com/api/v2',
    timeout: 5000,
    retries: 2,
    failureThreshold: 3,
    resetTimeout: 30000,
    retryDelay: 2000,
    maxRetryDelay: 8000
  },
  test: {
    baseURL: 'http://localhost:10000/api/v2',
    timeout: 1000,
    retries: 0,
    failureThreshold: 1,
    resetTimeout: 5000,
    retryDelay: 100,
    maxRetryDelay: 1000
  }
}[ENV];

if (!CONFIG.baseURL) {
  throw new Error('API_BASE_URL environment variable is not set');
}

// Extended Axios types
interface ExtendedAxiosRequestConfig extends AxiosRequestConfig {
  retry?: number;
  retryDelay?: number;
}

// Enhanced circuit breaker with persistence
class PersistentCircuitBreaker {
  private static readonly STORAGE_KEY = 'circuit_breaker_state';
  private static readonly STATE_TTL = 60 * 60 * 1000; // 1 hour

  private failures: number = 0;
  private lastFailureTime: number | null = null;
  private state: 'closed' | 'open' | 'half-open' = 'closed';

  constructor() {
    const savedState = this.loadState();
    if (savedState && Date.now() - savedState.timestamp < PersistentCircuitBreaker.STATE_TTL) {
      this.failures = savedState.failures;
      this.lastFailureTime = savedState.lastFailureTime;
      this.state = savedState.state;
    } else {
      this.reset();
    }
    
    // Always log the initial state
    console.log('Circuit breaker initialized with state:', this.getState());
  }

  private loadState() {
    try {
      const state = localStorage.getItem(PersistentCircuitBreaker.STORAGE_KEY);
      return state ? JSON.parse(state) : null;
    } catch (e) {
      console.error('Error loading circuit breaker state:', e);
      return null;
    }
  }

  private saveState() {
    try {
      localStorage.setItem(PersistentCircuitBreaker.STORAGE_KEY, JSON.stringify({
        failures: this.failures,
        lastFailureTime: this.lastFailureTime,
        state: this.state,
        timestamp: Date.now()
      }));
    } catch (e) {
      console.error('Error saving circuit breaker state:', e);
    }
  }

  recordFailure() {
    this.failures++;
    this.lastFailureTime = Date.now();
    if (this.failures >= CONFIG.failureThreshold) {
      this.state = 'open';
    }
    this.saveState();
    console.log('Circuit breaker recorded failure. New state:', this.getState());
  }

  recordSuccess() {
    this.failures = 0;
    this.state = 'closed';
    this.saveState();
    console.log('Circuit breaker recorded success. New state:', this.getState());
  }

  canMakeRequest(): boolean {
    // Clean up old cache before checking state
    clearOldCache();
    
    // ALWAYS allow requests regardless of circuit breaker state
    // This ensures the circuit breaker never blocks requests
    return true;
  }

  getState() {
    return {
      state: this.state,
      failures: this.failures,
      lastFailureTime: this.lastFailureTime
    };
  }

  reset() {
    this.failures = 0;
    this.lastFailureTime = null;
    this.state = 'closed';
    this.saveState();
    console.log('Circuit breaker has been reset. New state:', this.getState());
  }
}

const circuitBreaker = new PersistentCircuitBreaker();

// Function to reset the circuit breaker state
export function resetCircuitBreaker() {
  // Reset the circuit breaker
  circuitBreaker.reset();
  
  // Clear all cache items
  try {
    // Clear circuit breaker state
    localStorage.removeItem('circuit_breaker_state');
    localStorage.removeItem('successful_api_endpoint');
    
    // Clear all cache items
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && (key.startsWith('cache:') || key.includes('circuit') || key.includes('api_endpoint'))) {
        keysToRemove.push(key);
      }
    }
    
    keysToRemove.forEach(key => localStorage.removeItem(key));
    
    // Reset API client to default configuration
    api.defaults.baseURL = CONFIG.baseURL;
    
    // Clear axios cache and reset headers
    api.defaults.headers.common = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Client-Version': process.env.REACT_APP_VERSION || '1.0.0',
      'X-Client-Platform': 'web',
      'Accept-Encoding': 'gzip, deflate, br',
      'Origin': window.location.origin,
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache'
    };
    
    console.log(`Circuit breaker has been reset. Cleared ${keysToRemove.length} cache items.`);
    console.log(`API client baseURL reset to: ${CONFIG.baseURL}`);
    
    // Test connection after reset
    setTimeout(() => {
      testApiConnection().then(connected => {
        console.log(`Connection test after reset: ${connected ? 'successful' : 'failed'}`);
      });
    }, 500);
  } catch (e) {
    console.error('Error clearing localStorage:', e);
  }
}

// Get the successful endpoint from localStorage if available
const savedEndpoint = localStorage.getItem('successful_api_endpoint');
let apiBaseURL = CONFIG.baseURL;

// Use the saved endpoint if available
if (savedEndpoint) {
  console.log('Using saved API endpoint:', savedEndpoint);
  if (savedEndpoint.includes('/api/v2')) {
    apiBaseURL = savedEndpoint;
  } else {
    apiBaseURL = `${savedEndpoint}/api/v2`;
  }
}

// API client configuration
const api = axios.create({
  baseURL: CONFIG.baseURL,
  timeout: CONFIG.timeout,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Client-Version': process.env.REACT_APP_VERSION || '1.0.0',
    'X-Client-Platform': 'web'
  },
  withCredentials: false // Disable sending credentials for CORS
});

// Add request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    // Add CORS headers
    config.headers = {
      ...config.headers,
      'Origin': window.location.origin,
      'Access-Control-Request-Method': config.method?.toUpperCase() || 'GET',
    };
    console.log(`Making ${config.method?.toUpperCase()} request to: ${config.url}`, config);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for debugging and error handling
api.interceptors.response.use(
  (response) => {
    console.log(`Response from ${response.config.url}:`, response.data);
    return response;
  },
  (error) => {
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('Response error:', {
        status: error.response.status,
        data: error.response.data,
        headers: error.response.headers,
        url: error.config?.url
      });

      // Handle specific error cases
      if (error.response.status === 404) {
        return Promise.reject({
          message: 'Resource not found',
          details: error.response.data
        });
      } else if (error.response.status === 403) {
        return Promise.reject({
          message: 'Access forbidden',
          details: error.response.data
        });
      } else if (error.response.status === 429) {
        return Promise.reject({
          message: 'Too many requests. Please try again later.',
          details: error.response.data
        });
      }
    } else if (error.request) {
      // The request was made but no response was received
      console.error('Request error (no response):', {
        request: error.request,
        url: error.config?.url
      });
      return Promise.reject({
        message: 'No response received from server',
        details: 'The server is not responding. Please try again later.'
      });
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('Error:', error.message);
      return Promise.reject({
        message: 'Request failed',
        details: error.message
      });
    }
    return Promise.reject(error);
  }
);

interface ApiError extends Error {
  response?: {
    status: number;
    statusText: string;
    data?: {
      error?: {
        message?: string;
        code?: string;
        details?: any;
        request_id?: string;
      };
    };
  };
  isAxiosError?: boolean;
  code?: string;
}

async function handleApiError(error: unknown, context: string): Promise<never> {
  const apiError = error as ApiError;
  
  if (apiError.isAxiosError) {
    if (apiError.code === 'ECONNABORTED') {
      throw new Error(`Request timeout during ${context}`);
    }
    
    if (!apiError.response) {
      throw new Error(`Network error during ${context}`);
    }

    const status = apiError.response.status;
    const errorData = apiError.response.data?.error;
    const errorMessage = errorData?.message || apiError.response.statusText;
    const errorCode = errorData?.code;
    const requestId = errorData?.request_id;
    const details = errorData?.details;

    switch (status) {
      case 400:
        throw new Error(`Invalid request during ${context}: ${errorMessage}`);
      case 404:
        throw new Error(`Resource not found during ${context}`);
      case 429:
        throw new Error(`Rate limit exceeded during ${context}. Please try again later.`);
      case 503:
        throw new Error(`Service temporarily unavailable during ${context}. Please try again later.`);
      default:
        throw new Error(
          `Error during ${context}: ${errorMessage}${requestId ? ` (Request ID: ${requestId})` : ''}`
        );
    }
  }

  throw error;
}

export interface WordNetworkOptions {
  depth?: number;
  include_affixes?: boolean;
  include_etymology?: boolean;
  cluster_threshold?: number;
}

import { 
  WordInfo, Definition, Etymology, Relation, Affixation, Pronunciation, Credit, PartOfSpeech, RelatedWord, 
  RawWordComprehensiveData // Assuming a raw type exists, otherwise use 'any'
} from '../types';
import axios from 'axios'; // Ensure axios is imported if used directly below

// Helper function to safely parse JSON strings
function safeJsonParse(jsonString: string | null | undefined): Record<string, any> | null {
  if (!jsonString) return null;
  try {
    // Handle cases where the backend might already return an object
    if (typeof jsonString === 'object') return jsonString;
    return JSON.parse(jsonString);
  } catch (e) {
    console.warn('Failed to parse JSON string:', jsonString, e);
    return null;
  }
}

// Helper function to split strings by semicolon, trimming whitespace
function splitSemicolonSeparated(value: string | undefined | null): string[] {
  if (!value) return [];
  return value.split(';').map(s => s.trim()).filter(s => s !== '');
}

// Helper function to split strings by comma, trimming whitespace
function splitCommaSeparated(value: string | undefined | null): string[] {
  if (!value) return [];
  return value.split(',').map(s => s.trim()).filter(s => s !== '');
}

// Main normalization function - REVISED
function normalizeWordData(rawData: any): WordInfo {
  // Use the actual data, potentially nested under 'data'
  const wordData: RawWordComprehensiveData = rawData?.data || rawData;

  if (!wordData || typeof wordData !== 'object' || !wordData.id) {
    // If it's an array, maybe it's a search result list? Log warning.
    if (Array.isArray(wordData)) {
         console.warn("normalizeWordData received an array, expected object. Using first element.", wordData);
         // Attempt to use the first element if it looks like word data
         if (wordData.length > 0 && wordData[0] && wordData[0].id) {
             // Recurse with the first element
             return normalizeWordData(wordData[0]);
         } else {
             throw new Error('Invalid API response: Expected single word data object, received array with invalid content.');
         }
    }
    throw new Error('Invalid API response: Missing essential word data or ID.');
  }


  // --- Basic Word Info --- 
  const normalizedWord: WordInfo = {
    id: wordData.id,
    lemma: wordData.lemma || '',
    normalized_lemma: wordData.normalized_lemma || wordData.lemma || '',
    language_code: wordData.language_code || 'tl',
    has_baybayin: wordData.has_baybayin || false,
    baybayin_form: wordData.baybayin_form || null,
    romanized_form: wordData.romanized_form || null,
    root_word_id: wordData.root_word_id || null,
    preferred_spelling: wordData.preferred_spelling || null,
    tags: wordData.tags || null, // Keep as string or null from backend
    data_hash: wordData.data_hash || null,
    search_text: wordData.search_text || null,
    created_at: wordData.created_at || null,
    updated_at: wordData.updated_at || null,

    // --- Related Data Collections (Initialize empty/null) --- 
    definitions: [],
    etymologies: [],
    pronunciations: [],
    credits: [],
    root_word: null,
    derived_words: [],
    outgoing_relations: [],
    incoming_relations: [],
    root_affixations: [],
    affixed_affixations: [],
    data_completeness: wordData.data_completeness || null,
    relation_summary: wordData.relation_summary || null,
  };

  // --- Normalize Definitions --- 
  if (wordData.definitions && Array.isArray(wordData.definitions)) {
    normalizedWord.definitions = wordData.definitions.map((def: any): Definition => ({
      id: def.id,
      definition_text: def.definition_text || '',
      original_pos: def.original_pos || null,
      standardized_pos_id: def.standardized_pos_id || null,
      // Backend already provides the nested standardized_pos object
      standardized_pos: def.standardized_pos || null, 
      // Parse string fields from backend into arrays
      examples: splitSemicolonSeparated(def.examples), 
      usage_notes: splitSemicolonSeparated(def.usage_notes), 
      tags: splitCommaSeparated(def.tags), // Tags might be comma-separated? Adjust if needed
      sources: splitCommaSeparated(def.sources), // Sources might be comma-separated? Adjust if needed
      created_at: def.created_at || null,
      updated_at: def.updated_at || null,
    }));
  }

  // --- Normalize Etymologies --- 
  if (wordData.etymologies && Array.isArray(wordData.etymologies)) {
    normalizedWord.etymologies = wordData.etymologies.map((etym: any): Etymology => ({
      id: etym.id,
      etymology_text: etym.etymology_text || '',
      normalized_components: etym.normalized_components || null,
      etymology_structure: etym.etymology_structure || null,
      // Parse comma-separated string from backend
      language_codes: splitCommaSeparated(etym.language_codes), 
      // Backend 'components' might already be structured, pass through
      components: etym.components || [], 
       // Parse comma-separated string from backend
      sources: splitCommaSeparated(etym.sources),
      created_at: etym.created_at || null,
      updated_at: etym.updated_at || null,
    }));
  }

  // --- Normalize Pronunciations --- 
  if (wordData.pronunciations && Array.isArray(wordData.pronunciations)) {
    normalizedWord.pronunciations = wordData.pronunciations.map((pron: any): Pronunciation => ({
      id: pron.id,
      type: pron.type || '',
      value: pron.value || '',
      // Backend 'tags' is likely already an object/null
      tags: pron.tags || null, 
      // Backend 'sources' is a string
      sources: pron.sources || null, 
      created_at: pron.created_at || null,
      updated_at: pron.updated_at || null,
    }));
  }

  // --- Normalize Credits --- 
  if (wordData.credits && Array.isArray(wordData.credits)) {
    normalizedWord.credits = wordData.credits.map((cred: any): Credit => ({
      id: cred.id,
      credit: cred.credit || '',
      created_at: cred.created_at || null,
      updated_at: cred.updated_at || null,
    }));
  }

  // --- Normalize Root Word --- 
  // Backend provides a nested object directly
  if (wordData.root_word && typeof wordData.root_word === 'object') {
    normalizedWord.root_word = {
      id: wordData.root_word.id,
      lemma: wordData.root_word.lemma || '',
      normalized_lemma: wordData.root_word.normalized_lemma || null,
      language_code: wordData.root_word.language_code || null,
      has_baybayin: wordData.root_word.has_baybayin || false,
      baybayin_form: wordData.root_word.baybayin_form || null,
    };
  }

  // --- Normalize Derived Words --- 
  // Backend provides an array of nested objects
  if (wordData.derived_words && Array.isArray(wordData.derived_words)) {
    normalizedWord.derived_words = wordData.derived_words.map((dw: any): RelatedWord => ({
      id: dw.id,
      lemma: dw.lemma || '',
      normalized_lemma: dw.normalized_lemma || null,
      language_code: dw.language_code || null,
      has_baybayin: dw.has_baybayin || false,
      baybayin_form: dw.baybayin_form || null,
    }));
  }

  // --- Normalize Outgoing Relations --- 
  if (wordData.outgoing_relations && Array.isArray(wordData.outgoing_relations)) {
    normalizedWord.outgoing_relations = wordData.outgoing_relations.map((rel: any): Relation => ({
      id: rel.id,
      relation_type: rel.relation_type || '',
      // Backend 'metadata' is likely already an object/null
      metadata: rel.metadata || null, 
      // Backend 'sources' is a string
      sources: rel.sources || null, 
      // Backend provides nested target_word object
      target_word: rel.target_word ? {
        id: rel.target_word.id,
        lemma: rel.target_word.lemma || '',
        language_code: rel.target_word.language_code || null,
        has_baybayin: rel.target_word.has_baybayin || false,
        baybayin_form: rel.target_word.baybayin_form || null,
      } : { id: 0, lemma: 'Unknown' }, // Fallback
    }));
  }

  // --- Normalize Incoming Relations --- 
  if (wordData.incoming_relations && Array.isArray(wordData.incoming_relations)) {
    normalizedWord.incoming_relations = wordData.incoming_relations.map((rel: any): Relation => ({
      id: rel.id,
      relation_type: rel.relation_type || '',
       // Backend 'metadata' is likely already an object/null
      metadata: rel.metadata || null,
      // Backend 'sources' is a string
      sources: rel.sources || null, 
      // Backend provides nested source_word object
      source_word: rel.source_word ? {
        id: rel.source_word.id,
        lemma: rel.source_word.lemma || '',
        language_code: rel.source_word.language_code || null,
        has_baybayin: rel.source_word.has_baybayin || false,
        baybayin_form: rel.source_word.baybayin_form || null,
      } : { id: 0, lemma: 'Unknown' }, // Fallback
    }));
  }

  // --- Normalize Root Affixations --- 
  if (wordData.root_affixations && Array.isArray(wordData.root_affixations)) {
    normalizedWord.root_affixations = wordData.root_affixations.map((aff: any): Affixation => ({
      id: aff.id,
      affix_type: aff.affix_type || '',
      // Backend 'sources' is a string
      sources: aff.sources || null, 
      created_at: aff.created_at || null,
      updated_at: aff.updated_at || null,
      // Backend provides nested affixed_word object
      affixed_word: aff.affixed_word ? {
        id: aff.affixed_word.id,
        lemma: aff.affixed_word.lemma || '',
        language_code: aff.affixed_word.language_code || null,
        has_baybayin: aff.affixed_word.has_baybayin || false,
        baybayin_form: aff.affixed_word.baybayin_form || null,
      } : { id: 0, lemma: 'Unknown' }, // Fallback
    }));
  }

  // --- Normalize Affixed Affixations --- 
  if (wordData.affixed_affixations && Array.isArray(wordData.affixed_affixations)) {
    normalizedWord.affixed_affixations = wordData.affixed_affixations.map((aff: any): Affixation => ({
      id: aff.id,
      affix_type: aff.affix_type || '',
      // Backend 'sources' is a string
      sources: aff.sources || null, 
      created_at: aff.created_at || null,
      updated_at: aff.updated_at || null,
       // Backend provides nested root_word object
      root_word: aff.root_word ? {
        id: aff.root_word.id,
        lemma: aff.root_word.lemma || '',
        language_code: aff.root_word.language_code || null,
        has_baybayin: aff.root_word.has_baybayin || false,
        baybayin_form: aff.root_word.baybayin_form || null,
      } : { id: 0, lemma: 'Unknown' }, // Fallback
    }));
  }

  return normalizedWord;
}

// Function to fetch word details using the comprehensive endpoint
export async function fetchWordDetails(word: string): Promise<WordInfo> {
  const normalizedWord = word.toLowerCase();
  const cacheKey = `cache:wordDetails:${normalizedWord}`;

  // Try cache first
  const cachedData = getCache(cacheKey);
  if (cachedData) {
    console.log(`Cache hit for word details: ${normalizedWord}`);
    // Re-normalize cached data just in case structure changed
    try {
      // IMPORTANT: Ensure normalizeWordData is called here
      return normalizeWordData(cachedData); 
    } catch (e) {
      console.warn('Error normalizing cached data, fetching fresh:', e);
      clearCache(cacheKey); // Clear bad cache entry
    }
  }

  console.log(`Cache miss for word details: ${normalizedWord}. Fetching from API...`);

  // Check circuit breaker state before making the request
  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for word details.");
    // Consider throwing a specific error type if needed
    throw new Error("Circuit breaker is open. Please try again later.");
  }

  try {
    // Use the comprehensive endpoint
    const endpoint = `/words/${encodeURIComponent(normalizedWord)}/comprehensive`;
    const response = await api.get(endpoint); // Use the configured api instance

    // Check response status explicitly
    if (response.status !== 200) {
      // Handle non-200 status codes appropriately
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }

    // Record success for circuit breaker upon successful request
    circuitBreaker.recordSuccess();

    // Normalize the received data using the updated function
    const normalizedData = normalizeWordData(response.data);

    // Cache the raw response data before returning normalized data
    // Consider caching the *normalized* data if transformation is expensive
    setCache(cacheKey, response.data); 

    return normalizedData;

  } catch (error: unknown) {
     // Record failure for circuit breaker
    circuitBreaker.recordFailure();
    console.error(`Error fetching word details for '${normalizedWord}':`, error);
    
    // Provide more specific error handling based on AxiosError type
    if (axios.isAxiosError(error)) {
      if (error.response?.status === 404) {
        throw new Error(`Word not found: ${normalizedWord}`);
      }
      // Include status code in the error message if available
      const status = error.response?.status ? ` (Status ${error.response.status})` : '';
      throw new Error(`API error fetching details${status}: ${error.message}`);
    } else if (error instanceof Error) {
      // Handle errors thrown by normalizeWordData or other logic
       throw new Error(`Error processing word details: ${error.message}`);
    } else {
      // Fallback for unknown error types
      throw new Error('An unknown error occurred while fetching word details.');
    }
  }
}

export async function searchWords(query: string, options: SearchOptions): Promise<SearchResult> {
  const cacheKey = `cache:search:${query}:${JSON.stringify(options)}`;
  const cachedData = getCachedData<SearchResult>(cacheKey);

  if (cachedData) {
    return cachedData;
  }

  try {
    if (!circuitBreaker.canMakeRequest()) {
      throw new Error('Circuit breaker is open');
    }
    
    // Use snake_case for API parameters
    const apiParams = {
        q: query,
        limit: options.per_page,
        offset: options.page ? (options.page - 1) * (options.per_page || 20) : 0, // Calculate offset
        language: options.language,
        // Add other relevant params based on backend /search endpoint
    };

    const response = await api.get('/search', { params: apiParams });
    const data = response.data; // Assuming data = { total: number, words: RawWordSummary[] }
    
    // Transform the response into SearchResult format
    const searchResult: SearchResult = {
      // Map raw word summary to SearchWordResult
      words: (data.words || []).map((result: any): SearchWordResult => ({
        id: result.id,
        lemma: result.lemma,
        normalized_lemma: result.normalized_lemma,
        language_code: result.language_code,
        has_baybayin: result.has_baybayin,
        baybayin_form: result.baybayin_form,
        romanized_form: result.romanized_form,
        definitions: result.definitions // Assume definitions in search are already simple
      })),
      page: options.page || 1, // Use requested page or default
      perPage: options.per_page || (data.words?.length || 0), // Use requested per_page or actual count
      total: data.total || 0,
      query: query // <-- Add the query back
    };

    setCachedData(cacheKey, searchResult);
    circuitBreaker.recordSuccess();
    return searchResult;
  } catch (error) {
    circuitBreaker.recordFailure();
    throw handleApiError(error, `Error searching words with query "${query}"`);
  }
}

export async function getPartsOfSpeech(): Promise<PartOfSpeech[]> {
  const cacheKey = 'cache:parts_of_speech';
  const cachedData = getCachedData<PartOfSpeech[]>(cacheKey);

  if (cachedData) {
    return cachedData;
  }

  try {
    if (!circuitBreaker.canMakeRequest()) {
      throw new Error('Circuit breaker is open');
    }

    const response = await api.get('/parts-of-speech');
    const data = response.data;

    // Transform the response into PartOfSpeech array format
    const partsOfSpeech: PartOfSpeech[] = data.map((pos: any) => ({
      id: pos.id,
      code: pos.code,
      name_en: pos.name_en,
      name_tl: pos.name_tl,
      description: pos.description
    }));

    // Cache the transformed data
    setCachedData(cacheKey, partsOfSpeech);
    circuitBreaker.recordSuccess();

    return partsOfSpeech;
  } catch (error) {
    circuitBreaker.recordFailure();
    return handleApiError(error, 'Error fetching parts of speech');
  }
}

export async function testApiConnection(): Promise<boolean> {
  try {
    const response = await api.get('/test');
    console.log('API connection test successful:', response.data);
    return true;
  } catch (error) {
    console.error('API connection test failed:', error);
    return false;
  }
}

export async function getEtymologyTree(
  wordId: number, // Use word ID as per backend route
  maxDepth: number = 2 // Keep depth limited
): Promise<EtymologyTree> {
  const cacheKey = `cache:etymology:tree:${wordId}:${maxDepth}`;
  const cachedData = getCachedData<EtymologyTree>(cacheKey);

  if (cachedData) {
    return cachedData;
  }

  try {
    if (!circuitBreaker.canMakeRequest()) {
      throw new Error('Circuit breaker is open');
    }

    // Fetch using word ID
    const response = await api.get(`/words/${wordId}/etymology/tree`, {
      params: { max_depth: maxDepth }
    });
    const data = response.data; // Expects { word: string, etymology_tree: any, complete: boolean }

    // Basic validation
    if (!data || typeof data !== 'object' || !data.etymology_tree) {
        throw new Error('Invalid etymology tree data received');
    }

    // Directly use the backend structure as defined in EtymologyTree type
    const etymologyTree: EtymologyTree = {
      word: data.word,
      etymology_tree: data.etymology_tree,
      complete: data.complete,
      // Add metadata if needed
      metadata: {
          word: data.word,
          max_depth: maxDepth
      }
    };

    setCachedData(cacheKey, etymologyTree);
    circuitBreaker.recordSuccess();
    return etymologyTree;
  } catch (error) {
    circuitBreaker.recordFailure();
    throw handleApiError(error, `Error fetching etymology tree for word ID "${wordId}"`);
  }
}

export async function getRandomWord(): Promise<WordInfo> {
  try {
    const cacheKey = 'random-word';
    
    // Don't use cache for random words
    const response = await api.get<any>('/random', {
      params: {
        has_definitions: true,
        include_definitions: true,
        include_relations: true,
        include_etymology: true,
        include_metadata: true
      }
    });

    // Handle the API response format with data wrapper
    let data: WordInfo;
    
    if (response.data && response.data.data) {
      // New API format with data wrapper
      data = response.data.data;
      console.log('Using random word data from response.data.data');
    } else {
      // Old format or direct data
      data = response.data;
      console.log('Using random word data directly from response.data');
    }
    
    if (!data.lemma || !data.normalized_lemma || !data.language_code) {
      throw new Error('Invalid random word data received from server');
    }

    return data;
  } catch (error) {
    return handleApiError(error, 'fetching random word');
  }
}

export async function getStatistics(): Promise<Statistics> {
  const cacheKey = 'cache:statistics';
  const cachedData = getCachedData<Statistics>(cacheKey);

  if (cachedData) {
    return cachedData;
  }

  try {
    if (!circuitBreaker.canMakeRequest()) {
      throw new Error('Circuit breaker is open');
    }

    // Use /test endpoint which matches the Statistics type
    const response = await api.get('/test');
    const data = response.data;

    // Validate basic structure
    if (!data || typeof data !== 'object' || !data.status || !data.database) {
        throw new Error('Invalid statistics data received from /test endpoint');
    }

    // Data already matches the Statistics type defined in types.ts
    const statistics: Statistics = data;

    setCachedData(cacheKey, statistics);
    circuitBreaker.recordSuccess();
    return statistics;
  } catch (error) {
    circuitBreaker.recordFailure();
    throw handleApiError(error, 'Error fetching statistics');
  }
}

export async function getBaybayinWords(page: number = 1, limit: number = 20, language: string = 'tl'): Promise<any> {
  try {
    const cacheKey = `baybayin-${page}-${limit}-${language}`;
    
    const cachedData = getCachedData<any>(cacheKey);
    if (cachedData) {
      return cachedData;
    }

    const response = await api.get('/baybayin', {
      params: {
        page,
        limit,
        language
      }
    });
    
    const data = response.data;
    setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    return handleApiError(error, 'fetching baybayin words');
  }
}

export async function getAffixes(language: string = 'tl', type?: string): Promise<any> {
  try {
    const cacheKey = `affixes-${language}-${type || 'all'}`;
    
    const cachedData = getCachedData<any>(cacheKey);
    if (cachedData) {
      return cachedData;
    }

    const response = await api.get('/affixes', {
      params: {
        language,
        type
      }
    });
    
    const data = response.data;
    setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    return handleApiError(error, 'fetching affixes');
  }
}

export async function getRelations(language: string = 'tl', type?: string): Promise<any> {
  try {
    const cacheKey = `relations-${language}-${type || 'all'}`;
    
    const cachedData = getCachedData<any>(cacheKey);
    if (cachedData) {
      return cachedData;
    }

    const response = await api.get('/relations', {
      params: {
        language,
        type
      }
    });
    
    const data = response.data;
    setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    return handleApiError(error, 'fetching relations');
  }
}

export async function getAllWords(page: number = 1, perPage: number = 20, language: string = 'tl'): Promise<any> {
  try {
    const cacheKey = `all-words-${page}-${perPage}-${language}`;
    
    const cachedData = getCachedData<any>(cacheKey);
    if (cachedData) {
      return cachedData;
    }

    const response = await api.get('/words', {
      params: {
        page,
        per_page: perPage,
        language
      }
    });
    
    const data = response.data;
    setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    return handleApiError(error, 'fetching all words');
  }
}