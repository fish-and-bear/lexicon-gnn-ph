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

export async function fetchWordNetwork(
  word: string, 
  options: WordNetworkOptions = {}
): Promise<WordNetwork> {
  try {
    const sanitizedWord = sanitizeInput(word);
    if (!sanitizedWord) {
      throw new Error('Word is required');
    }

    const {
      depth = 2,
      include_affixes = true,
      include_etymology = true,
      cluster_threshold = 0.3
    } = options;

    const sanitizedDepth = Math.min(Math.max(1, depth), 3);
    const cacheKey = `wordNetwork-${sanitizedWord}-${sanitizedDepth}-${include_affixes}-${include_etymology}`;
    
    const cachedData = getCachedData<WordNetwork>(cacheKey);
    if (cachedData) {
      return cachedData;
    }

    const encodedWord = encodeURIComponent(sanitizedWord);
    console.log(`Fetching word network for: ${encodedWord} with depth: ${sanitizedDepth}`);
    
    const response = await api.get(`/words/${encodedWord}/related`, { 
      params: { 
        depth: sanitizedDepth, 
        include_affixes,
        include_etymology,
        cluster_threshold
      }
    });

    // Log the raw response data for debugging
    console.log('Raw network data response:', JSON.stringify(response.data, null, 2));
    
    // Use any type temporarily to handle the response data
    let networkData: any;
    
    // Handle the API response format with data wrapper
    if (response.data && response.data.data) {
      // New API format with data wrapper
      networkData = response.data.data;
      console.log('Using network data from response.data.data');
    } else {
      // Old format or direct data
      networkData = response.data;
      console.log('Using network data directly from response.data');
    }
    
    // More detailed validation
    if (!networkData) {
      console.error('Network data is null or undefined');
      throw new Error('Empty network data received from server');
    }
    
    // Check if we need to adapt the data structure
    if (networkData.meta && !networkData.metadata) {
      console.log('Adapting API response format: moving meta to metadata');
      networkData.metadata = networkData.meta;
      delete networkData.meta;
    }
    
    // Create default metadata if missing
    if (!networkData.metadata) {
      console.log('Creating default metadata');
      networkData.metadata = {
        root_word: sanitizedWord,
        normalized_lemma: sanitizedWord,
        language_code: 'tl',
        depth: sanitizedDepth,
        total_nodes: networkData.nodes?.length || 0,
        total_edges: networkData.edges?.length || 0,
        include_affixes,
        include_etymology,
        cluster_threshold
      };
    }
    
    // Validate the structure after adaptations
    if (!networkData.nodes) {
      console.error('Network data has no nodes property:', networkData);
      throw new Error('Invalid network data: missing nodes property');
    }
    
    if (!Array.isArray(networkData.nodes)) {
      console.error('Network data nodes is not an array:', networkData.nodes);
      throw new Error('Invalid network data: nodes is not an array');
    }
    
    // Create default clusters if missing
    if (!networkData.clusters) {
      console.log('Creating default clusters');
      networkData.clusters = {
        etymology: [],
        affixes: [],
        synonyms: [],
        antonyms: [],
        variants: [],
        root_words: [],
        derived_words: []
      };
    }
    
    // Now that we've validated and adapted the structure, we can cast it to WordNetwork
    const result = networkData as WordNetwork;
    
    setCachedData<WordNetwork>(cacheKey, result);
    return result;
  } catch (error) {
    console.error('Error fetching word network:', error);
    // Create a minimal valid network to avoid breaking the UI
    const emptyNetwork: WordNetwork = {
      word: word,
      nodes: [],
      edges: [],
      clusters: {
        etymology: [],
        affixes: [],
        synonyms: [],
        antonyms: [],
        variants: [],
        root_words: [],
        derived_words: []
      },
      metadata: {
        root_word: word,
        normalized_lemma: sanitizeInput(word) || word,
        language_code: 'tl',
        depth: options.depth || 2,
        total_nodes: 0,
        total_edges: 0
      }
    };
    return emptyNetwork;
  }
}

// Helper function to safely convert string or array to array
function toArray(value: string | string[] | undefined | null): string[] {
  if (Array.isArray(value)) {
    return value.filter(s => typeof s === 'string'); // Ensure all elements are strings
  } else if (typeof value === 'string') {
    return value.split(/[;,]/).map(s => s.trim()).filter(s => s.length > 0);
  }
  return [];
}

// Helper function specifically for splitting comma-separated strings
function splitCommaSeparated(value: string | undefined | null): string[] {
  if (typeof value === 'string') {
    return value.split(',').map(s => s.trim()).filter(s => s.length > 0);
  }
  return [];
}

// Helper function specifically for splitting semicolon-separated strings
function splitSemicolonSeparated(value: string | undefined | null): string[] {
  if (typeof value === 'string') {
    return value.split(';').map(s => s.trim()).filter(s => s.length > 0);
  }
  return [];
}

// Helper function to safely parse JSON strings
function safeJsonParse(jsonString: string | null | undefined): Record<string, any> | null {
  if (!jsonString) return null;
  try {
    return JSON.parse(jsonString);
  } catch (e) {
    console.warn('Failed to parse JSON string:', jsonString, e);
    return null;
  }
}

// Main normalization function - REVISED
function normalizeWordData(rawData: any): WordInfo {
  // Use the actual data, potentially nested under 'data'
  const wordData: RawWordComprehensiveData = rawData?.data || rawData;

  if (!wordData || typeof wordData !== 'object' || !wordData.id) {
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
    tags: wordData.tags || null, // Keep as string or null
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
      standardized_pos: def.standardized_pos || null, // Already an object from backend
      examples: splitSemicolonSeparated(def.examples), // Parse string
      usage_notes: splitSemicolonSeparated(def.usage_notes), // Parse string
      tags: splitSemicolonSeparated(def.tags), // Parse string
      sources: splitSemicolonSeparated(def.sources), // Parse string
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
      language_codes: splitCommaSeparated(etym.language_codes), // Parse comma-separated string
      components: etym.components || [], // Added by backend, might be complex
      sources: splitSemicolonSeparated(etym.sources), // Parse semicolon-separated string
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
      tags: pron.tags || null, // Backend provides JSON object already?
      sources: pron.sources || null, // Keep as string
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
      metadata: rel.metadata || null, // Already JSON object from backend?
      sources: rel.sources || null, // Keep as string
      target_word: rel.target_word ? {
        id: rel.target_word.id,
        lemma: rel.target_word.lemma || '',
        language_code: rel.target_word.language_code || null,
        has_baybayin: rel.target_word.has_baybayin || false,
        baybayin_form: rel.target_word.baybayin_form || null,
      } : { id: 0, lemma: 'Unknown' }, // Fallback if target_word is missing
      // source_word is implicitly the current word
    }));
  }

  // --- Normalize Incoming Relations --- 
  if (wordData.incoming_relations && Array.isArray(wordData.incoming_relations)) {
    normalizedWord.incoming_relations = wordData.incoming_relations.map((rel: any): Relation => ({
      id: rel.id,
      relation_type: rel.relation_type || '',
      metadata: rel.metadata || null, // Already JSON object from backend?
      sources: rel.sources || null, // Keep as string
      source_word: rel.source_word ? {
        id: rel.source_word.id,
        lemma: rel.source_word.lemma || '',
        language_code: rel.source_word.language_code || null,
        has_baybayin: rel.source_word.has_baybayin || false,
        baybayin_form: rel.source_word.baybayin_form || null,
      } : { id: 0, lemma: 'Unknown' }, // Fallback if source_word is missing
      // target_word is implicitly the current word
    }));
  }

  // --- Normalize Root Affixations --- 
  if (wordData.root_affixations && Array.isArray(wordData.root_affixations)) {
    normalizedWord.root_affixations = wordData.root_affixations.map((aff: any): Affixation => ({
      id: aff.id,
      affix_type: aff.affix_type || '',
      sources: aff.sources || null, // Keep as string
      created_at: aff.created_at || null,
      updated_at: aff.updated_at || null,
      affixed_word: aff.affixed_word ? {
        id: aff.affixed_word.id,
        lemma: aff.affixed_word.lemma || '',
        language_code: aff.affixed_word.language_code || null,
        has_baybayin: aff.affixed_word.has_baybayin || false,
        baybayin_form: aff.affixed_word.baybayin_form || null,
      } : { id: 0, lemma: 'Unknown' }, // Fallback
      // root_word is implicitly the current word
    }));
  }

  // --- Normalize Affixed Affixations --- 
  if (wordData.affixed_affixations && Array.isArray(wordData.affixed_affixations)) {
    normalizedWord.affixed_affixations = wordData.affixed_affixations.map((aff: any): Affixation => ({
      id: aff.id,
      affix_type: aff.affix_type || '',
      sources: aff.sources || null, // Keep as string
      created_at: aff.created_at || null,
      updated_at: aff.updated_at || null,
      root_word: aff.root_word ? {
        id: aff.root_word.id,
        lemma: aff.root_word.lemma || '',
        language_code: aff.root_word.language_code || null,
        has_baybayin: aff.root_word.has_baybayin || false,
        baybayin_form: aff.root_word.baybayin_form || null,
      } : { id: 0, lemma: 'Unknown' }, // Fallback
      // affixed_word is implicitly the current word
    }));
  }

  return normalizedWord;
}

// Function to fetch word details using the comprehensive endpoint
export async function fetchWordDetails(word: string): Promise<WordInfo> {
  const normalizedWord = word.toLowerCase();
  const cacheKey = `cache:wordDetails:${normalizedWord}`;

  // Try cache first
  const cachedData = getCachedData<WordInfo>(cacheKey);
  if (cachedData) {
    console.log(`Cache hit for word details: ${normalizedWord}`);
    // Re-normalize cached data just in case structure changed
    try {
      return normalizeWordData(cachedData);
    } catch (e) {
      console.warn('Error normalizing cached data, fetching fresh:', e);
      clearOldCache(cacheKey); // Clear bad cache entry
    }
  }

  console.log(`Cache miss for word details: ${normalizedWord}. Fetching from API...`);

  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for word details.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }

  try {
    // Use the comprehensive endpoint
    const endpoint = `/words/${encodeURIComponent(normalizedWord)}/comprehensive`;
    const response = await api.get(endpoint);

    // Check response status
    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }

    // Record success for circuit breaker
    circuitBreaker.recordSuccess();

    // Normalize the received data
    const normalizedData = normalizeWordData(response.data);

    // Cache the raw response data before returning normalized data
    setCachedData(cacheKey, normalizedData);

    return normalizedData;

  } catch (error: unknown) {
    circuitBreaker.recordFailure();
    console.error(`Error fetching word details for '${normalizedWord}':`, error);
    // Re-throw a more specific error or handle it
    if (axios.isAxiosError(error)) {
      if (error.response?.status === 404) {
        throw new Error(`Word not found: ${normalizedWord}`);
      }
      throw new Error(`API error fetching details: ${error.message}`);
    } else if (error instanceof Error) {
      throw new Error(`Error fetching details: ${error.message}`);
    } else {
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