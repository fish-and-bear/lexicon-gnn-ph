import axios, { AxiosError, AxiosResponse, AxiosRequestConfig } from 'axios';
import { 
  WordNetworkResponse, 
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
  RelatedWord
} from "../types";
import { sanitizeInput } from '../utils/sanitizer';
import { getCachedData, setCachedData, clearCache, clearOldCache } from '../utils/caching';

// Environment and configuration constants
const ENV = process.env.NODE_ENV || 'development';
// Define API_BASE_URL for use in direct fetch calls
export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL?.replace('/api/v1', '/api/v2') || 'http://localhost:10000/api/v2';

const CONFIG = {
  development: {
    baseURL: 'http://localhost:10000/api/v2',
    timeout: 10000,
    retries: 3,
    failureThreshold: 10,
    resetTimeout: 15000,
    retryDelay: 1000,
    maxRetryDelay: 10000
  },
  production: {
    baseURL: process.env.REACT_APP_API_BASE_URL?.replace('/api/v1', '/api/v2') || 'https://fil-relex.onrender.com/api/v2',
    timeout: 15000, // Increased timeout for production
    retries: 2,
    failureThreshold: 8, // Increased from 5 to 8 to be more tolerant
    resetTimeout: 15000,
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
    clearOldCache();
    // Simple check: if open and reset timeout hasn't passed, deny request
    if (this.state === 'open' && this.lastFailureTime && (Date.now() - this.lastFailureTime < CONFIG.resetTimeout)) {
        console.log('Circuit breaker is OPEN. Request denied.');
        return false;
    }
    // If open but timeout passed, move to half-open
    if (this.state === 'open') {
        this.state = 'half-open';
        this.saveState();
        console.log('Circuit breaker moved to HALF-OPEN.');
    }
    // Allow requests in closed or half-open state
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
  circuitBreaker.reset();
  try {
    localStorage.removeItem('circuit_breaker_state');
    localStorage.removeItem('successful_api_endpoint');
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && (key.startsWith('cache:') || key.includes('circuit') || key.includes('api_endpoint'))) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));
    api.defaults.baseURL = CONFIG.baseURL;
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
    setTimeout(() => {
      testApiConnection().then(connected => {
        console.log(`Connection test after reset: ${connected ? 'successful' : 'failed'}`);
      });
    }, 500);
  } catch (e) {
    console.error('Error clearing localStorage:', e);
  }
}

// Function to force the circuit breaker to allow requests
export function forceCircuitBreakerClosed() {
  // Reset the circuit breaker without clearing cache or other state
  circuitBreaker.reset();
  console.log('Circuit breaker has been forced closed. Will allow requests to proceed.');
}

// Get the successful endpoint from localStorage if available
const savedEndpoint = localStorage.getItem('successful_api_endpoint');
let apiBaseURL = CONFIG.baseURL;

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
  baseURL: apiBaseURL, // Use potentially saved endpoint
  timeout: CONFIG.timeout,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Client-Version': process.env.REACT_APP_VERSION || '1.0.0',
    'X-Client-Platform': 'web'
  },
  withCredentials: false
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    config.headers = {
      ...config.headers,
      'Origin': window.location.origin,
      'Access-Control-Request-Method': config.method?.toUpperCase() || 'GET',
    };
    console.log(`Making ${config.method?.toUpperCase()} request to: ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`Response from ${response.config.url}: Status ${response.status}`);
    // Record success only on successful requests (2xx status)
    if (response.status >= 200 && response.status < 300) {
        // Check if it was a half-open request that succeeded
        if (circuitBreaker.getState().state === 'half-open') {
            circuitBreaker.reset(); // Fully close the circuit
        } else {
            circuitBreaker.recordSuccess(); // Record success for closed state
        }
    }
    return response;
  },
  (error) => {
    console.error('Response error:', error);
    
    // Don't increment circuit breaker for network errors in search requests
    // (since we handle retries there separately)
    const isSearchRequest = error.config?.url?.includes('/search');
    const isNetworkError = !error.response && error.request;
    
    // Only record failure for circuit breaker if it's not a search network error
    if (!(isSearchRequest && isNetworkError)) {
      circuitBreaker.recordFailure();
    } else {
      console.log('Ignoring network error for search request in circuit breaker tracking');
    }
    
    // Handle specific HTTP errors if needed
    if (error.response) {
        console.error(`API Error: Status ${error.response.status}, Data:`, error.response.data);
        if (error.response.status === 404) {
            return Promise.reject(new Error('Resource not found (404)'));
        }
    } else if (error.request) {
        console.error('API Error: No response received', error.request);
    } else {
        console.error('API Error: Request setup failed', error.message);
    }
    // Don't throw here, let the calling function handle it via catch
    return Promise.reject(error);
  }
);

// Type guard for AxiosError
function isApiError(error: unknown): error is AxiosError {
  return axios.isAxiosError(error);
}

// Error handling function - REVISED again
async function handleApiError(error: unknown, context: string): Promise<never> {
  console.error(`API Error in ${context}:`, error);

  let errorMessage = `An unknown error occurred in ${context}.`;
  let errorDetails: any = {};

  if (isApiError(error)) {
    errorMessage = error.message;
    if (error.response) {
      const statusText = error.response.statusText || 'Unknown Status';
      errorMessage = `API request failed with status ${error.response.status} (${statusText}) in ${context}.`;
      errorDetails = error.response.data;
      if (error.response.status === 404) {
          errorMessage = `Resource not found in ${context}.`;
      }
       else if (error.response.status === 429) {
        errorMessage = `Too many requests. Please try again later (Rate Limit Exceeded) in ${context}.`;
      }
      const detailMessage = typeof errorDetails === 'object' && errorDetails !== null && errorDetails.message ? String(errorDetails.message) : null;
      if (detailMessage) {
          errorMessage += ` Server message: ${detailMessage}`;
      }
    } else if (error.request) {
      errorMessage = `No response received from server in ${context}. Check network connection and backend status.`;
    } else {
        errorMessage = `Request setup failed in ${context}: ${error.message}`;
    }
     // errorDetails.code = error.code; // REMOVED: Type of error.code is uncertain, avoid assigning directly to any
  } else if (error instanceof Error) {
    errorMessage = `Error in ${context}: ${error.message}`;
  }

  console.error("Final Error Message:", errorMessage, "Details:", errorDetails);
  throw new Error(errorMessage); 
}

// --- Word Network Fetching --- 
export interface WordNetworkOptions {
  depth?: number;
  include_affixes?: boolean;
  include_etymology?: boolean;
  cluster_threshold?: number;
}

export async function fetchWordNetwork(
  word: string, 
  options: WordNetworkOptions = {}
): Promise<WordNetworkResponse> {
  const sanitizedWord = word.toLowerCase(); // Use simple normalization
    const {
      depth = 2,
      include_affixes = true,
      include_etymology = true,
      cluster_threshold = 0.3
    } = options;
  const sanitizedDepth = Math.min(Math.max(1, depth), 3); // Limit depth
  const cacheKey = `cache:wordNetwork:${sanitizedWord}-${sanitizedDepth}-${include_affixes}-${include_etymology}`;
    
    const cachedData = getCachedData<WordNetworkResponse>(cacheKey);
    if (cachedData) {
    console.log(`Cache hit for word network: ${sanitizedWord}`);
      return cachedData;
    }

  console.log(`Cache miss for word network: ${sanitizedWord}. Fetching from API...`);

  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for word network.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }

  try {
    const encodedWord = encodeURIComponent(sanitizedWord);
    // Assuming backend uses /relations/graph endpoint for network
    const response = await api.get(`/words/${encodedWord}/relations/graph`, { 
      params: { 
        max_depth: sanitizedDepth, 
        // Add other params if the graph endpoint supports them
        // include_affixes: include_affixes,
        // include_etymology: include_etymology 
      }
    });

    if (response.status !== 200) {
        throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }

    const networkData = response.data; // Assuming direct data structure

    // Basic validation
    if (!networkData || !Array.isArray(networkData.nodes) || !Array.isArray(networkData.edges)) {
        console.error('Invalid network data received:', networkData);
        throw new Error('Invalid network data structure received from API.');
    }
    
    // Construct the WordNetwork object (may need more sophisticated mapping)
    const result: WordNetworkResponse = {
        nodes: networkData.nodes.map((n: any) => ({
          id: n.id,
          label: n.lemma || n.id?.toString() || 'unknown',
          word: n.lemma || n.id?.toString() || 'unknown',
          language: n.language_code,
        })),
        edges: networkData.edges.map((e: any) => ({
          id: `${e.source}-${e.target}-${e.type}`,
          source: e.source,
          target: e.target,
          type: e.type,
          metadata: e.metadata
        })),
        stats: networkData.stats || { node_count: networkData.nodes.length, edge_count: networkData.edges.length, depth: sanitizedDepth },
    };

    setCachedData(cacheKey, result);
    // Success already recorded by interceptor
    return result;

  } catch (error) {
    // Failure already recorded by interceptor
    // Throw a more specific error using handleApiError
    await handleApiError(error, `fetching word network for '${sanitizedWord}'`);
    // The line below won't be reached due to handleApiError throwing
    throw error; // Fallback if handleApiError doesn't throw as expected
  }
}

// --- Data Normalization Helpers --- 

// Helper function to safely parse JSON strings
function safeJsonParse(jsonString: string | null | undefined): Record<string, any> | null {
  if (!jsonString) return null;
  try {
    if (typeof jsonString === 'object') return jsonString; // Already an object
    return JSON.parse(jsonString);
  } catch (e) {
    console.warn('Failed to parse JSON string:', jsonString, e);
    return null;
  }
}

// Helper function to split strings by semicolon, trimming whitespace - Reverted Signature
function splitSemicolonSeparated(value: string | undefined): string[] { // Use string | undefined
  // No need to check for array input anymore
  if (typeof value === 'string') {
    return value.split(';').map(s => s.trim()).filter(s => s !== '');
  }
  return [];
}

// Helper function to split strings by comma, trimming whitespace - Reverted Signature
function splitCommaSeparated(value: string | undefined): string[] { // Use string | undefined
  // No need to check for array input anymore
  if (typeof value === 'string') {
    return value.split(',').map(s => s.trim()).filter(s => s !== '');
  }
  return [];
}

// --- Main Data Normalization Function --- 

function normalizeWordData(rawData: any): WordInfo {
  const wordData: RawWordComprehensiveData = rawData?.data || rawData;

  if (!wordData || typeof wordData !== 'object' || !wordData.id) {
    if (Array.isArray(wordData)) {
         console.warn("normalizeWordData received an array, expected object. Using first element.", wordData);
         if (wordData.length > 0 && wordData[0] && wordData[0].id) {
             return normalizeWordData(wordData[0]);
         } else {
             throw new Error('Invalid API response: Expected single word data object, received array with invalid content.');
         }
    }
    throw new Error('Invalid API response: Missing essential word data or ID.');
  }

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

  // Normalize Definitions
  if (wordData.definitions && Array.isArray(wordData.definitions)) {
    normalizedWord.definitions = wordData.definitions.map((def: any): Definition => ({ // Changed back to any
      // Raw fields needed by Omit base (RawDefinition)
      id: def.id,
      definition_text: def.definition_text || '', 
      original_pos: def.original_pos || null,
      standardized_pos: def.standardized_pos || null,
      created_at: def.created_at || null, 
      updated_at: def.updated_at || null, 
      // Fields required by the cleaned Definition type
      part_of_speech: def.standardized_pos || null, 
      examples: splitSemicolonSeparated(def.examples), 
      usage_notes: splitSemicolonSeparated(def.usage_notes), 
      tags: splitCommaSeparated(def.tags), 
      sources: splitCommaSeparated(def.sources), 
      // Other optional fields from RawDefinition if needed - REMOVE THESE
      // confidence_score: def.confidence_score,
      // is_verified: def.is_verified,
      // verified_by: def.verified_by,
      // verified_at: def.verified_at,
    }));
  }

  // Normalize Etymologies - Revert to ': any' and add cast
  if (wordData.etymologies && Array.isArray(wordData.etymologies)) {
    normalizedWord.etymologies = wordData.etymologies.map((etym: any): Etymology => ({ // Changed back to any
      // Raw fields needed by Omit base (RawEtymology)
      id: etym.id,
      etymology_text: etym.etymology_text || '', 
      etymology_structure: etym.etymology_structure || null,
      created_at: etym.created_at || null, // Reverted to null
      updated_at: etym.updated_at || null, // Reverted to null
      // Fields required by the cleaned Etymology type
      languages: splitCommaSeparated(etym.language_codes), 
      // Explicit cast to silence persistent linter error
      components: splitCommaSeparated(etym.components as string | undefined), 
      sources: splitCommaSeparated(etym.sources), 
      // Other optional fields from RawEtymology if needed
      // REMOVE: Not in Etymology type
      // confidence_level: etym.confidence_level,
      // verification_status: etym.verification_status,
      // verification_notes: etym.verification_notes,
    }));
  }

  // Normalize Pronunciations - Revert to ': any'
  if (wordData.pronunciations && Array.isArray(wordData.pronunciations)) {
    normalizedWord.pronunciations = wordData.pronunciations.map((pron: any): Pronunciation => ({ // Changed back to any
      id: pron.id,
      type: pron.type || '',
      value: pron.value || '',
      tags: pron.tags || null, 
      sources: pron.sources || null, 
      created_at: pron.created_at || null, // Reverted to null
      updated_at: pron.updated_at || null, // Reverted to null
    }));
  }

  // Normalize Credits - Revert to ': any'
  if (wordData.credits && Array.isArray(wordData.credits)) {
    normalizedWord.credits = wordData.credits.map((cred: any): Credit => ({ // Changed back to any
      id: cred.id,
      credit: cred.credit || '',
      created_at: cred.created_at || null, // Reverted to null
      updated_at: cred.updated_at || null, // Reverted to null
    }));
  }

  // Normalize Root Word
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

  // Normalize Derived Words - Revert to ': any'
  if (wordData.derived_words && Array.isArray(wordData.derived_words)) {
    normalizedWord.derived_words = wordData.derived_words.map((dw: any): RelatedWord => ({ // Changed back to any
      id: dw.id,
      lemma: dw.lemma || '',
      normalized_lemma: dw.normalized_lemma || null,
      language_code: dw.language_code || null,
      has_baybayin: dw.has_baybayin || false,
      baybayin_form: dw.baybayin_form || null,
    }));
  }

  // Normalize Outgoing Relations - Revert to ': any'
  if (wordData.outgoing_relations && Array.isArray(wordData.outgoing_relations)) {
    normalizedWord.outgoing_relations = wordData.outgoing_relations.map((rel: any): Relation => ({ // Changed back to any
        // Spread existing properties (assuming raw Relation matches cleaned enough)
        ...rel, 
        target_word: rel.target_word ? { ...rel.target_word } : undefined, 
        source_word: rel.source_word ? { ...rel.source_word } : undefined, 
    }));
  }

  // Normalize Incoming Relations - Revert to ': any'
  if (wordData.incoming_relations && Array.isArray(wordData.incoming_relations)) {
    normalizedWord.incoming_relations = wordData.incoming_relations.map((rel: any): Relation => ({ // Changed back to any
        ...rel, 
        target_word: rel.target_word ? { ...rel.target_word } : undefined, 
        source_word: rel.source_word ? { ...rel.source_word } : undefined, 
    }));
  }

  // Normalize Root Affixations - Revert to ': any'
  if (wordData.root_affixations && Array.isArray(wordData.root_affixations)) {
    normalizedWord.root_affixations = wordData.root_affixations.map((aff: any): Affixation => ({ // Changed back to any
        ...aff,
        affixed_word: aff.affixed_word ? { ...aff.affixed_word } : undefined,
        root_word: aff.root_word ? { ...aff.root_word } : undefined,
    }));
  }

  // Normalize Affixed Affixations - Revert to ': any'
  if (wordData.affixed_affixations && Array.isArray(wordData.affixed_affixations)) {
    normalizedWord.affixed_affixations = wordData.affixed_affixations.map((aff: any): Affixation => ({ // Changed back to any
      ...aff,
      affixed_word: aff.affixed_word ? { ...aff.affixed_word } : undefined,
      root_word: aff.root_word ? { ...aff.root_word } : undefined,
    }));
  }

  return normalizedWord;
}

// --- Word Details Fetching --- 

export async function fetchWordDetails(word: string): Promise<WordInfo> {
  const normalizedWord = word.toLowerCase();
  const cacheKey = `cache:wordDetails:${normalizedWord}`;

  const cachedData = getCachedData(cacheKey);
    if (cachedData) {
    console.log(`Cache hit for word details: ${normalizedWord}`);
    try {
      return normalizeWordData(cachedData); 
    } catch (e) {
      console.warn('Error normalizing cached data, fetching fresh:', e);
      clearCache();
    }
  }

  console.log(`Cache miss for word details: ${normalizedWord}. Fetching from API...`);

  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for word details.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }

  try {
    const endpoint = `/words/${encodeURIComponent(normalizedWord)}/comprehensive`;
    const response = await api.get(endpoint);

    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }

    // NOTE: Success is recorded by the interceptor
    const normalizedData = normalizeWordData(response.data);
    setCachedData(cacheKey, response.data); // Cache the raw data
    return normalizedData;

  } catch (error: unknown) {
    // NOTE: Failure is recorded by the interceptor
    // Throw a more specific error using handleApiError
    await handleApiError(error, `fetching word details for '${normalizedWord}'`);
    // This line likely won't be reached, but keeps TypeScript happy
    throw new Error('An unknown error occurred after handling API error.'); 
  }
}

// --- Search Functionality --- 

export async function searchWords(
  queryOrOptions: string | SearchOptions, 
  options?: SearchOptions,
  bypassCircuitBreaker: boolean = false
): Promise<SearchResult> {
  // Handle both function signatures:
  // searchWords(query: string, options: SearchOptions)
  // searchWords(options: SearchOptions)
  let query: string;
  let searchOptions: SearchOptions;
  
  if (typeof queryOrOptions === 'string') {
    query = queryOrOptions;
    searchOptions = options || { q: queryOrOptions }; // If options is missing, create minimal valid options
  } else {
    query = queryOrOptions.q || '';
    searchOptions = queryOrOptions;
  }
  
  // Normalize the query
  const normalizedQuery = query.trim();
  if (!normalizedQuery) {
    return {
      words: [],
      page: searchOptions.page || 1,
      perPage: searchOptions.per_page || 20,
      total: 0,
      query: ''
    };
  }
  
  const cacheKey = `cache:search:${normalizedQuery}:${JSON.stringify(searchOptions)}`;
  const cachedData = getCachedData<SearchResult>(cacheKey);

  if (cachedData) {
    console.log(`Cache hit for search: ${normalizedQuery}`);
    return cachedData;
  }
  console.log(`Cache miss for search: ${normalizedQuery}. Fetching from API...`);

  if (!circuitBreaker.canMakeRequest() && !bypassCircuitBreaker) {
    console.warn("Circuit breaker is open. Aborting API request for search.");
    throw new Error("Too many failed requests detected. The application is temporarily limiting API calls to protect the server. Please try again in 15 seconds, or click 'Force Search' to bypass this protection.");
  }

  try {
    // Use snake_case for API parameters to match backend expectations
    const apiParams: Record<string, any> = {
      q: normalizedQuery,
      limit: searchOptions.per_page || searchOptions.limit || 20, // Support both param styles
      offset: searchOptions.page ? (searchOptions.page - 1) * (searchOptions.per_page || 20) : searchOptions.offset || 0, 
    };
    
    // Remove any undefined or null params
    Object.keys(apiParams).forEach(key => {
      if (apiParams[key] === undefined || apiParams[key] === null) {
        delete apiParams[key];
      }
    });
    
    // Add optional parameters if defined
    if (searchOptions.language) apiParams.language = searchOptions.language;
    if (searchOptions.mode) apiParams.mode = searchOptions.mode;
    if (searchOptions.pos) apiParams.pos = searchOptions.pos;
    if (searchOptions.sort) apiParams.sort = searchOptions.sort;
    if (searchOptions.order) apiParams.order = searchOptions.order;
    if (searchOptions.exclude_baybayin !== undefined) apiParams.exclude_baybayin = searchOptions.exclude_baybayin;
    if (searchOptions.include_full !== undefined) apiParams.include_full = searchOptions.include_full;
    
    console.log('Search API request params:', apiParams);

    // Implement retry logic for search requests
    let response;
    let retryCount = 0;
    const maxRetries = 3;
    let lastError;
    
    while (retryCount <= maxRetries) {
      try {
        // Use increasing timeout for each retry attempt
        const currentTimeout = CONFIG.timeout * Math.pow(2, retryCount + 1); // 2x, 4x, 8x
        console.log(`Search attempt ${retryCount + 1}/${maxRetries + 1} for "${normalizedQuery}" with timeout: ${currentTimeout}ms`);
        
        response = await api.get('/search', { 
          params: apiParams,
          timeout: currentTimeout,
          maxContentLength: 1024 * 1024 * 10,
          maxBodyLength: 1024 * 1024 * 10
        });
        
        // Verify we got a successful response with data
        if (response.status === 200 && response.data) {
          // Add extra debug logging
          console.log('Search success, response keys:', Object.keys(response.data));
          
          if (Array.isArray(response.data.results)) {
            console.log(`Found ${response.data.results.length} results in API response`);
          } else {
            console.warn('API response does not contain results array:', response.data);
          }
        }
        
        // Success, exit retry loop
        break;
      } catch (error) {
        lastError = error;
        if (axios.isAxiosError(error) && !error.response && retryCount < maxRetries) {
          // Only retry network errors (timeouts, etc.), not server errors (4xx, 5xx)
          console.warn(`Search attempt ${retryCount + 1} failed. Retrying...`, error);
          retryCount++;
          // Wait before next retry (exponential backoff)
          await new Promise(resolve => setTimeout(resolve, CONFIG.retryDelay * Math.pow(2, retryCount)));
        } else {
          // Either not a network error or we've exhausted retries
          // Before giving up completely, try a fallback with just the basic parameters
          if (retryCount === maxRetries && axios.isAxiosError(error) && !error.response) {
            console.warn("All regular search attempts failed. Trying fallback minimal search...");
            try {
              // Simple fallback with minimal parameters
              const fallbackResponse = await api.get('/search', {
                params: { 
                  q: normalizedQuery,
                  limit: 5,
                  include_full: false,
                  mode: 'exact' // Try exact mode which might be faster
                },
                timeout: CONFIG.timeout * 8 // Very long timeout for last attempt
              });
              
              if (fallbackResponse.status === 200) {
                console.log("Fallback search succeeded with minimal parameters");
                response = fallbackResponse;
                break;
              }
            } catch (fallbackError) {
              console.error("Fallback search also failed:", fallbackError);
              // Still throw the original error
              throw error;
            }
          } else {
            throw error;
          }
        }
      }
    }
    
    if (!response) {
      throw lastError || new Error(`Failed after ${maxRetries} retry attempts`);
    }
    
    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }
    
    const data = response.data; // Backend response
    console.log('Search API response structure:', Object.keys(data));
    
    // Transform the response into SearchResult format
    const searchResult: SearchResult = {
      words: [],
      page: searchOptions.page || 1,
      perPage: searchOptions.per_page || searchOptions.limit || 20, 
      total: data.count || 0,
      query: normalizedQuery 
    };
    
    // Check if we have a results array from backend v2 API format
    if (Array.isArray(data.results)) {
      console.log(`Mapping ${data.results.length} results from 'results' array`);
      
      // Make sure to handle potentially incomplete result objects
      searchResult.words = data.results.map((result: any): SearchWordResult => ({
        id: result.id,
        lemma: result.lemma || `Unknown-${result.id}`,
        normalized_lemma: result.normalized_lemma || result.lemma || '',
        language_code: result.language_code || 'tl',
        has_baybayin: result.has_baybayin || false,
        baybayin_form: result.baybayin_form || null,
        romanized_form: result.romanized_form || null,
        definitions: (Array.isArray(result.definitions) ? result.definitions : []).map((def: any) => ({ 
          id: def.id || 0,
          definition_text: def.definition_text || '',
          part_of_speech: def.part_of_speech || null
        }))
      }));
      
      // Explicitly reset the circuit breaker on successful search with results
      if (searchResult.words.length > 0) {
        console.log('Search successful with results. Explicitly resetting circuit breaker.');
        circuitBreaker.reset();
      }
    } 
    // If "results" is empty, but "data" is an array itself
    else if (Array.isArray(data)) {
      console.log(`Mapping ${data.length} results from direct array response`);
      searchResult.words = data.map((result: any): SearchWordResult => ({
        id: result.id,
        lemma: result.lemma || `Unknown-${result.id}`,
        normalized_lemma: result.normalized_lemma || result.lemma || '',
        language_code: result.language_code || 'tl',
        has_baybayin: result.has_baybayin || false,
        baybayin_form: result.baybayin_form || null,
        romanized_form: result.romanized_form || null,
        definitions: (Array.isArray(result.definitions) ? result.definitions : []).map((def: any) => ({ 
          id: def.id || 0,
          definition_text: def.definition_text || '',
          part_of_speech: def.part_of_speech || null
        }))
      }));
    } 
    // In case we get another unexpected format, try to extract data reasonably
    else {
      // Log the unexpected format for debugging
      console.warn('Unexpected API response format:', data);
      
      // Look for any array property that might contain results
      const potentialResultArrays = Object.entries(data)
        .filter(([_, value]) => Array.isArray(value) && value.length > 0);
      
      if (potentialResultArrays.length > 0) {
        // Use the first array property as our results
        const [propertyName, resultsArray] = potentialResultArrays[0];
        console.log(`Using '${propertyName}' property as results array`);
        
        searchResult.words = (resultsArray as any[]).map((result: any): SearchWordResult => ({
          id: result.id || Math.floor(Math.random() * 10000),  // Fallback ID if missing
          lemma: result.lemma || result.word || result.term || 'Unknown',
          normalized_lemma: result.normalized_lemma || result.normalized || result.lemma || '',
          language_code: result.language_code || result.language || 'tl',
          has_baybayin: result.has_baybayin || false,
          baybayin_form: result.baybayin_form || null,
          romanized_form: result.romanized_form || null,
          definitions: []  // No definitions if we're using an unexpected format
        }));
      }
    }

    setCachedData(cacheKey, searchResult);
    // Success recorded by interceptor
    return searchResult;
  } catch (error) {
    // Failure recorded by interceptor
    console.error(`Search error details for "${normalizedQuery}":`, error);
    
    // Check if it's a timeout error
    if (axios.isAxiosError(error) && error.code === 'ECONNABORTED') {
      console.error('Search request timed out. Increasing future timeout values.');
      // You could adjust CONFIG.timeout here for future searches
    }
    
    await handleApiError(error, `searching words with query "${normalizedQuery}"`);
    throw new Error('An unknown error occurred after handling API error.');
  }
}

// --- Other Utility API Functions --- 

export async function getPartsOfSpeech(): Promise<PartOfSpeech[]> {
  const cacheKey = 'cache:parts_of_speech';
    const cachedData = getCachedData<PartOfSpeech[]>(cacheKey);
  if (cachedData) return cachedData;

  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");

  try {
    // Assuming an endpoint exists for this
    const response = await api.get('/parts_of_speech'); 
     if (response.status !== 200) throw new Error(`API status ${response.status}`);
    // Assuming the data is directly the array or nested under 'data'
    const data = response.data?.data || response.data || [];
    if (!Array.isArray(data)) throw new Error('Invalid data format for parts of speech');
    setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    await handleApiError(error, 'fetching parts of speech');
    throw new Error('An unknown error occurred after handling API error.');
  }
}

export async function testApiConnection(): Promise<boolean> {
  try {
    // First try the test endpoint directly
    const response = await api.get('/test', { timeout: 3000 }); // Short timeout
    if (response.status === 200) {
      return true;
    }
    return false;
  } catch (error) {
    console.error('API connection test failed:', error);
    try {
      // Try the full path as fallback
      const baseUrl = api.defaults.baseURL || CONFIG.baseURL;
      const fullPathResponse = await axios.get(`${baseUrl.split('/api/v2')[0]}/api/v2/test`, { 
        timeout: 3000,
        headers: {
          'Accept': 'application/json'
        }
      });
      const success = fullPathResponse.status === 200;
      if (success) {
        // If full path works, save it for future use
        localStorage.setItem('successful_api_endpoint', fullPathResponse.config.url?.split('/api/v2')[0] || '');
      }
      return success;
    } catch (fallbackError) {
      console.error('API connection test (fallback) failed:', fallbackError);
      return false;
    }
  }
}

export async function getEtymologyTree(
  wordId: number, 
  maxDepth: number = 2 
): Promise<EtymologyTree> {
  const cacheKey = `cache:etymologyTree:${wordId}-${maxDepth}`;
    const cachedData = getCachedData<EtymologyTree>(cacheKey);
  if (cachedData) return cachedData;

  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");

  try {
    const endpoint = `/words/${wordId}/etymology/tree`;
    const response = await api.get(endpoint, { params: { max_depth: maxDepth } });
     if (response.status !== 200) throw new Error(`API status ${response.status}`);
    // Assuming the tree is directly in response.data or response.data.etymology_tree
    const treeData = response.data?.etymology_tree || response.data;
    if (!treeData || typeof treeData !== 'object') throw new Error('Invalid etymology tree data');
    setCachedData(cacheKey, treeData);
    return treeData;
  } catch (error) {
    await handleApiError(error, `fetching etymology tree for word ID ${wordId}`);
    throw new Error('An unknown error occurred after handling API error.');
  }
}

export async function getRandomWord(): Promise<WordInfo> {
  // Random word shouldn't be cached aggressively
  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");

  try {
    // FIX: Correct endpoint for fetching a random word
    const response = await api.get('/random'); 
     if (response.status !== 200) throw new Error(`API status ${response.status}`);
    // Random word endpoint might return the full comprehensive structure already
    // Or it might return a simpler structure that needs fetching details separately.
    // Assuming it returns comprehensive data for now:
    return normalizeWordData(response.data);
  } catch (error) {
    await handleApiError(error, 'fetching random word');
     throw new Error('An unknown error occurred after handling API error.');
  }
}

export async function getStatistics(): Promise<Statistics> {
  const cacheKey = 'cache:statistics';
    const cachedData = getCachedData<Statistics>(cacheKey);
  if (cachedData) return cachedData;

  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");

  try {
    const response = await api.get('/statistics');
     if (response.status !== 200) throw new Error(`API status ${response.status}`);
    const statsData = response.data;
    if (!statsData || typeof statsData !== 'object') throw new Error('Invalid statistics data');
    // Add timestamp to stats data if not present
    statsData.timestamp = statsData.timestamp || new Date().toISOString();
    setCachedData(cacheKey, statsData);
    return statsData;
  } catch (error) {
    await handleApiError(error, 'fetching statistics');
     throw new Error('An unknown error occurred after handling API error.');
  }
}

// --- Potentially Less Used / Example Endpoints --- 

// Note: These might need adjustments based on actual backend implementation

export async function getBaybayinWords(page: number = 1, limit: number = 20, language: string = 'tl'): Promise<any> {
  // Example endpoint, adjust as needed
  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");
  try {
    const response = await api.get('/words/baybayin', { params: { page, limit, language } });
    return response.data;
  } catch (error) {
    await handleApiError(error, 'fetching baybayin words');
     throw new Error('An unknown error occurred after handling API error.');
  }
}

export async function getAffixes(language: string = 'tl', type?: string): Promise<any> {
  // Example endpoint, adjust as needed
  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");
  try {
    const params: any = { language };
    if (type) params.type = type;
    const response = await api.get('/affixes', { params });
    return response.data;
  } catch (error) {
    await handleApiError(error, 'fetching affixes');
     throw new Error('An unknown error occurred after handling API error.');
  }
}

export async function getRelations(language: string = 'tl', type?: string): Promise<any> {
  // Example endpoint, adjust as needed
  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");
  try {
    const params: any = { language };
    if (type) params.type = type;
    // Assuming endpoint might be /relationships or similar
    const response = await api.get('/relationships', { params }); 
    return response.data;
  } catch (error) {
    await handleApiError(error, 'fetching relations');
     throw new Error('An unknown error occurred after handling API error.');
  }
}

export async function getAllWords(page: number = 1, perPage: number = 20, language: string = 'tl'): Promise<any> {
  // Example endpoint, adjust as needed
  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");
  try {
    const response = await api.get('/words', { params: { page, per_page: perPage, language } });
    return response.data;
  } catch (error) {
    await handleApiError(error, 'fetching all words');
     throw new Error('An unknown error occurred after handling API error.');
  }
}