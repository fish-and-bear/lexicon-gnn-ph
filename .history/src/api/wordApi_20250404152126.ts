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
    timeout: 15000, // Increased timeout for production
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
    // Record failure for circuit breaker
      circuitBreaker.recordFailure();
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
): Promise<WordNetwork> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `network:${sanitizedWord}:${JSON.stringify(options)}`;
  const cached = getCachedData<WordNetwork>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = {
      params: {
        max_depth: options.max_depth ?? 1, // Use backend param name
        bidirectional: options.bidirectional ?? true, // Use backend param name
        // Remove params not used by backend
        // include_affixes: options.include_affixes ?? false,
        // include_etymology: options.include_etymology ?? false,
        // cluster_threshold: options.cluster_threshold ?? 0.5
      },
      retry: CONFIG.retries
    };
    const response = await api.get<WordNetwork>(`/words/${encodeURIComponent(sanitizedWord)}/relations/graph`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching network for ${sanitizedWord}`);
  }
}

// REMOVE the normalizeWordData function. Data cleaning/shaping should be
// handled by the backend or closer to the component rendering the data.
/*
function normalizeWordData(rawData: any): WordInfo { ... }
*/

// --- Endpoint Functions Aligned with /api/v2 --- 

// GET /words/{word}
export async function getWordBasic(word: string): Promise<WordInfo> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `word_basic:${sanitizedWord}`;
  const cached = getCachedData<WordInfo>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    // Assuming this endpoint returns a subset matching WordInfo structure, but possibly less detailed
    // Use RawWordComprehensiveData for now, components can pick what they need
    const response = await api.get<RawWordComprehensiveData>(`/words/${encodeURIComponent(sanitizedWord)}`, config);
    setCachedData(cacheKey, response.data); 
    circuitBreaker.recordSuccess();
    // No normalization needed here if types match backend closely
    return response.data as WordInfo; // Cast for now, assuming basic endpoint matches WordInfo structure reasonably
  } catch (error) {
    handleApiError(error, `fetching basic details for ${sanitizedWord}`);
  }
}

// GET /words/{word}/comprehensive (replaces old fetchWordDetails)
export async function fetchWordComprehensive(word: string): Promise<WordInfo> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `word_comprehensive:${sanitizedWord}`;
  const cached = getCachedData<WordInfo>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<RawWordComprehensiveData>(`/words/${encodeURIComponent(sanitizedWord)}/comprehensive`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    // No normalization needed
    return response.data as WordInfo; // Return the raw data, assuming WordInfo matches
  } catch (error) {
    handleApiError(error, `fetching comprehensive details for ${sanitizedWord}`);
  }
}

// GET /search
export async function searchWords(options: SearchOptions): Promise<SearchResult> {
  const sanitizedQuery = sanitizeInput(options.q); // Sanitize query param
  const cacheKey = `search:${sanitizedQuery}:${JSON.stringify(options)}`;
  const cached = getCachedData<SearchResult>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = {
      params: { 
        q: sanitizedQuery,
        limit: options.limit ?? 20, 
        offset: options.offset ?? 0,
        mode: options.mode,
        language: options.language,
        pos: options.pos,
        sort: options.sort,
        order: options.order
      },
      retry: CONFIG.retries
    };
    const response = await api.get<SearchResult>('/search', config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    // Ensure the response structure matches SearchResult type
    return response.data; 
  } catch (error) {
    handleApiError(error, `searching for ${sanitizedQuery}`);
  }
}

// GET /words/{word}/relations
export async function getWordRelations(word: string): Promise<{ outgoing_relations: Relation[], incoming_relations: Relation[] }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `relations:${sanitizedWord}`;
  const cached = getCachedData<{ outgoing_relations: Relation[], incoming_relations: Relation[] }>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ outgoing_relations: Relation[], incoming_relations: Relation[] }>(`/words/${encodeURIComponent(sanitizedWord)}/relations`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching relations for ${sanitizedWord}`);
  }
}

// GET /words/{word}/affixations
export async function getWordAffixations(word: string): Promise<{ as_root: Affixation[], as_affixed: Affixation[] }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `affixations:${sanitizedWord}`;
  const cached = getCachedData<{ as_root: Affixation[], as_affixed: Affixation[] }>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ as_root: Affixation[], as_affixed: Affixation[] }>(`/words/${encodeURIComponent(sanitizedWord)}/affixations`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching affixations for ${sanitizedWord}`);
  }
}

// GET /words/{word}/pronunciation
export async function getWordPronunciations(word: string): Promise<{ pronunciations: Pronunciation[], has_pronunciation: boolean }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `pronunciations:${sanitizedWord}`;
  const cached = getCachedData<{ pronunciations: Pronunciation[], has_pronunciation: boolean }>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ pronunciations: Pronunciation[], has_pronunciation: boolean }>(`/words/${encodeURIComponent(sanitizedWord)}/pronunciation`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching pronunciations for ${sanitizedWord}`);
  }
}

// GET /words/{word}/etymology
export async function getWordEtymology(word: string): Promise<{ etymologies: Etymology[], has_etymology: boolean }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `etymology:${sanitizedWord}`;
  const cached = getCachedData<{ etymologies: Etymology[], has_etymology: boolean }>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    // Use RawEtymology here as the component structure comes directly from backend
    const response = await api.get<{ etymologies: RawEtymology[], has_etymology: boolean }>(`/words/${encodeURIComponent(sanitizedWord)}/etymology`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    // Cast needed if components expect cleaned Etymology type
    return response.data as { etymologies: Etymology[], has_etymology: boolean }; 
  } catch (error) {
    handleApiError(error, `fetching etymology for ${sanitizedWord}`);
  }
}

// GET /words/{id}/etymology/tree (Keep this, uses ID)
export async function getEtymologyTree(
  wordId: number, 
  maxDepth: number = 2 
): Promise<EtymologyTree> {
  const cacheKey = `etymology_tree:${wordId}:${maxDepth}`;
  const cached = getCachedData<EtymologyTree>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = {
      params: { max_depth: maxDepth },
      retry: CONFIG.retries
    };
    // Note: Endpoint uses ID, not lemma
    const response = await api.get<EtymologyTree>(`/words/${wordId}/etymology/tree`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching etymology tree for ID ${wordId}`);
  }
}

// GET /random
export async function getRandomWord(): Promise<WordInfo> {
  // Random endpoint shouldn't be cached aggressively
  const cacheKey = `random_word:${Date.now()}`; // Very short-term cache key

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: 0 }; // No retries for random?
    const response = await api.get<RawWordComprehensiveData>('/random', config);
    // No caching for random word usually
    // setCachedData(cacheKey, response.data, 60 * 1000); // Cache for 1 min?
    circuitBreaker.recordSuccess();
    // No normalization needed
    return response.data as WordInfo;
  } catch (error) {
    handleApiError(error, 'fetching random word');
  }
}

// GET /statistics
export async function getStatistics(): Promise<Statistics> {
  const cacheKey = 'statistics';
  const cached = getCachedData<Statistics>(cacheKey, 5 * 60 * 1000); // Cache for 5 mins
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<Statistics>('/statistics', config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, 'fetching statistics');
  }
}

// GET /relationships/types
export async function getRelationshipTypes(): Promise<{ relationship_types: RelationshipTypeInfo[], total: number }> {
  const cacheKey = 'relationship_types';
  const cached = getCachedData<{ relationship_types: RelationshipTypeInfo[], total: number }>(cacheKey, 60 * 60 * 1000); // Cache for 1 hour
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ relationship_types: RelationshipTypeInfo[], total: number }>('/relationships/types', config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, 'fetching relationship types');
  }
}

// GET /relationships/{relationship_type}
export async function getWordsByRelationship(
  relationshipType: string,
  limit: number = 100,
  offset: number = 0
): Promise<{ relationship_type: string, relationships: Relation[], total: number, limit: number, offset: number }> {
  const cacheKey = `words_by_relationship:${relationshipType}:${limit}:${offset}`;
  const cachedResult = getCachedData<any>(cacheKey);
  if (cachedResult) return cachedResult;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = {
      params: { limit, offset },
      retry: CONFIG.retries,
    };
    const response = await api.get<any>(`/relationships/${encodeURIComponent(relationshipType)}`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching words for relationship type ${relationshipType}`);
  }
}

// --- Deprecated / To Be Reviewed --- 

// REMOVE getPartsOfSpeech - POS info should come nested within definitions
/*
export async function getPartsOfSpeech(): Promise<PartOfSpeech[]> { ... }
*/

// REMOVE getBaybayinWords - Can be achieved via search?
/*
export async function getBaybayinWords(...) { ... }
*/

// REMOVE getAffixes - Can be achieved via search?
/*
export async function getAffixes(...) { ... }
*/

// REMOVE getRelations - Replaced by getWordsByRelationship
/*
export async function getRelations(...) { ... }
*/

// REMOVE getAllWords - Can be achieved via search with large limit?
/*
export async function getAllWords(...) { ... }
*/

// Keep testApiConnection for diagnostics
export async function testApiConnection(): Promise<boolean> {
  if (!circuitBreaker.canMakeRequest()) {
    console.warn('Circuit breaker is open. Skipping connection test.');
    return false;
  }
  try {
    // Use the health endpoint for a quick check
    const response = await api.get('/health', { retry: 0, timeout: 3000 }); // Short timeout, no retry
    circuitBreaker.recordSuccess();
    console.log('API connection test successful.');
    return response.status === 200;
  } catch (error) {
    console.error('API connection test failed:', error);
    circuitBreaker.recordFailure();
    return false;
  }
}

// REMOVE old fetchWordDetails
/*
export async function fetchWordDetails(word: string): Promise<WordInfo> { ... }
*/


// ... existing code ...

// Environment and configuration constants
const ENV = process.env.NODE_ENV || 'development';
// Ensure URLs point to v2
const CONFIG = {
  development: {
    baseURL: 'http://localhost:10000/api/v2', // Already v2
    timeout: 10000,
    retries: 3,
// ... existing code ...
  },
  production: {
    baseURL: process.env.REACT_APP_API_BASE_URL?.replace('/api/v1', '/api/v2') || 'https://fil-relex.onrender.com/api/v2', // Ensure replacement or default is v2
    timeout: 15000, // Increased timeout for production
    retries: 2,
// ... existing code ...
    maxRetryDelay: 10000
  }
}[ENV];

// ... existing code ...
  depth?: number;
  include_affixes?: boolean;
  include_etymology?: boolean;
  cluster_threshold?: number;
}

// Update fetchWordNetwork to use the new endpoint and params
export async function fetchWordNetwork(
  word: string, 
  options: WordNetworkOptions = {}
): Promise<WordNetwork> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `network:${sanitizedWord}:${JSON.stringify(options)}`;
  const cached = getCachedData<WordNetwork>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = {
      params: {
        max_depth: options.max_depth ?? 1, // Use backend param name
        bidirectional: options.bidirectional ?? true, // Use backend param name
        // Remove params not used by backend
        // include_affixes: options.include_affixes ?? false,
        // include_etymology: options.include_etymology ?? false,
        // cluster_threshold: options.cluster_threshold ?? 0.5
      },
      retry: CONFIG.retries
    };
    const response = await api.get<WordNetwork>(`/words/${encodeURIComponent(sanitizedWord)}/relations/graph`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching network for ${sanitizedWord}`);
  }
}

// REMOVE the normalizeWordData function. Data cleaning/shaping should be
// handled by the backend or closer to the component rendering the data.
/*
function normalizeWordData(rawData: any): WordInfo { ... }
*/

// --- Endpoint Functions Aligned with /api/v2 --- 

// GET /words/{word}
export async function getWordBasic(word: string): Promise<WordInfo> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `word_basic:${sanitizedWord}`;
  const cached = getCachedData<WordInfo>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    // Assuming this endpoint returns a subset matching WordInfo structure, but possibly less detailed
    // Use RawWordComprehensiveData for now, components can pick what they need
    const response = await api.get<RawWordComprehensiveData>(`/words/${encodeURIComponent(sanitizedWord)}`, config);
    setCachedData(cacheKey, response.data); 
    circuitBreaker.recordSuccess();
    // No normalization needed here if types match backend closely
    return response.data as WordInfo; // Cast for now, assuming basic endpoint matches WordInfo structure reasonably
  } catch (error) {
    handleApiError(error, `fetching basic details for ${sanitizedWord}`);
  }
}

// GET /words/{word}/comprehensive (replaces old fetchWordDetails)
export async function fetchWordComprehensive(word: string): Promise<WordInfo> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `word_comprehensive:${sanitizedWord}`;
  const cached = getCachedData<WordInfo>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<RawWordComprehensiveData>(`/words/${encodeURIComponent(sanitizedWord)}/comprehensive`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    // No normalization needed
    return response.data as WordInfo; // Return the raw data, assuming WordInfo matches
  } catch (error) {
    handleApiError(error, `fetching comprehensive details for ${sanitizedWord}`);
  }
}

// GET /search
export async function searchWords(options: SearchOptions): Promise<SearchResult> {
  const sanitizedQuery = sanitizeInput(options.q); // Sanitize query param
  const cacheKey = `search:${sanitizedQuery}:${JSON.stringify(options)}`;
  const cached = getCachedData<SearchResult>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = {
      params: { 
        q: sanitizedQuery,
        limit: options.limit ?? 20, 
        offset: options.offset ?? 0,
        mode: options.mode,
        language: options.language,
        pos: options.pos,
        sort: options.sort,
        order: options.order
      },
      retry: CONFIG.retries
    };
    const response = await api.get<SearchResult>('/search', config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    // Ensure the response structure matches SearchResult type
    return response.data; 
  } catch (error) {
    handleApiError(error, `searching for ${sanitizedQuery}`);
  }
}

// GET /words/{word}/relations
export async function getWordRelations(word: string): Promise<{ outgoing_relations: Relation[], incoming_relations: Relation[] }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `relations:${sanitizedWord}`;
  const cached = getCachedData<{ outgoing_relations: Relation[], incoming_relations: Relation[] }>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ outgoing_relations: Relation[], incoming_relations: Relation[] }>(`/words/${encodeURIComponent(sanitizedWord)}/relations`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching relations for ${sanitizedWord}`);
  }
}

// GET /words/{word}/affixations
export async function getWordAffixations(word: string): Promise<{ as_root: Affixation[], as_affixed: Affixation[] }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `affixations:${sanitizedWord}`;
  const cached = getCachedData<{ as_root: Affixation[], as_affixed: Affixation[] }>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ as_root: Affixation[], as_affixed: Affixation[] }>(`/words/${encodeURIComponent(sanitizedWord)}/affixations`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching affixations for ${sanitizedWord}`);
  }
}

// GET /words/{word}/pronunciation
export async function getWordPronunciations(word: string): Promise<{ pronunciations: Pronunciation[], has_pronunciation: boolean }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `pronunciations:${sanitizedWord}`;
  const cached = getCachedData<{ pronunciations: Pronunciation[], has_pronunciation: boolean }>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ pronunciations: Pronunciation[], has_pronunciation: boolean }>(`/words/${encodeURIComponent(sanitizedWord)}/pronunciation`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    handleApiError(error, `fetching pronunciations for ${sanitizedWord}`);
  }
}

// GET /words/{word}/etymology
export async function getWordEtymology(word: string): Promise<{ etymologies: Etymology[], has_etymology: boolean }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `etymology:${sanitizedWord}`;
  const cached = getCachedData<{ etymologies: Etymology[], has_etymology: boolean }>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    // Use RawEtymology here as the component structure comes directly from backend
    const response = await api.get<{ etymologies: RawEtymology[], has_etymology: boolean }>(`/words/${encodeURIComponent(sanitizedWord)}/etymology`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    // Cast needed if components expect cleaned Etymology type
    return response.data as { etymologies: Etymology[], has_etymology: boolean }; 
  } catch (error) {
    handleApiError(error, `fetching etymology for ${sanitizedWord}`);
  }
}

// GET /words/{id}/etymology/tree (Keep this, uses ID)
export async function getEtymologyTree(
  wordId: number, 
  maxDepth: number = 2 
): Promise<EtymologyTree> {
// --- Search Functionality --- 

export async function searchWords(query: string, options: SearchOptions): Promise<SearchResult> {
  const cacheKey = `cache:search:${query}:${JSON.stringify(options)}`;
  const cachedData = getCachedData<SearchResult>(cacheKey);

  if (cachedData) {
    console.log(`Cache hit for search: ${query}`);
    return cachedData;
  }
  console.log(`Cache miss for search: ${query}. Fetching from API...`);

  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for search.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }

  try {
    // Use snake_case for API parameters to match backend expectations
    const apiParams: Record<string, any> = {
        q: query,
        limit: options.per_page || 20, // Default limit
        offset: options.page ? (options.page - 1) * (options.per_page || 20) : 0, 
    };
    if (options.language) apiParams.language = options.language;
    if (options.mode) apiParams.mode = options.mode;
    if (options.pos) apiParams.pos = options.pos;
    if (options.sort) apiParams.sort = options.sort;
    if (options.order) apiParams.order = options.order;
    if (options.exclude_baybayin !== undefined) apiParams.exclude_baybayin = options.exclude_baybayin;
    // Add any other supported options from SearchOptions

    const response = await api.get('/search', { params: apiParams });
     if (response.status !== 200) {
        throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }
    const data = response.data; // Assuming data = { total: number, words: RawWordSummary[] }
    
    // Transform the response into SearchResult format
    const searchResult: SearchResult = {
      words: (data.words || []).map((result: any): SearchWordResult => ({
        id: result.id,
        lemma: result.lemma,
        normalized_lemma: result.normalized_lemma,
        language_code: result.language_code,
        has_baybayin: result.has_baybayin,
        baybayin_form: result.baybayin_form,
        romanized_form: result.romanized_form,
        // Search results usually have simpler definition structures
        definitions: (result.definitions || []).map((def: any) => ({ 
            id: def.id || 0,
            definition_text: def.definition_text || '',
            part_of_speech: def.part_of_speech || null
        }))
      })),
        page: options.page || 1,
      perPage: options.per_page || (data.words?.length || 0), 
      total: data.total || 0,
      query: query 
    };

    setCachedData(cacheKey, searchResult);
    // Success recorded by interceptor
    return searchResult;
  } catch (error) {
    // Failure recorded by interceptor
    await handleApiError(error, `searching words with query "${query}"`);
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
    // Use a simple, fast endpoint like /health or /test
    const response = await api.get('/test', { timeout: 3000 }); // Short timeout
    return response.status === 200;
  } catch (error) {
    console.error('API connection test failed:', error);
    return false;
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