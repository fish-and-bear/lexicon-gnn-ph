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
  RelatedWord,
  RawEtymology,
  RelationshipTypeInfo,
  WordNetworkOptions
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
      testApiConnection().then((connected: boolean) => {
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
      },
      retry: CONFIG.retries
    };
    const response = await api.get<WordNetwork>(`/words/${encodeURIComponent(sanitizedWord)}/relations/graph`, config);
    setCachedData(cacheKey, response.data);
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    await handleApiError(error, `fetching network for ${sanitizedWord}`);
    throw new Error('Unreachable: handleApiError should have thrown');
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
    setCachedData(cacheKey, response.data as any); // Temp cast to any for caching
    circuitBreaker.recordSuccess();
    // No normalization needed here if types match backend closely
    return response.data as WordInfo; // Cast for now, assuming basic endpoint matches WordInfo structure reasonably
  } catch (error) {
    await handleApiError(error, `fetching basic details for ${sanitizedWord}`);
    throw new Error('Unreachable: handleApiError should have thrown');
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
    setCachedData(cacheKey, response.data as any); // Temp cast to any for caching
    circuitBreaker.recordSuccess();
    // No normalization needed
    return response.data as WordInfo; // Return the raw data, assuming WordInfo matches
  } catch (error) {
    await handleApiError(error, `fetching comprehensive details for ${sanitizedWord}`);
    throw new Error('Unreachable: handleApiError should have thrown');
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
        limit: options.limit ?? 20, // Use limit from SearchOptions
        offset: options.offset ?? 0, // Use offset from SearchOptions
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
    await handleApiError(error, `searching for ${sanitizedQuery}`);
    throw new Error('Unreachable: handleApiError should have thrown');
  }
}

// GET /words/{word}/relations
export async function getWordRelations(word: string): Promise<{ outgoing_relations: Relation[], incoming_relations: Relation[] }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `relations:${sanitizedWord}`;
  // Use 'any' for cache retrieval type due to complex object structure not fitting CacheableData easily
  const cached = getCachedData<any>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ outgoing_relations: Relation[], incoming_relations: Relation[] }>(`/words/${encodeURIComponent(sanitizedWord)}/relations`, config);
    setCachedData(cacheKey, response.data as any); // Temp cast to any for caching
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    await handleApiError(error, `fetching relations for ${sanitizedWord}`);
    throw new Error('Unreachable: handleApiError should have thrown');
  }
}

// GET /words/{word}/affixations
export async function getWordAffixations(word: string): Promise<{ as_root: Affixation[], as_affixed: Affixation[] }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `affixations:${sanitizedWord}`;
  // Use 'any' for cache retrieval type
  const cached = getCachedData<any>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ as_root: Affixation[], as_affixed: Affixation[] }>(`/words/${encodeURIComponent(sanitizedWord)}/affixations`, config);
    setCachedData(cacheKey, response.data as any); // Temp cast to any for caching
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    await handleApiError(error, `fetching affixations for ${sanitizedWord}`);
    throw new Error('Unreachable: handleApiError should have thrown');
  }
}

// GET /words/{word}/pronunciation
export async function getWordPronunciations(word: string): Promise<{ pronunciations: Pronunciation[], has_pronunciation: boolean }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `pronunciations:${sanitizedWord}`;
  // Use 'any' for cache retrieval type
  const cached = getCachedData<any>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ pronunciations: Pronunciation[], has_pronunciation: boolean }>(`/words/${encodeURIComponent(sanitizedWord)}/pronunciation`, config);
    setCachedData(cacheKey, response.data as any); // Temp cast to any for caching
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    await handleApiError(error, `fetching pronunciations for ${sanitizedWord}`);
    throw new Error('Unreachable: handleApiError should have thrown');
  }
}

// GET /words/{word}/etymology
export async function getWordEtymology(word: string): Promise<{ etymologies: RawEtymology[], has_etymology: boolean }> {
  const sanitizedWord = sanitizeInput(word);
  const cacheKey = `etymology:${sanitizedWord}`;
  const cached = getCachedData<any>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ etymologies: RawEtymology[], has_etymology: boolean }>(`/words/${encodeURIComponent(sanitizedWord)}/etymology`, config); 
    setCachedData(cacheKey, response.data as any); // Temp cast to any for caching
    circuitBreaker.recordSuccess();
    // Return the raw response data directly
    return response.data; 
  } catch (error) {
    await handleApiError(error, `fetching etymology for ${sanitizedWord}`);
    throw new Error('Unreachable: handleApiError should have thrown');
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
    await handleApiError(error, `fetching etymology tree for ID ${wordId}`);
    throw new Error('Unreachable: handleApiError should have thrown');
  }
}

// GET /random
export async function getRandomWord(): Promise<WordInfo> {
  // Random endpoint shouldn't be cached aggressively
  // const cacheKey = `random_word:${Date.now()}`; // Very short-term cache key

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
    await handleApiError(error, 'fetching random word');
    throw new Error('Unreachable: handleApiError should have thrown');
  }
}

// GET /statistics
export async function getStatistics(): Promise<Statistics> {
  const cacheKey = 'statistics';
  const cached = getCachedData<Statistics>(cacheKey);
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
    await handleApiError(error, 'fetching statistics');
    throw new Error('Unreachable: handleApiError should have thrown');
  }
}

// GET /relationships/types
export async function getRelationshipTypes(): Promise<{ relationship_types: RelationshipTypeInfo[], total: number }> {
  const cacheKey = 'relationship_types';
  // Use 'any' for cache retrieval type
  const cached = getCachedData<any>(cacheKey);
  if (cached) return cached;

  if (!circuitBreaker.canMakeRequest()) {
    throw new Error('Circuit breaker is open. Request blocked.');
  }

  try {
    const config: ExtendedAxiosRequestConfig = { retry: CONFIG.retries };
    const response = await api.get<{ relationship_types: RelationshipTypeInfo[], total: number }>('/relationships/types', config);
    setCachedData(cacheKey, response.data as any); // Temp cast to any for caching
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    await handleApiError(error, 'fetching relationship types');
    throw new Error('Unreachable: handleApiError should have thrown');
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
    setCachedData(cacheKey, response.data as any); // Temp cast to any for caching
    circuitBreaker.recordSuccess();
    return response.data;
  } catch (error) {
    await handleApiError(error, `fetching words for relationship type ${relationshipType}`);
    throw new Error('Unreachable: handleApiError should have thrown');
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
    const response = await api.get('/health', { timeout: 3000 }); // Remove non-standard 'retry' property
    circuitBreaker.recordSuccess();
    console.log('API connection test successful.');
    return response.status === 200;
  } catch (error) {
    console.error('API connection test failed:', error);
    // Don't record failure here as it might trigger CB unnecessarily on transient test failures
    // circuitBreaker.recordFailure(); 
    return false;
  }
}

// REMOVE old fetchWordDetails
/*
export async function fetchWordDetails(word: string): Promise<WordInfo> { ... }
*/

// REMOVE ALL DUPLICATED CODE THAT MIGHT HAVE BEEN LEFT BELOW THIS LINE