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
  RawWordData,
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
export const API_BASE_URL = 'http://localhost:10000/api/v2';

// Add a proxy URL for when direct connections fail
export const PROXY_API_URL = 'http://localhost:10000/api/v2';

// Log which endpoint we're using
console.log(`[API] Using API_BASE_URL: ${API_BASE_URL}`);

const CONFIG = {
  development: {
    baseURL: 'http://localhost:10000/api/v2',
    timeout: 15000,
    retries: 3,
    failureThreshold: 10,
    resetTimeout: 15000,
    retryDelay: 1000,
    maxRetryDelay: 10000
  },
  production: {
    baseURL: process.env.REACT_APP_API_BASE_URL?.replace('/api/v1', '/api/v2') || 'http://localhost:10000/api/v2',
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
    
    // New: Auto-reset if the circuit breaker has been open for too long
    const MAX_OPEN_TIME = 60000; // 1 minute maximum open time
    if (this.state === 'open' && this.lastFailureTime && 
        (Date.now() - this.lastFailureTime > MAX_OPEN_TIME)) {
      console.log(`Circuit breaker has been open for over ${MAX_OPEN_TIME/1000} seconds. Auto-resetting.`);
      this.reset();
      return true;
    }
    
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
  console.log('[API] Forcing circuit breaker to closed state');
  if (!circuitBreaker) {
    console.warn('[API] No circuit breaker instance exists');
    return false;
  }
  
  try {
    // Reset failure count and state
    circuitBreaker.reset();
    
    // Clear any cached errors
    localStorage.removeItem('api_last_error');
    localStorage.removeItem('api_error_count');
    
    console.log('[API] Circuit breaker successfully reset and forced closed');
    return true;
  } catch (e) {
    console.error('[API] Error forcing circuit breaker closed:', e);
    return false;
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
    // Any successful 2xx response indicates the API is working.
    if (response.status >= 200 && response.status < 300) {
      // Reset the circuit breaker to closed state on ANY success.
      console.log("Successful API response received. Explicitly resetting circuit breaker.");
      circuitBreaker.recordSuccess(); 
    }
    return response;
  },
  (error) => {
    console.error('Response error:', error);
    
    const isSearchRequest = error.config?.url?.includes('/search');
    const isNetworkError = !error.response && error.request;
    
    // Determine if this error should count as a failure for the circuit breaker
    let shouldRecordFailure = false;
    if (isSearchRequest) {
      // For search: Only count 5xx server errors as actual failures
      if (!isNetworkError && error.response && error.response.status >= 500) {
        console.warn(`Search request failed with server error ${error.response.status}. Recording failure.`);
        shouldRecordFailure = true;
      } else {
        console.log(`Ignoring search error (Status: ${error.response?.status}, Network: ${isNetworkError}) for circuit breaker.`);
      }
    } else {
      // For non-search requests: Count failures if it's not a pure network error
      if (!isNetworkError) { 
        console.warn(`Non-search request failed (Status: ${error.response?.status}). Recording failure.`);
        shouldRecordFailure = true;
      } else {
        console.log('Ignoring network error for non-search request for circuit breaker.');
      }
    }
    
    // Record failure if decided
    if (shouldRecordFailure) {
      circuitBreaker.recordFailure();
    }
    
    // Handle specific HTTP errors for user feedback (this part remains the same)
    if (error.response) {
        console.error(`API Error: Status ${error.response.status}, Data:`, error.response.data);
        if (error.response.status === 404) {
            // Return a rejected promise with a specific error type if needed, 
            // but don't necessarily throw here if the calling code handles rejection.
            return Promise.reject(new Error('Resource not found (404)')); 
        }
    } else if (error.request) {
        console.error('API Error: No response received', error.request);
    } else {
        console.error('API Error: Request setup failed', error.message);
    }
    
    // IMPORTANT: Always reject the promise so the calling function knows about the error
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
  depth?: number;   // How many relationship "hops" to traverse (1-5)
  relation_types?: string[]; // Which relationship types to include
  breadth?: number; // Maximum connections per word (1-100)
  timeout?: number; // Request timeout in ms
}

export const fetchWordNetwork = async (
  word: string,
  options: {
    depth?: number;
    breadth?: number;
    relation_types?: string[];
  } = {}
): Promise<WordNetworkResponse> => {
  const { depth = 1, breadth = 15, relation_types } = options;
  
  try {
    if (circuitBreaker && !circuitBreaker.canMakeRequest()) {
      throw new Error("Circuit breaker is open. Please try again later.");
    }
    
    // Fix: Use direct baseURL to ensure consistent endpoint
    const baseUrl = API_BASE_URL;
    let url = `${baseUrl}/words/${encodeURIComponent(word)}/semantic_network?depth=${depth}&breadth=${breadth}`;
    
    if (relation_types && relation_types.length > 0) {
      url += `&relation_types=${relation_types.join(',')}`;
    }
    
    console.log(`[API] Fetching word network from: ${url}`);
    
    // Try direct fetch with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout
    
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Origin': window.location.origin
        },
        mode: 'cors',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('[API] Network data received:', data);
      
      // Reset circuit breaker on success
      if (circuitBreaker) circuitBreaker.recordSuccess();
      
      // Normalize edges format   
      if (data.edges) { 
        data.edges = data.edges.map((edge: any) => ({
          ...edge,
          source: typeof edge.source === 'object' ? edge.source.id : edge.source,   
          target: typeof edge.target === 'object' ? edge.target.id : edge.target,   
          type: edge.type || 'default'
        }));
      } else if (data.links) {
        data.edges = data.links.map((edge: any) => ({
          ...edge,
          source: typeof edge.source === 'object' ? edge.source.id : edge.source,   
          target: typeof edge.target === 'object' ? edge.target.id : edge.target,   
          type: edge.type || 'default'
        }));
      } else {
        data.edges = [];
      }
      
      // Ensure we have nodes array
      if (!data.nodes) {
        data.nodes = [];
      }
      
      return data;
    } catch (fetchError: unknown) {
      clearTimeout(timeoutId);
      
      // Check if this was a timeout
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        console.warn('Direct fetch timed out for network data: ${word}. Falling back to axios.');
      } else {
        console.warn(`Direct fetch failed: ${fetchError instanceof Error ? fetchError.message : String(fetchError)}. Falling back to axios.`);
      }
      
      // If direct fetch fails, fall back to axios with extended timeout
      const response = await api.get(`/words/${encodeURIComponent(word)}/semantic_network`, {
        params: {
          depth,
          breadth,
          relation_types: relation_types ? relation_types.join(',') : undefined
        },
        timeout: 20000 // Extended timeout for fallback
      });
      
      const data = response.data;
      
      // Reset circuit breaker on success
      if (circuitBreaker) circuitBreaker.recordSuccess();
      
      // Normalize edges format for axios response
      if (data.edges) { 
        data.edges = data.edges.map((edge: any) => ({
          ...edge,
          source: typeof edge.source === 'object' ? edge.source.id : edge.source,   
          target: typeof edge.target === 'object' ? edge.target.id : edge.target,   
          type: edge.type || 'default'
        }));
      } else if (data.links) {
        data.edges = data.links.map((edge: any) => ({
          ...edge,
          source: typeof edge.source === 'object' ? edge.source.id : edge.source,   
          target: typeof edge.target === 'object' ? edge.target.id : edge.target,   
          type: edge.type || 'default'
        }));
      } else {
        data.edges = [];
      }
      
      // Ensure we have nodes array
      if (!data.nodes) {
        data.nodes = [];
      }
      
      return data;
    }
  } catch (error: unknown) {
    console.error(`Error fetching word network for "${word}":`, error);
    throw error;
  }
};

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
  // Handle different possible structures
  const wordData = rawData?.data || rawData;

  if (!wordData || typeof wordData !== 'object') {
    console.error("normalizeWordData received invalid data:", rawData);
    throw new Error('Invalid API response: Missing word data.');
  }
  
  // If ID is missing but we have a lemma, we can still create a partial record
  if (!wordData.id && !wordData.lemma) {
    console.error("normalizeWordData: Both ID and lemma are missing:", wordData);
    throw new Error('Invalid API response: Missing essential word data (both ID and lemma).');
  }

  try {
    // Create base WordInfo object with safe default values
    const normalizedWord: WordInfo = {
      id: wordData.id || 0,
      lemma: wordData.lemma || '',
      normalized_lemma: wordData.normalized_lemma || wordData.lemma || '',
      language_code: wordData.language_code || 'tl',
      has_baybayin: Boolean(wordData.has_baybayin) || false,
      baybayin_form: wordData.baybayin_form || null,
      romanized_form: wordData.romanized_form || null,
      root_word_id: wordData.root_word_id || null,
      preferred_spelling: wordData.preferred_spelling || null,
      tags: wordData.tags || null,
      data_hash: wordData.data_hash || null,
      search_text: wordData.search_text || null,
      created_at: wordData.created_at || null,
      updated_at: wordData.updated_at || null,
    };

    // Initialize empty arrays for collections 
    normalizedWord.definitions = [];
    normalizedWord.etymologies = [];
    normalizedWord.pronunciations = [];
    normalizedWord.credits = [];
    normalizedWord.derived_words = [];
    normalizedWord.outgoing_relations = [];
    normalizedWord.incoming_relations = [];
    normalizedWord.root_affixations = [];
    normalizedWord.affixed_affixations = [];
    normalizedWord.forms = wordData.forms || [];
    normalizedWord.templates = wordData.templates || [];

    // Safe handling of non-array data fields
    normalizedWord.data_completeness = wordData.data_completeness || null;
    normalizedWord.relation_summary = wordData.relation_summary || null;
    normalizedWord.completeness_score = wordData.completeness_score || 
                                        (wordData.data_completeness && 
                                        typeof wordData.data_completeness.completeness_score === 'number' ? 
                                        wordData.data_completeness.completeness_score : null);

    // Normalize Definitions - with safe handling for missing or malformed data
    if (wordData.definitions && Array.isArray(wordData.definitions)) {
      try {
        normalizedWord.definitions = wordData.definitions.map((def: any): Definition => {
          if (!def || typeof def !== 'object') return { 
            id: 0,
            definition_text: '', 
            examples: [], 
            usage_notes: [], 
            tags: [], 
            sources: [], 
            part_of_speech: null 
          };
          
          return {
            id: def.id || 0,
            definition_text: def.definition_text || '', 
            original_pos: def.original_pos || null,
            standardized_pos: def.standardized_pos || null,
            created_at: def.created_at || null, 
            updated_at: def.updated_at || null, 
            part_of_speech: def.standardized_pos || def.part_of_speech || null, 
            examples: Array.isArray(def.examples) ? def.examples : 
                      (typeof def.examples === 'string' ? def.examples.split(';').map((s: string) => s.trim()) : []),
            usage_notes: Array.isArray(def.usage_notes) ? def.usage_notes : splitSemicolonSeparated(def.usage_notes), 
            tags: Array.isArray(def.tags) ? def.tags : splitCommaSeparated(def.tags), 
            sources: Array.isArray(def.sources) ? def.sources : splitCommaSeparated(def.sources)
          };
        });
      } catch (e) {
        console.error("Error normalizing definitions:", e);
      }
    }

    // Normalize Etymologies - with safe handling
    if (wordData.etymologies && Array.isArray(wordData.etymologies)) {
      try {
        normalizedWord.etymologies = wordData.etymologies.map((etym: any): Etymology => {
          if (!etym || typeof etym !== 'object') return { 
            id: 0,
            etymology_text: '', 
            components: [], 
            languages: [], 
            sources: [] 
          };
          
          return {
            id: etym.id || 0,
            etymology_text: etym.etymology_text || '', 
            etymology_structure: etym.etymology_structure || null,
            created_at: etym.created_at || null,
            updated_at: etym.updated_at || null,
            etymology_data: etym.etymology_data || etym.etymology_metadata || {},
            languages: Array.isArray(etym.languages) ? etym.languages : 
                      (etym.language_codes ? splitCommaSeparated(etym.language_codes) : []), 
            components: Array.isArray(etym.components) ? etym.components :
                      (etym.normalized_components ? 
                        (typeof etym.normalized_components === 'string' ? 
                          splitCommaSeparated(etym.normalized_components) : 
                          Array.isArray(safeJsonParse(etym.normalized_components)) ? 
                            safeJsonParse(etym.normalized_components) : []
                        ) : []), 
            sources: Array.isArray(etym.sources) ? etym.sources : splitCommaSeparated(etym.sources)
          };
        });
      } catch (e) {
        console.error("Error normalizing etymologies:", e);
      }
    }

    // Normalize Pronunciations
    if (wordData.pronunciations && Array.isArray(wordData.pronunciations)) {
      normalizedWord.pronunciations = wordData.pronunciations.map((pron: any): Pronunciation => ({
        id: pron.id,
        type: pron.type || '',
        value: pron.value || '',
        sources: Array.isArray(pron.sources) ? pron.sources : 
                (typeof pron.sources === 'string' ? splitCommaSeparated(pron.sources) : []),
        tags: typeof pron.tags === 'object' ? pron.tags : 
             (typeof pron.tags === 'string' ? safeJsonParse(pron.tags) || {} : {}),
        pronunciation_metadata: typeof pron.pronunciation_metadata === 'object' ? pron.pronunciation_metadata : 
                               (typeof pron.metadata === 'object' ? pron.metadata : {}),
        created_at: pron.created_at || null,
        updated_at: pron.updated_at || null,
        // Add extracted fields for convenience
        ipa: pron.type === 'ipa' ? pron.value : undefined,
        audio_url: pron.type === 'audio' ? pron.value : undefined
      }));
    }

    // Normalize Credits
    if (wordData.credits && Array.isArray(wordData.credits)) {
      normalizedWord.credits = wordData.credits.map((cred: any): Credit => ({
        id: cred.id,
        credit: cred.credit || '',
        created_at: cred.created_at || null,
        updated_at: cred.updated_at || null
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
        baybayin_form: wordData.root_word.baybayin_form || null
      };
    }

    // Normalize Derived Words
    if (wordData.derived_words && Array.isArray(wordData.derived_words)) {
      normalizedWord.derived_words = wordData.derived_words.map((dw: any): RelatedWord => ({
        id: dw.id,
        lemma: dw.lemma || '',
        normalized_lemma: dw.normalized_lemma || null,
        language_code: dw.language_code || null,
        has_baybayin: dw.has_baybayin || false,
        baybayin_form: dw.baybayin_form || null
      }));
    }

    // Normalize Relations
    if (wordData.outgoing_relations && Array.isArray(wordData.outgoing_relations)) {
      normalizedWord.outgoing_relations = wordData.outgoing_relations.map((rel: any): Relation => ({
        id: rel.id,
        relation_type: rel.relation_type || '',
        // Handle both metadata and relation_data fields for compatibility
        metadata: rel.metadata || rel.relation_data || rel.relation_metadata || null,
        sources: rel.sources || null,
        target_word: rel.target_word ? { 
          id: rel.target_word.id,
          lemma: rel.target_word.lemma || '',
          normalized_lemma: rel.target_word.normalized_lemma || null,
          language_code: rel.target_word.language_code || null,
          has_baybayin: rel.target_word.has_baybayin || false,
          baybayin_form: rel.target_word.baybayin_form || null
        } : undefined,
        source_word: rel.source_word ? {
          id: rel.source_word.id,
          lemma: rel.source_word.lemma || '',
          normalized_lemma: rel.source_word.normalized_lemma || null,
          language_code: rel.source_word.language_code || null,
          has_baybayin: rel.source_word.has_baybayin || false,
          baybayin_form: rel.source_word.baybayin_form || null
        } : undefined,
        created_at: rel.created_at || null,
        updated_at: rel.updated_at || null
      }));
    }

    if (wordData.incoming_relations && Array.isArray(wordData.incoming_relations)) {
      normalizedWord.incoming_relations = wordData.incoming_relations.map((rel: any): Relation => ({
        id: rel.id,
        relation_type: rel.relation_type || '',
        // Handle both metadata and relation_data fields for compatibility
        metadata: rel.metadata || rel.relation_data || rel.relation_metadata || null,
        sources: rel.sources || null,
        target_word: rel.target_word ? { 
          id: rel.target_word.id,
          lemma: rel.target_word.lemma || '',
          normalized_lemma: rel.target_word.normalized_lemma || null,
          language_code: rel.target_word.language_code || null,
          has_baybayin: rel.target_word.has_baybayin || false,
          baybayin_form: rel.target_word.baybayin_form || null
        } : undefined,
        source_word: rel.source_word ? {
          id: rel.source_word.id,
          lemma: rel.source_word.lemma || '',
          normalized_lemma: rel.source_word.normalized_lemma || null,
          language_code: rel.source_word.language_code || null,
          has_baybayin: rel.source_word.has_baybayin || false,
          baybayin_form: rel.source_word.baybayin_form || null
        } : undefined,
        created_at: rel.created_at || null,
        updated_at: rel.updated_at || null
      }));
    }

    // Normalize Affixations
    if (wordData.root_affixations && Array.isArray(wordData.root_affixations)) {
      normalizedWord.root_affixations = wordData.root_affixations.map((aff: any): Affixation => ({
        id: aff.id,
        affix_type: aff.affix_type || '',
        sources: aff.sources || null,
        // Handle both metadata and affixation_data fields for compatibility
        affixation_data: aff.affixation_data || aff.affixation_metadata || {},
        created_at: aff.created_at || null,
        updated_at: aff.updated_at || null,
        affixed_word: aff.affixed_word ? {
          id: aff.affixed_word.id,
          lemma: aff.affixed_word.lemma || '',
          normalized_lemma: aff.affixed_word.normalized_lemma || null,
          language_code: aff.affixed_word.language_code || null,
          has_baybayin: aff.affixed_word.has_baybayin || false,
          baybayin_form: aff.affixed_word.baybayin_form || null
        } : undefined,
        root_word: aff.root_word ? {
          id: aff.root_word.id,
          lemma: aff.root_word.lemma || '',
          normalized_lemma: aff.root_word.normalized_lemma || null,
          language_code: aff.root_word.language_code || null,
          has_baybayin: aff.root_word.has_baybayin || false,
          baybayin_form: aff.root_word.baybayin_form || null
        } : undefined
      }));
    }

    if (wordData.affixed_affixations && Array.isArray(wordData.affixed_affixations)) {
      normalizedWord.affixed_affixations = wordData.affixed_affixations.map((aff: any): Affixation => ({
        id: aff.id,
        affix_type: aff.affix_type || '',
        sources: aff.sources || null,
        // Handle both metadata and affixation_data fields for compatibility
        affixation_data: aff.affixation_data || aff.affixation_metadata || {},
        created_at: aff.created_at || null,
        updated_at: aff.updated_at || null,
        affixed_word: aff.affixed_word ? {
          id: aff.affixed_word.id,
          lemma: aff.affixed_word.lemma || '',
          normalized_lemma: aff.affixed_word.normalized_lemma || null,
          language_code: aff.affixed_word.language_code || null,
          has_baybayin: aff.affixed_word.has_baybayin || false,
          baybayin_form: aff.affixed_word.baybayin_form || null
        } : undefined,
        root_word: aff.root_word ? {
          id: aff.root_word.id,
          lemma: aff.root_word.lemma || '',
          normalized_lemma: aff.root_word.normalized_lemma || null,
          language_code: aff.root_word.language_code || null,
          has_baybayin: aff.root_word.has_baybayin || false,
          baybayin_form: aff.root_word.baybayin_form || null
        } : undefined
      }));
    }

    return normalizedWord;
  } catch (e) {
    console.error("Error in normalizeWordData:", e);
    throw new Error(`Failed to normalize word data: ${e instanceof Error ? e.message : String(e)}`);
  }
}

// --- Word Details Fetching --- 

export async function fetchWordDetails(word: string): Promise<WordInfo> {
  // Handle potential numeric ID input or word string
  const isId = /^\d+$/.test(word);
  const normalizedWord = isId ? word : word.toLowerCase();
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
    // First try direct fetch with timeout
    const endpoint = `/words/${encodeURIComponent(normalizedWord)}`;
    const fetchUrl = `${API_BASE_URL}${endpoint}`;
    console.log(`Fetching word details with direct fetch from: ${fetchUrl}`);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout
    
    try {
      const response = await fetch(fetchUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Origin': window.location.origin
        },
        mode: 'cors',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        console.log("Direct fetch word lookup successful");
        const wordData = await response.json();
        const normalizedData = normalizeWordData(wordData);
        setCachedData(cacheKey, wordData);
        
        // Reset circuit breaker on success
        if (circuitBreaker) circuitBreaker.recordSuccess();
        
        return normalizedData;
      } else {
        throw new Error(`API returned status ${response.status}: ${response.statusText}`);
      }
    } catch (directFetchError: unknown) {
      clearTimeout(timeoutId);
      
      // Check if this was a timeout
      if (directFetchError instanceof Error && directFetchError.name === 'AbortError') {
        console.warn(`Direct fetch timed out for word details: ${normalizedWord}`);
      } else {
        console.warn(`Direct fetch lookup failed, error:`, directFetchError);
      }
      
      console.log(`Falling back to axios for word details`);
      
      // Fallback to axios with extended timeout
      const response = await api.get(endpoint, { timeout: 20000 });
      
      if (response.status === 200 && response.data) {
        console.log("Axios word lookup successful");
        const normalizedData = normalizeWordData(response.data);
        setCachedData(cacheKey, response.data);
        
        // Reset circuit breaker on success
        if (circuitBreaker) circuitBreaker.recordSuccess();
        
        return normalizedData;
      }
    }
    
    // If direct lookup failed, try search endpoint
    console.log(`Direct lookups failed, trying search endpoint for: ${normalizedWord}`);
    
    // Try direct fetch for search with timeout
    const searchController = new AbortController();
    const searchTimeoutId = setTimeout(() => searchController.abort(), 10000); // 10 second timeout
    
    try {
      const searchUrl = `${API_BASE_URL}/search?q=${encodeURIComponent(normalizedWord)}&limit=5&include_full=true`;
      console.log(`Searching with direct fetch from: ${searchUrl}`);
      
      const searchResponse = await fetch(searchUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Origin': window.location.origin
        },
        mode: 'cors',
        signal: searchController.signal
      });
      
      clearTimeout(searchTimeoutId);
      
      if (searchResponse.ok) {
        const searchData = await searchResponse.json();
        
        if (searchData.count > 0 && 
            searchData.results && 
            Array.isArray(searchData.results) && 
            searchData.results.length > 0) {
          
          const firstResult = searchData.results[0];
          console.log(`Search found match: ${firstResult.lemma} (ID: ${firstResult.id})`);
          
          // If the result has complete data, use it directly
          if (firstResult.definitions || firstResult.etymologies) {
            console.log(`Found complete word data in search result, normalizing`);
            const normalizedData = normalizeWordData(firstResult);
            setCachedData(cacheKey, firstResult);
            
            // Reset circuit breaker on success
            if (circuitBreaker) circuitBreaker.recordSuccess();
            
            return normalizedData;
          }
          
          // Otherwise fetch full details
          const detailController = new AbortController();
          const detailTimeoutId = setTimeout(() => detailController.abort(), 10000); // 10 second timeout
          
          try {
            const detailUrl = `${API_BASE_URL}/words/${firstResult.id}`;
            console.log(`Fetching details for search result from: ${detailUrl}`);
            
            const detailResponse = await fetch(detailUrl, {
              method: 'GET',
              headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Origin': window.location.origin
              },
              mode: 'cors',
              signal: detailController.signal
            });
            
            clearTimeout(detailTimeoutId);
            
            if (detailResponse.ok) {
              const detailData = await detailResponse.json();
              console.log(`Successfully fetched details for: ${firstResult.lemma}`);
              const normalizedData = normalizeWordData(detailData);
              setCachedData(cacheKey, detailData);
              
              // Reset circuit breaker on success
              if (circuitBreaker) circuitBreaker.recordSuccess();
              
              return normalizedData;
            }
          } catch (detailFetchError: unknown) {
            clearTimeout(detailTimeoutId);
            console.warn(`Detail fetch failed:`, detailFetchError);
          }
        }
      }
    } catch (searchFetchError: unknown) {
      clearTimeout(searchTimeoutId);
      
      // Check if this was a timeout
      if (searchFetchError instanceof Error && searchFetchError.name === 'AbortError') {
        console.warn(`Search fetch timed out for: ${normalizedWord}`);
      } else {
        console.warn(`Direct fetch search failed:`, searchFetchError);
      }
    }
    
    // Last resort: Try axios search if all direct fetch methods failed
    try {
      console.log(`Last resort: Trying axios search for: ${normalizedWord}`);
      const searchResponse = await api.get('/search', { 
        params: { 
          q: normalizedWord, 
          limit: 5, 
          include_full: false 
        },
        timeout: 20000 // Extended timeout
      });
      
      if (searchResponse.status === 200 && searchResponse.data) {
        const searchData = searchResponse.data;
        
        if (searchData.count > 0 && 
            searchData.results && 
            Array.isArray(searchData.results) && 
            searchData.results.length > 0) {
          
          const firstResult = searchData.results[0];
          console.log(`Axios search found match: ${firstResult.lemma} (ID: ${firstResult.id})`);
          
          // Now fetch full details for this word
          const detailResponse = await api.get(`/words/${firstResult.id}`, { timeout: 20000 });
          if (detailResponse.status === 200 && detailResponse.data) {
            console.log(`Successfully fetched details for: ${firstResult.lemma}`);
            const normalizedData = normalizeWordData(detailResponse.data);
            setCachedData(cacheKey, detailResponse.data);
            
            // Reset circuit breaker on success
            if (circuitBreaker) circuitBreaker.recordSuccess();
            
            return normalizedData;
          }
        }
      }
    } catch (axiosSearchError: unknown) {
      console.error(`Axios search failed:`, axiosSearchError);
    }
    
    // If we reach here, all methods failed
    throw new Error(`Failed to find word "${normalizedWord}" using multiple lookup methods`);

  } catch (error: unknown) {
    console.error(`Error fetching word details for '${normalizedWord}':`, error);
    
    if (isApiError(error) && error.response?.status === 404) {
      throw new Error(`Word "${normalizedWord}" not found in the dictionary`);
    }
    
    throw new Error(`Error searching for "${normalizedWord}": ${error instanceof Error ? error.message : 'Unknown error'}`);
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
      results: [],
      count: 0,
      query: '',
      page: searchOptions.page || 1,
      perPage: searchOptions.per_page || 20
    };
  }
  
  const cacheKey = `search:${normalizedQuery}:${JSON.stringify(searchOptions)}`;
  const cachedData = getCachedData<SearchResult>(cacheKey);
  if (cachedData) {
    console.log(`Cache hit for search: ${normalizedQuery}`);
    return cachedData;
  }
  
  console.log(`Cache miss for search: ${normalizedQuery}. Fetching from API...`);
  
  if (!bypassCircuitBreaker && !circuitBreaker.canMakeRequest()) {
    throw new Error("Circuit breaker is open. Try again later.");
  }
  
  try {
    // Convert SearchOptions into query parameters, including all valid options
    const apiOptions: Record<string, any> = {
      q: normalizedQuery,
      limit: searchOptions.limit || 20,
      offset: searchOptions.offset || 0,
      mode: searchOptions.mode || 'all',
      include_full: searchOptions.include_full || false,
      include_metadata: true,
    };
    
    // Add additional parameters if they exist in options
    if (searchOptions.language) apiOptions.language = searchOptions.language;
    if (searchOptions.pos) apiOptions.pos = searchOptions.pos;
    
    console.log('Making search API request with options:', apiOptions);
    
    // Try direct fetch with timeout handling
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
    
    try {
      // Direct fetch using the API_BASE_URL
      const queryParams = new URLSearchParams();
      Object.entries(apiOptions).forEach(([key, value]) => {
        if (value !== undefined) queryParams.append(key, String(value));
      });
      
      const fetchUrl = `${API_BASE_URL}/search?${queryParams.toString()}`;
      console.log(`Direct fetch URL: ${fetchUrl}`);
      
      const fetchResponse = await fetch(fetchUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Origin': window.location.origin
        },
        mode: 'cors',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!fetchResponse.ok) {
        throw new Error(`Search API returned status ${fetchResponse.status}: ${fetchResponse.statusText}`);
      }
      
      const responseData = await fetchResponse.json();
      console.log('Direct fetch search response:', responseData);
      
      // Create a clean SearchResult object from direct fetch
      const result: SearchResult = {
        query: normalizedQuery,
        count: responseData.count || 0,
        total: responseData.count || 0,
        offset: apiOptions.offset,
        limit: apiOptions.limit,
        page: Math.floor(apiOptions.offset / apiOptions.limit) + 1,
        perPage: apiOptions.limit,
        mode: apiOptions.mode,
        language: apiOptions.language,
        results: responseData.results || [],
        words: [] // Initialize empty words array
      };
      
      // Map API results to our frontend format
      if (responseData.results && responseData.results.length > 0) {
        result.words = responseData.results.map((item: any) => {
          return {
            id: item.id,
            lemma: item.lemma,
            normalized_lemma: item.normalized_lemma,
            language_code: item.language_code,
            has_baybayin: item.has_baybayin,
            baybayin_form: item.baybayin_form
          };
        });
      }
      
      setCachedData(cacheKey, result);
      
      // Reset circuit breaker on success
      if (circuitBreaker) circuitBreaker.recordSuccess();
      
      return result;
      
    } catch (fetchError: unknown) {
      clearTimeout(timeoutId);
      
      // Check if this was a timeout
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        console.warn('Direct fetch search timed out, falling back to axios');
      } else {
        console.warn('Direct fetch search failed, falling back to axios:', fetchError);
      }
      
      // Fall back to axios with a longer timeout
      const response = await api.get('/search', { 
        params: apiOptions,
        timeout: 20000 // Extended timeout for fallback
      });
      
      console.log('Search API raw response:', response.data);
      
      // Transform API response into our frontend SearchResult format
      const responseData = response.data;
      
      // Create a clean SearchResult object
      const result: SearchResult = {
        query: normalizedQuery,
        count: responseData.count || 0,
        total: responseData.count || 0,
        offset: apiOptions.offset,
        limit: apiOptions.limit,
        page: Math.floor(apiOptions.offset / apiOptions.limit) + 1,
        perPage: apiOptions.limit,
        mode: apiOptions.mode,
        language: apiOptions.language,
        results: responseData.results || [],
        words: [] // Initialize empty words array
      };
      
      // Map API results to our frontend format if they exist
      if (responseData.results && responseData.results.length > 0) {
        console.log(`Found ${responseData.results.length} results in API response`);
        
        // For include_full=true, the results contain full word objects
        if (apiOptions.include_full) {
          try {
            result.words = responseData.results.map((item: any) => {
              return normalizeWordData(item);
            });
            console.log(`Normalized ${result.words?.length || 0} full word results`);
          } catch (normalizeError) {
            console.warn('Error normalizing search results:', normalizeError);
            // Fallback to basic word info
            result.words = responseData.results.map((item: any) => {
              return {
                id: item.id,
                lemma: item.lemma,
                normalized_lemma: item.normalized_lemma,
                language_code: item.language_code,
                has_baybayin: item.has_baybayin,
                baybayin_form: item.baybayin_form
              };
            });
            console.log(`Created ${result.words?.length || 0} basic word results after normalization error`);
          }
        } 
        // For include_full=false, the results contain basic word info
        else {
          result.words = responseData.results.map((item: any) => {
            return {
              id: item.id,
              lemma: item.lemma,
              normalized_lemma: item.normalized_lemma,
              language_code: item.language_code,
              has_baybayin: item.has_baybayin,
              baybayin_form: item.baybayin_form
            };
          });
          console.log(`Created ${result.words?.length || 0} basic word results`);
        }
      }
      // If no results in the main property but we have words array, use it 
      else if (responseData.words && responseData.words.length > 0) {
        console.log(`Using ${responseData.words.length} items from words array instead of results`);
        result.results = responseData.words;
        result.words = responseData.words;
      }
      else {
        console.log(`API returned ${responseData.count || 0} count but no results or words array available`);
      }
      
      // Store in cache if we got valid results
      try {
        if ((result.count && result.count > 0) || 
            (result.results && result.results.length > 0) || 
            (result.words && result.words.length > 0)) {
          setCachedData(cacheKey, result);
        }
      } catch (e) {
        console.warn('Invalid data type for caching', e);
      }
      
      // Reset circuit breaker on success
      if (circuitBreaker) circuitBreaker.recordSuccess();
      
      return result;
    }
  } catch (error) {
    console.error(`Search error for "${normalizedQuery}":`, error);
    
    // Record circuit breaker failure
    if (circuitBreaker && !bypassCircuitBreaker) {
      circuitBreaker.recordFailure();
    }
    
    // Throw a more detailed error to help diagnose connectivity issues
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNABORTED') {
        throw new Error(`Search request timed out. Check network connection and backend status.`);
      } else if (!error.response) {
        throw new Error(`No response received from server in search. Check network connection and backend status.`);
      } else {
        const status = error.response.status;
        const message = error.response.data?.error || error.message;
        throw new Error(`Server returned ${status} error during search: ${message}`);
      }
    } else if (error instanceof Error) {
      throw new Error(`Search failed: ${error.message}`);
    } else {
      throw new Error(`Search failed with unknown error`);
    }
  }
}

// Direct search function that bypasses circuit breaker and caching completely
export async function directSearch(
  queryOrOptions: string | SearchOptions, 
  _options?: SearchOptions, // Mark as unused
  _bypassCircuitBreaker: boolean = false // Mark as unused
): Promise<SearchResult> {
  // Extract query string regardless of input type
  const query = typeof queryOrOptions === 'string' ? queryOrOptions : queryOrOptions.q || '';
  if (!query) {
    throw new Error('Direct search requires a non-empty query string.');
  }

  console.log(`Performing direct search for "${query}" (bypassing circuit breaker/cache)`);
  
  try {
    // Direct call without circuit breaker checks
    const response = await api.get('/search', { 
      params: { q: query, limit: 10 }, // Keep params simple
      timeout: 20000, // Extended timeout for direct search
    });
    
    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }
    
    console.log('Direct search response:', response.data);
    
    const data = response.data;
    
    // Basic search result structure
    const searchResult: SearchResult = {
      words: [],
      page: 1,
      perPage: 10,
      total: data.count || 0,
      query
    };
    
    // Map results - try different formats the API might return
    if (Array.isArray(data.results)) {
      searchResult.words = data.results.map((result: any) => ({
        id: result.id,
        lemma: result.lemma || `Unknown-${result.id}`,
        normalized_lemma: result.normalized_lemma || result.lemma || '',
        language_code: result.language_code || 'tl',
        has_baybayin: result.has_baybayin || false,
        baybayin_form: result.baybayin_form || null,
        romanized_form: result.romanized_form || null,
        definitions: []
      }));
      
      // Reset the circuit breaker since we got a good response
      circuitBreaker.reset();
    }
    
    return searchResult;
  } catch (error) {
    console.error(`Direct search error for "${query}":`, error);
    throw new Error(`Direct search failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

// --- Other Utility API Functions --- 

export async function getPartsOfSpeech(): Promise<PartOfSpeech[]> {
  const cacheKey = 'cache:parts_of_speech';
    const cachedData = getCachedData<PartOfSpeech[]>(cacheKey);
  if (cachedData) return cachedData;

  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");

  try {
    const response = await api.get('/parts_of_speech'); 
     if (response.status !== 200) throw new Error(`API status ${response.status}`);
    
    // The backend returns an array of part of speech objects
    // But we need to ensure they match our PartOfSpeech interface
    const data = response.data || [];
    
    if (!Array.isArray(data)) throw new Error('Invalid data format for parts of speech');
    
    // Transform if needed to match our interface
    const partsOfSpeech = data.map(pos => ({
      id: pos.id,
      code: pos.code,
      name_en: pos.name_en || pos.name,
      name_tl: pos.name_tl || pos.name,
      description: pos.description
    }));
    
    setCachedData(cacheKey, partsOfSpeech);
    return partsOfSpeech;
  } catch (error) {
    await handleApiError(error, 'fetching parts of speech');
    throw new Error('An unknown error occurred after handling API error.');
  }
}

export async function testApiConnection(): Promise<boolean> {
  try {
    // First try the test endpoint directly with fetch
    console.log("[API] Testing API connection via fetch");
    const testUrl = `${API_BASE_URL}/test`;
    
    try {
      // Try the test endpoint first
      console.log(`[API] Testing connection to ${testUrl}`);
      const response = await fetch(testUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Origin': window.location.origin
        },
        mode: 'cors'
      });
      
      if (response.ok) {
        console.log("[API] Direct fetch to /test successful");
        // Reset circuit breaker on success
        circuitBreaker?.recordSuccess();
        return true;
      }
      
      console.log(`[API] Fetch /test failed with status: ${response.status}`);
    } catch (fetchError) {
      console.warn("[API] Direct fetch /test failed:", fetchError);
    }
    
    // If test endpoint fails, try health check
    const healthUrl = `${API_BASE_URL.replace('/api/v2', '')}/health`;
    console.log(`[API] Testing health endpoint: ${healthUrl}`);
    
    try {
      const response = await fetch(healthUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Origin': window.location.origin
        },
        mode: 'cors'
      });
      
      if (response.ok) {
        console.log("[API] Health check successful");
        // Reset circuit breaker on success
        circuitBreaker?.recordSuccess();
        return true;
      }
      
      console.log(`[API] Health check failed with status: ${response.status}`);
    } catch (healthError) {
      console.warn("[API] Health check failed:", healthError);
    }
    
    // If direct tests fail, use axios fallback
    console.log("[API] Trying fallback with axios");
    try {
      const axiosResponse = await api.get('/test', { timeout: 5000 });
      if (axiosResponse.status === 200) {
        console.log("[API] Axios test successful");
        circuitBreaker?.recordSuccess();
        return true;
      }
    } catch (axiosError) {
      console.warn("[API] Axios test failed:", axiosError);
      
      // Try the search endpoint as last resort
      try {
        const searchResponse = await api.get('/search', { 
          params: { q: 'test', limit: 1 },
          timeout: 5000
        });
        
        if (searchResponse.status === 200) {
          console.log("[API] Search test successful");
          circuitBreaker?.recordSuccess();
          return true;
        }
      } catch (searchError) {
        console.error("[API] Search test failed:", searchError);
      }
    }
    
    console.log("[API] All connectivity tests failed");
    circuitBreaker?.recordFailure();
    return false;
  } catch (error) {
    console.error("API connection test error:", error);
    circuitBreaker?.recordFailure();
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
    const response = await api.get(endpoint, { params: { depth: maxDepth } });
     if (response.status !== 200) throw new Error(`API status ${response.status}`);
    
    // The API returns the tree structure directly
    const responseData = response.data;
    
    if (!responseData) {
      throw new Error('Invalid etymology tree data - empty response');
    }
    
    // Create a minimal tree structure with the response data
    const etymologyTree: EtymologyTree = {
      word: responseData.lemma || '',
      etymology_tree: responseData,
      complete: true,
      metadata: {
        word: responseData.lemma || '',
        max_depth: maxDepth
      }
    };
    
    setCachedData(cacheKey, etymologyTree);
    return etymologyTree;
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
    
    // Get raw data from response
    const rawStats = response.data || {};
    
    if (!rawStats || typeof rawStats !== 'object') throw new Error('Invalid statistics data');
    
    // Transform the data to match our Statistics interface
    const statsData: Statistics = {
      // Map fields from the API response to our interface
      total_words: rawStats.total_words || 0,
      total_definitions: rawStats.total_definitions || 0,
      total_etymologies: rawStats.total_etymologies || 0,
      total_relations: rawStats.total_relations || 0,
      total_affixations: rawStats.total_affixations || 0,
      words_with_examples: rawStats.words_with_examples || 0,
      words_with_etymology: rawStats.words_with_etymology || 0,
      words_with_relations: rawStats.words_with_relations || 0,
      words_with_baybayin: rawStats.words_with_baybayin || 0,
      // Map languages and parts_of_speech to words_by_language and words_by_pos
      words_by_language: rawStats.languages || {},
      words_by_pos: rawStats.parts_of_speech || {},
      // Add timestamp
      timestamp: rawStats.timestamp || new Date().toISOString()
    };
    
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