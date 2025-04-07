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
    // Any successful 2xx response indicates the API is working.
    if (response.status >= 200 && response.status < 300) {
      // Reset the circuit breaker to closed state on ANY success.
      console.log("Successful API response received. Explicitly resetting circuit breaker.");
      circuitBreaker.reset(); 
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
    const endpoint = `/words/${encodeURIComponent(sanitizedWord)}/semantic_network`;
    const params: Record<string, any> = {
      depth: sanitizedDepth
    };
    
    // Add relation types param if needed
    if (!include_affixes) {
      params.relation_types = 'synonym,antonym,hypernym,hyponym,related';
    }
    
    console.log(`Fetching semantic network from ${endpoint} with params:`, params);
    const response = await api.get(endpoint, { params });
    
    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }
    
    // Process the API response
    const networkData = response.data;
    
    // The API might return different formats, we need to normalize
    const formattedResponse: WordNetworkResponse = {
      nodes: [],
      edges: [],
      stats: {
        node_count: 0,
        edge_count: 0,
        depth: sanitizedDepth
      }
    };
    
    // Handle the current API response format
    if (networkData) {
      // Extract nodes from the response
      if (Array.isArray(networkData.nodes)) {
        formattedResponse.nodes = networkData.nodes.map((node: any) => ({
          id: node.id,
          lemma: node.lemma || '',
          language_code: node.language_code || 'tl'
        }));
      } else if (typeof networkData.nodes === 'object') {
        // Handle nodes as an object with IDs as keys
        formattedResponse.nodes = Object.values(networkData.nodes).map((node: any) => ({
          id: node.id,
          lemma: node.lemma || '',
          language_code: node.language_code || 'tl'
        }));
      }
      
      // Extract edges from the response
      if (Array.isArray(networkData.edges)) {
        formattedResponse.edges = networkData.edges.map((edge: any) => ({
          id: edge.id || `${edge.source}-${edge.type}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          type: edge.type,
          metadata: edge.metadata || null
        }));
      } else if (typeof networkData.edges === 'object') {
        // Handle edges as an object with IDs as keys
        formattedResponse.edges = Object.values(networkData.edges).map((edge: any) => ({
          id: edge.id || `${edge.source}-${edge.type}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          type: edge.type,
          metadata: edge.metadata || null
        }));
      }
      
      // Update stats
      formattedResponse.stats.node_count = formattedResponse.nodes.length;
      formattedResponse.stats.edge_count = formattedResponse.edges.length;
    }

    setCachedData(cacheKey, formattedResponse);
    return formattedResponse;

  } catch (error: unknown) {
    await handleApiError(error, `fetching semantic network for '${sanitizedWord}'`);
    throw new Error('An unknown error occurred after handling API error.');
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
  // Handle different possible structures
  const wordData = rawData?.data || rawData;

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

  // Create base WordInfo object
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
    tags: wordData.tags || null,
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
    forms: wordData.forms || [],
    templates: wordData.templates || [],
    data_completeness: wordData.data_completeness || null,
    relation_summary: wordData.relation_summary || null,
    completeness_score: wordData.completeness_score || 
                        (wordData.data_completeness && 
                         typeof wordData.data_completeness.completeness_score === 'number' ? 
                         wordData.data_completeness.completeness_score : null)
  };

  // Normalize Definitions
  if (wordData.definitions && Array.isArray(wordData.definitions)) {
    normalizedWord.definitions = wordData.definitions.map((def: any): Definition => ({
      id: def.id,
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
    }));
  }

  // Normalize Etymologies
  if (wordData.etymologies && Array.isArray(wordData.etymologies)) {
    normalizedWord.etymologies = wordData.etymologies.map((etym: any): Etymology => ({
      id: etym.id,
      etymology_text: etym.etymology_text || '', 
      etymology_structure: etym.etymology_structure || null,
      created_at: etym.created_at || null,
      updated_at: etym.updated_at || null,
      // Use etymology_data property for compatibility with backend changes
      etymology_data: etym.etymology_data || etym.etymology_metadata || {},
      languages: Array.isArray(etym.languages) ? etym.languages : 
               (etym.language_codes ? splitCommaSeparated(etym.language_codes) : []), 
      components: Array.isArray(etym.components) ? etym.components :
                (etym.normalized_components ? 
                  (typeof etym.normalized_components === 'string' ? 
                    splitCommaSeparated(etym.normalized_components) : 
                    // Handle if it's already a JSON object
                    Array.isArray(safeJsonParse(etym.normalized_components)) ? 
                      safeJsonParse(etym.normalized_components) : []
                  ) : []), 
      sources: Array.isArray(etym.sources) ? etym.sources : splitCommaSeparated(etym.sources)
    }));
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
    // Construct the correct endpoint URL based on whether we have an ID or word
    const endpoint = `/words/${encodeURIComponent(normalizedWord)}`;
    console.log(`Fetching word details from: ${endpoint}`);
    
    const response = await api.get(endpoint);

    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }

    // Process the API response
    if (!response.data) {
      throw new Error("API returned empty data");
    }

    try {
      // NOTE: Success is recorded by the interceptor
      const normalizedData = normalizeWordData(response.data);
      setCachedData(cacheKey, response.data); // Cache the raw data
      return normalizedData;
    } catch (normalizationError) {
      console.error(`Error normalizing data for word ${normalizedWord}:`, normalizationError);
      console.warn('Returning minimal word info due to normalization error');
      
      // Create a minimal word object from the raw data to ensure we return something useful
      const minimalWord: WordInfo = {
        id: response.data.id || parseInt(normalizedWord) || 0,
        lemma: response.data.lemma || normalizedWord,
        language_code: response.data.language_code || 'tl',
        normalized_lemma: response.data.normalized_lemma || response.data.lemma || normalizedWord,
        // Initialize empty arrays for nested data
        definitions: [],
        etymologies: [],
        pronunciations: []
      };
      
      // Safely extract basic definition data if available
      if (response.data.definitions && Array.isArray(response.data.definitions)) {
        try {
          minimalWord.definitions = response.data.definitions.map((def: any) => ({
            id: def.id || 0,
            definition_text: def.definition_text || '',
            examples: [],
            usage_notes: [],
            tags: [],
            sources: [],
            part_of_speech: null
          }));
        } catch (e) {
          console.warn('Error extracting definitions:', e);
        }
      }
      
      return minimalWord;
    }

  } catch (error: unknown) {
    // NOTE: Failure is recorded by the interceptor
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
    const response = await api.get('/search', { params: apiOptions });
    
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
      // For include_full=true, the results contain full word objects
      if (apiOptions.include_full) {
        result.words = responseData.results.map((item: any) => {
          return normalizeWordData(item);
        });
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
      }
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
    
    return result;
  } catch (error) {
    // Only count search errors for circuit breaker
    handleApiError(error, 'search');
    
    // Return empty result on error
    return {
      query: normalizedQuery,
      count: 0,
      total: 0,
      results: [],
      words: []
    };
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