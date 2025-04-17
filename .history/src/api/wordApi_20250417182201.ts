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
      'X-Client-Platform': 'web'
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
export interface NetworkOptions {
  depth?: number;
  breadth?: number;
  include_affixes?: boolean;
  include_etymology?: boolean;
  cluster_threshold?: number;
}

// Type for the actual network data structure expected from API
// (Update this based on the actual backend response for semantic_network)
export interface ApiWordNetwork {
    nodes: any[]; // Define more specific node type if possible
    links: any[]; // Define more specific link type if possible
    metadata: { // Ensure metadata matches backend structure
        root_word: string;
        normalized_lemma?: string;
        language_code?: string;
        depth?: number;
        total_nodes?: number;
        total_edges?: number;
        execution_time?: number;
        [key: string]: any; // Allow other metadata fields
    };
}

// Define a minimal valid fallback structure
const minimalApiWordNetwork = (rootWord: string): ApiWordNetwork => ({
    nodes: [],
    links: [],
    metadata: {
        root_word: rootWord,
        total_nodes: 0,
        total_edges: 0
    }
});

export async function fetchWordNetwork(word: string, options: NetworkOptions = {}, signal?: AbortSignal): Promise<ApiWordNetwork> {
  // Removed caching for ApiWordNetwork for now due to type constraints
  // const cacheKey = `network:${word}:${JSON.stringify(options)}`;
  // const cached = await getCachedData<ApiWordNetwork>(cacheKey);
  // if (cached) {
  //   if (cached.nodes && cached.links && cached.metadata) {
  //       return cached;
  //   } else {
  //       console.warn(`Invalid cached data structure for ${cacheKey}. Refetching.`);
  //   }
  // }

  try {
    const params = { ...options };
    // Pass signal correctly within the config object
    const config: AxiosRequestConfig = { params };
    if (signal) {
      config.signal = signal;
    }
    const response = await api.get<ApiWordNetwork>(`/words/${sanitizeInput(word)}/semantic_network`, config); 

    // Check response data structure before returning
    if (response.data && response.data.nodes && response.data.links && response.data.metadata) {
      // Removed caching call: await setCachedData(cacheKey, response.data);
      return response.data;
    } else {
      console.error(`Invalid data structure received from /semantic_network endpoint for word "${word}"`);
      return minimalApiWordNetwork(word); 
    }
  } catch (error) {
    // Check for cancel error first
    if (axios.isCancel(error)) {
      console.log('Request canceled:', error.message);
      return minimalApiWordNetwork(word); 
    }
    // Handle unknown error type
    let errorMessage = `Error fetching word network for "${word}"`;
    if (error instanceof Error) {
      errorMessage += `: ${error.message}`;
    }
    console.error(errorMessage, error); 
    return minimalApiWordNetwork(word); 
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
  console.log("Starting normalizeWordData with raw data:", {
    id: rawData?.id || (rawData?.data?.id),
    lemma: rawData?.lemma || (rawData?.data?.lemma),
    hasIncoming: Boolean(rawData?.incoming_relations || rawData?.data?.incoming_relations),
    hasOutgoing: Boolean(rawData?.outgoing_relations || rawData?.data?.outgoing_relations),
  });

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

  console.log("Normalizing word data:", JSON.stringify(wordData.id), wordData.lemma);

  // Create base normalized word structure
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

  // Normalize Definitions - Revert to ': any'
  if (wordData.definitions && Array.isArray(wordData.definitions)) {
    console.log(`Processing ${wordData.definitions.length} definitions`);
    normalizedWord.definitions = wordData.definitions.map((def: any): Definition => ({ // Changed back to any
      // Raw fields needed by Omit base (RawDefinition)
      id: def.id,
      definition_text: def.definition_text || '', 
      original_pos: def.original_pos || null,
      standardized_pos: def.standardized_pos || null,
      created_at: def.created_at || null, // Reverted to null
      updated_at: def.updated_at || null, // Reverted to null
      // Fields required by the cleaned Definition type
      text: def.definition_text || '', 
      part_of_speech: def.standardized_pos || null, 
      examples: splitSemicolonSeparated(def.examples), 
      usage_notes: splitSemicolonSeparated(def.usage_notes), 
      tags: splitCommaSeparated(def.tags), 
      sources: splitCommaSeparated(def.sources), 
      relations: [],
      // Other optional fields from RawDefinition if needed
      confidence_score: def.confidence_score,
      is_verified: def.is_verified,
      verified_by: def.verified_by,
      verified_at: def.verified_at,
    }));
  } else {
    console.warn("No definitions found in word data");
  }

  // Normalize Etymologies - Revert to ': any' and add cast
  if (wordData.etymologies && Array.isArray(wordData.etymologies)) {
    console.log(`Processing ${wordData.etymologies.length} etymologies`);
    normalizedWord.etymologies = wordData.etymologies.map((etym: any): Etymology => ({ // Changed back to any
      // Raw fields needed by Omit base (RawEtymology)
      id: etym.id,
      etymology_text: etym.etymology_text || '', 
      etymology_structure: etym.etymology_structure || null,
      created_at: etym.created_at || null, // Reverted to null
      updated_at: etym.updated_at || null, // Reverted to null
      // Fields required by the cleaned Etymology type
      text: etym.etymology_text || '', 
      languages: splitCommaSeparated(etym.language_codes), 
      // Explicit cast to silence persistent linter error
      components: splitCommaSeparated(etym.components as string | undefined), 
      sources: splitCommaSeparated(etym.sources), 
      // Other optional fields from RawEtymology if needed
      confidence_level: etym.confidence_level,
      verification_status: etym.verification_status,
      verification_notes: etym.verification_notes,
    }));
  } else {
    console.warn("No etymologies found in word data");
  }

  // Normalize Pronunciations - Revert to ': any'
  if (wordData.pronunciations && Array.isArray(wordData.pronunciations)) {
    console.log(`Processing ${wordData.pronunciations.length} pronunciations`);
    normalizedWord.pronunciations = wordData.pronunciations.map((pron: any): Pronunciation => ({ // Changed back to any
      id: pron.id,
      type: pron.type || '',
      value: pron.value || '',
      tags: pron.tags || null, 
      sources: pron.sources || null, 
      created_at: pron.created_at || null, // Reverted to null
      updated_at: pron.updated_at || null, // Reverted to null
    }));
  } else {
    console.warn("No pronunciations found in word data");
  }

  // Normalize Credits - Revert to ': any'
  if (wordData.credits && Array.isArray(wordData.credits)) {
    console.log(`Processing ${wordData.credits.length} credits`);
    normalizedWord.credits = wordData.credits.map((cred: any): Credit => ({ // Changed back to any
      id: cred.id,
      credit: cred.credit || '',
      created_at: cred.created_at || null, // Reverted to null
      updated_at: cred.updated_at || null, // Reverted to null
    }));
  } else {
    console.warn("No credits found in word data");
  }

  // Normalize Root Word
  if (wordData.root_word && typeof wordData.root_word === 'object') {
    console.log("Processing root word:", wordData.root_word.lemma);
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
    console.log(`Processing ${wordData.derived_words.length} derived words`);
    normalizedWord.derived_words = wordData.derived_words.map((dw: any): RelatedWord => ({ // Changed back to any
      id: dw.id,
      lemma: dw.lemma || '',
      normalized_lemma: dw.normalized_lemma || null,
      language_code: dw.language_code || null,
      has_baybayin: dw.has_baybayin || false,
      baybayin_form: dw.baybayin_form || null,
    }));
  } else {
    console.warn("No derived words found in word data");
  }

  // Normalize Outgoing Relations
  if (wordData.outgoing_relations && Array.isArray(wordData.outgoing_relations)) {
    console.log(`Processing ${wordData.outgoing_relations.length} outgoing relations`);
    normalizedWord.outgoing_relations = wordData.outgoing_relations.map((rel: any): Relation => {
      // Ensure the relation has a valid target_word
      if (!rel.target_word || !rel.target_word.lemma) {
        console.warn("Outgoing relation missing target_word:", rel);
        // Create a minimal target_word if possible
        let fixedTargetWord = null;
        if (typeof rel === 'object' && rel !== null) {
          // Try to extract from different possible fields
          const candidateFields = ['target', 'to', 'to_word', 'word'];
          for (const field of candidateFields) {
            if (rel[field]) {
              if (typeof rel[field] === 'string') {
                fixedTargetWord = { id: 0, lemma: rel[field] };
                break;
              } else if (typeof rel[field] === 'object' && rel[field].lemma) {
                fixedTargetWord = { 
                  id: rel[field].id || 0, 
                  lemma: rel[field].lemma 
                };
                break;
              }
            }
          }
        }
        
        // If we still couldn't create a target word, use a placeholder
        if (!fixedTargetWord) {
          fixedTargetWord = { id: 0, lemma: 'Unknown' };
        }
        
        return {
          id: rel.id || 0,
          relation_type: rel.relation_type || 'related',
          target_word: fixedTargetWord
        };
      }
      
      // Regular case - relation has target_word
      return {
        id: rel.id || 0,
        relation_type: rel.relation_type || 'related',
        target_word: {
          id: rel.target_word.id || 0,
          lemma: rel.target_word.lemma,
          has_baybayin: rel.target_word.has_baybayin,
          baybayin_form: rel.target_word.baybayin_form
        }
      };
    });
  } else {
    console.warn("No outgoing relations found in word data");
    normalizedWord.outgoing_relations = [];
  }

  // Normalize Incoming Relations
  if (wordData.incoming_relations && Array.isArray(wordData.incoming_relations)) {
    console.log(`Processing ${wordData.incoming_relations.length} incoming relations`);
    normalizedWord.incoming_relations = wordData.incoming_relations.map((rel: any): Relation => {
      // Ensure the relation has a valid source_word
      if (!rel.source_word || !rel.source_word.lemma) {
        console.warn("Incoming relation missing source_word:", rel);
        // Create a minimal source_word if possible
        let fixedSourceWord = null;
        if (typeof rel === 'object' && rel !== null) {
          // Try to extract from different possible fields
          const candidateFields = ['source', 'from', 'from_word', 'word'];
          for (const field of candidateFields) {
            if (rel[field]) {
              if (typeof rel[field] === 'string') {
                fixedSourceWord = { id: 0, lemma: rel[field] };
                break;
              } else if (typeof rel[field] === 'object' && rel[field].lemma) {
                fixedSourceWord = { 
                  id: rel[field].id || 0, 
                  lemma: rel[field].lemma 
                };
                break;
              }
            }
          }
        }
        
        // If we still couldn't create a source word, use a placeholder
        if (!fixedSourceWord) {
          fixedSourceWord = { id: 0, lemma: 'Unknown' };
        }
        
        return {
          id: rel.id || 0,
          relation_type: rel.relation_type || 'related',
          source_word: fixedSourceWord
        };
      }
      
      // Regular case - relation has source_word
      return {
        id: rel.id || 0,
        relation_type: rel.relation_type || 'related',
        source_word: {
          id: rel.source_word.id || 0,
          lemma: rel.source_word.lemma,
          has_baybayin: rel.source_word.has_baybayin,
          baybayin_form: rel.source_word.baybayin_form
        }
      };
    });
  } else {
    console.warn("No incoming relations found in word data");
    normalizedWord.incoming_relations = [];
  }

  // Normalize Root Affixations - Revert to ': any'
  if (wordData.root_affixations && Array.isArray(wordData.root_affixations)) {
    console.log(`Processing ${wordData.root_affixations.length} root affixations`);
    normalizedWord.root_affixations = wordData.root_affixations.map((aff: any): Affixation => ({ // Changed back to any
        ...aff,
        affixed_word: aff.affixed_word ? { ...aff.affixed_word } : undefined,
        root_word: aff.root_word ? { ...aff.root_word } : undefined,
    }));
  } else {
    console.warn("No root affixations found in word data");
  }

  // Normalize Affixed Affixations - Revert to ': any'
  if (wordData.affixed_affixations && Array.isArray(wordData.affixed_affixations)) {
    console.log(`Processing ${wordData.affixed_affixations.length} affixed affixations`);
    normalizedWord.affixed_affixations = wordData.affixed_affixations.map((aff: any): Affixation => ({ // Changed back to any
      ...aff,
      affixed_word: aff.affixed_word ? { ...aff.affixed_word } : undefined,
      root_word: aff.root_word ? { ...aff.root_word } : undefined,
    }));
  } else {
    console.warn("No affixed affixations found in word data");
  }

  console.log("Word data normalization complete");
  return normalizedWord;
}

// --- Word Details Fetching --- 

export async function fetchWordDetails(word: string): Promise<WordInfo> {
  const sanitizedWord = sanitizeInput(word);
  // Construct URL based on whether input is ID or lemma
  let url: string;
  const isIdFormat = word.startsWith('id:');
  const isNumeric = !isNaN(parseInt(word, 10));

  if (isIdFormat) {
    const wordId = word.substring(3);
    if (isNaN(parseInt(wordId, 10))) throw new Error('Invalid ID format');
    url = `/words/id/${wordId}`; // Use ID endpoint
  } else if (isNumeric) {
    url = `/words/id/${word}`; // Use ID endpoint if it's just a number
  } else {
    url = `/words/${encodeURIComponent(sanitizedWord)}`; // Use lemma endpoint
  }

  // Removed caching logic - rely on react-query
  try {
    const response = await api.get(url);
    if (!response.data) {
      throw new Error("No data received from API");
    }
    // Still normalize the data structure after fetching
    return normalizeWordData(response.data);
  } catch (error) {
    // Ensure handleApiError always throws, so this function matches Promise<never>
    await handleApiError(error, `fetching word details for ${word}`);
    // This line should be unreachable if handleApiError throws, but satisfies TS
    throw new Error('handleApiError did not throw');
  }
}

// --- Search Functionality --- 

export async function searchWords(query: string, options: SearchOptions): Promise<SearchResult> {
  const cacheKey = `search:${query}:${JSON.stringify(options)}`;
  const cached = await getCachedData<SearchResult>(cacheKey);
  if (cached) return cached;

  try {
    // Use page and per_page from SearchOptions, not limit/offset
    const params = { 
      q: query, 
      ...options,
      page: options.page || 1, // Default page 1
      per_page: options.per_page || 10 // Default 10 per page
    };
    // Remove limit/offset if they were incorrectly added to params previously
    // delete params.limit; 
    // delete params.offset;

    const response = await api.get<{ words: SearchWordResult[]; total: number; page: number; perPage: number; query: string }>('/search', { params });
    
    // Access response.data.words, not response.data.results
    const searchResult: SearchResult = {
        words: response.data.words,
        total: response.data.total,
        page: response.data.page,
        perPage: response.data.perPage,
        query: response.data.query
    };

    await setCachedData(cacheKey, searchResult);
    return searchResult;
  } catch (error) {
    console.error(`Error searching words for query "${query}":`, error);
    return minimalSearchResult(query, options); // Return minimal result on error
  }
}

/**
 * Advanced search with additional filtering capabilities
 */
export async function advancedSearch(query: string, options: SearchOptions): Promise<SearchResult> {
  const cacheKey = `advanced_search:${query}:${JSON.stringify(options)}`;
  const cached = await getCachedData<SearchResult>(cacheKey);
  if (cached) return cached;

  try {
    // Use page and per_page from SearchOptions
    const params = { 
        q: query, 
        ...options,
        page: options.page || 1,
        per_page: options.per_page || 10
    };
    // delete params.limit;
    // delete params.offset;

    const response = await api.get<{ words: SearchWordResult[]; total: number; page: number; perPage: number; query: string }>('/search/advanced', { params });

    // Access response.data.words
     const searchResult: SearchResult = {
        words: response.data.words,
        total: response.data.total,
        page: response.data.page,
        perPage: response.data.perPage,
        query: response.data.query
    };
    
    await setCachedData(cacheKey, searchResult);
    return searchResult;
  } catch (error) {
    console.error(`Error in advanced search for query "${query}":`, error);
    return minimalSearchResult(query, options); // Return minimal result on error
  }
}

// --- Other Utility API Functions --- 

export async function getPartsOfSpeech(): Promise<PartOfSpeech[]> {
  // Removed caching logic
  try {
    const response = await api.get<PartOfSpeech[]>('/parts_of_speech');
    return response.data || []; // Return empty array if data is null/undefined
  } catch (error) {
    // Ensure handleApiError always throws, so this function matches Promise<never>
    await handleApiError(error, 'fetching parts of speech');
    throw new Error('handleApiError did not throw');
  }
}

export const testApiConnection = async (): Promise<boolean> => {
  try {
    const response = await api.get('/test');
    const isSuccess = response.status === 200 && response.data && response.data.message === "API v2 is running";
    if (isSuccess) {
      localStorage.setItem('successful_api_endpoint', api.defaults.baseURL || CONFIG.baseURL);
    } else {
      localStorage.removeItem('successful_api_endpoint');
    }
    return isSuccess; // Ensure boolean is returned
  } catch (error) {
    console.error('API connection test failed:', error);
    localStorage.removeItem('successful_api_endpoint');
    return false; // Return false on error
  }
};

export async function getEtymologyTree(
  wordId: number, 
  maxDepth: number = 2 
): Promise<EtymologyTree> {
  // Removed caching logic
  try {
    const response = await api.get<EtymologyTree>(`/words/${wordId}/etymology/tree`, { params: { depth: maxDepth } });
    if (!response.data /* add more checks based on EtymologyTree structure */) {
        console.warn(`Received incomplete etymology tree for word ID ${wordId}`);
        // Return a default empty tree structure matching the EtymologyTree type
        // Update this default based on the actual type definition
        return { word: '', etymologies: [], components: [], id: wordId, language_code: '' } as unknown as EtymologyTree;
    }
    return response.data;
  } catch (error) {
    // Ensure handleApiError always throws, so this function matches Promise<never>
    await handleApiError(error, `fetching etymology tree for word ID ${wordId}`);
    throw new Error('handleApiError did not throw');
  }
}

export async function getRandomWord(): Promise<WordInfo> {
  // Keep NO caching here for random word, it should always be fresh unless handled by caller
  try {
    const response = await api.get('/random');
     if (!response.data) {
       throw new Error("No data received from API for random word");
     }
    // Normalize the potentially raw data from the random endpoint
    return normalizeWordData(response.data);
  } catch (error) {
    // Ensure handleApiError always throws, so this function matches Promise<never>
    await handleApiError(error, 'fetching random word');
    throw new Error('handleApiError did not throw');
  }
}

export async function getStatistics(): Promise<Statistics> {
  // Removed caching logic
  try {
    const response = await api.get<Statistics>('/statistics');
    return response.data; // Add validation if needed
  } catch (error) {
    // Ensure handleApiError always throws, so this function matches Promise<never>
    await handleApiError(error, 'fetching statistics');
    throw new Error('handleApiError did not throw');
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

// --- Word Relations Fetching ---
export async function fetchWordRelations(word: string): Promise<{
  outgoing_relations: Relation[],
  incoming_relations: Relation[]
}> {
  const sanitizedWord = sanitizeInput(word);
  // Construct URL based on whether input is ID or lemma
  let url: string;
  const isIdFormat = word.startsWith('id:');
  const isNumeric = !isNaN(parseInt(word, 10));

  if (isIdFormat) {
    const wordId = word.substring(3);
    if (isNaN(parseInt(wordId, 10))) throw new Error('Invalid ID format');
    url = `/words/id/${wordId}/relations`; // Assuming this endpoint exists
  } else if (isNumeric) {
    url = `/words/id/${word}/relations`; // Assuming this endpoint exists
  } else {
     url = `/words/${encodeURIComponent(sanitizedWord)}/relations`; // Assuming this endpoint exists
  }

  // No caching here unless specifically decided otherwise
  try {
    const response = await api.get<{ outgoing_relations: Relation[], incoming_relations: Relation[] }>(url);
    // Assuming the endpoint returns { outgoing_relations: [], incoming_relations: [] }
    // Return data or default empty structure
    return response.data || { outgoing_relations: [], incoming_relations: [] };
  } catch (error) {
    // Ensure this path is unreachable if handleApiError throws
    throw new Error('Control should not reach here after relation fetch error');
  }
}

// Add these new API functions for Baybayin endpoints

/**
 * Search for words with specific Baybayin characters
 */
export const searchBaybayin = async (query: string, options: { 
  language_code?: string, 
  limit?: number, 
  offset?: number 
} = {}): Promise<any> => {
  const sanitizedQuery = sanitizeInput(query);
  try {
    const response = await api.get('/baybayin/search', { params: { query: sanitizedQuery, ...options } });
    return response.data;
  } catch (error) {
    // Ensure handleApiError always throws, so this function matches Promise<never>
    await handleApiError(error, `searching baybayin for "${sanitizedQuery}"`);
    throw new Error('handleApiError did not throw');
  }
};

/**
 * Get detailed statistics about Baybayin usage in the dictionary
 */
export const getBaybayinStatistics = async (): Promise<any> => {
  try {
    const response = await api.get('/baybayin/statistics');
    return response.data;
  } catch (error) {
    // Ensure handleApiError always throws, so this function matches Promise<never>
    await handleApiError(error, 'fetching baybayin statistics');
    throw new Error('handleApiError did not throw');
  }
};

/**
 * Convert text to Baybayin script
 */
export const convertToBaybayin = async (text: string, language: string = 'fil'): Promise<any> => {
  const sanitizedText = sanitizeInput(text);
  try {
    const response = await api.post('/baybayin/convert', { text: sanitizedText, language_code: language });
    return response.data;
  } catch (error) {
    // Ensure handleApiError always throws, so this function matches Promise<never>
    await handleApiError(error, 'converting text to baybayin');
    throw new Error('handleApiError did not throw');
  }
};

// Define a minimal valid SearchResult fallback
const minimalSearchResult = (query: string, options: SearchOptions): SearchResult => ({
    words: [],
    total: 0,
    page: options.page || 1,
    perPage: options.per_page || 50, // Use per_page, provide default
    query: query,
    // Removed filters property which is not in SearchResult type
    // filters: options, 
    // Add other potentially required fields from SearchResult if needed
    // count: 0, // Assuming count is NOT part of SearchResult 
    // results: [] // Assuming results is NOT part of SearchResult
});