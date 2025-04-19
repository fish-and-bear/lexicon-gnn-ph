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

// const circuitBreaker = new PersistentCircuitBreaker(); // Temporarily comment out

// Function to reset the circuit breaker state
export function resetCircuitBreaker() {
  // circuitBreaker.reset(); // Temporarily comment out
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
// const savedEndpoint = localStorage.getItem('successful_api_endpoint'); // Temporarily comment out
let apiBaseURL = CONFIG.baseURL;

/* Temporarily comment out saved endpoint logic
if (savedEndpoint) {
  console.log('Using saved API endpoint:', savedEndpoint);
  if (savedEndpoint.includes('/api/v2')) {
    apiBaseURL = savedEndpoint;
  } else {
    apiBaseURL = `${savedEndpoint}/api/v2`;
  }
}
*/

// API client configuration
/* Temporarily comment out axios instance creation and interceptors
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
    // Modify headers directly instead of reassigning
    if (config.headers) {
        config.headers['Origin'] = window.location.origin;
        config.headers['Access-Control-Request-Method'] = config.method?.toUpperCase() || 'GET';
    }
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
    /* Temporarily comment out circuit breaker interaction
    // ... circuit breaker logic ...
    */
/*
    return response;
  },
  (error) => {
    console.error('Response error:', error);
    // circuitBreaker.recordFailure(); // Temporarily comment out
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
    return Promise.reject(error);
  }
);
*/

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
  breadth?: number;
  include_affixes?: boolean;
  include_etymology?: boolean;
  cluster_threshold?: number;
  relation_types?: string[];
}

interface NetworkMetadata {
  root_word: string;
  normalized_lemma: string;
  language_code: string;
  depth: number;
  total_nodes: number;
  total_edges: number;
  query_time?: number | null;
  filters_applied?: {
    depth: number;
    breadth: number;
    include_affixes: boolean;
    include_etymology: boolean;
    relation_types: string[];
  };
}

interface NetworkNode {
  id: string;
  label: string;
  word: string;
  language: string;
  type: string;
  depth: number;
  has_baybayin: boolean;
  baybayin_form: string | null;
  normalized_lemma: string;
  main: boolean;
}

interface NetworkEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  directed: boolean;
  weight: number;
}

interface LocalWordNetwork {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  metadata: NetworkMetadata;
}

export async function fetchWordNetwork(
  word: string, 
  options: WordNetworkOptions = {}
): Promise<ImportedWordNetwork> {
  const sanitizedWord = word.toLowerCase();
  const {
    depth = 2,
    breadth = 10,
    include_affixes = true,
    include_etymology = true,
    cluster_threshold = 0.3,
    relation_types
  } = options;
  
  const sanitizedDepth = Math.min(Math.max(1, depth), 4); // Max depth 4
  const sanitizedBreadth = Math.min(Math.max(5, breadth), 50); // Between 5 and 50
  
  const cacheKey = `cache:wordNetwork:${sanitizedWord}-${sanitizedDepth}-${sanitizedBreadth}-${include_affixes}-${include_etymology}-${relation_types?.join(',')}`;
    
  const cachedData = getCachedData<ImportedWordNetwork>(cacheKey);
  if (cachedData) {
    console.log(`Cache hit for word network: ${sanitizedWord}`);
    return cachedData;
  }

  console.log(`Cache miss for word network: ${sanitizedWord}. Fetching from API...`);

  /* Temporarily comment out circuit breaker check
  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for word network.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }
  */

  try {
    const encodedWord = encodeURIComponent(sanitizedWord);
    const endpoint = `/words/${encodedWord}/semantic_network`;
    
    const params: Record<string, any> = {
      depth: sanitizedDepth,
      breadth: sanitizedBreadth,
      include_affixes,
      include_etymology,
      cluster_threshold
    };

    if (relation_types && relation_types.length > 0) {
      params.relation_types = relation_types.join(',');
    }
    
    const response = await api.get(endpoint, {
      params,
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      }
    });

    if (!response.data) {
      throw new Error('No data received from API');
    }

    console.log('Network data received:', response.data);

    // Check for required properties and validate structure
    if (!response.data.nodes || !Array.isArray(response.data.nodes) || 
        !response.data.links || !Array.isArray(response.data.links)) {
      console.error('Invalid response data:', response.data);
      throw new Error('Invalid network data structure received from API');
    }

    // Normalize and validate each node
    const nodes = response.data.nodes.map((n: any) => {
      const node = n as Partial<NetworkNode>;
      if (!node.id) {
        console.warn('Node missing ID:', node);
        throw new Error('Invalid node data: missing ID');
      }

      return {
        id: String(node.id),
        label: node.label || node.word || String(node.id),
        word: node.word || node.label || String(node.id),
        language: node.language || 'tl',
        type: node.type || (node.main ? 'main' : 'related'),
        depth: typeof node.depth === 'number' ? node.depth : 0,
        has_baybayin: Boolean(node.has_baybayin),
        baybayin_form: node.baybayin_form || null,
        normalized_lemma: node.normalized_lemma || node.label || String(node.id),
        main: Boolean(node.main)
      } as NetworkNode;
    });

    // Normalize and validate each edge
    const edges = response.data.links.map((e: any) => {
      if (!e.source || !e.target) {
        console.warn('Edge missing source or target:', e);
        throw new Error('Invalid edge data: missing source or target');
      }

      return {
        id: e.id || `${e.source}-${e.target}-${e.type || 'related'}`,
        source: String(e.source),
        target: String(e.target),
        type: e.type || 'related',
        directed: e.directed ?? false,
        weight: typeof e.weight === 'number' ? e.weight : 1
      };
    });

    const wordNetwork: LocalWordNetwork = {
      nodes,
      edges,
      metadata: {
        root_word: word,
        normalized_lemma: sanitizedWord,
        language_code: response.data.metadata?.language_code || 'tl',
        depth: sanitizedDepth,
        total_nodes: nodes.length,
        total_edges: edges.length,
        query_time: response.data.metadata?.execution_time || null,
        filters_applied: {
          depth: sanitizedDepth,
          breadth: sanitizedBreadth,
          include_affixes,
          include_etymology,
          relation_types: relation_types || []
        }
      }
    };

    // Validate network connectivity
    const hasMainNode = nodes.some((n: NetworkNode) => n.main || n.type === 'main');
    if (!hasMainNode) {
      console.warn('Network missing main node');
      throw new Error('Invalid network: missing main node');
    }

    // Record success and cache the data
    // circuitBreaker.recordSuccess();
    setCachedData(cacheKey, wordNetwork as unknown as ImportedWordNetwork);

    return wordNetwork as unknown as ImportedWordNetwork;
  } catch (error) {
    console.error('Error in fetchWordNetwork:', error);
    
    // Record failure for circuit breaker
    // circuitBreaker.recordFailure();
    
    if (error instanceof Error) {
      if (error.message.includes('404')) {
        throw new Error(`Word '${word}' not found in the dictionary.`);
      }
      if (error.message.includes('Failed to fetch') || 
          error.message.includes('Network Error') || 
          error.message.includes('ECONNREFUSED')) {
        throw new Error(
          'Cannot connect to the dictionary server. Please check your internet connection and try again.'
        );
      }
      throw new Error(`Failed to fetch word network: ${error.message}`);
    }
    
    throw new Error('An unexpected error occurred while fetching the word network');
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
  // Check if the word parameter uses the ID format (id:12345)
  let isIdRequest = word.startsWith('id:');
  let endpoint;
  
  if (isIdRequest) {
    // Extract the numeric ID from the format
    const wordIdStr = word.substring(3); // Skip the 'id:' prefix
    const wordId = parseInt(wordIdStr, 10);
    
    // Validate that the ID is actually a number
    if (isNaN(wordId)) {
      console.error(`Invalid word ID format: "${wordIdStr}" is not a number`);
      throw new Error(`Invalid word ID format: "${wordIdStr}" is not a number. ID must be numeric.`);
    }
    
    console.log(`Fetching word details by ID: ${wordId}`);
    endpoint = `/words/id/${wordId}`; // Use the ID-specific endpoint
    
    // Don't normalize or cache ID-based requests using the word text
  } else if (!isNaN(parseInt(word, 10))) {
    // Handle case where a numeric ID is passed directly without 'id:' prefix
    const wordId = parseInt(word, 10);
    console.log(`Numeric ID detected, fetching word details by ID: ${wordId}`);
    endpoint = `/words/id/${wordId}`;
    
    // Treat this as an ID request for error handling
    isIdRequest = true;
  } else {
    // Normal word text request - use the original flow
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
    endpoint = `/words/${encodeURIComponent(normalizedWord)}`;
  }

  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for word details.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }

  try {
    console.log(`Making API request to endpoint: ${endpoint}`);
    // Add query parameters to include all related data
    const response = await api.get(endpoint, {
      params: {
        include_relations: true,
        include_etymologies: true,
        include_root: true,
        include_derived: true,
        include_affixations: true,
        include_definition_relations: true,
        include_forms: true,
        include_templates: true
      }
    });

    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }

    // Validate response data - make sure it has the minimum required properties
    if (!response.data || !response.data.id || !response.data.lemma) {
      console.error('Invalid or empty response data:', response.data);
      if (isIdRequest) {
        throw new Error(`Word with ID "${word.startsWith('id:') ? word.substring(3) : word}" not found or has invalid data.`);
      } else {
        throw new Error(`Word "${word}" not found or has invalid data.`);
      }
    }

    console.log('Word details API response received:', response.data);
    
    // Check for expected relation data structures
    if (!response.data.outgoing_relations) {
      console.warn('Response missing outgoing_relations:', response.data);
    } else {
      console.log(`Response has ${response.data.outgoing_relations.length} outgoing relations. Sample:`, 
        response.data.outgoing_relations.length > 0 ? JSON.stringify(response.data.outgoing_relations[0]) : 'none');
    }
    
    if (!response.data.incoming_relations) {
      console.warn('Response missing incoming_relations:', response.data);
    } else {
      console.log(`Response has ${response.data.incoming_relations.length} incoming relations. Sample:`, 
        response.data.incoming_relations.length > 0 ? JSON.stringify(response.data.incoming_relations[0]) : 'none');
    }
    
    if (!response.data.etymologies) {
      console.warn('Response missing etymologies:', response.data);
    }

    // NOTE: Success is recorded by the interceptor
    const normalizedData = normalizeWordData(response.data);
    
    // Log the normalized data structure for debugging
    console.log('Normalized word data with relations:', {
      id: normalizedData.id,
      lemma: normalizedData.lemma,
      incomingRelations: normalizedData.incoming_relations?.length || 0,
      outgoingRelations: normalizedData.outgoing_relations?.length || 0,
      incomingSample: normalizedData.incoming_relations && normalizedData.incoming_relations.length > 0 
        ? JSON.stringify(normalizedData.incoming_relations[0]) : 'none',
      outgoingSample: normalizedData.outgoing_relations && normalizedData.outgoing_relations.length > 0 
        ? JSON.stringify(normalizedData.outgoing_relations[0]) : 'none'
    });
    
    // Only cache non-ID requests
    if (!isIdRequest) {
      setCachedData(`cache:wordDetails:${word.toLowerCase()}`, response.data); // Cache the raw data
    }
    
    return normalizedData;

  } catch (error) {
    // Error handling logic
    circuitBreaker.recordFailure();
    
    // Create a custom error with more context for ID-based requests
    if (isIdRequest && axios.isAxiosError(error) && error.response?.status === 404) {
      console.error(`Word with ID '${word.startsWith('id:') ? word.substring(3) : word}' not found:`, error);
      throw new Error(`Word with ID '${word.startsWith('id:') ? word.substring(3) : word}' not found in the database.`);
    }
    
    // Check for database schema errors related to relation_data column
    if (axios.isAxiosError(error) && error.response?.status === 500) {
      const errorMessage = error.response.data?.error || 'Internal server error';
      
      // Handle database relation_data missing column error
      if (errorMessage.includes('column r.relation_data does not exist') || 
          errorMessage.includes('relation_data') ||
          errorMessage.includes('SQL error')) {
        
        console.log('Detected database schema error with relation_data column. Creating partial word info with error flag.');
        
        // Try to fetch the semantic network for fallback
        try {
          // Get basic word info from the error response if possible
          const basicData = error.response.data?.word || {
            id: isIdRequest ? parseInt(word.replace('id:', ''), 10) : 0,
            lemma: word
          };
          
          // Create a partial word info object with the error message
          const partialWordInfo: WordInfo = {
            id: basicData.id,
            lemma: basicData.lemma || word,
            server_error: `Server database error: ${errorMessage}`,
            incoming_relations: [],
            outgoing_relations: []
          };
          
          // Try to fetch semantic network data as fallback
          try {
            if (basicData.id || basicData.lemma) {
              const idOrWord = basicData.id ? `id:${basicData.id}` : basicData.lemma;
              const networkData = await fetchWordNetwork(idOrWord);
              
              if (networkData && networkData.nodes && networkData.edges) {
                partialWordInfo.semantic_network = {
                  nodes: networkData.nodes,
                  links: networkData.edges
                };
                console.log('Successfully added semantic network fallback data');
              }
            }
          } catch (networkError) {
            console.error('Failed to fetch semantic network fallback:', networkError);
          }
          
          return partialWordInfo;
        } catch (fallbackError) {
          console.error('Error creating fallback word info:', fallbackError);
        }
      }
      
      // Check if the error is a database error when looking up by word
      if (!isIdRequest && (
        errorMessage.includes('Database error') || 
        errorMessage.includes('SqliteError') ||
        errorMessage.includes('SQL error')
      )) {
        console.log('General database error detected. Creating partial word info with error flag.');
        
        // Try to fetch the semantic network for fallback
        try {
          // Create a partial word info object with the error message
          const partialWordInfo: WordInfo = {
            id: isIdRequest ? parseInt(word.replace('id:', ''), 10) : 0,
            lemma: word,
            server_error: `Database error: ${errorMessage}`,
            incoming_relations: [],
            outgoing_relations: []
          };
          
          // Try to fetch semantic network data as fallback
          try {
            const networkData = await fetchWordNetwork(word);
            
            if (networkData && networkData.nodes && networkData.edges) {
              partialWordInfo.semantic_network = {
                nodes: networkData.nodes,
                links: networkData.edges
              };
              console.log('Successfully added semantic network fallback data for general database error');
            }
          } catch (networkError) {
            console.error('Failed to fetch semantic network fallback:', networkError);
          }
          
          return partialWordInfo;
        } catch (fallbackError) {
          console.error('Error creating fallback word info:', fallbackError);
        }
        
        throw new Error(`Database error when retrieving details for word '${word}'. Try searching for this word instead.`);
      }
      
      // Handle dictionary update sequence error specifically
      if (errorMessage.includes('dictionary update sequence element')) {
        console.error('Dictionary update sequence error detected, this is a backend database issue');
        throw new Error(`Server database error. Please try searching for this word instead of using direct lookup.`);
      }
      
      throw new Error(`Server error: ${errorMessage}`);
    }
    
    // NOTE: Failure is recorded by the interceptor
    // Throw a more specific error using handleApiError
    await handleApiError(error, `fetching word details for '${word}'`);
    // This line likely won't be reached, but keeps TypeScript happy
    throw new Error('An unknown error occurred after handling API error.'); 
  }
}

// --- Search Functionality --- 

export async function searchWords(query: string, options: SearchOptions): Promise<SearchResult> {
  console.log(`[DEBUG] searchWords called with query: "${query}" and options:`, options);
  
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
        q: encodeURIComponent(query), // Explicitly encode the query parameter
        limit: options.per_page || 20, // Default limit
        offset: options.page ? (options.page - 1) * (options.per_page || 20) : 0, 
    };
    
    // Basic filters
    if (options.language) apiParams.language = options.language;
    if (options.mode) apiParams.mode = options.mode;
    if (options.pos) apiParams.pos = options.pos;
    if (options.sort) apiParams.sort = options.sort;
    if (options.order) apiParams.order = options.order;
    
    // Feature filters
    if (options.has_baybayin !== undefined) apiParams.has_baybayin = options.has_baybayin;
    if (options.exclude_baybayin !== undefined) apiParams.exclude_baybayin = options.exclude_baybayin;
    if (options.has_etymology !== undefined) apiParams.has_etymology = options.has_etymology;
    if (options.has_pronunciation !== undefined) apiParams.has_pronunciation = options.has_pronunciation;
    if (options.has_forms !== undefined) apiParams.has_forms = options.has_forms;
    if (options.has_templates !== undefined) apiParams.has_templates = options.has_templates;
    
    // Advanced boolean filters
    if (options.is_root !== undefined) apiParams.is_root = options.is_root;
    if (options.is_proper_noun !== undefined) apiParams.is_proper_noun = options.is_proper_noun;
    if (options.is_abbreviation !== undefined) apiParams.is_abbreviation = options.is_abbreviation;
    if (options.is_initialism !== undefined) apiParams.is_initialism = options.is_initialism;
    
    // Date range filters
    if (options.date_added_from) apiParams.date_added_from = options.date_added_from;
    if (options.date_added_to) apiParams.date_added_to = options.date_added_to;
    if (options.date_modified_from) apiParams.date_modified_from = options.date_modified_from;
    if (options.date_modified_to) apiParams.date_modified_to = options.date_modified_to;
    
    // Definition and relation count filters
    if (options.min_definition_count !== undefined) apiParams.min_definition_count = options.min_definition_count;
    if (options.max_definition_count !== undefined) apiParams.max_definition_count = options.max_definition_count;
    if (options.min_relation_count !== undefined) apiParams.min_relation_count = options.min_relation_count;
    if (options.max_relation_count !== undefined) apiParams.max_relation_count = options.max_relation_count;
    
    // Completeness score range
    if (options.min_completeness !== undefined) apiParams.min_completeness = options.min_completeness;
    if (options.max_completeness !== undefined) apiParams.max_completeness = options.max_completeness;
    
    // Include options
    if (options.include_full !== undefined) apiParams.include_full = options.include_full;
    if (options.include_definitions !== undefined) apiParams.include_definitions = options.include_definitions;
    if (options.include_pronunciations !== undefined) apiParams.include_pronunciations = options.include_pronunciations;
    if (options.include_etymologies !== undefined) apiParams.include_etymologies = options.include_etymologies;
    if (options.include_relations !== undefined) apiParams.include_relations = options.include_relations;
    if (options.include_forms !== undefined) apiParams.include_forms = options.include_forms;
    if (options.include_templates !== undefined) apiParams.include_templates = options.include_templates;
    if (options.include_metadata !== undefined) apiParams.include_metadata = options.include_metadata;
    if (options.include_related_words !== undefined) apiParams.include_related_words = options.include_related_words;
    if (options.include_definition_relations !== undefined) apiParams.include_definition_relations = options.include_definition_relations;
    
    // For debugging, log the exact URL that will be called
    const searchUrl = `${CONFIG.baseURL}/search?q=${apiParams.q}&limit=${apiParams.limit}`;
    console.log(`Making search GET request to URL: ${searchUrl}`);

    console.log(`[DEBUG] Making search API request with params:`, apiParams);
    
    // Try with direct fetch first for improved reliability
    let searchResult: SearchResult | null = null;
    let fetchError: any = null;
    
    try {
      console.log(`Trying direct fetch for search: ${query}`);
      const directResponse = await fetch(`${CONFIG.baseURL}/search?q=${apiParams.q}&limit=${apiParams.limit}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        mode: 'cors'
      });
      
      if (directResponse.ok) {
        const data = await directResponse.json();
        console.log(`Direct fetch successful:`, data);
        
        // Format the response like standard searchResult
        searchResult = {
          words: (data.results || []).map((result: any): SearchWordResult => ({
            id: result.id,
            lemma: result.lemma,
            normalized_lemma: result.normalized_lemma,
            language_code: result.language_code,
            has_baybayin: result.has_baybayin,
            baybayin_form: result.baybayin_form,
            romanized_form: result.romanized_form,
            definitions: []
          })),
          page: options.page || 1,
          perPage: options.per_page || (data.results?.length || 0), 
          total: data.count || 0,
          query: query 
        };
      } else {
        console.log(`Direct fetch failed with status: ${directResponse.status}`);
        fetchError = new Error(`Failed direct fetch with status: ${directResponse.status}`);
      }
    } catch (error) {
      console.error(`Error with direct fetch:`, error);
      fetchError = error;
    }
    
    // If direct fetch succeeded, return the result
    if (searchResult) {
      setCachedData(cacheKey, searchResult);
      return searchResult;
    }
    
    // If direct fetch failed, try with axios
    try {
      // Continue with normal axios request if direct fetch didn't succeed
      const response = await api.get('/search', { params: apiParams });
      console.log(`[DEBUG] Search API responded with status: ${response.status}`);
      
      if (response.status !== 200) {
          throw new Error(`API returned status ${response.status}: ${response.statusText}`);
      }
      
      const data = response.data; // Assuming data = { total: number, words: RawWordSummary[] }
      console.log(`[DEBUG] Search API raw response data:`, data);
      
      // Transform the response into SearchResult format
      searchResult = {
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
      
      console.log(`[DEBUG] Transformed search result:`, searchResult);
      setCachedData(cacheKey, searchResult);
      return searchResult;
      
    } catch (axiosError) {
      console.error(`[DEBUG] Axios search error:`, axiosError);
      // If we have a fetchError from the direct fetch attempt, include it in the error message
      if (fetchError) {
        console.error(`[DEBUG] Both direct fetch and axios failed. Direct fetch error:`, fetchError);
      }
      throw axiosError;
    }
    
  } catch (error) {
    // Failure recorded by interceptor
    console.error(`[DEBUG] Search error for query "${query}":`, error);
    
    // Handle specific error cases
    if (axios.isAxiosError(error) && error.response?.status === 500) {
      const errorMessage = error.response.data?.error || 'Internal server error';
      
      // Handle any database schema error related to undefined columns
      if (errorMessage.includes('column') && 
          (errorMessage.includes('does not exist') || 
           errorMessage.includes('undefined column'))) {
        console.error('Database schema error detected:', errorMessage);
        
        // Create a minimal result with an error message
        const errorResult: SearchResult = {
          words: [],
          page: options.page || 1,
          perPage: options.per_page || 10,
          total: 0,
          query: query,
          error: 'Database schema error: The database schema doesn\'t match what the application expects. Please contact the administrator.'
        };
        
        return errorResult;
      }
      
      // Handle dictionary update sequence error specifically
      if (errorMessage.includes('dictionary update sequence element')) {
        console.error('Dictionary update sequence error detected, this is a backend database issue');
        throw new Error(`Server database error. Please try a different search query.`);
      }
    }
    
    await handleApiError(error, `searching words with query "${query}"`);
    throw new Error('An unknown error occurred during search. Please try with a different query.');
  }
}

/**
 * Advanced search with additional filtering capabilities
 */
export async function advancedSearch(query: string, options: SearchOptions): Promise<SearchResult> {
  console.log(`[DEBUG] advancedSearch called with query: "${query}" and options:`, options);
  
  const cacheKey = `cache:advanced_search:${query}:${JSON.stringify(options)}`;
  const cachedData = getCachedData<SearchResult>(cacheKey);

  if (cachedData) {
    console.log(`Cache hit for advanced search: ${query}`);
    return cachedData;
  }
  console.log(`Cache miss for advanced search: ${query}. Fetching from API...`);

  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for advanced search.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }

  try {
    // Use snake_case for API parameters to match backend expectations
    const apiParams: Record<string, any> = {
        query: encodeURIComponent(query), // Explicitly encode the query parameter
        limit: options.per_page || 50, // Default limit
        offset: options.page ? (options.page - 1) * (options.per_page || 50) : 0, 
        include_details: options.include_full || false,
    };
    
    // Advanced filters
    if (options.language) apiParams.language = options.language;
    if (options.pos) apiParams.pos = options.pos;
    
    // Feature filters
    if (options.has_baybayin !== undefined) apiParams.has_baybayin = options.has_baybayin;
    if (options.has_etymology !== undefined) apiParams.has_etymology = options.has_etymology;
    if (options.has_pronunciation !== undefined) apiParams.has_pronunciation = options.has_pronunciation;
    
    // Date range filters
    if (options.date_added_from) apiParams.date_added_from = options.date_added_from;
    if (options.date_added_to) apiParams.date_added_to = options.date_added_to;
    if (options.date_modified_from) apiParams.date_modified_from = options.date_modified_from;
    if (options.date_modified_to) apiParams.date_modified_to = options.date_modified_to;
    
    // Definition and relation count filters
    if (options.min_definition_count !== undefined) apiParams.min_definition_count = options.min_definition_count;
    if (options.max_definition_count !== undefined) apiParams.max_definition_count = options.max_definition_count;
    if (options.min_relation_count !== undefined) apiParams.min_relation_count = options.min_relation_count;
    if (options.max_relation_count !== undefined) apiParams.max_relation_count = options.max_relation_count;
    
    // Completeness score range
    if (options.min_completeness !== undefined) apiParams.min_completeness = options.min_completeness;
    if (options.max_completeness !== undefined) apiParams.max_completeness = options.max_completeness;
    
    console.log(`[DEBUG] Making advanced search API request with params:`, apiParams);
    
    const response = await api.get('/search/advanced', { params: apiParams });
    console.log(`[DEBUG] Advanced search API responded with status: ${response.status}`);
    
    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }
    
    const data = response.data;
    console.log(`[DEBUG] Advanced search API raw response data:`, data);
    
    // Transform the response into SearchResult format
    const searchResult: SearchResult = {
      words: (data.results || []).map((result: any): SearchWordResult => ({
        id: result.id,
        lemma: result.lemma,
        normalized_lemma: result.normalized_lemma,
        language_code: result.language_code,
        has_baybayin: result.has_baybayin,
        baybayin_form: result.baybayin_form,
        romanized_form: result.romanized_form,
        completeness_score: result.completeness_score,
        definitions: result.definitions || []
      })),
      page: options.page || 1,
      perPage: options.per_page || (data.results?.length || 0), 
      total: data.count || 0,
      query: query 
    };
    
    console.log(`[DEBUG] Transformed advanced search result:`, searchResult);
    setCachedData(cacheKey, searchResult);
    return searchResult;
      
  } catch (error) {
    console.error(`[DEBUG] Advanced search error:`, error);
    await handleApiError(error, `performing advanced search with query "${query}"`);
    throw new Error('An unexpected error occurred during advanced search.');
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

export const testApiConnection = async (): Promise<boolean> => {
  try {
    // First try with direct fetch for better error handling
    try {
      const response = await fetch(`${CONFIG.baseURL}/test`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        mode: 'cors',
        cache: 'no-cache'
      });

      if (response.ok) {
        const data = await response.json();
        if (data.status === 'ok') {
          // Save the successful endpoint
          localStorage.setItem('successful_api_endpoint', CONFIG.baseURL);
          return true;
        }
      }
      return false;
    } catch (fetchError) {
      console.error('Direct fetch test failed:', fetchError);
      
      // Fall back to axios if fetch fails
      const response = await axios.get(`${CONFIG.baseURL}/test`, {
        timeout: 5000 // 5 second timeout
      });
      
      if (response.status === 200 && response.data.status === 'ok') {
        // Save the successful endpoint
        localStorage.setItem('successful_api_endpoint', CONFIG.baseURL);
        return true;
      }
      return false;
    }
  } catch (error) {
    console.error('Backend server connection test failed:', error);
    return false;
  }
};

export async function getEtymologyTree(
  wordId: number, 
  maxDepth: number = 2 
): Promise<EtymologyTree> {
  console.log(`Fetching etymology tree for wordId=${wordId}, maxDepth=${maxDepth}`);
  const cacheKey = `cache:etymologyTree:${wordId}-${maxDepth}`;
  const cachedData = getCachedData<EtymologyTree>(cacheKey);
  if (cachedData) {
    console.log(`Using cached etymology tree data for wordId=${wordId}`);
    return cachedData;
  }

  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");

  try {
    const endpoint = `/words/${wordId}/etymology/tree`;
    console.log(`Making API request to: ${endpoint} with maxDepth=${maxDepth}`);
    
    const response = await api.get(endpoint, { params: { max_depth: maxDepth } });
    console.log(`Etymology tree API response status: ${response.status}`);
    
    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }
    
    console.log('Etymology tree raw response:', response.data);
    
    // Try to extract the tree data, with more detailed validation
    const treeData = response.data?.etymology_tree || response.data;
    
    if (!treeData) {
      console.error('Etymology tree response is empty or null');
      throw new Error('Empty etymology tree data received from API');
    }
    
    if (typeof treeData !== 'object') {
      console.error('Etymology tree response is not an object:', treeData);
      throw new Error('Invalid etymology tree data: Not an object');
    }
    
    // Check for empty tree (no nodes)
    if (!treeData.nodes || !Array.isArray(treeData.nodes) || treeData.nodes.length === 0) {
      console.warn('Etymology tree has no nodes:', treeData);
      // Return empty tree structure instead of throwing error
      const emptyTree: EtymologyTree = { 
        nodes: [], 
        edges: [],
        word: '',
        etymology_tree: {},
        complete: false
      };
      setCachedData(cacheKey, emptyTree);
      return emptyTree;
    }
    
    console.log(`Etymology tree data received with ${treeData.nodes.length} nodes and ${treeData.edges?.length || 0} edges`);
    setCachedData(cacheKey, treeData);
    return treeData;
  } catch (error) {
    console.error(`Error fetching etymology tree for wordId=${wordId}:`, error);
    // Return empty tree structure instead of throwing error
    const emptyTree: EtymologyTree = { 
      nodes: [], 
      edges: [],
      word: '',
      etymology_tree: {},
      complete: false
    };
    return emptyTree;
  }
}

export async function getRandomWord(): Promise<WordInfo> {
  // Random word shouldn't be cached aggressively
  // Don't use circuit breaker for random word requests - removed check
  
  try {
    console.log("Fetching random word from API...");
    
    // Use the base random word URL without filtering for baybayin
    const randomWordUrl = `${CONFIG.baseURL}/random`;
    console.log("Random word URL:", randomWordUrl);
    
    // Use direct fetch instead of axios for more control
    const response = await fetch(randomWordUrl, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      mode: 'cors'
    });
    
    if (!response.ok) {
      // Try to parse the error response
      try {
        const errorData = await response.json();
        const errorMessage = errorData.error || `API returned status ${response.status}: ${response.statusText}`;
        const errorDetails = errorData.details ? `\nDetails: ${JSON.stringify(errorData.details)}` : '';
        throw new Error(`${errorMessage}${errorDetails}`);
      } catch (parseError) {
        // If we can't parse the error JSON, use the raw status
        throw new Error(`API returned status ${response.status}: ${response.statusText}`);
      }
    }
    
    const data = await response.json();
    console.log("Random word response data:", data);
    
    // Validate data before normalization
    if (!data || !data.lemma) {
      throw new Error('Invalid random word data: Missing required fields');
    }
    
    // Ensure baybayin_form is handled safely
    if (data.has_baybayin && data.baybayin_form) {
      try {
        // Verify baybayin form is valid by checking for valid Unicode range
        const hasBaybayin = data.baybayin_form.length > 0 && 
                          data.baybayin_form.split('').every((c: string) => 
                            c === ' ' || ('\u1700' <= c && c <= '\u171F'));
        
        if (!hasBaybayin) {
          console.warn('Invalid baybayin characters detected, clearing baybayin_form');
          data.baybayin_form = null;
          data.has_baybayin = false;
        }
      } catch (error) {
        console.warn('Error validating baybayin_form, clearing it to prevent issues', error);
        data.baybayin_form = null;
        data.has_baybayin = false;
      }
    }
    
    return normalizeWordData(data);
  } catch (error) {
    console.error("Error fetching random word:", error);
    
    if (error instanceof Error) {
      if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch')) {
        throw new Error(`Network error while fetching random word. Please check that the backend server is running on port 10000.`);
      }
      
      // Don't record failures for random word requests
      // Pass through the error message for better debugging
      throw error;
    }
    
    // Don't use handleApiError for random word to avoid circuit breaker
    console.warn('Non-Error type caught in getRandomWord:', error);
    throw new Error('Failed to fetch a random word. Please try again later.');
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

// --- Word Relations Fetching ---
export async function fetchWordRelations(word: string): Promise<{
  outgoing_relations: Relation[],
  incoming_relations: Relation[]
}> {
  console.log(`Fetching relations directly for word: ${word}`);
  
  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for word relations.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }
  
  try {
    const sanitizedWord = typeof word === 'string' ? word.trim() : String(word);
    
    // Handle ID-based requests
    let endpoint;
    if (sanitizedWord.startsWith('id:')) {
      const wordId = sanitizedWord.substring(3);
      endpoint = `/words/id/${wordId}/relations`;
    } else if (!isNaN(parseInt(sanitizedWord, 10))) {
      // Handle numeric ID without 'id:' prefix
      endpoint = `/words/id/${sanitizedWord}/relations`;
    } else {
      endpoint = `/words/${encodeURIComponent(sanitizedWord)}/relations`;
    }
    
    console.log(`Making API request for relations to: ${endpoint}`);
    const response = await api.get(endpoint);
    
    if (response.status !== 200) {
      throw new Error(`Failed to fetch relations: ${response.statusText}`);
    }
    
    console.log("Raw relations response:", response.data);
    
    return {
      incoming_relations: response.data.incoming_relations || [],
      outgoing_relations: response.data.outgoing_relations || []
    };
  } catch (error) {
    console.error('Error in fetchWordRelations:', error);
    circuitBreaker.recordFailure();
    
    if (error instanceof Error) {
      if (error.message.includes('404')) {
        console.warn(`No relations found for word '${word}'`);
        return { incoming_relations: [], outgoing_relations: [] };
      }
      throw new Error(`Failed to fetch relations: ${error.message}`);
    }
    
    throw new Error('An unexpected error occurred while fetching relations');
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
  try {
    const params = {
      query,
      ...options
    };
    
    const response = await api.get('/baybayin/search', { params });
    return response.data;
  } catch (error) {
    return handleApiError(error, 'searching Baybayin characters');
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
    return handleApiError(error, 'fetching Baybayin statistics');
  }
};

/**
 * Convert text to Baybayin script
 */
export const convertToBaybayin = async (text: string, language: string = 'fil'): Promise<any> => {
  try {
    console.log(`Converting "${text}" to baybayin with language "${language}"`);
    
    // Make sure we're using the right API endpoint
    const url = `${api.defaults.baseURL}/convert/baybayin`;
    
    // Log the URL and request data
    console.log(`Making Baybayin conversion request to: ${url}`);
    console.log(`Request data: { text: "${text}", language: "${language}" }`);
    
    const response = await api.post(url, {
      text: text,
      language: language
    });
    
    // Log successful conversion
    console.log(`Baybayin conversion successful:`, response.data);
    
    return response.data;
  } catch (error) {
    console.error('Error converting to baybayin:', error);
    // Return a helpful error object instead of throwing
    return { 
      error: true, 
      message: error instanceof Error ? error.message : 'Unknown error during Baybayin conversion' 
    };
  }
};