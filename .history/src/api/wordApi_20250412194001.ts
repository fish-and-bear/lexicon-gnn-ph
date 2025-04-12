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

  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for word network.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }

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
    circuitBreaker.recordSuccess();
    setCachedData(cacheKey, wordNetwork as unknown as ImportedWordNetwork);

    return wordNetwork as unknown as ImportedWordNetwork;
  } catch (error) {
    console.error('Error in fetchWordNetwork:', error);
    
    // Record failure for circuit breaker
    circuitBreaker.recordFailure();
    
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

  // Normalize Definitions - Revert to ': any'
  if (wordData.definitions && Array.isArray(wordData.definitions)) {
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
  // Check if the word parameter uses the ID format (id:12345)
  const isIdRequest = word.startsWith('id:');
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
    endpoint = `/words/${encodeURIComponent(normalizedWord)}/comprehensive`;
  }

  if (!circuitBreaker.canMakeRequest()) {
    console.warn("Circuit breaker is open. Aborting API request for word details.");
    throw new Error("Circuit breaker is open. Please try again later.");
  }

  try {
    const response = await api.get(endpoint);

    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }

    // Validate response data - make sure it has the minimum required properties
    if (!response.data || !response.data.id || !response.data.lemma) {
      console.error('Invalid or empty response data:', response.data);
      if (isIdRequest) {
        throw new Error(`Word with ID "${word.substring(3)}" not found or has invalid data.`);
      } else {
        throw new Error(`Word "${word}" not found or has invalid data.`);
      }
    }

    // NOTE: Success is recorded by the interceptor
    const normalizedData = normalizeWordData(response.data);
    
    // Only cache non-ID requests
    if (!isIdRequest) {
      setCachedData(`cache:wordDetails:${word.toLowerCase()}`, response.data); // Cache the raw data
    }
    
    return normalizedData;

  } catch (error: unknown) {
    // Create a custom error with more context for ID-based requests
    if (isIdRequest && axios.isAxiosError(error) && error.response?.status === 404) {
      console.error(`Word with ID '${word.substring(3)}' not found:`, error);
      throw new Error(`Word with ID '${word.substring(3)}' not found in the database.`);
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
  if (!circuitBreaker.canMakeRequest()) {
    throw new Error("Circuit breaker is open.");
  }

  try {
    console.log("Fetching random word from API...");
    
    // Construct the full random word URL with parameters to avoid baybayin validation issues
    const randomWordUrl = `${CONFIG.baseURL}/random?has_baybayin=false`;
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
      
      // Pass through the error message for better debugging
      throw error;
    }
    
    await handleApiError(error, 'fetching random word');
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