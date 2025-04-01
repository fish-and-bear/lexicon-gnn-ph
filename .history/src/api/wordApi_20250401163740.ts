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

/**
 * Normalizes API word data to ensure consistent structure matching WordInfo interface
 * Handles nested data structures and different property naming conventions
 */
function normalizeWordData(data: any): WordInfo {
  // Handle nested structure with data property
  const wordData = data?.data || data;
  
  if (!wordData) {
    throw new Error('Invalid API response: no word data found');
  }
  
  // Extract core word details
  const normalizedWord: Partial<WordInfo> = {
    id: wordData.id || 0,
    lemma: wordData.lemma || '',
    normalized_lemma: wordData.normalized_lemma || wordData.lemma || '',
    language_code: wordData.language_code || 'tl',
    preferred_spelling: wordData.preferred_spelling || null,
    tags: wordData.tags || [],
    has_baybayin: wordData.has_baybayin || false,
    baybayin_form: wordData.baybayin_form || null,
    romanized_form: wordData.romanized_form || null,
    
    // Handle different formats for pronunciation
    pronunciation: wordData.pronunciation || null,
    
    // Initialize empty arrays for collections
    definitions: [],
    etymologies: [],
    relations: {
      synonyms: [],
      antonyms: [],
      variants: [],
      related: [],
      component_of: [],
      cognate: [],
      root_of: [],
      derived_from: [],
      derived: [],
      root: null,
      // Add special Filipino relation type
      kaugnay: []
    },
    
    // Copy remaining fields from original data
    source_info: wordData.source_info || {},
    data_hash: wordData.data_hash || '',
    complexity_score: wordData.complexity_score || 0,
    usage_frequency: wordData.usage_frequency || 0,
    created_at: wordData.created_at || '',
    updated_at: wordData.updated_at || '',
    last_lookup_at: wordData.last_lookup_at || null,
    view_count: wordData.view_count || 0,
    last_viewed_at: wordData.last_viewed_at || null,
    is_verified: wordData.is_verified || false,
    verification_notes: wordData.verification_notes || null,
    data_quality_score: wordData.data_quality_score || 0,
    
    // Add extra properties that might be useful
    idioms: wordData.idioms || []
  };
  
  // Add is_root_word as a property to be carried over (not in WordInfo interface)
  if (wordData.is_root_word !== undefined) {
    (normalizedWord as any).is_root_word = wordData.is_root_word;
  }
  
  // Normalize definitions 
  if (wordData.definitions && Array.isArray(wordData.definitions)) {
    normalizedWord.definitions = wordData.definitions.map((def: any) => ({
      ...def,
      text: def.definition_text || def.text || '', // Ensure text exists
      part_of_speech: def.standardized_pos || null,
      examples: def.examples || [],
      usage_notes: def.usage_notes || [],
      tags: def.tags || [],
      sources: def.sources || [],
      relations: (def.relations || []).map((rel: { id: number; type: string; word?: string; sources?: string | string[] }) => ({
        ...rel,
        word: rel.word || '', // Ensure word exists
        sources: toArray(rel.sources) // Ensure sources is array
      }))
    }));
  } else if (wordData.definition) {
    // Handle single definition case
    const def = wordData.definition;
    normalizedWord.definitions = [{
      id: 0,
      text: typeof def === 'string' ? def : (def.text || def.definition_text || ''),
      definition_text: typeof def === 'string' ? def : (def.text || def.definition_text || ''),
      original_pos: null,
      part_of_speech: null,
      examples: [],
      usage_notes: [],
      tags: [],
      sources: [],
      relations: [],
      confidence_score: 0,
      is_verified: false
    }];
  }
  
  // Normalize etymologies
  if (wordData.etymologies && Array.isArray(wordData.etymologies)) {
    normalizedWord.etymologies = wordData.etymologies.map((etym: any) => ({
      ...etym,
      text: etym.etymology_text || etym.text || '', // Ensure text exists
      components: toArray(etym.components), // Use helper
      languages: toArray(etym.language_codes), // Use helper (renamed)
      sources: toArray(etym.sources), // Use helper
      // Keep other fields like confidence_level, verification_status as is
    }));
  } else if (wordData.etymology) {
    // Handle single etymology case
    const etym = wordData.etymology;
    normalizedWord.etymologies = [{
      id: 0,
      text: typeof etym === 'string' ? etym : (etym.text || etym.etymology_text || ''),
      etymology_text: typeof etym === 'string' ? etym : (etym.text || etym.etymology_text || ''),
      components: typeof etym === 'object' ? (etym.components || []) : [],
      languages: typeof etym === 'object' ? (etym.language_codes || etym.languages || []) : [],
      sources: typeof etym === 'object' ? (etym.sources || []) : [],
      confidence_level: 'medium',
      verification_status: 'unverified'
    }];
  }
  
  // Normalize relations
  if (wordData.relations) {
    // Map each relation type, ensuring consistent format
    const relationTypes = [
      // Primary relation types from database
      'synonyms', 'antonyms', 'variants', 'related', 'kaugnay',
      'derived_from', 'root_of', 'component_of', 'cognate',
      
      // Semantically organized categories
      'main', 'derivative', 'etymology', 'associated', 'other'
    ];
    
    relationTypes.forEach(type => {
      if (wordData.relations[type]) {
        const relations = wordData.relations[type];
        normalizedWord.relations![type] = Array.isArray(relations) 
          ? relations.map((rel: any) => {
              if (typeof rel === 'string') {
                return { word: rel };
              } else {
                return { word: rel.word || rel.to_word || '' };
              }
            })
          : [];
      }
    });
    
    // Special handling for derived_from -> derived mapping
    if (wordData.relations.derived_from && !normalizedWord.relations!.derived) {
      normalizedWord.relations!.derived = Array.isArray(wordData.relations.derived_from)
        ? wordData.relations.derived_from.map((rel: any) => {
            if (typeof rel === 'string') {
              return { word: rel };
            } else {
              return { word: rel.word || rel.to_word || '' };
            }
          })
        : [];
    }
    
    // Special handling for root_of -> root mapping
    if (wordData.relations.root_of) {
      const rootOf = wordData.relations.root_of;
      if (Array.isArray(rootOf) && rootOf.length > 0) {
        const rootRel = rootOf[0];
        normalizedWord.relations!.root = typeof rootRel === 'string'
          ? { word: rootRel }
          : { word: rootRel.word || rootRel.to_word || '' };
      }
    }
    
    // Handle root word
    if (wordData.relations.root && !normalizedWord.relations!.root) {
      const root = wordData.relations.root;
      if (typeof root === 'string') {
        normalizedWord.relations!.root = { word: root };
      } else if (root && typeof root === 'object') {
        normalizedWord.relations!.root = { 
          word: root.word || '' 
        };
      }
    }
  }
  
  // Handle affixations if they exist
  if (wordData.relations && wordData.relations.affixations) {
    (normalizedWord.relations as any).affixations = wordData.relations.affixations;
  }
  
  // Ensure all relations are initialized (even if empty)
  // This prevents undefined errors in the UI
  const allRelationTypes = [
    'synonyms', 'antonyms', 'variants', 'related', 'derived', 'root_of', 'component_of', 
    'cognate', 'main', 'derivative', 'etymology', 'associated', 'other', 'kaugnay', 'derived_from'
  ];
  
  allRelationTypes.forEach(type => {
    if (!normalizedWord.relations![type]) {
      normalizedWord.relations![type] = [];
    }
  });
  
  return normalizedWord as WordInfo;
}

export async function fetchWordDetails(
  word: string
): Promise<WordInfo> {
  const cacheKey = `wordDetails:${word}`;
  const cachedData = getCachedData<WordInfo>(cacheKey);
  
  if (cachedData) {
    console.log("Serving word details from cache:", word);
    return cachedData;
  }

  console.log("Fetching word details from API:", word);
  try {
    if (!circuitBreaker.canMakeRequest()) {
      throw new Error('Circuit breaker is open');
    }

    const response = await api.get<RawWordComprehensiveData>(`/words/${encodeURIComponent(word)}/comprehensive`);
    const rawData = response.data;

    if (!rawData || typeof rawData !== 'object' || !rawData.id || !rawData.lemma) {
      throw new Error('Invalid comprehensive word data received from API');
    }

    // --- Transformation Logic --- 

    // 1. Transform Definitions
    const cleanDefinitions: Definition[] = (rawData.definitions || []).map((def: RawDefinition): Definition => {
      const tags = toArray(def.tags);
      const sources = toArray(def.sources);
      const examples = toArray(def.examples);
      const usage_notes = toArray(def.usage_notes);
      const relations = (def.relations || []).map((rel: { id: number; type: string; word?: string; sources?: string | string[] }) => ({ 
        id: rel.id,
        type: rel.type,
        word: rel.word || '', 
        sources: toArray(rel.sources) 
      }));
      
      return {
        id: def.id,
        text: def.definition_text || def.text || '', 
        definition_text: def.definition_text || def.text,
        original_pos: def.original_pos,
        part_of_speech: def.standardized_pos || null,
        examples,
        usage_notes,
        tags,
        sources,
        relations,
        confidence_score: def.confidence_score || 0,
        is_verified: def.is_verified || false,
        verified_by: def.verified_by,
        verified_at: def.verified_at,
        created_at: def.created_at,
        updated_at: def.updated_at
      };
    });

    // 2. Transform Etymologies
    const cleanEtymologies: Etymology[] = (rawData.etymologies || []).map((etym: RawEtymology): Etymology => ({
      ...etym,
      text: etym.etymology_text || etym.text || '', // Ensure text exists
      components: toArray(etym.components), // Use helper
      languages: toArray(etym.language_codes), // Use helper (renamed)
      sources: toArray(etym.sources), // Use helper
      // Keep other fields like confidence_level, verification_status as is
    }));

    // 3. Transform Pronunciation (pick first one as primary)
    let cleanPronunciation: CleanPronunciation | null = null;
    if (rawData.pronunciations && rawData.pronunciations.length > 0) {
      const firstPron: Pronunciation = rawData.pronunciations[0]; // Use raw type here
      cleanPronunciation = {
        // Map fields from Pronunciation to CleanPronunciation
        id: firstPron.id,
        type: firstPron.type,
        tags: firstPron.tags,
        created_at: firstPron.created_at,
        updated_at: firstPron.updated_at,
        // Transformed fields
        text: firstPron.value || '',
        ipa: firstPron.type === 'ipa' ? firstPron.value : undefined,
        audio_url: firstPron.type === 'audio' ? firstPron.value : undefined,
        sources: toArray(firstPron.sources)
      };
    }

    // 4. Transform Relations (combine incoming/outgoing)
    const cleanRelations: CleanRelation[] = [
      ...(rawData.outgoing_relations || []),
      ...(rawData.incoming_relations || [])
    ].map((rel: Relation): CleanRelation => ({
      ...rel,
      sources: toArray(rel.sources)
    }));

    // 5. Transform Affixations (combine root/affixed)
    const cleanAffixations: CleanAffixation[] = [
      ...(rawData.root_affixations || []),
      ...(rawData.affixed_affixations || [])
    ].map((aff: Affixation): CleanAffixation => ({
      ...aff,
      sources: toArray(aff.sources)
    }));

    // 6. Build the final WordInfo object
    const wordInfo: WordInfo = {
      // --- Core Identifiers (Guaranteed) ---
      id: rawData.id,
      lemma: rawData.lemma,
      normalized_lemma: rawData.normalized_lemma || rawData.lemma, // Fallback
      language_code: rawData.language_code || 'unknown', // Fallback
      
      // --- Basic Word Properties (Mostly Guaranteed) ---
      has_baybayin: rawData.has_baybayin || false,
      baybayin_form: rawData.baybayin_form,
      romanized_form: rawData.romanized_form,
      tags: toArray(rawData.tags),
      created_at: rawData.created_at,
      updated_at: rawData.updated_at,
      data_hash: rawData.data_hash,
      
      // --- Spelling & Root Info ---
      preferred_spelling: rawData.preferred_spelling,
      root_word_id: rawData.root_word_id,
      root_word: rawData.root_word, // Already BasicWord or null

      // --- Pronunciation (Transformed) ---
      pronunciation: cleanPronunciation,
      
      // --- Definitions (Transformed) ---
      definitions: cleanDefinitions,
      
      // --- Etymology (Transformed) ---
      etymologies: cleanEtymologies,
      
      // --- Relations (Transformed & Structured) ---
      relations: buildRelationsObject(cleanRelations), // Structured object by type
      all_relations: cleanRelations, // Flat list
      relation_summary: rawData.relation_summary, // Counts by type
      
      // --- Affixations (Transformed) ---
      affixations: cleanAffixations, // Flat list
      derived_words: rawData.derived_words || [], // Words derived *from* this one
      
      // --- Other Associated Data ---
      credits: rawData.credits || [],
      idioms: (rawData.idioms || []).map((idiom: string | Idiom): Idiom => 
        typeof idiom === 'string' ? { text: idiom } : idiom
      ),
      source_info: rawData.source_info, // Raw source info
      search_text: rawData.search_text, // Raw search text

      // --- Metadata & Scores (Optional) ---
      data_completeness: rawData.data_completeness,
      complexity_score: rawData.complexity_score,
      usage_frequency: rawData.usage_frequency,
      last_lookup_at: rawData.last_lookup_at,
      view_count: rawData.view_count,
      last_viewed_at: rawData.last_viewed_at,
      is_verified: rawData.is_verified,
      verification_notes: rawData.verification_notes,
      data_quality_score: rawData.data_quality_score,
      
      // --- Additional Boolean Flags/Other fields ---
      is_proper_noun: rawData.is_proper_noun,
      is_abbreviation: rawData.is_abbreviation,
      is_initialism: rawData.is_initialism,
      is_root: rawData.is_root,
      badlit_form: rawData.badlit_form,
      hyphenation: rawData.hyphenation
    };
    
    // Optional: Clean up undefined fields if necessary
    // Object.keys(wordInfo).forEach(key => wordInfo[key] === undefined && delete wordInfo[key]);

    setCachedData(cacheKey, wordInfo);
    circuitBreaker.recordSuccess();
    console.log("Successfully fetched and transformed word details:", word);

    return wordInfo;
  } catch (error) {
    circuitBreaker.recordFailure();
    console.error(`Error fetching word details for \"${word}\":`, error);
    // Rethrow or return a custom error structure
    throw handleApiError(error, `fetching word details for "${word}"`); 
  }
}

// Helper function to build the structured Relations object
function buildRelationsObject(cleanRelations: CleanRelation[]): Relations {
  const relations: Relations = {};
  cleanRelations.forEach(rel => {
    const category = rel.relation_type as keyof Relations;
    const relatedWord: RelatedWord = {
      word: rel.target_word?.lemma || rel.source_word?.lemma || '' // Prefer target, fallback to source
      // No sources here as per the type definition
    };
    if (relatedWord.word) { // Only add if we have a word
      if (!relations[category]) {
        relations[category] = [];
      }
      // Check if it's an array before pushing
      const relationList = relations[category];
      if (Array.isArray(relationList)) {
         relationList.push(relatedWord);
      }
      // Handle potential single root object (though unlikely with this logic)
      else if(category === 'root' && !relationList) {
          relations[category] = relatedWord;
      }
    }
  });
  return relations;
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