import axios, { AxiosError, AxiosResponse, AxiosRequestConfig } from 'axios';
import { 
  WordNetwork, 
  WordInfo, 
  SearchOptions, 
  SearchResult, 
  Etymology,
  PartOfSpeech,
  Statistics,
  EtymologyTree
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
  baseURL: apiBaseURL,
  timeout: CONFIG.timeout,
  withCredentials: false, // Don't send cookies with cross-origin requests
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Client-Version': process.env.REACT_APP_VERSION || '1.0.0',
    'X-Client-Platform': 'web',
    'Accept-Encoding': 'gzip, deflate, br',
    'Origin': window.location.origin
  },
  // Add CORS headers
  validateStatus: (status) => status >= 200 && status < 300,
  transformRequest: [
    (data, headers) => {
      // Add request ID for tracking
      headers['X-Request-ID'] = Math.random().toString(36).substring(7);
      // Log the request for debugging
      console.log('Making request with headers:', headers);
      return JSON.stringify(data);
    }
  ],
  transformResponse: [
    (data) => {
      try {
        // Log raw response for debugging
        console.log('Raw API response:', data);
        
        // Handle empty response
        if (!data || data.trim() === '') {
          console.log('Empty response received, returning empty object');
          return {};
        }
        
        // Validate response structure
        const parsed = JSON.parse(data);
        if (!parsed || typeof parsed !== 'object') {
          throw new Error('Invalid response format');
        }
        
        // Return the full response object to allow individual functions to handle the data wrapper
        return parsed;
      } catch (e) {
        console.error('Error parsing response:', e);
        console.error('Raw response:', data);
        throw e;
      }
    }
  ]
});

// Request interceptor for circuit breaker
api.interceptors.request.use(
  async (config: ExtendedAxiosRequestConfig) => {
    if (!circuitBreaker.canMakeRequest()) {
      throw new Error('Circuit breaker is open');
    }
    
    // Debug logging
    console.log(`Making API request to: ${config.url}`, config);
    
    return config;
  },
  error => Promise.reject(error)
);

// Response interceptor for error handling and circuit breaker
api.interceptors.response.use(
  response => {
    circuitBreaker.recordSuccess();
    console.log(`Successful response from: ${response.config.url}`, response.data);
    return response;
  },
  async (error) => {
    console.error('API error:', error);
    
    if (axios.isAxiosError(error)) {
      console.error('Axios error details:', {
        message: error.message,
        code: error.code,
        config: error.config,
        response: error.response ? {
          status: error.response.status,
          statusText: error.response.statusText,
          data: error.response.data
        } : 'No response'
      });
      
      circuitBreaker.recordFailure();
      
      const config = error.config as ExtendedAxiosRequestConfig;
      if (!config || !config.retry) {
        config.retry = 0;
      }

      if (config.retry >= CONFIG.retries) {
        return Promise.reject(error);
      }

      config.retry += 1;
      const delay = Math.min(
        CONFIG.retryDelay * Math.pow(2, config.retry - 1),
        CONFIG.maxRetryDelay
      );
      
      // Only retry on network errors or 5xx responses
      if (!error.response || (error.response.status >= 500 && error.response.status <= 599)) {
        console.log(`Retrying request to ${config.url} (attempt ${config.retry}/${CONFIG.retries}) after ${delay}ms`);
        await new Promise(resolve => setTimeout(resolve, delay));
        return api(config);
      }
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
      root: null,
      derived: []
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
      id: def.id || 0,
      text: def.definition_text || def.text || def.definition || def.meaning || '',
      definition_text: def.definition_text || def.text || def.definition || def.meaning || '',
      original_pos: def.original_pos || null,
      part_of_speech: def.part_of_speech || null,
      examples: def.examples || [],
      usage_notes: def.usage_notes || [],
      sources: def.sources || [],
      relations: def.relations || [],
      confidence_score: def.confidence_score || 0,
      is_verified: def.is_verified || false,
      verified_by: def.verified_by || undefined,
      verified_at: def.verified_at || undefined
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
      sources: [],
      relations: [],
      confidence_score: 0,
      is_verified: false
    }];
  }
  
  // Normalize etymologies
  if (wordData.etymologies && Array.isArray(wordData.etymologies)) {
    normalizedWord.etymologies = wordData.etymologies.map((etym: any) => ({
      id: etym.id || 0,
      text: etym.etymology_text || etym.text || '',
      etymology_text: etym.etymology_text || etym.text || '',
      components: etym.components || [],
      languages: etym.language_codes || etym.languages || [],
      sources: etym.sources || [],
      confidence_level: etym.confidence_level || 'medium',
      verification_status: etym.verification_status || 'unverified',
      verification_notes: etym.verification_notes || undefined
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
      // Primary relation types from backend
      'synonyms', 'antonyms', 'variants', 'related', 
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
                return { word: rel, sources: [] };
              } else {
                return { word: rel.word || rel.to_word || '', sources: rel.sources || [] };
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
              return { word: rel, sources: [] };
            } else {
              return { word: rel.word || rel.to_word || '', sources: rel.sources || [] };
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
          ? { word: rootRel, sources: [] }
          : { word: rootRel.word || rootRel.to_word || '', sources: rootRel.sources || [] };
      }
    }
    
    // Handle root word
    if (wordData.relations.root && !normalizedWord.relations!.root) {
      const root = wordData.relations.root;
      if (typeof root === 'string') {
        normalizedWord.relations!.root = { word: root, sources: [] };
      } else if (root && typeof root === 'object') {
        normalizedWord.relations!.root = { 
          word: root.word || '', 
          sources: root.sources || [] 
        };
      }
    }
  }
  
  // Handle affixations if they exist
  if (wordData.relations && wordData.relations.affixations) {
    (normalizedWord.relations as any).affixations = wordData.relations.affixations;
  }
  
  // Handle derived words if they exist
  if (wordData.relations && wordData.relations.derived) {
    normalizedWord.relations!.derived = wordData.relations.derived;
  }
  
  return normalizedWord as WordInfo;
}

export async function fetchWordDetails(
  word: string,
  include_definitions: boolean = true,
  include_relations: boolean = true,
  include_etymology: boolean = true
): Promise<WordInfo> {
  try {
    const sanitizedWord = sanitizeInput(word);
    if (!sanitizedWord) {
      throw new Error('Word is required');
    }

    const cacheKey = `wordDetails-${sanitizedWord}-${include_definitions}-${include_relations}-${include_etymology}`;
    const cachedData = getCachedData<WordInfo>(cacheKey);
    if (cachedData) {
      return cachedData;
    }

    const encodedWord = encodeURIComponent(sanitizedWord);
    const response = await api.get<any>(`/words/${encodedWord}`, {
      params: {
        include_definitions,
        include_relations,
        include_etymology
      }
    });
    
    // Normalize the API response to ensure consistent structure
    const normalizedWordData = normalizeWordData(response.data);
    
    // Validate the normalized data
    if (!normalizedWordData.lemma || !normalizedWordData.normalized_lemma) {
      throw new Error('Invalid word data received from server');
    }

    // Cache the normalized data
    setCachedData<WordInfo>(cacheKey, normalizedWordData);
    return normalizedWordData;
  } catch (error) {
    return handleApiError(error, 'fetching word details');
  }
}

export async function searchWords(query: string, options: SearchOptions): Promise<SearchResult> {
  try {
    const sanitizedQuery = sanitizeInput(query);
    if (!sanitizedQuery) {
      throw new Error('Search query is required');
    }

    // Create a clean params object with only valid parameters
    const params: Record<string, any> = {
      q: sanitizedQuery,
      limit: Math.min(100, Math.max(1, options.per_page || 20)),
      language: options.language || 'tl',
      include_baybayin: options.exclude_baybayin === true ? false : true,
      min_similarity: 0.3,
      mode: options.mode || 'all',
      sort: options.sort || 'relevance',
      order: options.order || 'desc'
    };

    // Only add optional parameters if they are defined
    if (options.page && options.page > 0) {
      params.page = options.page;
    }
    
    if (options.pos) {
      params.pos = options.pos;
    }

    console.log('Search params:', params);

    const response = await api.get<SearchResult>('/search', { params });

    // Transform the response to match the expected format
    const data = response.data;
    
    // Handle different response formats
    if (Array.isArray(data)) {
      return {
        words: data,
        page: options.page || 1,
        perPage: options.per_page || 20,
        total: data.length
      };
    } else if (data.words) {
      return {
        words: data.words,
        page: data.page || options.page || 1,
        perPage: data.perPage || options.per_page || 20,
        total: data.total || data.words.length
      };
    } else {
      // If the response doesn't match expected format, create a compatible structure
      return {
        words: [],
        page: options.page || 1,
        perPage: options.per_page || 20,
        total: 0
      };
    }
  } catch (error) {
    console.error('Search error:', error);
    // Return empty result instead of throwing error to improve user experience
    return {
      words: [],
      page: options.page || 1,
      perPage: options.per_page || 20,
      total: 0
    };
  }
}

export async function getPartsOfSpeech(): Promise<PartOfSpeech[]> {
  try {
    const cacheKey = 'parts-of-speech';
    const cachedData = getCachedData<PartOfSpeech[]>(cacheKey);
    if (cachedData) {
      return cachedData;
    }

    const response = await api.get<PartOfSpeech[]>('/parts-of-speech');
    const data = response.data;
    
    if (!Array.isArray(data) || !data.every(pos => pos.code && pos.name_en && pos.name_tl)) {
      throw new Error('Invalid parts of speech data received from server');
    }

    setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    return handleApiError(error, 'fetching parts of speech');
  }
}

export async function testApiConnection(): Promise<boolean> {
  try {
    console.log('Testing API connection...');
    // Try to connect to the base URL without the /api/v2 path
    const baseUrl = CONFIG.baseURL.replace('/api/v2', '');
    
    const response = await fetch(baseUrl, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      mode: 'cors',
      cache: 'no-cache'
    });
    
    if (response.ok) {
      console.log('API connection successful!');
      // Store the successful endpoint for future use
      localStorage.setItem('successful_api_endpoint', baseUrl);
      return true;
    } else {
      console.error(`API connection failed: ${response.status} ${response.statusText}`);
      return false;
    }
  } catch (error) {
    console.error('API connection error:', error);
    return false;
  }
}

export async function getEtymologyTree(
  word: string,
  maxDepth: number = 3,
  includeUncertain: boolean = false,
  groupByLanguage: boolean = true
): Promise<EtymologyTree> {
  try {
    const sanitizedWord = sanitizeInput(word);
    if (!sanitizedWord) {
      throw new Error('Word is required');
    }

    const cacheKey = `etymology-tree-${sanitizedWord}-${maxDepth}-${includeUncertain}-${groupByLanguage}`;
    
    const cachedData = getCachedData<EtymologyTree>(cacheKey);
    if (cachedData) {
      return cachedData;
    }

    const encodedWord = encodeURIComponent(sanitizedWord);
    const response = await api.get<any>(`/words/${encodedWord}/etymology-tree`, {
      params: {
        max_depth: maxDepth,
        include_uncertain: includeUncertain,
        group_by_language: groupByLanguage
      }
    });

    // Handle the API response format with data wrapper
    let data: EtymologyTree;
    
    if (response.data && response.data.data) {
      // New API format with data wrapper
      data = response.data.data;
      console.log('Using etymology tree data from response.data.data');
    } else {
      // Old format or direct data
      data = response.data;
      console.log('Using etymology tree data directly from response.data');
    }
    
    setCachedData<EtymologyTree>(cacheKey, data);
    return data;
  } catch (error) {
    return handleApiError(error, 'fetching etymology tree');
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
  try {
    const cacheKey = 'statistics';
    
    const cachedData = getCachedData<Statistics>(cacheKey);
    if (cachedData) {
      return cachedData;
    }

    const response = await api.get<Statistics>('/statistics');
    const data = response.data;
    
    setCachedData<Statistics>(cacheKey, data);
    return data;
  } catch (error) {
    return handleApiError(error, 'fetching statistics');
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