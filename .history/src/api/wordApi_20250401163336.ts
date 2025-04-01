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
  word: string,
  include_definitions: boolean = true,
  include_relations: boolean = true,
  include_etymology: boolean = true
): Promise<WordInfo> {
  const cacheKey = `cache:word:${word}:${include_definitions}:${include_relations}:${include_etymology}`;
  const cachedData = getCachedData<WordInfo>(cacheKey);
  
  if (cachedData) {
    return cachedData;
  }

  try {
    if (!circuitBreaker.canMakeRequest()) {
      throw new Error('Circuit breaker is open');
    }

    const response = await api.get(`/words/${encodeURIComponent(word)}/comprehensive`);
    const data = response.data;

    // Transform the comprehensive response into WordInfo format
    const wordInfo: WordInfo = {
      id: data.id,
      lemma: data.lemma,
      normalized_lemma: data.normalized_lemma,
      language_code: data.language_code,
      preferred_spelling: data.preferred_spelling,
      tags: Array.isArray(data.tags) ? data.tags : (data.tags ? data.tags.split(',') : []),
      has_baybayin: data.has_baybayin || false,
      baybayin_form: data.baybayin_form,
      romanized_form: data.romanized_form,
      pronunciation: data.pronunciations?.[0] ? {
        text: data.pronunciations[0].value || data.pronunciations[0].text,
        ipa: data.pronunciations[0].ipa,
        audio_url: data.pronunciations[0].audio_url
      } : null,
      source_info: data.source_info || {},
      data_hash: data.data_hash || '',
      complexity_score: data.complexity_score || 0,
      usage_frequency: data.usage_frequency || 0,
      created_at: data.created_at,
      updated_at: data.updated_at,
      last_lookup_at: data.last_lookup_at,
      view_count: data.view_count || 0,
      last_viewed_at: data.last_viewed_at,
      is_verified: data.is_verified || false,
      verification_notes: data.verification_notes,
      data_quality_score: data.data_quality_score || 0,
      definitions: (data.definitions || []).map((def: any) => ({
        id: def.id,
        text: def.definition_text || def.text,
        definition_text: def.definition_text || def.text,
        original_pos: def.original_pos,
        part_of_speech: def.standardized_pos,
        examples: Array.isArray(def.examples) ? def.examples : (def.examples ? def.examples.split(';') : []),
        usage_notes: Array.isArray(def.usage_notes) ? def.usage_notes : (def.usage_notes ? def.usage_notes.split(';') : []),
        sources: Array.isArray(def.sources) ? def.sources : (def.sources ? def.sources.split(',').map((s: string): string => s.trim()) : []),
        relations: Array.isArray(def.relations) ? def.relations : [],
        confidence_score: def.confidence_score || 0,
        is_verified: def.is_verified || false,
        verified_by: def.verified_by,
        verified_at: def.verified_at
      })),
      etymologies: (data.etymologies || []).map((etym: any) => ({
        id: etym.id,
        text: etym.etymology_text || etym.text,
        etymology_text: etym.etymology_text || etym.text,
        components: Array.isArray(etym.components) ? etym.components : [],
        languages: Array.isArray(etym.language_codes) ? etym.language_codes : (etym.language_codes ? etym.language_codes.split(',') : []),
        sources: Array.isArray(etym.sources) ? etym.sources : (etym.sources ? etym.sources.split(',').map((s: string): string => s.trim()) : []),
        confidence_level: etym.confidence_level || 'medium',
        verification_status: etym.verification_status || 'unverified',
        verification_notes: etym.verification_notes
      })),
      relations: data.relations || {},
      idioms: Array.isArray(data.idioms) ? data.idioms : []
    };

    // Cache the transformed data
    setCachedData(cacheKey, wordInfo);
    circuitBreaker.recordSuccess();

    return wordInfo;
  } catch (error) {
    circuitBreaker.recordFailure();
    return handleApiError(error, `Error fetching word details for "${word}"`);
  }
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

    const response = await api.get('/search', {
      params: {
        q: query,
        page: options.page,
        per_page: options.per_page,
        exclude_baybayin: options.exclude_baybayin,
        pos: options.pos,
        source: options.source,
        language: options.language,
        mode: options.mode,
        sort: options.sort,
        order: options.order,
        is_real_word: options.is_real_word
      }
    });

    const data = response.data;
    
    // Transform the response into SearchResult format
    const searchResult: SearchResult = {
      words: data.results.map((result: any) => ({
        id: result.id,
        word: result.lemma
      })),
      page: data.page,
      perPage: data.per_page,
      total: data.total
    };

    // Cache the transformed data
    setCachedData(cacheKey, searchResult);
    circuitBreaker.recordSuccess();

    return searchResult;
  } catch (error) {
    circuitBreaker.recordFailure();
    return handleApiError(error, `Error searching words with query "${query}"`);
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
  word: string,
  maxDepth: number = 3,
  includeUncertain: boolean = false,
  groupByLanguage: boolean = true
): Promise<EtymologyTree> {
  const cacheKey = `cache:etymology:tree:${word}:${maxDepth}:${includeUncertain}:${groupByLanguage}`;
  const cachedData = getCachedData<EtymologyTree>(cacheKey);

  if (cachedData) {
    return cachedData;
  }

  try {
    if (!circuitBreaker.canMakeRequest()) {
      throw new Error('Circuit breaker is open');
    }

    const response = await api.get(`/words/${encodeURIComponent(word)}/etymology/tree`, {
      params: {
        max_depth: maxDepth,
        include_uncertain: includeUncertain,
        group_by_language: groupByLanguage
      }
    });

    const data = response.data;
    
    // Transform and validate the response
    const etymologyTree: EtymologyTree = {
      id: data.id,
      word: data.word,
      normalized_lemma: data.normalized_lemma,
      language: data.language,
      has_baybayin: data.has_baybayin || false,
      baybayin_form: data.baybayin_form,
      romanized_form: data.romanized_form,
      etymologies: data.etymologies?.map((etym: any) => ({
        id: etym.id,
        text: etym.text,
        languages: etym.languages || [],
        sources: etym.sources || []
      })) || [],
      components: data.components || [],
      component_words: data.component_words?.map((comp: any) => ({
        id: comp.id,
        word: comp.word,
        normalized_lemma: comp.normalized_lemma,
        language: comp.language,
        has_baybayin: comp.has_baybayin || false,
        baybayin_form: comp.baybayin_form,
        romanized_form: comp.romanized_form,
        etymologies: comp.etymologies?.map((etym: any) => ({
          id: etym.id,
          text: etym.text,
          languages: etym.languages || [],
          sources: etym.sources || []
        })) || [],
        components: comp.components || [],
        component_words: comp.component_words || []
      })) || [],
      metadata: {
        word: data.word,
        normalized_lemma: data.normalized_lemma,
        language: data.language,
        max_depth: maxDepth,
        group_by_language: groupByLanguage
      }
    };

    // Cache the transformed data
    setCachedData(cacheKey, etymologyTree);
    circuitBreaker.recordSuccess();

    return etymologyTree;
  } catch (error) {
    circuitBreaker.recordFailure();
    return handleApiError(error, `Error fetching etymology tree for "${word}"`);
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

    const response = await api.get('/test');
    const data = response.data;

    // Transform the response into Statistics format
    const statistics: Statistics = {
      words: {
        total: data.words.total,
        with_definitions: data.words.with_definitions,
        with_etymology: data.words.with_etymology,
        with_baybayin: data.words.with_baybayin,
        by_language: data.words.by_language || {}
      },
      definitions: {
        total: data.definitions.total,
        verified: data.definitions.verified,
        by_pos: data.definitions.by_pos || {}
      },
      relations: {
        total: data.relations.total,
        by_type: data.relations.by_type || {}
      },
      sources: {
        total: data.sources.total,
        by_name: data.sources.by_name || {}
      },
      last_updated: data.last_updated
    };

    // Cache the transformed data
    setCachedData(cacheKey, statistics);
    circuitBreaker.recordSuccess();

    return statistics;
  } catch (error) {
    circuitBreaker.recordFailure();
    return handleApiError(error, 'Error fetching statistics');
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