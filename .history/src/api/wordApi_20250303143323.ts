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
  WordNetworkGraph
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

export async function fetchWordNetwork(word: string, options?: WordNetworkOptions): Promise<WordNetworkGraph> {
  try {
    console.log('Fetching word network for:', word, 'with options:', options);
    
    const sanitizedWord = sanitizeInput(word);
    if (!sanitizedWord) {
      throw new Error('Word is required');
    }

    const response = await api.get(`/words/${encodeURIComponent(sanitizedWord)}/network`, {
      params: {
        depth: options?.depth || 2,
        include_affixes: options?.include_affixes ?? true,
        include_etymology: options?.include_etymology ?? true,
        cluster_threshold: options?.cluster_threshold || 0.3
      }
    });

    console.log('Raw API response:', response.data);

    // Handle both data wrapper and direct response formats
    const rawData = response.data?.data || response.data;

    if (!rawData) {
      throw new Error('No data received from API');
    }

    // Transform the data to match our WordNetworkGraph type
    const transformedData: WordNetworkGraph = {
      word: word,
      nodes: rawData.nodes?.map((node: any) => ({
        id: node.id || Math.random().toString(),
        word: node.word || node.lemma || '',
        normalized_lemma: node.normalized_lemma || node.word || '',
        language: node.language || 'tl',
        has_baybayin: node.has_baybayin || false,
        baybayin_form: node.baybayin_form || null,
        type: node.type || node.relation_type || 'other',
        path: node.path || [],
        definitions: node.definitions || []
      })) || [],
      edges: rawData.edges?.map((edge: any) => ({
        source: edge.source || edge.from_id || '',
        target: edge.target || edge.to_id || '',
        type: edge.type || edge.relation_type || 'other',
        weight: edge.weight || 1
      })) || [],
      clusters: {
        etymology: rawData.clusters?.etymology || [],
        affixes: rawData.clusters?.affixes || [],
        synonyms: rawData.clusters?.synonyms || [],
        antonyms: rawData.clusters?.antonyms || [],
        variants: rawData.clusters?.variants || [],
        root_words: rawData.clusters?.root_words || [],
        derived_words: rawData.clusters?.derived_words || []
      },
      metadata: {
        root_word: rawData.metadata?.root_word || word,
        max_depth: rawData.metadata?.max_depth || options?.depth || 2,
        total_nodes: rawData.metadata?.total_nodes || 0,
        total_edges: rawData.metadata?.total_edges || 0,
        cluster_count: rawData.metadata?.cluster_count || 0
      }
    };

    console.log('Transformed network data:', transformedData);
    return transformedData;
  } catch (error) {
    console.error('Error fetching word network:', error);
    // Return a minimal valid network structure
    return {
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
        max_depth: options?.depth || 2,
        total_nodes: 0,
        total_edges: 0,
        cluster_count: 0
      }
    };
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
  
  console.log('Normalizing word data:', wordData);
  
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
    pronunciation: wordData.pronunciation || null,
    definitions: [],
    etymologies: [],
    relations: {
      synonyms: [],
      antonyms: [],
      variants: [],
      related: [],
      kaugnay: [],
      derived: [],
      derived_from: [],
      root_of: [],
      component_of: [],
      cognate: [],
      main: [],
      derivative: [],
      etymology: [],
      associated: [],
      other: []
    },
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
    idioms: wordData.idioms || []
  };

  // Normalize definitions
  if (Array.isArray(wordData.definitions)) {
    normalizedWord.definitions = wordData.definitions.map((def: any) => ({
      id: def.id || 0,
      text: def.definition_text || def.text || def.definition || '',
      definition_text: def.definition_text || def.text || def.definition || '',
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
  }

  // Normalize etymologies
  if (Array.isArray(wordData.etymologies)) {
    normalizedWord.etymologies = wordData.etymologies.map((etym: any) => ({
      id: etym.id || 0,
      text: etym.etymology_text || etym.text || '',
      etymology_text: etym.etymology_text || etym.text || '',
      components: etym.components || [],
      languages: etym.languages || [],
      sources: etym.sources || [],
      confidence_level: etym.confidence_level || 'medium',
      verification_status: etym.verification_status || 'unverified',
      verification_notes: etym.verification_notes || undefined
    }));
  }

  // Normalize relations
  const relations = wordData.relations || wordData.word_relations || {};
  console.log('Processing relations:', relations);

  // Helper function to normalize relation array
  const normalizeRelations = (relArray: any[]): Array<{ word: string; sources: string[] }> => {
    return relArray.map(rel => {
      if (typeof rel === 'string') {
        return { word: rel, sources: [] };
      }
      return {
        word: rel.word || rel.lemma || rel.to_word || '',
        sources: rel.sources || []
      };
    });
  };

  // Process each relation type
  Object.entries(relations).forEach(([type, words]) => {
    if (Array.isArray(words) && words.length > 0) {
      const normalizedType = type.toLowerCase().replace(/-/g, '_');
      if (normalizedWord.relations![normalizedType] !== undefined) {
        normalizedWord.relations![normalizedType] = normalizeRelations(words);
      } else if (normalizedType === 'root') {
        // Special handling for root relation
        normalizedWord.relations!.root = normalizeRelations(words)[0];
      }
    }
  });

  // Additional processing for network-style relations if they exist
  if (wordData.network) {
    const network = wordData.network;
    if (network.nodes) {
      network.nodes.forEach((node: any) => {
        const type = node.type?.toLowerCase() || 'other';
        if (normalizedWord.relations![type]) {
          normalizedWord.relations![type].push({
            word: node.word,
            sources: node.sources || []
          });
        }
      });
    }
  }

  console.log('Normalized word data:', normalizedWord);
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
}
