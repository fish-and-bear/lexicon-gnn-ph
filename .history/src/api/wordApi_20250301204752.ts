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
  }

  recordSuccess() {
    this.failures = 0;
    this.state = 'closed';
    this.saveState();
  }

  canMakeRequest(): boolean {
    // Clean up old cache before checking state
    clearOldCache();
    
    // TEMPORARY FIX: Always allow requests regardless of circuit breaker state
    console.log('Circuit breaker state:', this.state, 'but allowing request anyway');
    return true;
    
    /* Original implementation:
    if (this.state === 'closed') return true;
    if (this.state === 'open' && this.lastFailureTime && 
        Date.now() - this.lastFailureTime > CONFIG.resetTimeout) {
      this.state = 'half-open';
      this.saveState();
      return true;
    }
    return this.state === 'half-open';
    */
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
    
    // Clear axios cache
    api.defaults.headers.common = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Client-Version': process.env.REACT_APP_VERSION || '1.0.0',
      'X-Client-Platform': 'web',
      'Accept-Encoding': 'gzip, deflate, br',
      'Origin': window.location.origin
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
        
        // Handle the new API response format which wraps data in a data property
        // with meta information
        if (parsed.data !== undefined && parsed.meta !== undefined) {
          return parsed.data;
        }
        
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
    const response = await api.get<WordNetwork>(`/words/${encodedWord}/related`, { 
      params: { 
        depth: sanitizedDepth, 
        include_affixes,
        include_etymology,
        cluster_threshold
      }
    });

    const networkData = response.data;
    if (!networkData.nodes || !networkData.clusters || !networkData.metadata) {
      throw new Error('Invalid network data received from server');
    }

    setCachedData<WordNetwork>(cacheKey, networkData);
    return networkData;
  } catch (error) {
    return handleApiError(error, 'fetching word network');
  }
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
    const response = await api.get<WordInfo>(`/words/${encodedWord}`, {
      params: {
        include_definitions,
        include_relations,
        include_etymology
      }
    });
    
    const wordData = response.data;
    if (!wordData.lemma || !wordData.normalized_lemma || !wordData.language_code) {
      throw new Error('Invalid word data received from server');
    }

    setCachedData<WordInfo>(cacheKey, wordData);
    return wordData;
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

    const response = await api.get<SearchResult>('/search', {
      params: {
        q: sanitizedQuery,
        page: Math.max(1, options.page),
        per_page: Math.min(100, Math.max(1, options.per_page)),
        pos: options.pos,
        language: options.language || 'tl',
        include_baybayin: !options.exclude_baybayin,
        min_similarity: 0.3,
        mode: options.mode || 'all',
        sort: options.sort || 'relevance',
        order: options.order || 'desc'
      }
    });

    // Transform the response to match the expected format
    const data = response.data;
    return {
      words: Array.isArray(data) ? data : (data.words || []),
      page: data.page || options.page,
      perPage: data.perPage || options.per_page || options.per_page,
      total: data.total || 0
    };
  } catch (error) {
    return handleApiError(error, 'searching words');
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
    const response = await api.get<EtymologyTree>(`/words/${encodedWord}/etymology-tree`, {
      params: {
        max_depth: Math.min(5, Math.max(1, maxDepth)),
        include_uncertain: includeUncertain,
        group_by_language: groupByLanguage
      }
    });

    const data = response.data;
    if (!data.word || !data.components) {
      throw new Error('Invalid etymology tree data received from server');
    }

    setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    return handleApiError(error, 'fetching etymology tree');
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
    
    if (!data.words || !data.definitions || !data.relations || !data.sources) {
      throw new Error('Invalid statistics data received from server');
    }

    setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    return handleApiError(error, 'fetching statistics');
  }
}

// Export types for better type safety
export type { 
  WordNetwork,
  WordInfo,
  SearchOptions,
  SearchResult,
  Etymology,
  PartOfSpeech,
  Statistics,
  EtymologyTree
};

// Function to update API client configuration with a successful endpoint
function updateApiClient(endpoint: string) {
  let apiUrl: string;
  
  if (endpoint.includes('/api/v2')) {
    apiUrl = endpoint;
  } else {
    apiUrl = `${endpoint}/api/v2`;
  }
  
  console.log(`Updating API client baseURL to: ${apiUrl}`);
  
  // Update the axios instance baseURL
  api.defaults.baseURL = apiUrl;
  
  // Store for future use
  localStorage.setItem('successful_api_endpoint', endpoint);
  
  return apiUrl;
}

// Function to test API connectivity
export async function testApiConnection(): Promise<boolean> {
  // Try multiple possible API endpoints with exact URLs from the backend logs
  const possibleEndpoints = [
    'http://127.0.0.1:10000',            // Local server root (exact match from backend logs)
    'http://localhost:10000',            // Local server root with localhost
    'http://127.0.0.1:10000/api/v2',     // Local API path with IP
    'http://localhost:10000/api/v2',     // Local API path with localhost
    CONFIG.baseURL,                      // Configured API URL
    CONFIG.baseURL.replace('/api/v2', '') // Root without API path
  ];
  
  console.log('Testing API connection with endpoints:', possibleEndpoints);
  
  for (const endpoint of possibleEndpoints) {
    try {
      console.log(`Trying to connect to: ${endpoint}`);
      
      // Use fetch API for simpler CORS handling
      const response = await fetch(endpoint, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Origin': window.location.origin
        },
        mode: 'cors'  // Important for CORS
      });
      
      console.log(`Response from ${endpoint}:`, {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok
      });
      
      if (response.ok) {
        console.log('API connection test successful with endpoint:', endpoint);
        
        // Update the API client with the successful endpoint
        updateApiClient(endpoint);
        
        return true;
      }
    } catch (error) {
      console.error(`API connection test error for ${endpoint}:`, error);
    }
  }
  
  console.error('All API connection tests failed');
  return false;
}