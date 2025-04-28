import axios, { AxiosError, AxiosInstance } from 'axios';
import { 
  WordInfo, 
  SearchOptions, 
  SearchResults, 
  Etymology,
  PartOfSpeech,
  Statistics,
  EtymologyTree,
  RawDefinition,
  Credit,
  SearchResultItem,
  RawWordComprehensiveData,
  Pronunciation,
  Relation,
  Affixation,
  BasicWord,
  WordNetwork as ImportedWordNetwork,
  WordSuggestion // Import the correct type
} from "../types";
import { getCachedData, setCachedData, clearCache, clearOldCache } from '../utils/caching';

// --- Simple Rate Limiter --- 
function rateLimit(axiosInstance: AxiosInstance, { maxRequests, perMilliseconds }: { maxRequests: number; perMilliseconds: number }): AxiosInstance {
  let requests: number[] = [];

  axiosInstance.interceptors.request.use(config => {
    const now = Date.now();
    // Remove requests older than the time window
    requests = requests.filter(timestamp => now - timestamp < perMilliseconds);

    if (requests.length >= maxRequests) {
      // Too many requests, reject the new one
      console.warn(`Rate limit exceeded: ${requests.length}/${maxRequests} requests in ${perMilliseconds}ms. Rejecting request.`);
      return Promise.reject(new Error('Rate limit exceeded'));
    } else {
      // Add the current request timestamp
      requests.push(now);
      return config;
    }
  });

  return axiosInstance;
}
// --- End Rate Limiter ---

// Define cache expiration time (copy from caching.ts or use a shared constant)
const CACHE_EXPIRATION = 5 * 60 * 1000; // 5 minutes

// Environment and configuration constants
const ENV = import.meta.env.MODE || 'development'; // Use Vite's MODE

// Define the type for the config structure explicitly
type AppConfig = {
  baseURL: string;
  timeout: number;
  retries: number;
  failureThreshold: number;
  resetTimeout: number;
  retryDelay: number;
  maxRetryDelay: number;
};

// Define the full configuration map
const configMap: { [key: string]: AppConfig } = {
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
    baseURL: import.meta.env.VITE_API_BASE_URL?.replace('/api/v1', '/api/v2') || 'https://fil-relex.onrender.com/api/v2',
    timeout: 15000,
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
};

// Get the configuration for the current environment, defaulting to development
const CONFIG: AppConfig = configMap[ENV] || configMap.development;

// Check if baseURL is valid after determining CONFIG
if (!CONFIG.baseURL) {
  throw new Error('API_BASE_URL could not be determined for the current environment');
}

// --- Lazy Initialized Circuit Breaker ---
class PersistentCircuitBreaker {
  private static readonly STORAGE_KEY = 'circuit_breaker_state';
  private static readonly STATE_TTL = 60 * 60 * 1000; // 1 hour

  private failures: number = 0;
  private lastFailureTime: number | null = null;
  private state: 'closed' | 'open' | 'half-open' = 'closed';
  private isInitialized: boolean = false; // Flag to track initialization

  constructor() {
    // Defer initialization
  }

  private initializeIfNeeded() {
    if (this.isInitialized) return;

    const savedState = this.loadState();
    if (savedState && Date.now() - savedState.timestamp < PersistentCircuitBreaker.STATE_TTL) {
      this.failures = savedState.failures;
      this.lastFailureTime = savedState.lastFailureTime;
      this.state = savedState.state;
      console.log('Circuit breaker initialized from saved state:', this.getState());
    } else {
      this.resetState(); // Reset internal state variables
      console.log('Circuit breaker initialized with default state.');
    }
    this.isInitialized = true;
  }

  private loadState() {
    try {
      // Check if localStorage is available
      if (typeof localStorage === 'undefined') {
        console.warn('localStorage not available for circuit breaker state.');
        return null;
      }
      const state = localStorage.getItem(PersistentCircuitBreaker.STORAGE_KEY);
      return state ? JSON.parse(state) : null;
    } catch (e) {
      console.error('Error loading circuit breaker state:', e);
      return null;
    }
  }

  private saveState() {
    if (!this.isInitialized) return; // Don't save if not initialized
    try {
      // Check if localStorage is available
      if (typeof localStorage === 'undefined') return;

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
    this.initializeIfNeeded();
    this.failures++;
    this.lastFailureTime = Date.now();
    if (this.failures >= CONFIG.failureThreshold) {
      this.state = 'open';
    }
    this.saveState();
    console.log('Circuit breaker recorded failure. New state:', this.getState());
  }

  recordSuccess() {
    this.initializeIfNeeded();
    // Only reset if state was not already closed
    if (this.state !== 'closed' || this.failures > 0) {
        this.failures = 0;
        this.state = 'closed';
        this.saveState();
        console.log('Circuit breaker recorded success. New state:', this.getState());
    }
  }

  canMakeRequest(): boolean {
    this.initializeIfNeeded(); 
    clearOldCache(); // Assuming clearOldCache also checks for localStorage
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
    this.initializeIfNeeded();
    return {
      state: this.state,
      failures: this.failures,
      lastFailureTime: this.lastFailureTime
    };
  }

  // Renamed from reset to avoid conflict, just resets internal state vars
  private resetState() {
    this.failures = 0;
    this.lastFailureTime = null;
    this.state = 'closed';
  }

  // Public reset function also clears storage
  reset() {
    this.resetState();
    this.isInitialized = true; // Mark as initialized after reset
    try {
       if (typeof localStorage !== 'undefined') {
         localStorage.removeItem(PersistentCircuitBreaker.STORAGE_KEY);
       }
    } catch (e) {
        console.error('Error removing circuit breaker state from storage:', e);
    }
    this.saveState(); // Save the reset state
    console.log('Circuit breaker has been reset. New state:', this.getState());
  }
}

// Instantiate the circuit breaker (constructor does nothing now)
const circuitBreaker = new PersistentCircuitBreaker(); 

// Function to reset the circuit breaker state (adjust to handle lazy init)
export function resetCircuitBreaker() {
  circuitBreaker.reset(); // Call the public reset method
  try {
    // Check localStorage availability
    if (typeof localStorage === 'undefined') {
        console.warn('localStorage not available, cannot clear related items.');
        return;
    }
    localStorage.removeItem('successful_api_endpoint');
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      // More specific check to avoid removing unrelated items
      if (key && (key.startsWith('cache:') || key === 'circuit_breaker_state' || key === 'successful_api_endpoint')) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));
    // Reset base URL lazily if needed, or ensure Axios instance is updated
    // This might require making `api` mutable or providing a function to update it.
    // For now, let's assume the initial creation uses the default.
    console.log(`Circuit breaker reset. Cleared ${keysToRemove.length} cache/state items.`);
    // Re-test connection after a delay
    setTimeout(() => {
      testApiConnection().then(connected => {
        console.log(`Connection test after reset: ${connected ? 'successful' : 'failed'}`);
      });
    }, 500);
  } catch (e) {
    console.error('Error clearing localStorage during circuit breaker reset:', e);
  }
}

// --- Lazy Loaded API Base URL ---
let apiBaseURL = CONFIG.baseURL; // Start with default
let endpointChecked = false;

function getApiBaseURL(): string {
  if (!endpointChecked && typeof localStorage !== 'undefined') {
    try {
      const savedEndpoint = localStorage.getItem('successful_api_endpoint');
      if (savedEndpoint) {
        console.log('Using saved API endpoint:', savedEndpoint);
        // Basic validation
        if (savedEndpoint.startsWith('http')) {
             if (savedEndpoint.includes('/api/v2')) {
               apiBaseURL = savedEndpoint;
             } else {
               apiBaseURL = `${savedEndpoint}/api/v2`; // Append /api/v2 if missing
             }
        } else {
             console.warn('Saved endpoint looks invalid, using default:', savedEndpoint);
             localStorage.removeItem('successful_api_endpoint'); // Remove invalid endpoint
             apiBaseURL = CONFIG.baseURL;
        }
      }
    } catch (e) {
      console.error('Error reading saved API endpoint from localStorage:', e);
      apiBaseURL = CONFIG.baseURL; // Fallback to default
    }
    endpointChecked = true;
  }
  return apiBaseURL;
}

// API client configuration
const api = axios.create({
  // Use Vite environment variable syntax (import.meta.env.VITE_...)
  // Ensure you create a .env file in the project root with VITE_API_BASE_URL and VITE_VERSION
  baseURL: import.meta.env.VITE_API_BASE_URL?.replace('/api/v1', '/api/v2') || 'https://fil-relex.onrender.com/api/v2',
  timeout: CONFIG.timeout,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Client-Version': import.meta.env.VITE_VERSION || '1.0.0',
    'X-Client-Platform': 'web'
  },
  withCredentials: false
});

// Apply rate limiting - 10 requests per second
//@ts-ignore
const limitedFetchWordDetails = rateLimit(api, { maxRequests: 10, perMilliseconds: 1000 });

// Inject client version from environment variable
api.interceptors.request.use(config => {
  config.headers['X-Client-Version'] = import.meta.env.VITE_VERSION || '1.0.0';
  return config;
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Ensure baseURL is up-to-date before each request
    config.baseURL = getApiBaseURL(); 
    // config.headers.set('Origin', window.location.origin); // REMOVE: Forbidden header
    // config.headers.set('Access-Control-Request-Method', config.method?.toUpperCase() || 'GET'); // REMOVE: Forbidden header
    console.log(`Making ${config.method?.toUpperCase()} request to: ${config.baseURL}${config.url}`);
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
  total_links: number;
  query_time?: number | null;
  filters_applied?: {
    depth: number;
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

interface NetworkLink {
  id: string;
  source: string;
  target: string;
  type: string;
  directed: boolean;
  weight: number;
}

interface LocalWordNetwork {
  nodes: NetworkNode[];
  links: NetworkLink[];
  metadata: NetworkMetadata;
}

export async function fetchWordNetwork(
  word: string, 
  options: WordNetworkOptions = {},
  signal?: AbortSignal // Add optional signal parameter
): Promise<ImportedWordNetwork> {
  const sanitizedWord = word.toLowerCase();
  const {
    depth = 2,
    include_affixes = true,
    include_etymology = true,
    cluster_threshold = 0.3,
    relation_types
  } = options;
  
  const sanitizedDepth = Math.min(Math.max(1, depth), 4); // Max depth 4
  
  const cacheKey = `cache:wordNetwork:${sanitizedWord}-${sanitizedDepth}-${include_affixes}-${include_etymology}-${cluster_threshold}-${relation_types?.join(',')}`;
    
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
      },
      signal // Pass signal to axios
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
    const links = response.data.links.map((e: any) => {
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
      links,
      metadata: {
        root_word: word,
        normalized_lemma: sanitizedWord,
        language_code: response.data.metadata?.language_code || 'tl',
        depth: sanitizedDepth,
        total_nodes: nodes.length,
        total_links: links.length,
        query_time: response.data.metadata?.execution_time || null,
        filters_applied: {
          depth: sanitizedDepth,
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
  // Replace undefined/null arrays with empty arrays for consistency
  const data = {
    ...rawData,
    definitions: rawData.definitions || [],
    etymologies: rawData.etymologies || [],
    pronunciations: rawData.pronunciations || [],
    forms: rawData.forms || [],
    templates: rawData.templates || [],
    credits: rawData.credits || [],
    outgoing_relations: rawData.outgoing_relations || [],
    incoming_relations: rawData.incoming_relations || [],
    root_affixations: rawData.root_affixations || [],
    affixed_affixations: rawData.affixed_affixations || [],
    definition_relations: rawData.definition_relations || [],
    related_definitions: rawData.related_definitions || [],
  };

  // Ensure examples have consistent structure in all definitions
  if (data.definitions && Array.isArray(data.definitions)) {
    data.definitions = data.definitions.map((def: any) => {
      if (!def) return def;
      
      // Handle examples - ensure examples exist and has proper metadata
      if (def.examples && Array.isArray(def.examples)) {
        def.examples = def.examples.map((example: any) => {
          if (!example) return example;
          
          // Convert example_metadata to example_metadata if needed
          if (example.metadata && !example.example_metadata) {
            example.example_metadata = example.metadata;
            // Keep the old property to avoid breaking changes during transition
            // delete example.metadata; // Uncomment later when all code is updated
          }
          
          // Ensure romanization is accessible
          if (example.example_metadata && example.example_metadata.romanization && !example.romanization) {
            example.romanization = example.example_metadata.romanization;
          }
          
          return example;
        });
      } else {
        def.examples = [];
      }
      
      return def;
    });
  }
  
  // Convert properties with special handling needs
  // Ensure proper relations structure by combining incoming/outgoing
  if (!data.relations) {
    data.relations = [
      ...(data.outgoing_relations || []),
      ...(data.incoming_relations || [])
    ];
  }

  // Ensure proper affixations structure by combining
  if (!data.affixations) {
    data.affixations = [
      ...(data.root_affixations || []),
      ...(data.affixed_affixations || [])
    ];
  }

  // Ensure is_root property
  if (typeof data.is_root !== 'boolean') {
    data.is_root = !data.root_word_id;
  }

  return data as WordInfo;
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
        // Use the complete parameter to get all data in one request
        complete: true,
        // Include specific parameters as fallback
        include_relations: true,
        include_etymologies: true,
        include_pronunciation: true,
        include_root: true,
        include_derived: true,
        include_affixations: true,
        include_definition_relations: true,
        include_forms: true,
        include_templates: true,
        include_credits: true,
        include_metadata: true
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
            // Initialize other required fields to avoid type errors, even if empty
            normalized_lemma: basicData.lemma || word,
            language_code: 'tl', 
            has_baybayin: false,
            is_root: !basicData.root_word_id,
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
            // Add other optional fields as null or default
            baybayin_form: null,
            romanized_form: null,
            root_word_id: null,
            preferred_spelling: null,
            tags: null,
            data_hash: null,
            created_at: null,
            updated_at: null,
            word_metadata: null,
            source_info: null,
            idioms: null,
            badlit_form: null,
            hyphenation: null,
            is_proper_noun: false,
            is_abbreviation: false,
            is_initialism: false,
          };
          
          console.warn(`Returning partial word info for '${word}' due to database error.`);
          
          // Try to fetch semantic network data as fallback
          try {
            if (basicData.id || basicData.lemma) {
              const idOrWord = basicData.id ? `id:${basicData.id}` : basicData.lemma;
              const networkData = await fetchWordNetwork(idOrWord);
              
              if (networkData && networkData.nodes && networkData.links) {
                console.log('Successfully fetched semantic network fallback data (but not added to WordInfo)');
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
        errorMessage.includes('SQL error'))) {
        console.log('General database error detected. Creating partial word info with error flag.');
        
        // Try to fetch the semantic network for fallback
        try {
          // Create a partial word info object with the error message
          const partialWordInfo: WordInfo = {
            id: isIdRequest ? parseInt(word.replace('id:', ''), 10) : 0,
            lemma: word,
            // Initialize other required fields to avoid type errors, even if empty
            normalized_lemma: word,
            language_code: 'tl', 
            has_baybayin: false,
            is_root: false, // Default value when basicData is unavailable
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
            // Add other optional fields as null or default
            baybayin_form: null,
            romanized_form: null,
            root_word_id: null,
            preferred_spelling: null,
            tags: null,
            data_hash: null,
            created_at: null,
            updated_at: null,
            word_metadata: null,
            source_info: null,
            idioms: null,
            badlit_form: null,
            hyphenation: null,
            is_proper_noun: false,
            is_abbreviation: false,
            is_initialism: false,
          };
          
          console.warn(`Returning partial word info for '${word}' due to database error.`);
          
          // Try to fetch semantic network data as fallback
          try {
            const networkData = await fetchWordNetwork(word);
            
            if (networkData && networkData.nodes && networkData.links) {
              console.log('Successfully fetched semantic network fallback data for general database error (but not added to WordInfo)');
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

export async function searchWords(query: string, options: SearchOptions): Promise<SearchResults> {
  console.log(`[DEBUG] searchWords called with query: "${query}" and options:`, options);
  
  const cacheKey = `cache:search:${query}:${JSON.stringify(options)}`;
  const cachedData = getCachedData<SearchResults>(cacheKey);

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
    
    // Tags and categories
    if (options.tags && options.tags.length > 0) apiParams.tags = options.tags.join(',');
    if (options.categories && options.categories.length > 0) apiParams.categories = options.categories.join(',');
    
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
    let searchResult: SearchResults | null = null;
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
          results: (data.results || []).map((result: any): SearchResultItem => ({
            word_id: result.id, // FIX: Map id to word_id
            lemma: result.lemma,
            lang_code: result.language_code, // Use result.language_code
            lang_name: result.lang_name || '', // Add lang_name
            gloss: result.gloss || '', // Add gloss
            pos: result.pos || '', // Add pos
            score: result.score || 0, // Add score
            word: result.word || null, // Add optional full word data
          })),
          page: options.page || 1,
          per_page: options.per_page || (data.results?.length || 0), // FIX: Use per_page
          total: data.total || data.count || 0, // Use total or count
          query_details: data.query_details || null // Add query_details
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
        results: (data.results || []).map((result: any): SearchResultItem => ({
          word_id: result.id, // FIX: Map id to word_id
          lemma: result.lemma,
          lang_code: result.language_code, // Use result.language_code
          lang_name: result.lang_name || '', // Add lang_name
          gloss: result.gloss || '', // Add gloss
          pos: result.pos || '', // Add pos
          score: result.score || 0, // Add score
          word: result.word || null, // Add optional full word data
        })),
        page: options.page || 1,
        per_page: options.per_page || (data.results?.length || 0), 
        total: data.total || 0,
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
        const errorResult: SearchResults = {
          results: [],
          page: options.page || 1,
          per_page: options.per_page || 10,
          total: 0,
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
export async function advancedSearch(query: string, options: SearchOptions): Promise<SearchResults> {
  console.log(`[DEBUG] advancedSearch called with query: "${query}" and options:`, options);
  
  const cacheKey = `cache:advanced_search:${query}:${JSON.stringify(options)}`;
  const cachedData = getCachedData<SearchResults>(cacheKey);

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
        q: encodeURIComponent(query), // Explicitly encode the query parameter
        limit: options.per_page || 50, // Default limit
        offset: options.page ? (options.page - 1) * (options.per_page || 50) : 0, 
        include_details: options.include_full || false,
    };
    
    // Advanced filters
    if (options.language) apiParams.language = options.language;
    if (options.pos) apiParams.pos = options.pos;
    if (options.sort) apiParams.sort = options.sort;
    if (options.order) apiParams.order = options.order;
    
    // Feature filters
    if (options.has_baybayin !== undefined) apiParams.has_baybayin = options.has_baybayin;
    if (options.has_etymology !== undefined) apiParams.has_etymology = options.has_etymology;
    if (options.has_pronunciation !== undefined) apiParams.has_pronunciation = options.has_pronunciation;
    if (options.has_forms !== undefined) apiParams.has_forms = options.has_forms;
    if (options.has_templates !== undefined) apiParams.has_templates = options.has_templates;
    
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
    
    // Tags and categories
    if (options.tags && options.tags.length > 0) apiParams.tags = options.tags.join(',');
    if (options.categories && options.categories.length > 0) apiParams.categories = options.categories.join(',');
    
    console.log(`[DEBUG] Making advanced search API request with params:`, apiParams);
    
    const response = await api.get('/search/advanced', { params: apiParams });
    console.log(`[DEBUG] Advanced search API responded with status: ${response.status}`);
    
    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.statusText}`);
    }
    
    const data = response.data;
    console.log(`[DEBUG] Advanced search API raw response data:`, data);
    
    // Transform the response into SearchResult format
    const searchResult: SearchResults = {
      // Handle both 'results' and legacy 'words' key from API response
      results: (data.results || data.words || []).map((result: any): SearchResultItem => ({
        word_id: result.word_id || result.id, // Ensure this line maps id or word_id to word_id
        lemma: result.lemma,
        lang_code: result.lang_code || result.language_code,
        lang_name: result.lang_name || '',
        gloss: result.gloss || '',
        pos: result.pos || '',
        score: result.score || 0,
        word: result.word || null,
      })),
      page: options.page || 1,
      // Use per_page from options, or API's limit, or fallback to results length
      per_page: options.per_page || data.limit || (data.results || data.words || []).length,
      // Use total or count from API response
      total: data.total || data.count || 0,
      query_details: data.query_details || null // Include query details if provided
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
      console.log('Etymology tree response has no nodes (this is valid):', treeData); // Changed from console.warn
      // Return empty tree structure instead of throwing error
      const emptyTree: EtymologyTree = { 
        nodes: [], 
        links: [],
        word: '',
        etymology_tree: {},
        complete: false
      };
      setCachedData(cacheKey, emptyTree);
      return emptyTree;
    }
    
    console.log(`Etymology tree data received with ${treeData.nodes.length} nodes and ${treeData.links?.length || 0} links`);
    setCachedData(cacheKey, treeData);
    return treeData;
  } catch (error) {
    console.error(`Error fetching etymology tree for wordId=${wordId}:`, error);
    // Return empty tree structure instead of throwing error
    const emptyTree: EtymologyTree = { 
      nodes: [], 
      links: [],
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
    // Add a cache-busting parameter (_=timestamp)
    const randomWordUrl = `${CONFIG.baseURL}/random?_=${Date.now()}`;
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
    
  // Try to get stringified data from cache
  const cachedString = localStorage.getItem(cacheKey);
  if (cachedString) {
    try {
      const cachedData: { data: Statistics, timestamp: number } = JSON.parse(cachedString);
      if (Date.now() - cachedData.timestamp < CACHE_EXPIRATION) { // Use CACHE_EXPIRATION defined in caching.ts (assuming it's accessible or re-defined here)
        console.log('Using cached statistics data');
        return cachedData.data;
      } else {
        localStorage.removeItem(cacheKey); // Remove expired item
      }
    } catch (e) {
      console.error('Error parsing cached statistics:', e);
      localStorage.removeItem(cacheKey); // Remove corrupted item
    }
  }

  if (!circuitBreaker.canMakeRequest()) throw new Error("Circuit breaker is open.");

  try {
    const response = await api.get('/statistics');
     if (response.status !== 200) throw new Error(`API status ${response.status}`);
    const statsData = response.data;
    if (!statsData || typeof statsData !== 'object') throw new Error('Invalid statistics data');
    // Add timestamp to stats data if not present
    statsData.timestamp = statsData.timestamp || new Date().toISOString();
    
    // Store stringified data with timestamp
    const cacheItem = {
      data: statsData,
      timestamp: Date.now()
    };
    localStorage.setItem(cacheKey, JSON.stringify(cacheItem));
    
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

// Use relative path for production, Vite proxy handles it in development
const API_BASE_URL = import.meta.env.PROD ? '/api/v2' : (import.meta.env.VITE_API_URL || 'http://localhost:10000/api/v2');

let successfulApiEndpoint = API_BASE_URL;

/**
 * Fetches word suggestions based on the search query.
 * @param query - The search term prefix.
 * @param limit - Maximum number of suggestions to fetch (defaults to 10).
 * @returns A promise that resolves to an array of suggestion objects.
 */
export const fetchSuggestions = async (query: string, limit: number = 10): Promise<WordSuggestion[]> => {
  if (!query || query.length < 2) {
    return []; // Don't fetch for empty or very short queries
  }
  try {
    // Ensure the generic type here uses the imported WordSuggestion
    const response = await api.get<WordSuggestion[]>('/suggestions', { 
      params: { query, limit },
    });
    console.log(`Suggestions fetched for "${query}":`, response.data);
    return response.data; // Return the array of objects
  } catch (error) {
    console.error('Error fetching suggestions:', error);
    return []; // Return empty array on error
  }
};