import axios, { AxiosError, AxiosResponse } from 'axios';
import { 
  WordNetwork, 
  WordInfo, 
  SearchOptions, 
  SearchResult, 
  Etymology,
  PartOfSpeech
} from "../types";
import { sanitizeInput } from '../utils/sanitizer';
import { getCachedData, setCachedData } from '../utils/caching';

declare const process: {
  env: {
    NEXT_PUBLIC_API_BASE_URL?: string;
  };
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://api.hapinas.net/api/v1';

// Type definition for rate limiting
type RateLimitedAxiosInstance = ReturnType<typeof axios.create> & {
  maxRequests?: number;
  perMilliseconds?: number;
};

// Create rate-limited axios instance
const api: RateLimitedAxiosInstance = axios.create({
  baseURL: API_BASE_URL,
});

api.maxRequests = 50;
api.perMilliseconds = 60000;

interface ApiError {
  response?: {
    status: number;
    statusText: string;
  };
  request?: any;
  message?: string;
}

interface NetworkNode {
  word: string;
  definition: string;
  synonyms: string[];
  antonyms: string[];
  derived: string[];
  related: string[];
  root: string | null;
}

export async function fetchWordNetwork(
  word: string, 
  depth: number = 2, 
  breadth: number = 10,
  relation_types?: string[]
): Promise<WordNetwork> {
  try {
    const sanitizedWord = sanitizeInput(word);
    const sanitizedDepth = Math.min(Math.max(1, depth), 5);
    const sanitizedBreadth = Math.min(Math.max(5, breadth), 20);

    const cacheKey = `wordNetwork-${sanitizedWord}-${sanitizedDepth}-${sanitizedBreadth}-${relation_types?.join(',')}`;
    const cachedData = getCachedData(cacheKey);
    if (cachedData) {
      return cachedData as WordNetwork;
    }

    const encodedWord = encodeURIComponent(sanitizedWord);
    const response = await api.get<Record<string, NetworkNode>>(`/word_network/${encodedWord}`, { 
      params: { 
        depth: sanitizedDepth, 
        breadth: sanitizedBreadth,
        relation_types
      },
      timeout: 10000
    });

    const networkData: WordNetwork = response.data;
    setCachedData(cacheKey, networkData);
    return networkData;
  } catch (error) {
    console.error('Error fetching word network:', error);
    const apiError = error as ApiError;
    if (axios.isAxiosError(error)) {
      if (apiError.response) {
        throw new Error(`Failed to fetch word network: ${apiError.response.status} ${apiError.response.statusText}`);
      } else if (apiError.request) {
        throw new Error('Failed to fetch word network: No response received');
      }
    }
    throw error;
  }
}

export async function fetchWordDetails(word: string): Promise<WordInfo> {
  try {
    const sanitizedWord = sanitizeInput(word);
    const cacheKey = `wordDetails-${sanitizedWord}`;
    const cachedData = getCachedData(cacheKey);
    if (cachedData) {
      if ('meta' in cachedData && 'data' in cachedData) {
        return cachedData as WordInfo;
      }
    }

    const encodedWord = encodeURIComponent(sanitizedWord);
    const response = await api.get<WordInfo>(`/words/${encodedWord}`);
    
    setCachedData(cacheKey, response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching word details:', error);
    const apiError = error as ApiError;
    if (axios.isAxiosError(error)) {
      if (apiError.response?.status === 404) {
        throw new Error('Word not found');
      }
      throw new Error(`Failed to fetch word details: ${apiError.message}`);
    }
    throw error;
  }
}

export async function bulkFetchWordDetails(words: string[]): Promise<WordInfo[]> {
  try {
    const sanitizedWords = words.map(sanitizeInput);
    const response = await api.post<{ words: WordInfo[] }>('/bulk_words', { words: sanitizedWords });
    return response.data.words;
  } catch (error) {
    console.error('Error fetching bulk word details:', error);
    throw error;
  }
}

interface SearchResponse {
  results: Array<{ id: number; word: string; score: number }>;
  total?: number;
}

export async function searchWords(query: string, options: SearchOptions): Promise<SearchResult> {
  try {
    const sanitizedQuery = sanitizeInput(query);
    const response = await api.get<SearchResponse>('/search', {
      params: {
        q: sanitizedQuery,
        limit: options.per_page,
        min_similarity: 0.3,
        ...options
      }
    });
    return {
      words: response.data.results.map((r) => ({
        id: r.id,
        word: r.word
      })),
      page: options.page,
      perPage: options.per_page,
      total: response.data.total || response.data.results.length
    };
  } catch (error) {
    console.error("Error searching words:", error);
    throw error;
  }
}

export async function checkWord(word: string): Promise<{ exists: boolean; word: string | null }> {
  try {
    const sanitizedWord = sanitizeInput(word);
    const encodedWord = encodeURIComponent(sanitizedWord);
    const response = await api.get<{ exists: boolean; word: string | null }>(`/check_word/${encodedWord}`);
    return response.data;
  } catch (error) {
    console.error("Error checking word:", error);
    throw error;
  }
}

export async function getEtymology(word: string): Promise<{ word: string; etymologies: Etymology[] }> {
  try {
    const sanitizedWord = sanitizeInput(word);
    const encodedWord = encodeURIComponent(sanitizedWord);
    const response = await api.get<{ word: string; etymologies: Etymology[] }>(`/etymology/${encodedWord}`);
    return response.data;
  } catch (error) {
    console.error("Error fetching etymology:", error);
    throw error;
  }
}

export async function getPartsOfSpeech(): Promise<PartOfSpeech[]> {
  try {
    const response = await api.get<{ parts_of_speech: PartOfSpeech[] }>('/pos');
    return response.data.parts_of_speech;
  } catch (error) {
    console.error("Error fetching parts of speech:", error);
    throw error;
  }
}