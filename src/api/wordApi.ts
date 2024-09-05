import axios from 'axios';
import { WordNetwork, WordInfo } from "../types";
import rateLimit from 'axios-rate-limit';
import { sanitizeInput } from '../utils/sanitizer';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://api.hapinas.net/api/v1';

const api = rateLimit(axios.create({
  baseURL: API_BASE_URL,
}), { maxRequests: 100, perMilliseconds: 60000 });

const cache = new Map<string, any>();

export async function fetchWordNetwork(word: string, depth: number, breadth: number): Promise<WordNetwork> {
  try {
    const sanitizedWord = sanitizeInput(word);
    const sanitizedDepth = Math.min(Math.max(1, depth), 5);
    const sanitizedBreadth = Math.min(Math.max(5, breadth), 20);

    const cacheKey = `wordNetwork-${sanitizedWord}-${sanitizedDepth}-${sanitizedBreadth}`;
    if (cache.has(cacheKey)) {
      return cache.get(cacheKey);
    }
    const encodedWord = encodeURIComponent(sanitizedWord);
    console.log(`Fetching word network for: ${encodedWord}, depth: ${sanitizedDepth}, breadth: ${sanitizedBreadth}`);
    const response = await api.get(`/word_network/${encodedWord}`, { 
      params: { depth: sanitizedDepth, breadth: sanitizedBreadth },
      timeout: 5000
    });
    console.log('Response received:', response.status, response.statusText);
    console.log('Response data:', response.data);
    cache.set(cacheKey, response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching word network:', error);
    if (axios.isAxiosError(error)) {
      if (error.response) {
        console.error('Response error data:', error.response.data);
        throw new Error(`Failed to fetch word network: ${error.response.status} ${error.response.statusText}`);
      } else if (error.request) {
        throw new Error('Failed to fetch word network: No response received');
      } else {
        throw new Error(`Failed to fetch word network: ${error.message}`);
      }
    }
    throw error; // Re-throw the original error if it's not an Axios error
  }
}

export async function fetchWordDetails(word: string): Promise<WordInfo> {
  const cacheKey = `wordDetails-${word}`;
  if (cache.has(cacheKey)) {
    return cache.get(cacheKey);
  }
  const response = await api.get(`/words/${word}`);
  cache.set(cacheKey, response.data);
  return response.data;
}

export async function bulkFetchWordDetails(words: string[]): Promise<WordInfo[]> {
  const response = await api.post('/bulk_words', { words });
  return response.data.words;
}

export async function searchWords(query: string, page: number = 1, perPage: number = 20): Promise<{ words: { word: string, id: number }[], total: number }> {
  const response = await api.get('/words', { params: { search: query, page, per_page: perPage } });
  return response.data;
}