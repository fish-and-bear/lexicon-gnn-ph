import axios from 'axios';
import { WordNetwork, WordInfo } from "../types";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://54.252.249.125:10000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL
});

const cache = new Map<string, any>();

export async function fetchWordNetwork(word: string, depth: number, breadth: number): Promise<WordNetwork> {
  const cacheKey = `wordNetwork-${word}-${depth}-${breadth}`;
  if (cache.has(cacheKey)) {
    return cache.get(cacheKey);
  }
  const response = await api.get(`/word_network/${word}`, { params: { depth, breadth } });
  cache.set(cacheKey, response.data);
  return response.data;
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