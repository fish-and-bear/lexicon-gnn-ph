import { WordNetwork } from "../types";

const CACHE_EXPIRATION = 5 * 60 * 1000; // 5 minutes

interface CacheItem {
  data: WordNetwork;
  timestamp: number;
}

export function getCachedData(key: string): WordNetwork | null {
  const item = localStorage.getItem(key);
  if (!item) return null;

  const { data, timestamp }: CacheItem = JSON.parse(item);
  if (Date.now() - timestamp > CACHE_EXPIRATION) {
    localStorage.removeItem(key);
    return null;
  }

  return data;
}

export function setCachedData(key: string, data: WordNetwork): void {
  const cacheItem: CacheItem = {
    data,
    timestamp: Date.now(),
  };
  localStorage.setItem(key, JSON.stringify(cacheItem));
}