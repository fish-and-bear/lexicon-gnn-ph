import { WordNetwork, WordInfo, Statistics, PartOfSpeech, EtymologyTree, SearchResult } from "../types";

const CACHE_EXPIRATION = 5 * 60 * 1000; // 5 minutes
const MAX_CACHE_SIZE = 5 * 1024 * 1024; // 5MB
const MAX_CACHE_ITEMS = 100;
const CACHE_CLEANUP_INTERVAL = 60 * 1000; // 1 minute

interface CacheItem<T> {
  data: T;
  timestamp: number;
  type: string;
  size: number;
}

export type CacheableData = WordInfo | WordNetwork | Statistics | SearchResult | EtymologyTree | PartOfSpeech[];

interface CacheStats {
  totalItems: number;
  totalSize: number;  // Size in KB
  oldestTimestamp: number | null;
  newestTimestamp: number | null;
}

function calculateCacheStats(): CacheStats {
  let totalSize = 0;
  let itemCount = 0;
  let oldestTimestamp = Date.now();

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith('cache:')) {
      try {
        const item = JSON.parse(localStorage.getItem(key) || '');
        totalSize += item.size || 0;
        itemCount++;
        oldestTimestamp = Math.min(oldestTimestamp, item.timestamp);
      } catch (e) {
        console.error('Error calculating cache stats:', e);
      }
    }
  }

  return { totalSize, itemCount, oldestTimestamp };
}

export function clearOldCache() {
  try {
    const keys = Object.keys(localStorage);
    const cacheKeys = keys.filter(key => key.startsWith('cache:'));
    const currentTime = Date.now();

    cacheKeys.forEach(key => {
      const item = localStorage.getItem(key);
      if (item) {
        try {
          const { timestamp } = JSON.parse(item);
          if (currentTime - timestamp > CACHE_EXPIRATION) {
            localStorage.removeItem(key);
          }
        } catch (e) {
          // If we can't parse the item, remove it
          localStorage.removeItem(key);
        }
      }
    });
  } catch (e) {
    console.error('Error clearing old cache:', e);
  }
}

export function getCachedData<T extends CacheableData>(key: string): T | null {
  const cacheKey = `cache:${key}`;
  const item = localStorage.getItem(cacheKey);
  if (!item) return null;

  try {
    const { data, timestamp, type, size }: CacheItem<T> = JSON.parse(item);
    
  if (Date.now() - timestamp > CACHE_EXPIRATION) {
      localStorage.removeItem(cacheKey);
      return null;
    }

    // Type validation
    if (isWordNetwork(data) && type === 'WordNetwork') {
      return data as T;
    } else if (isWordInfo(data) && type === 'WordInfo') {
      return data as T;
    } else if (isStatistics(data) && type === 'Statistics') {
      return data as T;
    } else if (isPartsOfSpeech(data) && type === 'PartOfSpeech[]') {
      return data as T;
    } else if (isEtymologyTree(data) && type === 'EtymologyTree') {
      return data as T;
    }

    return null;
  } catch (error) {
    console.error('Error parsing cached data:', error);
    localStorage.removeItem(cacheKey);
    return null;
  }
}

export function setCachedData<T extends CacheableData>(key: string, data: T): void {
  try {
    const type = getDataType(data);
    if (!type) {
      console.error('Invalid data type for caching');
      return;
    }

    // Calculate item size
    const serialized = JSON.stringify(data);
    const size = new Blob([serialized]).size;

    // Check if item is too large
    if (size > MAX_CACHE_SIZE * 0.1) { // Single item shouldn't exceed 10% of total cache
      console.warn('Cache item too large, skipping cache');
      return;
    }

    const cacheKey = `cache:${key}`;
    const cacheItem: CacheItem<T> = {
    data,
    timestamp: Date.now(),
      type,
      size
    };

    // Clear old cache if needed
    clearOldCache();

    localStorage.setItem(cacheKey, JSON.stringify(cacheItem));
  } catch (error) {
    console.error('Error caching data:', error);
  }
}

// Start cache cleanup interval
setInterval(clearOldCache, CACHE_CLEANUP_INTERVAL);

// Type guards
function isWordNetwork(data: any): data is WordNetwork {
  return (
    data &&
    typeof data === 'object' &&
    'nodes' in data &&
    'clusters' in data &&
    'metadata' in data
  );
}

function isWordInfo(data: any): data is WordInfo {
  return (
    data &&
    typeof data === 'object' &&
    'lemma' in data &&
    'normalized_lemma' in data &&
    'language_code' in data
  );
}

function isStatistics(data: any): data is Statistics {
  return (
    data &&
    typeof data === 'object' &&
    'words' in data &&
    'definitions' in data &&
    'relations' in data
  );
}

function isPartsOfSpeech(data: any): data is PartOfSpeech[] {
  return (
    Array.isArray(data) &&
    data.every(item => 
      item &&
      typeof item === 'object' &&
      'code' in item &&
      'name_en' in item &&
      'name_tl' in item
    )
  );
}

function isEtymologyTree(data: any): data is EtymologyTree {
  return (
    data &&
    typeof data === 'object' &&
    'word' in data &&
    'normalized_lemma' in data &&
    'components' in data &&
    'component_words' in data &&
    'metadata' in data
  );
}

function getDataType(data: CacheableData): string | null {
  if (isWordNetwork(data)) return 'WordNetwork';
  if (isWordInfo(data)) return 'WordInfo';
  if (isStatistics(data)) return 'Statistics';
  if (isPartsOfSpeech(data)) return 'PartOfSpeech[]';
  if (isEtymologyTree(data)) return 'EtymologyTree';
  return null;
}

// Export cache management functions
export function getCacheStats(): CacheStats {
  const keys = Object.keys(localStorage);
  const cacheKeys = keys.filter(key => key.startsWith('cache:'));
  const totalItems = cacheKeys.length;
  let totalSize = 0;

  cacheKeys.forEach(key => {
    const item = localStorage.getItem(key);
    if (item) {
      totalSize += item.length;
    }
  });

  return {
    totalItems,
    totalSize: Math.round(totalSize / 1024), // Convert to KB
    oldestTimestamp: 0, // We'll implement this if needed
    newestTimestamp: 0  // We'll implement this if needed
  };
}

export function clearCache(): void {
  const keys = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith('cache:')) {
      keys.push(key);
    }
  }
  keys.forEach(key => localStorage.removeItem(key));
}