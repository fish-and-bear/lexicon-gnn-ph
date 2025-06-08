import { WordNetwork, WordInfo, Statistics, PartOfSpeech, EtymologyTree, SearchResults } from "../types";

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

export type CacheableData = WordInfo | WordNetwork | Statistics | SearchResults | EtymologyTree | PartOfSpeech[];

interface CacheStats {
  totalItems: number;
  totalSize: number;  // Size in KB
  oldestTimestamp: number | null;
  newestTimestamp: number | null;
}

// Function to calculate cache statistics (optional)
// function calculateCacheStats() {
//   let totalSize = 0;
//   let itemCount = 0;
//   const stats: Record<string, { size: number, expires: number | null }> = {};
//
//   for (let i = 0; i < localStorage.length; i++) {
//     const key = localStorage.key(i);
//     if (key && key.startsWith(CACHE_PREFIX)) {
//       try {
//         const item = localStorage.getItem(key);
//         if (item) {
//           const parsed = JSON.parse(item);
//           const size = (item.length * 2) / 1024; // Size in KB (approx)
//           totalSize += size;
//           itemCount++;
//           stats[key.substring(CACHE_PREFIX.length)] = {
//             size: parseFloat(size.toFixed(2)),
//             expires: parsed.expires || null
//           };
//         }
//       } catch (e) {
//         console.warn(`Error processing cache item ${key}:`, e);
//       }
//     }
//   }
//
//   console.log(`Cache Stats: ${itemCount} items, Total Size: ${totalSize.toFixed(2)} KB`);
//   // console.table(stats);
//   return { itemCount, totalSize, stats };
// }

/**
 * Clears expired cache entries based on their TTL.
 */
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
            // console.log(`Cache item ${key} removed due to expiration.`);
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

// Flag to ensure interval is only set once
let cleanupIntervalStarted = false;

// Function to start the cleanup interval if not already started
function ensureCacheCleanupInterval() {
  if (!cleanupIntervalStarted) {
    setInterval(clearOldCache, CACHE_CLEANUP_INTERVAL);
    cleanupIntervalStarted = true;
    // console.log('Cache cleanup interval started.');
  }
}

export function getCachedData<T extends CacheableData>(key: string): T | null {
  ensureCacheCleanupInterval(); // Ensure interval is running
  const cacheKey = `cache:${key}`;
  const item = localStorage.getItem(cacheKey);
  if (!item) return null;

  try {
    const { data, timestamp, type/*, size*/ }: CacheItem<T> = JSON.parse(item);
    
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
  ensureCacheCleanupInterval(); // Ensure interval is running
  try {
    const stats = getCacheStats();
    
    // Clear old cache if we're getting too full
    if (stats.totalSize > MAX_CACHE_SIZE || stats.totalItems > MAX_CACHE_ITEMS) {
      clearOldCache();
    }

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

    localStorage.setItem(cacheKey, JSON.stringify(cacheItem));
  } catch (e) {
    console.error('Error setting cache data:', e);
  }
}

// Type guards
function isWordNetwork(data: any): data is WordNetwork {
  return (
    data &&
    typeof data === 'object' &&
    'nodes' in data && Array.isArray(data.nodes) &&
    'links' in data && Array.isArray(data.links) &&
    'metadata' in data && typeof data.metadata === 'object'
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
    'total_words' in data &&
    'total_definitions' in data &&
    'total_relations' in data &&
    'words_by_language' in data
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
    'word' in data && typeof data.word === 'string' &&
    'nodes' in data && Array.isArray(data.nodes) &&
    'links' in data && Array.isArray(data.links) &&
    'etymology_tree' in data &&
    'complete' in data
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
  let oldestTimestamp: number | null = null;
  let newestTimestamp: number | null = null;

  cacheKeys.forEach(key => {
    const item = localStorage.getItem(key);
    if (item) {
      totalSize += item.length;
      try {
        const { timestamp } = JSON.parse(item);
        if (oldestTimestamp === null || timestamp < oldestTimestamp) {
          oldestTimestamp = timestamp;
        }
        if (newestTimestamp === null || timestamp > newestTimestamp) {
          newestTimestamp = timestamp;
        }
      } catch (e) {
        // Ignore parse errors
      }
    }
  });

  return {
    totalItems,
    totalSize: Math.round(totalSize / 1024), // Convert to KB
    oldestTimestamp,
    newestTimestamp
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

export function initializeCache(): void {
  try {
    cleanupExpiredItems();
    // console.log('Cache cleanup interval started.');
  } catch (error) {
    console.error('Failed to initialize cache:', error);
  }
}

export function cleanupExpiredItems(): void {
  const now = Date.now();
  const keys = Object.keys(localStorage);
  const cacheKeys = keys.filter(key => key.startsWith('cache:'));
  
  cacheKeys.forEach(key => {
    const item = localStorage.getItem(key);
    if (item) {
      try {
        const cacheItem: CacheItem<any> = JSON.parse(item);
        if (cacheItem.timestamp && (now - cacheItem.timestamp) > CACHE_EXPIRATION) {
          localStorage.removeItem(key);
          // console.log(`Cache item ${key} removed due to expiration.`);
        }
      } catch (e) {
        // Invalid cache item, remove it
        localStorage.removeItem(key);
      }
    }
  });
}

function logCacheStats(): void {
  // Development-only cache statistics
  if (process.env.NODE_ENV === 'development') {
    const stats = getCacheStats();
    // console.log('Cache Stats:', stats);
  }
}