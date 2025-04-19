/**
 * Frontend Performance Fix Script
 * 
 * This script improves application performance by:
 * 1. Cleaning up stale or corrupted localStorage entries
 * 2. Setting startup flags to defer non-critical operations
 * 3. Preventing multiple API calls during initialization
 */

console.log('Running frontend performance optimizations...');

// Function to safely clear localStorage items
function safeCleanLocalStorage() {
  try {
    // Keep track of cleared items for reporting
    const cleared = [];
    const preserved = [];
    
    // Items to preserve (keep critical user settings)
    const preserveKeys = ['theme', 'language_preference', 'word_history'];
    
    // Identify stale items to remove
    const now = Date.now();
    const ONE_DAY = 24 * 60 * 60 * 1000; // 1 day in milliseconds
    
    // Check each item in localStorage
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key) continue;
      
      // Always preserve important user settings
      if (preserveKeys.includes(key)) {
        preserved.push(key);
        continue;
      }
      
      // Clear circuit breaker state to prevent startup issues
      if (key === 'circuit_breaker_state') {
        localStorage.removeItem(key);
        cleared.push(key);
        continue;
      }
      
      // Clean up old cache entries
      if (key.startsWith('cache:')) {
        try {
          const cacheData = JSON.parse(localStorage.getItem(key) || '{}');
          // Remove cache items older than 1 day
          if (cacheData.timestamp && (now - cacheData.timestamp > ONE_DAY)) {
            localStorage.removeItem(key);
            cleared.push(key);
          } else {
            preserved.push(key);
          }
        } catch (e) {
          // If JSON parsing fails, the cache is corrupted - remove it
          localStorage.removeItem(key);
          cleared.push(`${key} (corrupted)`);
        }
      }
    }
    
    // Set performance optimization flags
    localStorage.setItem('app_startup_time', now.toString());
    localStorage.setItem('defer_api_checks', 'true');
    
    console.log(`Storage cleanup complete. Cleared ${cleared.length} items, preserved ${preserved.length} items.`);
  } catch (e) {
    console.error('Error during localStorage cleanup:', e);
  }
}

// Function to preload critical CSS files
function preloadCriticalCSS() {
  const criticalCSSFiles = [
    '/styles/utilities/_reset.css',
    '/styles/utilities/_variables.css',
    '/components/common.css'
  ];
  
  criticalCSSFiles.forEach(file => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'style';
    link.href = file;
    document.head.appendChild(link);
  });
  
  console.log('Preloaded critical CSS files');
}

// Run cleanup immediately
safeCleanLocalStorage();

// Execute when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', preloadCriticalCSS);
} else {
  preloadCriticalCSS();
}

// Flag to prevent duplicate API initialization
window.API_INITIALIZED = false;

console.log('Frontend performance optimizations complete.'); 