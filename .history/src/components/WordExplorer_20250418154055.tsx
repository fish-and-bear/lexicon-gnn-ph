import React, { useEffect, useState, useCallback } from 'react'; // Added useCallback
import { useTheme } from "../contexts/ThemeContext"; // Restore useTheme import
import { testApiConnection, searchWords, fetchWordDetails, getEtymologyTree } from "../api/wordApi"; // Added fetchWordDetails, getEtymologyTree
import { WordInfo, SearchWordResult, SearchOptions, EtymologyTree } from '../types'; // Added EtymologyTree
import WordDetails from './WordDetails'; // Import WordDetails component

const WordExplorer: React.FC = () => {
  // --- DEBUG: Restore useTheme hook call and use its value ---
  const { theme: contextTheme, toggleTheme } = useTheme(); // Call hook
  // const theme = 'light'; // Remove default theme for now
  // const toggleTheme = () => {}; // Remove dummy function
  // --- END DEBUG ---
  
  // Basic state variables
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null); // Start as null
  const [inputValue, setInputValue] = useState<string>("");
  const [mainWord, setMainWord] = useState<string>("");
  const [selectedWordInfo, setSelectedWordInfo] = useState<WordInfo | null>(null);

  // Search state variables
  const [searchResults, setSearchResults] = useState<SearchWordResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Etymology state variables
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);

  // --- DEBUG: Re-enable initial useEffect --- 
  useEffect(() => {
    console.log("Testing API connection..."); // Restore original log
    setIsLoading(true);
    setError(null);

    // --- DEBUG: Restore API call and promise handling ---
    testApiConnection()
      .then(connected => {
        console.log("API Connection:", connected);
        setApiConnected(connected); // <-- RE-ENABLE
        if (!connected) {
          setError("API connection failed. Please check the backend server.");
          console.error("API connection failed.");
        }
      })
      .catch(error => {
        console.error("API Error:", error);
        setError(error.message || "An unknown error occurred during API connection test.");
        setApiConnected(false); // <-- RE-ENABLE
      })
      .finally(() => {
        console.log("API test finally block reached.");
        setIsLoading(false);   // <-- RE-ENABLE
      });
    // --- END DEBUG ---

    /*
    // Directly set state as if API succeeded
    console.log("Directly setting apiConnected=true and isLoading=false");
    setApiConnected(true); // <-- RE-ENABLE
    setIsLoading(false);   // <-- RE-ENABLE
    */

  }, []);
  // --- END DEBUG ---

  // --- DEBUG: Keep handlers commented out for now --- 
  /*
  // Fetch Etymology Tree
  const fetchEtymologyData = useCallback(async (wordId: number) => {
    // ... implementation ...
  }, []);

  // Load Word Details
  const loadWordData = useCallback(async (wordIdentifier: string | number) => {
    // ... implementation ...
  }, [fetchEtymologyData]); // Add fetchEtymologyData dependency

  // Handle Search Function (updated)
  const handleSearch = useCallback(async (query: string) => {
    // ... implementation ...
  }, [apiConnected, loadWordData]); // Add loadWordData dependency
  
  // Handler for clicking links within WordDetails (or search results later)
  const handleWordLinkClick = useCallback((wordIdentifier: string | number) => {
    // For now, just call loadWordData
    loadWordData(wordIdentifier);
  }, [loadWordData]);
  */
  // --- END DEBUG ---

  // Dummy handlers for search button
  const handleSearch = () => console.log("Search clicked (dummy)");
  const handleWordLinkClick = (id: string | number) => console.log("Word link clicked (dummy):", id);

  return (
    <div className={`word-explorer ${contextTheme}`}>
      <header>
        <h1>Filipino Root Word Explorer</h1>
        {/* --- DEBUG: Restore theme toggle button --- */}
         <button onClick={toggleTheme}> {/* Restore button */}
           {/* Use contextTheme here if needed, or keep hardcoded for now */}
           {contextTheme === "dark" ? "☀️" : "🌙"}
         </button>
        {/* --- END DEBUG --- */}

        {/* Display API Connection Status */}
        <div>
          API Status: {apiConnected === null ? 'Checking...' : apiConnected ? 'Connected' : 'Disconnected'}
        </div>
      </header>

      {/* Search Bar Area */} 
      <div className="search-container" style={{ padding: '1rem', display: 'flex', gap: '0.5rem' }}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => {
            // if (e.key === 'Enter') {
            //   handleSearch(inputValue);
            // }
          }}
          placeholder="Search for a word..."
          disabled={!apiConnected} // Disable if API not connected
          style={{ flexGrow: 1, padding: '0.5rem' }}
        />
        <button 
          onClick={handleSearch} // Use dummy handler
          disabled={isLoading || !apiConnected || !inputValue.trim()}
        >
          {isLoading ? 'Checking API...' : 'Search'} // Keep loading text for now
        </button>
      </div>

      <main>
        {/* --- DEBUG: Show simplified content --- */}
        <div>
          <p>WordExplorer component rendered (simplified).</p>
          <p>Initial data fetching and main content rendering are disabled.</p>
        </div>
        {/* 
        // Display Loading State (main loading, not etymology)
        {isLoading && <div>Loading Word Details...</div>}
        // Display General Error Message
        {error && <div style={{ color: 'red' }}>Error: {error}</div>}
        // Display Search Error Message
        {searchError && !error && <div style={{ color: 'orange' }}>Search Error: {searchError}</div>} // Hide if general error exists
        
        // Conditionally render WordDetails or search results/placeholder
        {!isLoading && !error && (
          <div>
            // Restore WordDetails rendering, but pass minimal props
            {selectedWordInfo ? ( 
              <WordDetails 
                wordInfo={selectedWordInfo} 
                // Remove Etymology related props for testing
                etymologyTree={null} // Pass null explicitly
                isLoadingEtymology={false}
                etymologyError={null}
                onWordLinkClick={handleWordLinkClick} 
                onEtymologyNodeClick={() => {}} // Pass dummy function
              />
            ) : searchResults.length > 0 ? (
              // Still show search results if no word is selected yet
              <div>
                <h2>Search Results:</h2>
                <ul>
                  {searchResults.map(word => (
                    <li key={word.id} 
                        onClick={() => handleWordLinkClick(word.id)} 
                        style={{ cursor: 'pointer', textDecoration: 'underline', color: 'blue' }}
                    >
                      {word.lemma}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              // Initial placeholder or no results message
              <div>{searchError ? '' : 'Enter a word to search, or click a result.'}</div>
            )}
            // Keep the raw data display commented out for now
            // {selectedWordInfo && <pre>Selected: {JSON.stringify(selectedWordInfo, null, 2)}</pre>}
          </div>
        )}
        */}
        {/* --- END DEBUG --- */}
      </main>
    </div>
  );
};

export default WordExplorer;
