import React, { useEffect, useState, useCallback } from 'react'; // Added useCallback
import { useTheme } from "../contexts/ThemeContext"; // Restore useTheme import
import { testApiConnection, searchWords } from "../api/wordApi"; // Added searchWords
import { WordInfo, SearchWordResult, SearchOptions } from '../types'; // Added SearchWordResult, SearchOptions

const WordExplorer: React.FC = () => {
  const { theme, toggleTheme } = useTheme(); // Restore useTheme hook
  // const theme = 'light'; // Remove default theme
  // const toggleTheme = () => {}; // Remove dummy function
  
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

  // Updated useEffect for API connection test
  useEffect(() => {
    console.log("Testing API connection...");
    setIsLoading(true); // Set loading state
    setError(null); // Clear previous errors
    testApiConnection()
      .then(connected => {
        console.log("API Connection:", connected);
        setApiConnected(connected);
        if (!connected) {
          setError("API connection failed. Please check the backend server.");
        }
      })
      .catch(error => {
        console.error("API Error:", error);
        setError(error.message || "An unknown error occurred during API connection test.");
        setApiConnected(false);
      })
      .finally(() => {
        setIsLoading(false); // Clear loading state
      });
  }, []);

  // Handle Search Function
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim() || !apiConnected) {
      setSearchResults([]);
      setShowSuggestions(false);
      if (!apiConnected) setError("Cannot search: API is disconnected.");
      return;
    }

    console.log(`Starting search for: "${query}"`);
    setIsLoading(true); // Use main loading indicator for now
    setSearchError(null);
    setError(null); 

    try {
      const searchOptions: SearchOptions = {
        page: 1,
        per_page: 10, // Limit results initially
        mode: 'all',
        sort: 'relevance'
      };
      const result = await searchWords(query, searchOptions);
      console.log("Search results:", result);

      if (result.error) {
        setSearchError(result.error);
        setSearchResults([]);
      } else if (result && result.words && result.words.length > 0) {
        setSearchResults(result.words);
        // For now, just log. We'll handle selecting/displaying later.
        console.log(`Found ${result.words.length} words.`);
      } else {
        setSearchError(`No results found for "${query}".`);
        setSearchResults([]);
      }
    } catch (err: any) {
      console.error("Search error:", err);
      setSearchError(err.message || "An error occurred during search.");
      setSearchResults([]);
    } finally {
      setIsLoading(false);
      setIsLoadingSuggestions(false); // Ensure this is reset
      setShowSuggestions(false); // Hide suggestions after search for now
    }
  }, [apiConnected]); // Depend on apiConnected
  
  return (
    <div className={`word-explorer ${theme}`}>
      <header>
        <h1>Filipino Root Word Explorer</h1>
        <button onClick={toggleTheme}> {/* Restore button */}
          {theme === "dark" ? "☀️" : "🌙"}
        </button>
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
            if (e.key === 'Enter') {
              handleSearch(inputValue);
            }
          }}
          placeholder="Search for a word..."
          disabled={!apiConnected} // Disable if API not connected
          style={{ flexGrow: 1, padding: '0.5rem' }}
        />
        <button 
          onClick={() => handleSearch(inputValue)} 
          disabled={isLoading || !apiConnected || !inputValue.trim()}
        >
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </div>

      <main>
        {/* Display Loading State */}
        {isLoading && !isLoadingSuggestions && <div>Loading...</div>} {/* Adjusted loading display */}
        {/* Display General Error Message */}
        {error && <div style={{ color: 'red' }}>Error: {error}</div>}
        {/* Display Search Error Message */}
        {searchError && <div style={{ color: 'orange' }}>Search Error: {searchError}</div>}
        
        {/* Basic content or search results placeholder */}
        {!isLoading && !error && (
          <div>
            {/* Display search results simply for now */}            
            {searchResults.length > 0 ? (
              <ul>
                {searchResults.map(word => (
                  <li key={word.id}>{word.lemma}</li>
                ))}
              </ul>
            ) : (
              <div>Enter a word and click Search.</div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default WordExplorer;
