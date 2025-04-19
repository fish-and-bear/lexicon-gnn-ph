import React, { useEffect, useState, useCallback } from 'react'; // Added useCallback
import { useTheme } from "../contexts/ThemeContext"; // Restore useTheme import
import { testApiConnection, searchWords, fetchWordDetails, getEtymologyTree } from "../api/wordApi"; // Added fetchWordDetails, getEtymologyTree
import { WordInfo, SearchWordResult, SearchOptions, EtymologyTree } from '../types'; // Added EtymologyTree
import WordDetails from './WordDetails'; // Import WordDetails component

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

  // Etymology state variables
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);

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

  // Fetch Etymology Tree
  const fetchEtymologyData = useCallback(async (wordId: number) => {
    if (!wordId) return;
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    setEtymologyTree(null); // Clear previous tree
    try {
      console.log("Fetching etymology for word ID:", wordId);
      const treeData = await getEtymologyTree(wordId, 3); // Use default depth 3 for now
      setEtymologyTree(treeData);
    } catch (err: any) {
      console.error("Error fetching etymology tree:", err);
      setEtymologyError(err.message || "Failed to fetch etymology.");
    } finally {
      setIsLoadingEtymology(false);
    }
  }, []);

  // Load Word Details
  const loadWordData = useCallback(async (wordIdentifier: string | number) => {
    console.log(`Loading data for word: ${wordIdentifier}`);
    setIsLoading(true);
    setError(null);
    setSelectedWordInfo(null); // Clear previous selection
    setEtymologyTree(null);   // Clear previous etymology
    setEtymologyError(null);

    try {
      // Ensure identifier is a string for fetchWordDetails
      const identifierString = String(wordIdentifier).startsWith('id:') 
                               ? String(wordIdentifier) 
                               : `id:${wordIdentifier}`;
      
      const wordData = await fetchWordDetails(identifierString);
      setSelectedWordInfo(wordData);
      setMainWord(wordData.lemma); // Update mainWord as well
      setInputValue(wordData.lemma); // Sync input field

      // Fetch etymology in the background
      if (wordData.id) {
         // Convert potential 'id:123' back to number for etymology fetch
         const etymologyIdString = String(wordData.id);
         const etymologyId = etymologyIdString.startsWith('id:') 
           ? parseInt(etymologyIdString.substring(3), 10) 
           : wordData.id;
        fetchEtymologyData(etymologyId);
      }

    } catch (err: any) {
      console.error("Error loading word data:", err);
      setError(err.message || `Failed to load details for ${wordIdentifier}.`);
    } finally {
      setIsLoading(false);
    }
  }, [fetchEtymologyData]); // Add fetchEtymologyData dependency

  // Handle Search Function (updated)
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim() || !apiConnected) {
      setSearchResults([]);
      setShowSuggestions(false);
      setSelectedWordInfo(null); // Clear selection on empty/failed search
      setEtymologyTree(null);
      if (!apiConnected) setError("Cannot search: API is disconnected.");
      return;
    }

    console.log(`Starting search for: "${query}"`);
    setIsLoading(true); 
    setSearchError(null);
    setError(null); 
    setSelectedWordInfo(null); // Clear previous selection
    setEtymologyTree(null);

    try {
      const searchOptions: SearchOptions = {
        page: 1,
        per_page: 10, 
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
        // Automatically load the first result
        await loadWordData(result.words[0].id); 
      } else {
        setSearchError(`No results found for "${query}".`);
        setSearchResults([]);
      }
    } catch (err: any) {
      console.error("Search error:", err);
      setSearchError(err.message || "An error occurred during search.");
      setSearchResults([]);
    } finally {
      // Don't set isLoading false here if loadWordData was called,
      // as loadWordData will handle its own loading state.
      // setIsLoading(false); 
      setIsLoadingSuggestions(false); 
      setShowSuggestions(false); 
    }
  }, [apiConnected, loadWordData]); // Add loadWordData dependency
  
  // Handler for clicking links within WordDetails (or search results later)
  const handleWordLinkClick = useCallback((wordIdentifier: string | number) => {
    // For now, just call loadWordData
    loadWordData(wordIdentifier);
  }, [loadWordData]);

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
        {/* Display Loading State (main loading, not etymology) */}
        {isLoading && <div>Loading Word Details...</div>}
        {/* Display General Error Message */}
        {error && <div style={{ color: 'red' }}>Error: {error}</div>}
        {/* Display Search Error Message */}
        {searchError && !error && <div style={{ color: 'orange' }}>Search Error: {searchError}</div>} {/* Hide if general error exists */}
        
        {/* Conditionally render WordDetails or search results/placeholder */} 
        {!isLoading && !error && (
          <div>
            {/* Restore WordDetails rendering, but pass minimal props */}
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
            {/* Keep the raw data display commented out for now */}
            {/* {selectedWordInfo && <pre>Selected: {JSON.stringify(selectedWordInfo, null, 2)}</pre>} */}
          </div>
        )}
      </main>
    </div>
  );
};

export default WordExplorer;
