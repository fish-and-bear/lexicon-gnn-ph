import React, { useState, useCallback, useEffect } from "react";
import WordGraph from "./WordGraph";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchResult, SearchOptions } from "../types";
import unidecode from "unidecode";
import { fetchWordNetwork, fetchWordDetails, searchWords, resetCircuitBreaker, testApiConnection } from "../api/wordApi";
import axios from 'axios';
import DOMPurify from 'dompurify';
import { debounce } from "lodash";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://api.hapinas.net/api/v1';

const WordExplorer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [wordNetwork, setWordNetwork] = useState<WordNetwork | null>(null);
  const [mainWord, setMainWord] = useState<string>("");
  const [selectedWordInfo, setSelectedWordInfo] = useState<WordInfo | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const { theme, toggleTheme } = useTheme();
  const [inputValue, setInputValue] = useState<string>("");
  const [depth, setDepth] = useState<number>(2);
  const [breadth, setBreadth] = useState<number>(10);
  const [wordHistory, setWordHistory] = useState<string[]>([]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState<number>(-1);
  const [searchResults, setSearchResults] = useState<Array<{ id: number; word: string }>>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [apiEndpoint, setApiEndpoint] = useState<string | null>(localStorage.getItem('successful_api_endpoint'));

  // Function to normalize input
  const normalizeInput = (input: string) => unidecode(input.trim().toLowerCase());

  const fetchWordNetworkData = useCallback(async (word: string, depth: number, breadth: number) => {
    try {
      return await fetchWordNetwork(word, { 
        depth,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.3
      });
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
      throw error; // Pass the error through instead of creating a new one
    }
  }, []);

  const handleSearch = useCallback(async (searchWord?: string) => {
    const wordToSearch = searchWord || inputValue.trim();
    if (!wordToSearch) {
      setError("Please enter a word to search");
      return;
    }
  
    const sanitizedInput = DOMPurify.sanitize(wordToSearch);
    const normalizedInput = normalizeInput(sanitizedInput);

    setIsLoading(true);
    setError(null);
    setWordNetwork(null);

    try {
      console.log('Searching for word:', normalizedInput);
      
      // First try to get word details directly
      try {
        console.log('Fetching word details for:', normalizedInput);
        const detailsData = await fetchWordDetails(normalizedInput);
        console.log('Word details:', detailsData);
        
        if (detailsData && detailsData.lemma) {
          setSelectedWordInfo(detailsData);
          setMainWord(detailsData.lemma);
          
          console.log('Fetching word network for:', normalizedInput);
          let networkData = await fetchWordNetworkData(normalizedInput, depth, breadth);
          console.log('Word network:', networkData);
          
          setWordNetwork(networkData);
          setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), detailsData.lemma]);
          setCurrentHistoryIndex(prevIndex => prevIndex + 1);
          setInputValue(detailsData.lemma);
          setShowSuggestions(false);
          setIsLoading(false);
          return;
        }
      } catch (detailsError) {
        console.log('Direct word details fetch failed, trying search:', detailsError);
        // Continue to search if direct fetch fails
      }
      
      // If direct fetch fails, try search
      const searchResults = await searchWords(normalizedInput, { 
        page: 1, 
        per_page: 20, 
        exclude_baybayin: true,
        language: 'tl',
        mode: 'all',
        sort: 'relevance',
        order: 'desc'
      });
      
      console.log('Search results:', searchResults);
      
      if (searchResults && searchResults.words && searchResults.words.length > 0) {
        // Use the first search result
        const firstResult = searchResults.words[0];
        console.log('Using first search result:', firstResult);
        
        // Fetch details for the first result
        const detailsData = await fetchWordDetails(firstResult.word);
        console.log('Word details for search result:', detailsData);
        
        if (detailsData && detailsData.lemma) {
          setSelectedWordInfo(detailsData);
          setMainWord(detailsData.lemma);
          
          console.log('Fetching word network for search result:', detailsData.lemma);
          let networkData = await fetchWordNetworkData(detailsData.lemma, depth, breadth);
          console.log('Word network for search result:', networkData);
          
          setWordNetwork(networkData);
          setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), detailsData.lemma]);
          setCurrentHistoryIndex(prevIndex => prevIndex + 1);
          setInputValue(detailsData.lemma);
          setShowSuggestions(false);
        } else {
          throw new Error("Could not fetch details for the search result.");
        }
      } else {
        throw new Error("No results found. Try a different search term.");
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      let errorMessage = "Failed to fetch word data.";
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (axios.isAxiosError(error) && error.response) {
        errorMessage = error.response.data?.error?.message || error.message;
      }
      setError(errorMessage);
      setWordNetwork(null);
      setSelectedWordInfo(null);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, depth, breadth, currentHistoryIndex, fetchWordNetworkData]);

  const debouncedSearch = useCallback(
    debounce(async (query: string) => {
      if (query.length > 1) {
        setIsLoading(true);
        setError(null);
        console.log('Searching for:', query);
        try {
          const results = await searchWords(query, { 
            page: 1, 
            per_page: 20, 
            exclude_baybayin: true,
            language: 'tl',
            mode: 'all',
            sort: 'relevance',
            order: 'desc',
            is_real_word: true
          });
          console.log('API search response:', results);
          
          if (results && results.words && results.words.length > 0) {
            const searchResults = results.words
              .filter(word => word && word.word && typeof word.word === 'string')
              .map(word => ({
                id: word.id || Math.random(),
                word: word.word.trim()
              }))
              .filter(result => result.word !== '');
            
            console.log('Processed search results:', searchResults);
            
            setSearchResults(searchResults);
            setShowSuggestions(searchResults.length > 0);
            
            if (searchResults.length === 0) {
              setError("No results found. Try a different search term.");
            }
          } else {
            console.error('Invalid API response or no results:', results);
            setSearchResults([]);
            setShowSuggestions(false);
            
            // Try direct word lookup if search returns no results
            try {
              const normalizedQuery = normalizeInput(query);
              const wordDetails = await fetchWordDetails(normalizedQuery);
              
              if (wordDetails && wordDetails.lemma) {
                console.log('Found word via direct lookup:', wordDetails);
                // If we found the word directly, trigger the search with it
                handleSearch(wordDetails.lemma);
                return;
              }
            } catch (directLookupError) {
              console.log('Direct lookup failed:', directLookupError);
              setError("No results found. Try a different search term.");
            }
          }
        } catch (error) {
          console.error("Error fetching search results:", error);
          let errorMessage = "Failed to fetch search results.";
          if (error instanceof Error) {
            errorMessage = error.message;
          } else if (axios.isAxiosError(error) && error.response) {
            errorMessage = error.response.data?.error?.message || error.message;
          }
          setError(errorMessage);
          setSearchResults([]);
          setShowSuggestions(false);
        } finally {
          setIsLoading(false);
        }
      } else {
        setSearchResults([]);
        setShowSuggestions(false);
      }
    }, 300),
    [handleSearch]
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setInputValue(value);
    setError(null);
    if (value.length > 1) {
      debouncedSearch(value);
      setShowSuggestions(true);
    } else {
      setSearchResults([]);
      setShowSuggestions(false);
    }
  };

  const handleClickOutside = useCallback((event: MouseEvent) => {
    const searchContainer = document.querySelector('.search-input-container');
    if (searchContainer && !searchContainer.contains(event.target as Node)) {
      setShowSuggestions(false);
    }
  }, []);

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [handleClickOutside]);

  const handleSuggestionClick = (word: string) => {
    setInputValue(word);
    setShowSuggestions(false);
    handleSearch(word);
  };

  const handleNodeClick = useCallback(async (word: string) => {
    setError(null);
    setIsLoading(true);
    
    try {
      // First, get the word details
      const detailsData = await fetchWordDetails(word);
      setSelectedWordInfo(detailsData);
      
      // Then, get the word network
      const networkData = await fetchWordNetworkData(word, depth, breadth);
      
      // Update state with the new data
      setWordNetwork(networkData);
      setMainWord(detailsData.lemma);
      
      // Update history
      setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), detailsData.lemma]);
      setCurrentHistoryIndex(prevIndex => prevIndex + 1);
      
      // Update input value
      setInputValue(detailsData.lemma);
    } catch (error) {
      console.error("Error fetching node data:", error);
      let errorMessage = "Failed to fetch word data.";
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (axios.isAxiosError(error) && error.response) {
        errorMessage = error.response.data?.error?.message || error.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [depth, breadth, currentHistoryIndex, fetchWordNetworkData]);

  const handleNetworkChange = useCallback((newDepth: number, newBreadth: number) => {
    setDepth(newDepth);
    setBreadth(newBreadth);
    if (mainWord) {
      setIsLoading(true);
      fetchWordNetworkData(normalizeInput(mainWord), newDepth, newBreadth)
        .then(networkData => {
          setWordNetwork(networkData);
        })
        .catch(error => {
          console.error("Error updating network:", error);
          setError("Failed to update network. Please try again.");
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [mainWord, fetchWordNetworkData]);

  const renderDefinitions = useCallback((wordInfo: WordInfo) => {
    if (!wordInfo.definitions || wordInfo.definitions.length === 0) {
      return <div className="no-definitions">No definitions available for this word.</div>;
    }

    // Group definitions by part of speech
    const definitionsByPos: Record<string, any[]> = {};
    
    wordInfo.definitions.forEach(definition => {
      const posName = definition.part_of_speech?.name_en || 'Other';
      if (!definitionsByPos[posName]) {
        definitionsByPos[posName] = [];
      }
      definitionsByPos[posName].push(definition);
    });

    return (
      <div className="definitions-section">
        <div className="definitions-section-header">
          <h3>Definitions</h3>
          <span className="definition-count">{wordInfo.definitions.length}</span>
        </div>
        
        {Object.entries(definitionsByPos).map(([posName, definitions]: [string, any[]]) => (
          <div key={posName} className="pos-group">
            <div className="pos-group-header">
              {posName}
              <span className="pos-count">{definitions.length}</span>
            </div>
            
            {definitions.map((definition: any, index: number) => (
              <div key={index} className="definition-card">
                <p className="definition-text">{definition.text}</p>
                
                {definition.examples && definition.examples.length > 0 && (
                  <div className="examples">
                    <h4>Examples</h4>
                    <ul>
                      {definition.examples.map((example: string, idx: number) => {
                        return <li key={idx}>{example}</li>;
                      })}
                    </ul>
                  </div>
                )}
                
                {definition.usage_notes && definition.usage_notes.length > 0 && (
                  <div className="usage-notes">
                    <h4>Usage Notes</h4>
                    <ul>
                      {definition.usage_notes.map((note: string, idx: number) => {
                        return <li key={idx}>{note}</li>;
                      })}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
    );
  }, []);

  const renderArraySection = useCallback((title: string, items?: string[]) => {
    if (!items || items.length === 0) return null;
    return (
      <div className={title.toLowerCase().replace(/\s+/g, "-")}>
        <h3>{title}</h3>
        <ul className="word-list">
          {items
            .filter((item) => item.trim() !== "" && item.trim() !== "0")
            .map((item, index) => (
              <li
                key={index}
                onClick={() => handleNodeClick(item)}
                className="clickable-word"
              >
                {item}
              </li>
            ))}
        </ul>
      </div>
    );
  }, [handleNodeClick]);

  const handleBack = useCallback(() => {
    if (currentHistoryIndex > 0) {
      setCurrentHistoryIndex(prevIndex => prevIndex - 1);
      const previousWord = wordHistory[currentHistoryIndex - 1];
      handleNodeClick(previousWord);
    }
  }, [currentHistoryIndex, wordHistory, handleNodeClick]);

  const handleForward = useCallback(() => {
    if (currentHistoryIndex < wordHistory.length - 1) {
      setCurrentHistoryIndex(prevIndex => prevIndex + 1);
      const nextWord = wordHistory[currentHistoryIndex + 1];
      handleNodeClick(nextWord);
    }
  }, [currentHistoryIndex, wordHistory, handleNodeClick]);

  // Function to reset the circuit breaker
  const handleResetCircuitBreaker = () => {
    resetCircuitBreaker();
    setError(null);
    // Retry the last search if there was one
    if (inputValue) {
      handleSearch(inputValue);
    }
  };

  // Function to manually test API connection
  const handleTestApiConnection = async () => {
    setError(null);
    setApiConnected(null); // Set to checking state
    
    try {
      console.log("Manually testing API connection...");
      
      // First try the testApiConnection function
      const isConnected = await testApiConnection();
      setApiConnected(isConnected);
      
      if (!isConnected) {
        console.log("API connection test failed, trying direct fetch...");
        
        // Reset circuit breaker before trying direct fetch
        resetCircuitBreaker();
        
        // Try direct fetch as a fallback
        try {
          const response = await fetch('http://127.0.0.1:10000/', {
            method: 'GET',
            headers: { 'Accept': 'application/json' },
            mode: 'cors',
            cache: 'no-cache'
          });
          
          const directFetchSuccess = response.ok;
          setApiConnected(directFetchSuccess);
          
          if (directFetchSuccess) {
            console.log("API connection successful with direct fetch!");
            setError(null);
            // Update API client with the successful endpoint
            localStorage.setItem('successful_api_endpoint', 'http://127.0.0.1:10000');
            
            // If connection is successful, try to use the API
            if (inputValue) {
              handleSearch(inputValue);
            }
          } else {
            setError("Cannot connect to the API server. Please check that the backend server is running.");
          }
        } catch (e) {
          console.error("Direct fetch attempt failed:", e);
          setApiConnected(false);
          setError("Cannot connect to the API server. Please check that the backend server is running and network settings allow the connection.");
        }
      } else {
        // If connection is successful, try to use the API
        if (inputValue) {
          handleSearch(inputValue);
        }
      }
    } catch (e) {
      console.error("Error testing API connection:", e);
      setApiConnected(false);
      setError("Error testing API connection. Please try again later.");
    }
  };

  // Test API connectivity on mount
  useEffect(() => {
    const checkApiConnection = async () => {
      try {
        console.log("Checking API connection...");
        setApiConnected(null); // Set to checking state
        setError(null); // Clear any previous errors
        
        // First attempt
        const isConnected = await testApiConnection();
        setApiConnected(isConnected);
        
        if (!isConnected) {
          console.log("Initial API connection failed, retrying in 1 second...");
          setError("API connection failed. Retrying...");
          
          // Reset circuit breaker before retry
          resetCircuitBreaker();
          
          // Retry after 1 second
          setTimeout(async () => {
            console.log("Retrying API connection...");
            setApiConnected(null); // Set to checking state again
            const retryResult = await testApiConnection();
            setApiConnected(retryResult);
            
            if (!retryResult) {
              // Try one more time with a different approach
              setTimeout(async () => {
                console.log("Final API connection retry...");
                setApiConnected(null);
                
                // Try with direct fetch to avoid any axios/circuit breaker issues
                try {
                  const response = await fetch('http://127.0.0.1:10000/', {
                    method: 'GET',
                    headers: { 'Accept': 'application/json' },
                    mode: 'cors',
                    cache: 'no-cache'
                  });
                  
                  const finalRetrySuccess = response.ok;
                  setApiConnected(finalRetrySuccess);
                  
                  if (finalRetrySuccess) {
                    console.log("API connection successful on final retry!");
                    setError(null);
                    // Update API client with the successful endpoint
                    localStorage.setItem('successful_api_endpoint', 'http://127.0.0.1:10000');
                  } else {
                    const errorMsg = "Cannot connect to the API server. The backend server may not be running. Please start the backend server and click 'Test API'.";
                    console.error(errorMsg);
                    setError(errorMsg);
                  }
                } catch (e) {
                  console.error("Final API connection attempt failed:", e);
                  const errorMsg = "Cannot connect to the API server. The backend server may not be running. Please start the backend server and click 'Test API'.";
                  setError(errorMsg);
                  setApiConnected(false);
                }
              }, 1000);
            } else {
              // Clear any previous errors if connection is successful
              setError(null);
              console.log("API connection successful on retry!");
            }
          }, 1000);
        } else {
          // Clear any previous errors if connection is successful
          setError(null);
          console.log("API connection successful on first attempt!");
        }
      } catch (e) {
        console.error("Error checking API connection:", e);
        setApiConnected(false);
        setError("Error checking API connection. Please try again by clicking 'Test API'.");
      }
    };
    
    checkApiConnection();
  }, []);

  console.log('Search query:', inputValue);
  console.log('Search results:', searchResults);
  console.log('Show suggestions:', showSuggestions);

  return (
    <div className={`word-explorer ${theme}`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <div className="header-buttons">
          <button
            onClick={handleResetCircuitBreaker}
            className="debug-button"
            title="Reset API connection"
          >
            🔄 Reset API
          </button>
          <button
            onClick={handleTestApiConnection}
            className="debug-button"
            title="Test API connection"
          >
            🔌 Test API
          </button>
          <div className={`api-status ${
            apiConnected === null ? 'checking' : 
            apiConnected ? 'connected' : 'disconnected'
          }`}>
            API: {apiConnected === null ? 'Checking...' : 
                 apiConnected ? '✅ Connected' : '❌ Disconnected'}
          </div>
          <button
            onClick={toggleTheme}
            className="theme-toggle"
            aria-label="Toggle theme"
          >
            {theme === "light" ? "🌙" : "☀️"}
          </button>
        </div>
      </header>
      <div className="search-container">
        <button
          onClick={handleBack}
          disabled={currentHistoryIndex <= 0}
          className="history-button"
          aria-label="Go back"
        >
          ←
        </button>
        <button
          onClick={handleForward}
          disabled={currentHistoryIndex >= wordHistory.length - 1}
          className="history-button"
          aria-label="Go forward"
        >
          →
        </button>
        <div className="search-input-container">
          <input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleSearch();
              }
            }}
            placeholder="Enter a word"
            className="search-input"
            aria-label="Search word"
          />
          {isLoading && <div className="search-loading">Loading...</div>}
          {showSuggestions && searchResults.length > 0 && (
            <ul className="search-suggestions">
              {searchResults.map((result) => (
                <li key={result.id} onClick={() => handleSuggestionClick(result.word)}>
                  {result.word}
                </li>
              ))}
            </ul>
          )}
        </div>
        <button
          onClick={() => handleSearch()}
          disabled={isLoading}
          className="search-button"
        >
          Search
        </button>
      </div>
      {error && (
        <div className="error-message">
          <p>{error}</p>
          {error.includes('Circuit breaker') && (
            <button onClick={handleResetCircuitBreaker} className="reset-button">
              Reset Connection
            </button>
          )}
          {error.includes('API connection') && (
            <div className="error-actions">
              <button onClick={handleResetCircuitBreaker} className="reset-button">
                Reset Connection
              </button>
              <button 
                onClick={handleTestApiConnection} 
                className="retry-button"
              >
                Test API Connection
              </button>
            </div>
          )}
          {error.includes('backend server') && (
            <div className="error-actions">
              <div className="backend-instructions">
                <p><strong>To start the backend server:</strong></p>
                <ol>
                  <li>Open a new terminal/command prompt</li>
                  <li>Navigate to the project directory</li>
                  <li>Run: <code>cd backend</code></li>
                  <li>Run: <code>python serve.py</code></li>
                </ol>
              </div>
              <button 
                onClick={handleTestApiConnection} 
                className="retry-button"
              >
                Test API Connection
              </button>
            </div>
          )}
          {error.includes('Network error') && (
            <div className="error-actions">
              <button onClick={handleResetCircuitBreaker} className="reset-button">
                Reset Connection
              </button>
              <button 
                onClick={async () => {
                  const isConnected = await testApiConnection();
                  setApiConnected(isConnected);
                  if (isConnected && inputValue) {
                    handleSearch(inputValue);
                  }
                }} 
                className="retry-button"
              >
                Test Connection & Retry
              </button>
            </div>
          )}
        </div>
      )}
      <main>
        <div className="graph-container">
          <div className="graph-content">
            {wordNetwork && mainWord && wordNetwork.nodes && wordNetwork.nodes.length > 0 && (
              <WordGraph
                wordNetwork={wordNetwork}
                mainWord={mainWord}
                onNodeClick={handleNodeClick}
                onNetworkChange={handleNetworkChange}
                initialDepth={depth}
                initialBreadth={breadth}
              />
            )}
          </div>
        </div>
        <div className="details-container">
          <div className="details-content">
            {isLoading ? (
              <div className="loading-spinner">Loading word data...</div>
            ) : selectedWordInfo ? (
              <div className="word-details">
                <div className="word-header">
                  <h2>{selectedWordInfo.lemma}</h2>
                  {selectedWordInfo.language_code && (
                    <span className="language">{selectedWordInfo.language_code}</span>
                  )}
                </div>
                
                {selectedWordInfo.pronunciation?.text && (
                  <p className="pronunciation">
                    <strong>Pronunciation:</strong>{" "}
                    {selectedWordInfo.pronunciation.text}
                    {selectedWordInfo.pronunciation.ipa && (
                      <span className="ipa">[{selectedWordInfo.pronunciation.ipa}]</span>
                    )}
                    {selectedWordInfo.pronunciation.audio_url && (
                      <button 
                        className="play-audio"
                        onClick={() => {
                          const audio = new Audio(selectedWordInfo.pronunciation?.audio_url);
                          audio.play().catch(console.error);
                        }}
                      >
                        🔊
                      </button>
                    )}
                  </p>
                )}
                
                {selectedWordInfo.etymologies && selectedWordInfo.etymologies.length > 0 && (
                  <div className="etymology-section">
                    <h3>Etymology</h3>
                    <p>{selectedWordInfo.etymologies[0].text}</p>
                  </div>
                )}
                
                {renderDefinitions(selectedWordInfo)}
                
                {selectedWordInfo.relations && (
                  Object.values(selectedWordInfo.relations).some(val => 
                    Array.isArray(val) ? val.length > 0 : val !== null
                  ) && (
                    <div className="relations-section">
                      {renderArraySection(
                        "Synonyms",
                        selectedWordInfo.relations.synonyms?.map(item => item.word) || []
                      )}
                      {renderArraySection(
                        "Antonyms",
                        selectedWordInfo.relations.antonyms?.map(item => item.word) || []
                      )}
                      {renderArraySection(
                        "Related Words",
                        selectedWordInfo.relations.related?.map(item => item.word) || []
                      )}
                      {renderArraySection(
                        "Derivatives",
                        selectedWordInfo.relations.derived?.map(item => item.word) || []
                      )}
                      {selectedWordInfo.relations.root && (
                        <div className="root-word">
                          <h3>Root Word</h3>
                          <span
                            className="clickable-word"
                            onClick={() =>
                              handleNodeClick(
                                selectedWordInfo.relations.root!.word
                              )
                            }
                          >
                            {selectedWordInfo.relations.root.word}
                          </span>
                        </div>
                      )}
                    </div>
                  )
                )}
              </div>
            ) : (
              <div className="empty-state">
                <p>Enter a word to explore or click on a node in the graph to see details.</p>
              </div>
            )}
          </div>
        </div>
      </main>
      <footer className="footer">
        © {new Date().getFullYear()} Filipino Root Word Explorer. All Rights
        Reserved.
      </footer>
    </div>
  );
};

export default WordExplorer;
