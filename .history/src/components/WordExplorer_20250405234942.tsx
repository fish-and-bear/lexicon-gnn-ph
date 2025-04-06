import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import {
  WordNetworkResponse,
  WordInfo,
  SearchResult,
  SearchWordResult,
  SearchOptions,
  EtymologyTree,
  Statistics,
  Definition,
  Etymology,
  Pronunciation,
  PartOfSpeech,
  Language,
  WordForm,
  WordTemplate,
  BasicWord,
  Relation,
  Affixation,
  Credit,
} from "../types";
import unidecode from "unidecode";
import { 
  fetchWordNetwork, 
  fetchWordDetails, 
  searchWords, 
  resetCircuitBreaker, 
  forceCircuitBreakerClosed,
  testApiConnection,
  getEtymologyTree,
  getPartsOfSpeech,
  getRandomWord,
  getStatistics,
  API_BASE_URL
} from "../api/wordApi";
import axios from 'axios';
import DOMPurify from 'dompurify';
import { debounce } from "lodash";

const WordExplorer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [wordNetwork, setWordNetwork] = useState<WordNetworkResponse | null>(null);
  const [selectedWordInfo, setSelectedWordInfo] = useState<WordInfo | null>(null);
  const [mainWord, setMainWord] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [depth, setDepth] = useState<number>(2);
  const [inputValue, setInputValue] = useState<string>("");
  const [wordHistory, setWordHistory] = useState<string[]>([]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState<number>(-1);
  const [searchResults, setSearchResults] = useState<SearchWordResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [partsOfSpeech, setPartsOfSpeech] = useState<PartOfSpeech[]>([]);
  const [languages, setLanguages] = useState<Language[]>([]);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [isLoadingStatistics, setIsLoadingStatistics] = useState<boolean>(false);

  const detailsContainerRef = useRef<HTMLDivElement>(null);
  const searchContainerRef = useRef<HTMLDivElement>(null);

  const { theme, toggleTheme } = useTheme();
  const [showMetadata, setShowMetadata] = useState(false);

  const normalizeInput = (input: string): string => {
    return unidecode(input.toLowerCase().trim());
  };

  const fetchWordNetworkData = useCallback(async (word: string, currentDepth: number = 2) => {
    try {
      setIsLoading(true);
      const data = await fetchWordNetwork(word, { depth: currentDepth });
      setWordNetwork(data);
      return data;
    } catch (fetchError) {
      console.error("Error in fetchWordNetworkData:", fetchError);
      setWordNetwork(null);
      throw fetchError;
    } finally {
    }
  }, []);

  const fetchEtymologyTree = useCallback(async (wordId: number) => {
    if (!wordId) return;
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    try {
      const data = await getEtymologyTree(wordId, 3);
      setEtymologyTree(data);
    } catch (fetchError) {
      console.error("Error fetching etymology tree:", fetchError);
      let errorMessage = "Failed to fetch etymology tree.";
      if (fetchError instanceof Error) errorMessage = fetchError.message;
      else if (axios.isAxiosError(fetchError)) errorMessage = fetchError.response?.data?.error || fetchError.message;
      setEtymologyError(errorMessage);
      setEtymologyTree(null);
    } finally {
      setIsLoadingEtymology(false);
    }
  }, []);

  const handleSearch = useCallback(async (wordToSearchParam?: string) => {
    const wordToSearch = wordToSearchParam ?? inputValue;
    const normalizedInput = normalizeInput(wordToSearch);
    if (!normalizedInput) return;

    console.log('Starting search for normalized input:', normalizedInput);
    setIsLoading(true);
    setError(null);
    setWordNetwork(null);
    setEtymologyTree(null); 
    setSelectedWordInfo(null);
    setMainWord(normalizedInput);

    try {
      console.log('Attempting direct fetch for:', normalizedInput);
      let wordData: WordInfo | null = null;
      try {
          wordData = await fetchWordDetails(normalizedInput);
          console.log('Direct fetch result:', wordData);
      } catch (detailsError) {
          console.warn('Direct fetch failed (may be expected):', detailsError);
      }
        
      if (wordData && wordData.lemma) {
        console.log('Direct fetch successful, word data:', wordData);
        setSelectedWordInfo(wordData);
        setMainWord(wordData.lemma);
        setInputValue(wordData.lemma);

        setWordHistory(prev => [...prev.slice(0, currentHistoryIndex + 1), wordData!.lemma]);
        setCurrentHistoryIndex(prev => prev + 1);
        setShowSuggestions(false);

        await Promise.all([
          fetchWordNetworkData(wordData.lemma, depth),
          fetchEtymologyTree(wordData.id)
        ]);
      } else {
        console.log('Direct fetch failed, performing search API call...');
        const searchOptions: SearchOptions = { 
          q: normalizedInput,
          limit: 20,
          offset: 0,
          include_full: false
        };
          
        console.log('Search options:', searchOptions);
        const searchResult: SearchResult = await searchWords(searchOptions);
        console.log('Search API response:', searchResult);

        if (searchResult && Array.isArray(searchResult.words) && searchResult.words.length > 0) {
          const firstResult = searchResult.words[0];
          console.log('Using first search result:', firstResult);
          
          const minimalWordInfo: WordInfo = {
            id: firstResult.id,
            lemma: firstResult.lemma,
          };
          setSelectedWordInfo(minimalWordInfo);
          setMainWord(firstResult.lemma);
          setInputValue(firstResult.lemma);
          setWordHistory(prev => [...prev.slice(0, currentHistoryIndex + 1), firstResult.lemma]);
          setCurrentHistoryIndex(prev => prev + 1);
          setShowSuggestions(false);

          console.log('Fetching full details for search result ID:', firstResult.id);
          fetchWordDetails(firstResult.id.toString())
            .then(fullWordData => {
              if (fullWordData && selectedWordInfo && fullWordData.id === selectedWordInfo.id) {
                console.log('Full details received for search result', fullWordData);
                setSelectedWordInfo(fullWordData);
                // Run these promises but don't return their results to the chain
                fetchWordNetworkData(fullWordData.lemma, depth)
                  .then(networkData => {
                    setWordNetwork(networkData);
                  })
                  .catch(err => console.error("Error fetching network data in force search:", err));
                  
                fetchEtymologyTree(fullWordData.id)
                  .catch(err => console.error("Error fetching etymology in force search:", err));
              }
            })
            .catch(err => {
              console.error("Error fetching word details in force search:", err);
              setError(`Error fetching details: ${err.message}`);
            });
        } else {
          setError(`No results found for "${wordToSearch}"`);
          setSelectedWordInfo(null);
        setWordNetwork(null);
          setEtymologyTree(null);
        }
      }
    } catch (searchError: any) {
      console.error("Error during search phase:", searchError);
      let errorMessage = "An error occurred during the search.";
      
      // Create more user-friendly error messages
      if (searchError.message.includes("No response received")) {
        errorMessage = `Search for "${wordToSearch}" is taking too long to respond. The backend may be overloaded. Try a simpler or shorter word, or try again later.`;
      } else if (searchError.message.includes("Network Error")) {
        errorMessage = `Network error while searching for "${wordToSearch}". Check your internet connection and that the backend server is running.`;
      } else if (searchError.message.includes("Circuit breaker")) {
        errorMessage = `Too many failed requests. The system has temporarily stopped making requests to protect against overload. Please wait a moment and try again, or try a different word.`;
      } else if (searchError.message) {
        errorMessage = searchError.message;
      }
      
      setError(errorMessage);
      setSelectedWordInfo(null);
      setWordNetwork(null);
      setEtymologyTree(null);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, depth, fetchWordNetworkData, fetchEtymologyTree, currentHistoryIndex, selectedWordInfo]);

  const handleWordLinkClick = useCallback((word: string) => {
    console.log(`Word link clicked: ${word}`);
    const normalizedWord = normalizeInput(word);
    handleSearch(normalizedWord);
  }, [handleSearch]);

  const handleNodeClick = handleWordLinkClick;

  const debouncedSearch = useCallback(
    debounce(async (query: string) => {
      if (query.length < 2) {
            setSearchResults([]);
            setShowSuggestions(false);
                return;
              }
      try {
        console.log("Fetching search suggestions for:", query);
        const searchOptions: SearchOptions = { q: query, limit: 10, include_full: false };
        const searchResult: SearchResult = await searchWords(searchOptions);
        console.log("Search suggestions response:", searchResult);
        
        if (searchResult && Array.isArray(searchResult.words)) {
          console.log("Setting search results:", searchResult.words.length, "items");
          setSearchResults(searchResult.words);
          setShowSuggestions(searchResult.words.length > 0);
        } else {
          console.log("No valid search results found in response:", searchResult);
          setSearchResults([]);
          setShowSuggestions(false);
          }
        } catch (error) {
        console.error("Error fetching search suggestions:", error);
          setSearchResults([]);
          setShowSuggestions(false);
        }
    }, 300),
    []
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setInputValue(query);
    debouncedSearch(query);
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
    setShowSuggestions(false);
    setSearchResults([]);
    handleSearch(word);
  };

  const handleNetworkChange = useCallback((newDepth: number) => {
    setDepth(newDepth);
    if (mainWord) {
      fetchWordNetworkData(normalizeInput(mainWord), newDepth);
    }
  }, [mainWord, fetchWordNetworkData]);

  const handleBack = useCallback(() => {
    if (currentHistoryIndex > 0) {
      const newIndex = currentHistoryIndex - 1;
      setCurrentHistoryIndex(newIndex);
      const previousWord = wordHistory[newIndex];
      console.log(`Navigating back to: ${previousWord} (index ${newIndex})`);
      
      setIsLoading(true);
      setError(null);
      setSelectedWordInfo(null);
      setWordNetwork(null);
      setEtymologyTree(null);

      fetchWordDetails(previousWord)
        .then(rawWordData => {
          if (!rawWordData) throw new Error("Word details not found");
          setSelectedWordInfo(rawWordData);
          setMainWord(rawWordData.lemma);
          setInputValue(rawWordData.lemma);
          return Promise.all([
            fetchWordNetworkData(previousWord, depth),
            fetchEtymologyTree(rawWordData.id)
          ]);
        })
        .then(([networkData]) => {
        setWordNetwork(networkData);
      })
      .catch(error => {
        console.error("Error navigating back:", error);
          setError(`Failed to navigate back to "${previousWord}": ${error.message}`);
      })
      .finally(() => {
        setIsLoading(false);
      });
    }
  }, [currentHistoryIndex, wordHistory, depth, fetchWordNetworkData, fetchEtymologyTree]);

  const handleForward = useCallback(() => {
    if (currentHistoryIndex < wordHistory.length - 1) {
      const newIndex = currentHistoryIndex + 1;
      setCurrentHistoryIndex(newIndex);
      const nextWord = wordHistory[newIndex];
      console.log(`Navigating forward to: ${nextWord} (index ${newIndex})`);
      
      setIsLoading(true);
      setError(null);
      setSelectedWordInfo(null);
      setWordNetwork(null);
      setEtymologyTree(null);

      fetchWordDetails(nextWord)
        .then(rawWordData => {
          if (!rawWordData) throw new Error("Word details not found");
          setSelectedWordInfo(rawWordData);
          setMainWord(rawWordData.lemma);
          setInputValue(rawWordData.lemma);
          return Promise.all([
            fetchWordNetworkData(nextWord, depth),
            fetchEtymologyTree(rawWordData.id)
          ]);
        })
        .then(([networkData]) => {
        setWordNetwork(networkData);
      })
      .catch(error => {
        console.error("Error navigating forward:", error);
          setError(`Failed to navigate forward to "${nextWord}": ${error.message}`);
      })
      .finally(() => {
        setIsLoading(false);
      });
    }
  }, [currentHistoryIndex, wordHistory, depth, fetchWordNetworkData, fetchEtymologyTree]);

  const handleResetCircuitBreaker = () => {
    resetCircuitBreaker();
    setError(null);
    if (inputValue) {
      handleSearch(inputValue);
    }
  };

  const handleTestApiConnection = async () => {
    setError(null);
    setApiConnected(null);
    
    try {
      console.log("Manually testing API connection...");
      
      const isConnected = await testApiConnection();
      setApiConnected(isConnected);
      
      if (!isConnected) {
        console.log("API connection test failed, trying direct fetch...");
        
        resetCircuitBreaker();
        
        try {
          const response = await fetch('http://127.0.0.1:10000/api/v2/test', {
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
            localStorage.setItem('successful_api_endpoint', 'http://127.0.0.1:10000');
            
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

  useEffect(() => {
    const checkApiConnection = async () => {
      try {
        console.log("Checking API connection...");
        setApiConnected(null);
        setError(null);
        
        const isConnected = await testApiConnection();
        setApiConnected(isConnected);
        
        if (!isConnected) {
          console.log("Initial API connection failed, retrying in 1 second...");
          setError("API connection failed. Retrying...");
          
          resetCircuitBreaker();
          
          setTimeout(async () => {
            console.log("Retrying API connection...");
            setApiConnected(null);
            const retryResult = await testApiConnection();
            setApiConnected(retryResult);
            
            if (!retryResult) {
              setTimeout(async () => {
                console.log("Final API connection retry...");
                setApiConnected(null);
                
                try {
                  const response = await fetch('http://127.0.0.1:10000/api/v2/test', {
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
              setError(null);
              console.log("API connection successful on retry!");
            }
          }, 1000);
        } else {
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

  useEffect(() => {
    const fetchInitialData = async () => {
      setIsLoadingStatistics(true);
      try {
        const [posData, langData, statsData] = await Promise.all([
          getPartsOfSpeech(),
          Promise.resolve([{ code: 'tl', name_en: 'Tagalog' }, { code: 'en', name_en: 'English' }]),
          getStatistics(),
        ]);
        setPartsOfSpeech(posData || []);
        setLanguages(langData || []);
        setStatistics(statsData || null);
      } catch (error) {
        console.error("Error fetching initial data:", error);
        setLanguages([{ code: 'tl', name_en: 'Tagalog' }, { code: 'en', name_en: 'English' }]);
      } finally {
        setIsLoadingStatistics(false);
      }
    };
    fetchInitialData();
  }, []);

  const handleRandomWord = useCallback(async () => {
    setError(null);
    setIsLoading(true);
    setEtymologyTree(null);
    setSelectedWordInfo(null);
    setWordNetwork(null);

    try {
      console.log('Fetching random word (V2)');
      const rawWordData = await getRandomWord();
      console.log('Raw random word data:', rawWordData);

      if (rawWordData && rawWordData.lemma) {
        setSelectedWordInfo(rawWordData);
        setMainWord(rawWordData.lemma);
        setInputValue(rawWordData.lemma);
        setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), rawWordData.lemma]);
        setCurrentHistoryIndex(prevIndex => prevIndex + 1);

        await Promise.all([
          fetchWordNetworkData(rawWordData.lemma, depth),
          fetchEtymologyTree(rawWordData.id)
        ]);
      } else {
        throw new Error("Could not fetch valid random word data.");
      }
    } catch (error) {
      console.error("Error fetching random word:", error);
      let errorMessage = "Failed to fetch random word.";
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (axios.isAxiosError(error)) {
        errorMessage = error.response?.data?.error || error.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [depth, currentHistoryIndex, fetchWordNetworkData, fetchEtymologyTree]);

  const toggleMetadata = useCallback(() => {
    setShowMetadata(prev => !prev);
  }, []);

  const resetDisplay = useCallback(() => {
    setInputValue("");
    setWordNetwork(null); 
    setSelectedWordInfo(null);
    setMainWord("");
    setError(null);
    setIsLoading(false);
    setSearchResults([]);
    setShowSuggestions(false);
    setWordHistory([]);
    setCurrentHistoryIndex(-1);
    setEtymologyTree(null);
    setIsLoadingEtymology(false);
    setEtymologyError(null);
  }, []);

  useEffect(() => {
    // Example: Fetch initial word or reset on mount
    // resetDisplay(); 
  }, []);

  const handleForceSearch = useCallback(async () => {
    setError(null);
    setIsLoading(true);
    
    try {
      // Reset the circuit breaker state
      forceCircuitBreakerClosed();
      
      const normalizedInput = normalizeInput(inputValue);
      console.log('Forcing search to bypass circuit breaker for:', normalizedInput);
      
      // Use the circuit breaker bypass flag
      const searchOptions: SearchOptions = { 
        q: normalizedInput,
        limit: 20,
        offset: 0,
        include_full: false
      };
      
      // Call with the bypass flag set to true
      const searchResult = await searchWords(searchOptions, undefined, true);
      
      if (searchResult && searchResult.words && searchResult.words.length > 0) {
        const firstResult = searchResult.words[0];
        
        // Process the results similarly to the normal search
        const minimalWordInfo: WordInfo = {
          id: firstResult.id,
          lemma: firstResult.lemma,
        };
        
        setSelectedWordInfo(minimalWordInfo);
        setMainWord(firstResult.lemma);
        setInputValue(firstResult.lemma);
        setWordHistory(prev => [...prev.slice(0, currentHistoryIndex + 1), firstResult.lemma]);
        setCurrentHistoryIndex(prev => prev + 1);
        setShowSuggestions(false);
        
        // Fetch full details
        fetchWordDetails(firstResult.id.toString())
          .then(fullWordData => {
            if (fullWordData && selectedWordInfo && fullWordData.id === selectedWordInfo.id) {
              setSelectedWordInfo(fullWordData);
              // Run these promises but don't return their results to the chain
              fetchWordNetworkData(fullWordData.lemma, depth)
                .then(networkData => {
                  setWordNetwork(networkData);
                })
                .catch(err => console.error("Error fetching network data in force search:", err));
                
              fetchEtymologyTree(fullWordData.id)
                .catch(err => console.error("Error fetching etymology in force search:", err));
            }
          })
          .catch(err => {
            console.error("Error fetching word details in force search:", err);
            setError(`Error fetching details: ${err.message}`);
          });
      } else {
        setError(`No results found for "${inputValue}" even with forced search.`);
      }
    } catch (err) {
      console.error("Force search failed:", err);
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred during forced search.";
      setError(`Forced search failed: ${errorMsg}`);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, currentHistoryIndex, depth, fetchWordNetworkData, fetchEtymologyTree, selectedWordInfo]);

  const handleTestSearch = useCallback(async () => {
    if (!inputValue) return;
    
    setError(null);
    setIsLoading(true);
    
    try {
      console.log("Manually testing search API with query:", inputValue);
      
      // Make a direct fetch to avoid any processing in our API functions
      const apiUrl = `${API_BASE_URL}/search?q=${encodeURIComponent(inputValue)}&limit=5`;
      console.log("Fetching from URL:", apiUrl);
      
      const response = await fetch(apiUrl, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        mode: 'cors',
        cache: 'no-cache'
      });
      
      if (response.ok) {
        const rawData = await response.json();
        console.log("Raw API Response:", rawData);
        
        // Check the structure
        if (rawData && Array.isArray(rawData.results) && rawData.results.length > 0) {
          console.log("Search successful! First result:", rawData.results[0]);
          setError("Search API returned results. See console for details.");
        } else {
          console.warn("API response doesn't have expected 'results' array:", rawData);
          setError("API response doesn't have expected format. See console.");
        }
      } else {
        const errorText = await response.text();
        console.error("API error:", response.status, errorText);
        setError(`API returned error ${response.status}: ${errorText || 'No error details'}`);
      }
    } catch (e) {
      console.error("Manual search test error:", e);
      setError(`Manual search test failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue]);

  return (
    <div className={`word-explorer ${theme} ${isLoading ? 'loading' : ''}`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <div className="header-buttons">
          <button
            onClick={handleRandomWord}
            className="random-button"
            title="Explore a random word"
            disabled={isLoading}
          >
            🎲 Random Word
          </button>
          <button
            onClick={handleResetCircuitBreaker}
            className="debug-button"
            title="Reset API connection"
          >
            🔄 Reset API
          </button>
          <button
            onClick={handleTestSearch}
            className="debug-button"
            title="Test Search API directly"
          >
            🔍 Test Search
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
                console.log("Enter key pressed! Calling handleSearch(). Input value:", inputValue);
                e.preventDefault();
                handleSearch();
              }
            }}
            placeholder="Enter a word"
            className="search-input"
            aria-label="Search word"
          />
          {showSuggestions && searchResults.length > 0 && (
            <ul className="search-suggestions">
              {searchResults.map((result: SearchWordResult) => (
                <li key={result.id} onClick={() => handleSuggestionClick(result.lemma)}>
                  {result.lemma}
                  {result.language_code && ` (${result.language_code})`}
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
            <div className="error-actions">
              <button onClick={handleResetCircuitBreaker} className="reset-button">
                Reset Connection
              </button>
              <button onClick={handleForceSearch} className="force-button">
                Force Search Anyway
              </button>
            </div>
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
            {isLoading && <div className="loading">Loading Network...</div>}
            {!isLoading && wordNetwork && (
              <WordGraph
                wordNetwork={wordNetwork}
                mainWord={mainWord}
                onNodeClick={handleNodeClick}
                onNetworkChange={handleNetworkChange}
                initialDepth={depth}
              />
            )}
            {!isLoading && !wordNetwork && !error && (
                <div className="empty-graph">Enter a word to explore its network.</div>
            )}
          </div>
        </div>
        <div ref={detailsContainerRef} className="details-container">
          {isLoading && <div className="loading-spinner">Loading Details...</div>} 
          {!isLoading && selectedWordInfo && (
            <WordDetails 
              wordInfo={selectedWordInfo} 
              etymologyTree={etymologyTree}
              isLoadingEtymology={isLoadingEtymology}
              etymologyError={etymologyError}
              onWordLinkClick={handleWordLinkClick}
              onEtymologyNodeClick={handleNodeClick}
            />
          )}
          {!isLoading && !selectedWordInfo && (
                <div className="no-word-selected">Select a word or search to see details.</div>
            )}
            {error && <div className="error-message">Error: {error}</div>}
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
