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
  PartOfSpeech,
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
  API_BASE_URL,
  directSearch
} from "../api/wordApi";
import axios from 'axios';
import { debounce } from "lodash";

const WordExplorer: React.FC = () => {
  const [wordNetwork, setWordNetwork] = useState<WordNetworkResponse | null>(null);
  const [selectedWordInfo, setSelectedWordInfo] = useState<WordInfo | null>(null);
  const [mainWord, setMainWord] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [depth, setDepth] = useState<number>(2);
  const [width, setWidth] = useState<number>(200);
  const [colorIntensity, setColorIntensity] = useState<number>(100);
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
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [isLoadingStatistics, setIsLoadingStatistics] = useState<boolean>(false);
  const [diagnostic, setDiagnostic] = useState<{data: any, shown: boolean}>({ data: null, shown: false });

  const detailsContainerRef = useRef<HTMLDivElement>(null);

  const { theme, toggleTheme } = useTheme();

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

    console.log('[handleSearch] Starting for:', normalizedInput);
    setIsLoading(true);
    setError(null);
    setWordNetwork(null);
    setEtymologyTree(null);
    setSelectedWordInfo(null); // Clear previous info
    setShowSuggestions(false); // Hide suggestions on explicit search

    try {
      // Use simple search with just plain text - this is the most reliable
      const searchOptions: SearchOptions = { 
        q: normalizedInput, 
        limit: 10, 
        include_full: false // Don't try to load full details in the search results
      };
      
      console.log('[handleSearch] Performing simple search with options:', searchOptions);
      const searchResponse = await searchWords(searchOptions);
      console.log('[handleSearch] Search API response:', searchResponse);
      
      if ((searchResponse.count && searchResponse.count > 0) && searchResponse.results && searchResponse.results.length > 0) {
        // We found results! Get the first one and fetch its details
        const firstResult = searchResponse.results[0];
        console.log('[handleSearch] Found result, fetching details for:', firstResult.id);
        
        try {
          // Use the ID to get full details - this is more reliable than guessing
          const wordDetails = await fetchWordDetails(firstResult.id.toString());
          setSelectedWordInfo(wordDetails);
          setMainWord(wordDetails.lemma);
          
          // Now fetch network data and etymology tree in the background
          fetchWordNetworkData(wordDetails.lemma, depth).catch(e => 
            console.error('[handleSearch] Non-critical error fetching network:', e));
            
          if (wordDetails.id) {
            fetchEtymologyTree(wordDetails.id).catch(e => 
              console.error('[handleSearch] Non-critical error fetching etymology:', e));
          }
        } catch (detailsError) {
          console.error('[handleSearch] Error fetching details:', detailsError);
          // Even if we can't get details, show something basic
          setSelectedWordInfo({
            id: firstResult.id,
            lemma: firstResult.lemma,
            language_code: firstResult.language_code,
            normalized_lemma: firstResult.normalized_lemma
          });
          setError(`Found "${firstResult.lemma}" but could not load all details`);
        }
      } 
      else if (searchResponse.count && searchResponse.count > 0) {
        // API reports results but didn't return them - try direct fetch
        console.log('[handleSearch] API reports results but none returned. Trying direct lookup.');
        try {
          const wordDetails = await fetchWordDetails(normalizedInput);
          setSelectedWordInfo(wordDetails);
          setMainWord(wordDetails.lemma);
          
          // Fetch network in background
          fetchWordNetworkData(wordDetails.lemma, depth).catch(e => 
            console.error('[handleSearch] Non-critical error fetching network:', e));
        } catch (directError) {
          console.error('[handleSearch] Direct lookup failed:', directError);
          setError(`API found matches for "${normalizedInput}" but could not retrieve details`);
        }
      }
      else {
        // No results found
        console.error('[handleSearch] No results found for:', normalizedInput);
        setError(`No results found for "${normalizedInput}"`);
      }
    } catch (searchError) {
      console.error('[handleSearch] Search error:', searchError);
      setError(`Error searching for "${normalizedInput}": ${searchError instanceof Error ? searchError.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
      console.log('[handleSearch] Finished.');
    }
  }, [inputValue, fetchWordNetworkData, depth, fetchEtymologyTree]);

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
        
        // Call with minimal search options for faster response
        const searchOptions: SearchOptions = { 
          q: query, 
          limit: 10, 
          include_full: false,
          mode: 'all' // Use 'all' mode for broader results
        };
        
        // Try the API call
        const searchResult = await searchWords(searchOptions);
        console.log("Search suggestions response:", searchResult);
        
        // Check if we have valid results
        if (searchResult && Array.isArray(searchResult.words) && searchResult.words.length > 0) {
          console.log("Setting search results:", searchResult.words.length, "items");
          setSearchResults(searchResult.words);
          setShowSuggestions(true);
        } else {
          console.log("No valid search results found in response:", searchResult);
          setSearchResults([]);
          setShowSuggestions(false);
        }
      } catch (error) {
        console.error("Error fetching search suggestions:", error);
        // Don't show an error message for suggestions, just log it
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
    console.log("Suggestion clicked:", word);
    setShowSuggestions(false);
    setSearchResults([]);
    setInputValue(word);
    
    // Direct search with the selected word
    setIsLoading(true);
    setError(null);
    setWordNetwork(null);
    setEtymologyTree(null);
    setSelectedWordInfo(null);
    
    fetchWordDetails(word)
      .then(wordDetails => {
        setSelectedWordInfo(wordDetails);
        setMainWord(wordDetails.lemma);
        
        // Update search history
        setWordHistory(prev => {
          if (prev.length > 0 && prev[prev.length - 1] === word) return prev;
          return [...prev.slice(0, currentHistoryIndex + 1), word];
        });
        setCurrentHistoryIndex(prev => prev + 1);
        
        // After getting details, fetch network
        return fetchWordNetworkData(wordDetails.lemma, depth).catch(e => 
          console.error("Network fetch error in suggestion click:", e));
      })
      .catch(error => {
        console.error("Error fetching details for suggestion:", error);
        setError(`Could not find details for "${word}".`);
      })
      .finally(() => {
        setIsLoading(false);
      });
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
          return Promise.allSettled([
            fetchWordNetworkData(previousWord, depth),
            fetchEtymologyTree(rawWordData.id)
          ]);
        })
        .then(results => {
          results.forEach(result => {
             if (result.status === 'rejected') console.error("Error during back navigation fetch:", result.reason);
          });
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
          return Promise.allSettled([
            fetchWordNetworkData(nextWord, depth),
            fetchEtymologyTree(rawWordData.id)
          ]);
        })
        .then(results => {
          results.forEach(result => {
             if (result.status === 'rejected') console.error("Error during forward navigation fetch:", result.reason);
          });
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
        // Fetch Parts of Speech and Statistics concurrently
        const [posData, statsData] = await Promise.all([
          getPartsOfSpeech(),
          getStatistics(),
        ]);
        setPartsOfSpeech(posData || []);
        setStatistics(statsData || null);
      } catch (error) {
        console.error("Error fetching initial data:", error);
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

        await Promise.allSettled([
          fetchWordNetworkData(rawWordData.lemma, depth),
          fetchEtymologyTree(rawWordData.id)
        ]).then(results => {
           results.forEach(result => {
             if (result.status === 'rejected') console.error("Error during random word fetch:", result.reason);
          });
        });
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
            } else if (selectedWordInfo && fullWordData?.id !== selectedWordInfo.id) {
                console.warn("Stale data fetched in force search, ignoring.");
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

  const toggleDiagnostic = useCallback(() => {
    setDiagnostic(prev => ({ ...prev, shown: !prev.shown }));
  }, []);

  const handleDiagnosticSearch = useCallback(async () => {
    if (!inputValue) {
      setError("Please enter a search term first");
      return;
    }
    
    setError(null);
    setIsLoading(true);
    setDiagnostic({ data: null, shown: true });
    
    try {
      // Direct fetch with minimal processing
      const response = await fetch(`${API_BASE_URL}/search?q=${encodeURIComponent(inputValue)}&limit=5`);
      const data = await response.json();
      console.log("Direct API response:", data);
      setDiagnostic({ data, shown: true });
      
      // Try to extract results for debugging
      if (data && data.results && Array.isArray(data.results)) {
        setError(`Found ${data.results.length} results in API response`);
      } else if (data && data.count !== undefined) {
        setError(`API returned count: ${data.count} but possible issue with results format`);
      } else {
        setError("API returned 200 OK but unexpected response format");
      }
    } catch (e) {
      console.error("Diagnostic search error:", e);
      setError(`Diagnostic error: ${e instanceof Error ? e.message : String(e)}`);
      setDiagnostic({ data: { error: String(e) }, shown: true });
    } finally {
      setIsLoading(false);
    }
  }, [inputValue]);

  const handleWidthChange = useCallback((newWidth: number) => {
    setWidth(newWidth);
    if (wordNetwork) {
      // If we already have a network, just update its width
      // This assumes the WordGraph component can handle width changes
    }
  }, [wordNetwork]);

  const handleColorIntensityChange = useCallback((newIntensity: number) => {
    setColorIntensity(newIntensity);
  }, []);

  const handleNetworkReset = useCallback(() => {
    if (mainWord) {
      setDepth(2);  // Reset to default depth
      setWidth(200);  // Reset to default width
      fetchWordNetworkData(normalizeInput(mainWord), 2);
    }
  }, [mainWord, fetchWordNetworkData]);

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
          <button
            onClick={handleDiagnosticSearch}
            className="debug-button"
            title="Test API response"
          >
            🔬 Diagnostics
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
          disabled={currentHistoryIndex <= 0 || isLoading}
          className="history-button"
          aria-label="Go back"
        >
          ←
        </button>
        <button
          onClick={handleForward}
          disabled={currentHistoryIndex >= wordHistory.length - 1 || isLoading}
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
                console.log("Enter key pressed! Input value:", inputValue);
                e.preventDefault();
                if (inputValue.trim()) {
                  console.log("Executing search for:", inputValue);
                  setMainWord(inputValue);
                  setWordHistory(prev => {
                    if (prev.length > 0 && prev[prev.length - 1] === inputValue) return prev;
                    return [...prev.slice(0, currentHistoryIndex + 1), inputValue];
                  });
                  setCurrentHistoryIndex(prev => prev + 1);
                  
                  // Direct search call with the input value
                  fetchWordDetails(inputValue.trim())
                    .then(wordDetails => {
                      setSelectedWordInfo(wordDetails);
                      setMainWord(wordDetails.lemma);
                      
                      // After getting details, fetch network
                      return fetchWordNetworkData(wordDetails.lemma, depth).catch(e => 
                        console.error("Network fetch error:", e));
                    })
                    .catch(error => {
                      console.error("Search error:", error);
                      setError(`Could not find "${inputValue}". Try a different spelling.`);
                    })
                    .finally(() => {
                      setIsLoading(false);
                    });
                  
                  // Show loading state
                  setIsLoading(true);
                  setError(null);
                  setWordNetwork(null);
                  setEtymologyTree(null);
                }
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
                </li>
              ))}
            </ul>
          )}
        </div>
        <button
          onClick={() => {
            if (inputValue.trim()) {
              console.log("Search button clicked! Input value:", inputValue);
              setMainWord(inputValue);
              setWordHistory(prev => {
                if (prev.length > 0 && prev[prev.length - 1] === inputValue) return prev;
                return [...prev.slice(0, currentHistoryIndex + 1), inputValue];
              });
              setCurrentHistoryIndex(prev => prev + 1);
              
              // Direct search call with the input value
              setIsLoading(true);
              setError(null);
              setWordNetwork(null);
              setEtymologyTree(null);
              setSelectedWordInfo(null);
              
              fetchWordDetails(inputValue.trim())
                .then(wordDetails => {
                  setSelectedWordInfo(wordDetails);
                  setMainWord(wordDetails.lemma);
                  
                  // After getting details, fetch network
                  return fetchWordNetworkData(wordDetails.lemma, depth).catch(e => 
                    console.error("Network fetch error:", e));
                })
                .catch(error => {
                  console.error("Search error:", error);
                  setError(`Could not find "${inputValue}". Try a different spelling.`);
                })
                .finally(() => {
                  setIsLoading(false);
                });
            }
          }}
          disabled={isLoading || !inputValue}
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
              <button onClick={handleResetCircuitBreaker} className="reset-button" disabled={isLoading}>
                Reset Connection
              </button>
              <button onClick={handleForceSearch} className="force-button" disabled={isLoading}>
                Force Search Anyway
              </button>
            </div>
          )}
          {error.includes('API connection') && (
            <div className="error-actions">
              <button onClick={handleResetCircuitBreaker} className="reset-button" disabled={isLoading}>
                Reset Connection
              </button>
              <button 
                onClick={handleTestApiConnection} 
                className="retry-button"
                disabled={isLoading}
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
                disabled={isLoading}
              >
                Test API Connection
              </button>
            </div>
          )}
          {error.includes('Network error') && (
            <div className="error-actions">
              <button onClick={handleResetCircuitBreaker} className="reset-button" disabled={isLoading}>
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
                disabled={isLoading}
              >
                Test Connection & Retry
              </button>
            </div>
          )}
        </div>
      )}
      
      {diagnostic.shown && diagnostic.data && (
        <div className="diagnostic-container">
          <div className="diagnostic-header">
            <h3>API Response Diagnostic</h3>
            <button onClick={toggleDiagnostic} className="close-button">×</button>
          </div>
          <div className="diagnostic-content">
            <pre>{JSON.stringify(diagnostic.data, null, 2)}</pre>
          </div>
        </div>
      )}
      <main>
        <div className="graph-container">
          <div className="graph-content">
            {isLoading && <div className="loading-message">
              <div className="loading-spinner"></div>
              <p>Searching for "{inputValue}"...</p>
            </div>}
            {!isLoading && wordNetwork && (
              <WordGraph
                wordNetwork={wordNetwork}
                mainWord={mainWord}
                onNodeClick={handleNodeClick}
                onNetworkChange={handleNetworkChange}
                initialDepth={depth}
                width={width}
              />
            )}
            {!isLoading && !wordNetwork && !error && !selectedWordInfo && (
                <div className="empty-graph">
                  <p>Enter a Filipino word to explore its network</p>
                  <p className="empty-suggestion">Try: "bahay", "araw", "tao"</p>
                </div>
            )}
          </div>
          
          {wordNetwork && (
            <div className="controls-container">
              <div className="graph-controls">
                <div className="slider-container">
                  <label htmlFor="depth-slider">Relation Depth: {depth}</label>
                  <input
                    id="depth-slider"
                    type="range"
                    min={1}
                    max={3}
                    step={1}
                    value={depth}
                    onChange={(e) => {
                      const newDepth = parseInt(e.target.value);
                      setDepth(newDepth);
                      if (mainWord) {
                        fetchWordNetworkData(normalizeInput(mainWord), newDepth);
                      }
                    }}
                  />
                </div>
                <div className="slider-container">
                  <label htmlFor="width-slider">Display Width: {width}</label>
                  <input
                    id="width-slider"
                    type="range"
                    min={100}
                    max={500}
                    step={10}
                    value={width}
                    onChange={(e) => handleWidthChange(parseInt(e.target.value))}
                  />
                </div>
                <div className="slider-container">
                  <label htmlFor="color-slider">Color Intensity: {colorIntensity}%</label>
                  <input
                    id="color-slider"
                    type="range"
                    min={20}
                    max={150}
                    step={10}
                    value={colorIntensity}
                    onChange={(e) => handleColorIntensityChange(parseInt(e.target.value))}
                  />
                </div>
                <button 
                  onClick={handleNetworkReset}
                  className="reset-button"
                  title="Reset network display"
                >
                  Reset View
                </button>
              </div>
            </div>
          )}
        </div>
        <div ref={detailsContainerRef} className="details-container">
          {isLoading && <div className="loading-details">
            <div className="loading-spinner"></div>
            <p>Loading word details...</p>
          </div>} 
          {selectedWordInfo && (
            <WordDetails 
              wordInfo={selectedWordInfo} 
              etymologyTree={etymologyTree}
              isLoadingEtymology={isLoadingEtymology}
              etymologyError={etymologyError}
              onWordLinkClick={handleWordLinkClick}
              onEtymologyNodeClick={handleNodeClick}
            />
          )}
          {!isLoading && !selectedWordInfo && !error && (
                <div className="no-word-selected">
                  <p>Enter a word in the search bar above</p>
                  <p className="empty-tip">Filipino dictionary with over 60,000 words</p>
                </div>
          )}
            {error && <div className="error-details">
              <p>{error}</p>
              <p className="error-suggestion">Try a different spelling or another word</p>
            </div>}
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
