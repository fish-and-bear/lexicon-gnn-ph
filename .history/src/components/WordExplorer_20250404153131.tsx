import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchResult, SearchOptions, EtymologyTree, Statistics, Definition, WordNetworkOptions } from "../types";
import unidecode from "unidecode";
import { 
  fetchWordNetwork, 
  fetchWordComprehensive,
  searchWords, 
  resetCircuitBreaker, 
  testApiConnection,
  getEtymologyTree,
  getRandomWord,
  getStatistics,
} from "../api/wordApi";
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
  const [searchResults, setSearchResults] = useState<SearchResult | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [apiEndpoint, setApiEndpoint] = useState<string | null>(localStorage.getItem('successful_api_endpoint'));
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [isLoadingStatistics, setIsLoadingStatistics] = useState<boolean>(false);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('tl');

  const detailsContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const savedWidth = localStorage.getItem('wordDetailsWidth');
    if (savedWidth && detailsContainerRef.current) {
      detailsContainerRef.current.style.width = `${savedWidth}px`;
    }
  }, []);

  useEffect(() => {
    if (!detailsContainerRef.current) return;
    
    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        if (entry.target === detailsContainerRef.current) {
          const newWidth = entry.contentRect.width;
          localStorage.setItem('wordDetailsWidth', newWidth.toString());
        }
      }
    });
    
    resizeObserver.observe(detailsContainerRef.current);
    
    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  const normalizeInput = (input: string) => unidecode(input.trim().toLowerCase());

  const fetchWordNetworkData = useCallback(async (word: string, options: WordNetworkOptions = {}) => {
    try {
      setIsLoading(true);
      const data = await fetchWordNetwork(word, { 
        max_depth: options.max_depth ?? 2,
        bidirectional: options.bidirectional ?? true,
      });
      
      setWordNetwork(data);
      return data;
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchEtymologyTree = useCallback(async (wordId: number) => {
    if (!wordId) return;
    
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    
    try {
      console.log(`Fetching etymology tree for word ID: ${wordId}`);
      const data = await getEtymologyTree(wordId, 3);
      console.log('Etymology tree data:', data);
      setEtymologyTree(data);
    } catch (error) {
      console.error("Error fetching etymology tree:", error);
      let errorMessage = "Failed to fetch etymology tree.";
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      setEtymologyError(errorMessage);
      setEtymologyTree(null);
    } finally {
      setIsLoadingEtymology(false);
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

    console.log('Starting search for normalized input:', normalizedInput);
    
    setIsLoading(true);
    setError(null);
    setWordNetwork(null);
    setSelectedWordInfo(null);
    setEtymologyTree(null); 

    try {
      console.log('Attempting to fetch comprehensive details for:', normalizedInput);
      const wordData = await fetchWordComprehensive(normalizedInput);
      console.log('Comprehensive word details received:', wordData);
      
      if (wordData && wordData.lemma) {
        console.log('Setting selected word info with valid data from comprehensive fetch');
        setSelectedWordInfo(wordData);
        setMainWord(wordData.lemma);
        
        await Promise.all([
          fetchWordNetworkData(normalizedInput, { max_depth: depth }),
          wordData.id ? fetchEtymologyTree(wordData.id) : Promise.resolve()
        ]);
        
        setWordHistory(prevHistory => {
          const newHistory = [...prevHistory.slice(0, currentHistoryIndex + 1), wordData.lemma];
          return newHistory.slice(-20);
        });
        setCurrentHistoryIndex(prevIndex => prevIndex + 1);
        setInputValue(wordData.lemma);
        setShowSuggestions(false);
      } else {
        console.log('Comprehensive details fetch failed or invalid, trying search...');
        const searchOptions: SearchOptions = { 
          q: normalizedInput,
          limit: 1,
          offset: 0,
          language: selectedLanguage,
          mode: 'all', 
          sort: 'relevance', 
        };
        const searchResultsData = await searchWords(searchOptions);
        setSearchResults(searchResultsData);
        
        console.log('Search results:', searchResultsData);
        
        if (searchResultsData?.words?.length > 0) {
          const firstResultLemma = searchResultsData.words[0].lemma;
          console.log('Found search result, fetching comprehensive details for:', firstResultLemma);
          const firstResultData = await fetchWordComprehensive(firstResultLemma);
          
          if (firstResultData?.lemma) {
            setSelectedWordInfo(firstResultData);
            setMainWord(firstResultData.lemma);
            
            await Promise.all([
              fetchWordNetworkData(firstResultLemma, { max_depth: depth }),
              firstResultData.id ? fetchEtymologyTree(firstResultData.id) : Promise.resolve()
            ]);

            setWordHistory(prevHistory => {
                const newHistory = [...prevHistory.slice(0, currentHistoryIndex + 1), firstResultData.lemma];
                return newHistory.slice(-20);
            });
            setCurrentHistoryIndex(prevIndex => prevIndex + 1);
            setInputValue(firstResultData.lemma);
            setShowSuggestions(false);
          } else {
             console.log('Could not fetch details for the first search result lemma:', firstResultLemma);
             setError(`Word "${normalizedInput}" not found.`);
          }
        } else {
          console.log('No results found from search either.');
          setError(`Word "${normalizedInput}" not found.`);
          setMainWord(normalizedInput);
        }
      }
    } catch (err) {
      console.error('Error during handleSearch:', err);
      let message = "An error occurred during search.";
      if (err instanceof Error) {
        message = err.message;
      } else if (typeof err === 'string') {
        message = err;
      }
      setError(message);
      setMainWord(inputValue.trim());
      setSelectedWordInfo(null);
      setWordNetwork(null);
      setEtymologyTree(null);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, depth, selectedLanguage, currentHistoryIndex, fetchWordNetworkData, fetchEtymologyTree]);

  const handleNodeClickDetailed = useCallback(async (nodeId: number, nodeLabel: string) => {
    console.log("Node clicked (detailed):", nodeId, nodeLabel);
    const normalizedLabel = normalizeInput(nodeLabel);
    setIsLoading(true);
    setError(null);
    try {
      const wordData = await fetchWordComprehensive(normalizedLabel);
      if (wordData?.lemma) {
        setSelectedWordInfo(wordData);
        setMainWord(wordData.lemma);
        setInputValue(wordData.lemma);
        await Promise.all([
          fetchWordNetworkData(normalizedLabel, { max_depth: depth }),
          wordData.id ? fetchEtymologyTree(wordData.id) : Promise.resolve()
        ]);
        setWordHistory(prevHistory => {
            const newHistory = [...prevHistory.slice(0, currentHistoryIndex + 1), wordData.lemma];
            return newHistory.slice(-20);
        });
        setCurrentHistoryIndex(prevIndex => prevIndex + 1);
        setShowSuggestions(false);
      } else {
        setError(`Could not load details for ${nodeLabel}`);
      }
    } catch (err) {
      console.error("Error fetching details on node click:", err);
      let message = "Failed to load details.";
       if (err instanceof Error) message = err.message;
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [depth, currentHistoryIndex, fetchWordNetworkData, fetchEtymologyTree]);

  const handleSimpleWordClick = useCallback((word: string) => {
      console.log("Simple word link clicked:", word);
      setInputValue(word);
      handleSearch(word);
  }, [handleSearch]);

  const handleEtymologyNodeClickWrapper = useCallback((node: any) => {
    const word = node?.label || node?.lemma || node?.name;
    console.log("Etymology node clicked:", node, "Extracted word:", word);
    if (word && typeof word === 'string') {
       setInputValue(word);
       handleSearch(word); 
    } else {
      console.warn("Could not extract word label from etymology node:", node);
    }
  }, [handleSearch]);

  const handleSuggestionClick = (word: string) => {
    console.log('Suggestion clicked:', word);
    setInputValue(word);
    setShowSuggestions(false);
    handleSearch(word);
  };

  const debouncedSearch = useCallback(
    debounce(async (currentQuery: string) => {
      if (currentQuery.length < 2) {
        setSearchResults(null);
        setShowSuggestions(false);
        return;
      }
      try {
        const results = await searchWords({ 
            q: currentQuery, 
            limit: 10,
            offset: 0,
            language: selectedLanguage 
        });
        setSearchResults(results);
        setShowSuggestions(true);
      } catch (error) {
        console.error("Error fetching search suggestions:", error);
        setSearchResults(null);
        setShowSuggestions(false);
      }
    }, 300),
    [selectedLanguage]
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);
    debouncedSearch(newValue); 
  };

  const handleHistoryNavigation = useCallback((direction: 'back' | 'forward') => {
    let newIndex = currentHistoryIndex;
    if (direction === 'back' && currentHistoryIndex > 0) {
      newIndex--;
    } else if (direction === 'forward' && currentHistoryIndex < wordHistory.length - 1) {
      newIndex++;
    }

    if (newIndex !== currentHistoryIndex && wordHistory[newIndex]) {
      setCurrentHistoryIndex(newIndex);
      const wordToLoad = wordHistory[newIndex];
      setInputValue(wordToLoad);
      handleSearch(wordToLoad);
    }
  }, [currentHistoryIndex, wordHistory, handleSearch]);

  const handleResetCircuitBreaker = useCallback(() => {
    resetCircuitBreaker();
    testApiConnection().then(setApiConnected);
    setError("Circuit breaker reset. Connection test initiated.");
  }, []);

  const handleTestApiConnection = useCallback(async () => {
    setError("Testing API connection...");
    const connected = await testApiConnection();
    setApiConnected(connected);
    setError(connected ? "API connection successful." : "API connection failed.");
  }, []);

  const handleGraphParamsChange = useCallback((newDepth: number /*, newBreadth: number */) => {
    console.log("Graph params changed - New Depth:", newDepth);
    setDepth(newDepth);
  }, []);

  useEffect(() => {
    const checkApi = async () => {
      const connected = await testApiConnection();
      setApiConnected(connected);
      if (!connected) {
        setError("API connection failed. Please check the backend server.");
      }
    };
    checkApi();
  }, []);

  return (
    <div className={`word-explorer ${theme} ${isLoading ? 'loading' : ''}`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <div className="header-buttons">
          <button
            onClick={() => handleSearch()}
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
          onClick={() => handleHistoryNavigation('back')}
          disabled={currentHistoryIndex <= 0 || isLoading}
          className="history-button"
          aria-label="Go back"
        >
          ←
        </button>
        <button
          onClick={() => handleHistoryNavigation('forward')}
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
                console.log("Enter key pressed! Calling handleSearch(). Input value:", inputValue);
                e.preventDefault();
                handleSearch();
              }
            }}
            placeholder="Enter a word"
            className="search-input"
            aria-label="Search word"
          />
          {isLoading && <div className="search-loading">Loading...</div>}
          {showSuggestions && searchResults && searchResults.words && searchResults.words.length > 0 && (
            <ul className="search-suggestions">
              {searchResults.words.map((result) => (
                <li key={result.id} onClick={() => handleSuggestionClick(result.lemma)}>
                  {result.lemma}
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
            {isLoading && <div className="loading">Loading Network...</div>}
            {!isLoading && wordNetwork && (
              <WordGraph
                wordNetwork={wordNetwork}
                mainWord={mainWord}
                onNodeClick={handleSimpleWordClick}
                onNetworkChange={handleGraphParamsChange}
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
              onWordLinkClick={handleSimpleWordClick}
              onEtymologyNodeClick={handleEtymologyNodeClickWrapper}
            />
          )}
          {!isLoading && !selectedWordInfo && (
                <div className="no-word-selected">Select a word or search to see details.</div>
            )}
             {/* Display general error messages */}
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
