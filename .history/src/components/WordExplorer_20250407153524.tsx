import React, { useState, useCallback, useEffect, useRef, useMemo } from "react";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import {
  WordNetworkResponse,
  WordInfo,
  SearchWordResult,
  SearchResultItem,
  SearchOptions,
  EtymologyTree,
  PartOfSpeech,
  Statistics,
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
} from "../api/wordApi";
import axios from 'axios';
import { debounce } from "lodash";
import WordGraph, { NodeData } from './WordGraph';
import NetworkControls from './NetworkControls';
import './NetworkControls.css';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import Alert from '@mui/material/Alert';
import Header from './Header';
import SearchBar from './SearchBar';
import Footer from './Footer';

const MIN_DEPTH = 1;
const MAX_DEPTH = 5;
const MIN_BREADTH = 5;
const MAX_BREADTH = 50;
const INITIAL_DEPTH = 3;
const INITIAL_BREADTH = 15;

interface DiagnosticData {
  network?: WordNetworkResponse;
  details?: WordInfo;
  etymology?: EtymologyTree;
  error?: any;
}

// Helper components for loading and error states
const Loader: React.FC<{ message: string }> = ({ message }) => (
  <div className="loading-details">
    <div className="loading-spinner"></div>
    <p>{message}</p>
  </div>
);

const ErrorDisplay: React.FC<{ message: string }> = ({ message }) => (
  <div className="error-details">
    <p>{message}</p>
    <p className="error-suggestion">Try searching for a different word or check your connection.</p>
  </div>
);

const WordExplorer: React.FC = () => {
  const [wordNetwork, setWordNetwork] = useState<WordNetworkResponse | null>(null);
  const [selectedWordInfo, setSelectedWordInfo] = useState<WordInfo | null>(null);
  const [mainWord, setMainWord] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [depth, setDepth] = useState<number>(INITIAL_DEPTH);
  const [breadth, setBreadth] = useState<number>(INITIAL_BREADTH);
  const [inputValue, setInputValue] = useState<string>("salita");
  const [wordHistory, setWordHistory] = useState<string[]>([]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState<number>(-1);
  const [searchResults, setSearchResults] = useState<SearchWordResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [diagnostic, setDiagnostic] = useState<{ shown: boolean; data: DiagnosticData | null }>({ shown: false, data: null });
  const [leftPanelWidth, setLeftPanelWidth] = useState<string>('60%');
  const [isResizing, setIsResizing] = useState<boolean>(false);
  const mainRef = useRef<HTMLDivElement>(null);
  const resizerRef = useRef<HTMLDivElement>(null);
  const [isLoadingNetwork, setIsLoadingNetwork] = useState<boolean>(false);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [isLoadingDetails, setIsLoadingDetails] = useState<boolean>(false);
  const [detailsError, setDetailsError] = useState<string | null>(null);
  const [partsOfSpeech, setPartsOfSpeech] = useState<PartOfSpeech[]>([]);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [isLoadingStatistics, setIsLoadingStatistics] = useState<boolean>(false);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState<boolean>(false);

  const { theme } = useTheme();

  const normalizeInput = useCallback((input: string): string => {
    if (!input) return ''; // Handle undefined or empty input
    return unidecode(input.toLowerCase().trim());
  }, []);

  const updateHistory = useCallback((newWord: string) => {
    setWordHistory(prevHistory => {
      // Avoid duplicates at the end
      if (prevHistory.length > 0 && prevHistory[currentHistoryIndex] === newWord) {
        return prevHistory;
      }
      const newHistory = [...prevHistory.slice(0, currentHistoryIndex + 1), newWord];
      setCurrentHistoryIndex(newHistory.length - 1);
      return newHistory;
    });
  }, [currentHistoryIndex]);

  const fetchEtymologyTreeData = useCallback(async (wordId: number) => {
    if (!wordId) return;
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    try {
      console.log(`Fetching etymology tree for word ID: ${wordId}`);
      const data = await getEtymologyTree(wordId, 3); // Use getEtymologyTree
      console.log('Etymology API response:', data);
      setEtymologyTree(data);
      setDiagnostic(prev => ({ ...prev, data: { ...prev.data, etymology: data } }));
    } catch (fetchError) {
      console.error("Error fetching etymology tree:", fetchError);
      let errorMessage = "Failed to fetch etymology tree.";
      if (fetchError instanceof Error) errorMessage = fetchError.message;
      else if (axios.isAxiosError(fetchError)) errorMessage = fetchError.response?.data?.error || fetchError.message;
      setEtymologyError(errorMessage);
      setEtymologyTree(null);
      setDiagnostic(prev => ({ ...prev, data: { ...prev.data, error: fetchError } }));
    } finally {
      setIsLoadingEtymology(false);
    }
  }, []);

  const fetchWordNetworkData = useCallback(async (word: string, currentDepth: number = depth, currentBreadth: number = breadth) => {
    if (!word) return null; // Don't fetch if word is empty
    setIsLoadingNetwork(true);
    setNetworkError(null);
    try {
      console.log(`Fetching word network for ${word} with depth=${currentDepth}, breadth=${currentBreadth}`);
      
      // Add retries for better resilience
      let retryCount = 0;
      const maxRetries = 3;
      let success = false;
      let data;
      
      while (!success && retryCount < maxRetries) {
        try {
          const currentRetry = retryCount; // Create a closure-safe copy
          data = await fetchWordNetwork(word, { 
            depth: currentDepth, 
            breadth: currentBreadth,
            relation_types: ['synonym', 'antonym', 'hypernym', 'hyponym', 'related']
          });
          success = true;
        } catch (retryError) {
          retryCount++;
          console.warn(`Attempt ${retryCount}/${maxRetries} failed. ${retryCount < maxRetries ? 'Retrying...' : 'Giving up.'}`);
          if (retryCount >= maxRetries) throw retryError;
          // Wait before retrying (exponential backoff)
          await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
        }
      }
      
      if (!data) throw new Error("Network data is empty after successful fetch");
      
      console.log('Network API response:', data);
      
      if (data && data.nodes && data.nodes.length > 0) {
        const processedData = {
          nodes: data.nodes.map((node: any) => ({
            id: typeof node.id === 'number' ? node.id : parseInt(String(node.id), 10) || Math.floor(Math.random() * 10000),
            lemma: node.lemma || node.label || '',
            language_code: node.language_code || 'tl'
          })),
          // Use links if available, fallback to edges
          edges: (data.links || data.edges || []).map((edge: any) => {
            const source = typeof edge.source === 'object' ? edge.source.id : edge.source;
            const target = typeof edge.target === 'object' ? edge.target.id : edge.target;
            return {
              ...edge,
              source: source,
              target: target,
              type: edge.type || 'default'
            };
          }),
          stats: data.stats || {
            node_count: data.nodes.length,
            edge_count: (data.links || data.edges || []).length,
            depth: currentDepth,
            breadth: currentBreadth
          }
        };
        console.log('Processed network data:', processedData);
        setWordNetwork(processedData);
        setDiagnostic(prev => ({ ...prev, data: { ...prev.data, network: processedData } }));
        return processedData;
      } else {
        console.warn("API returned empty or invalid network for:", word);
        const minimalNetwork = {
          nodes: [{ id: 1, lemma: word, language_code: 'tl' }],
          edges: [],
          stats: { node_count: 1, edge_count: 0, depth: currentDepth, breadth: currentBreadth }
        };
        setWordNetwork(minimalNetwork);
        setDiagnostic(prev => ({ ...prev, data: { ...prev.data, network: minimalNetwork } }));
        return minimalNetwork;
      }
    } catch (fetchError) {
      console.error("Error in fetchWordNetworkData:", fetchError);
      setNetworkError(`Failed to load word connections: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}`);
      const fallbackNetwork = {
        nodes: [{ id: 1, lemma: word, language_code: 'tl' }],
        edges: [],
        stats: { node_count: 1, edge_count: 0, depth: currentDepth, breadth: currentBreadth }
      };
      setWordNetwork(fallbackNetwork);
      setDiagnostic(prev => ({ ...prev, data: { ...prev.data, error: fetchError } }));
      return fallbackNetwork;
    } finally {
      setIsLoadingNetwork(false);
    }
  }, [depth, breadth]);

  const fetchWordDetailsData = useCallback(async (wordOrId: string | number) => {
      if (!wordOrId) return null;
      setIsLoadingDetails(true);
      setDetailsError(null);
      setEtymologyTree(null); // Clear previous etymology
      try {
          console.log(`Fetching details for: ${wordOrId}`);
          
          // Add retries for better resilience
          let retryCount = 0;
          const maxRetries = 3;
          let success = false;
          let details;
          
          while (!success && retryCount < maxRetries) {
            try {
              details = await fetchWordDetails(String(wordOrId));
              success = true;
            } catch (retryError) {
              retryCount++;
              console.warn(`Attempt ${retryCount}/${maxRetries} failed for details. ${retryCount < maxRetries ? 'Retrying...' : 'Giving up.'}`);
              if (retryCount >= maxRetries) throw retryError;
              // Wait before retrying (exponential backoff)
              await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
            }
          }
          
          if (!details) throw new Error('Word details not found.');
          
          console.log('Details API response:', details);
          setSelectedWordInfo(details);
          setDiagnostic(prev => ({ ...prev, data: { ...prev.data, details: details } }));
          
          // Fetch etymology after details are loaded
          if (details.id) {
            fetchEtymologyTreeData(details.id); // Call the separate etymology fetch function
          }
          return details;
      } catch (error) {
          console.error("Error fetching word details:", error);
          setDetailsError(error instanceof Error ? error.message : 'Failed to load details.');
          setSelectedWordInfo(null);
          setDiagnostic(prev => ({ ...prev, data: { ...prev.data, error: error } }));
          return null;
      } finally {
          setIsLoadingDetails(false);
    }
  }, []);

  const loadWordData = useCallback(async (word: string) => {
    if (!word) return;
    console.log(`[loadWordData] Loading data for: ${word}`);
    setMainWord(word);
    setInputValue(word);
    updateHistory(word);
    
    setIsLoading(true); // Single loading state for combined fetch
    setError(null);
    setNetworkError(null);
    setDetailsError(null);
    setEtymologyError(null);
    setWordNetwork(null);
    setSelectedWordInfo(null);
    setEtymologyTree(null);
    setDiagnostic({ shown: false, data: null }); // Reset diagnostic

    try {
      // Fetch details first
      const details = await fetchWordDetailsData(word);
      
      // If details fetch is successful, fetch network
      if (details) {
         fetchWordNetworkData(details.lemma, depth, breadth); // Use fetched lemma
      } else {
          // If details fail, try fetching network with the original word anyway
          console.warn("Details fetch failed, attempting network fetch with original word");
          fetchWordNetworkData(word, depth, breadth);
      }
    } catch (error) {
        console.error("[loadWordData] Error loading word data:", error);
        setError(error instanceof Error ? error.message : "Failed to load word data.");
        setDiagnostic(prev => ({ ...prev, data: { ...prev.data, error: error } }));
    } finally {
      setIsLoading(false);
    }
  }, [fetchWordDetailsData, fetchWordNetworkData, updateHistory, depth, breadth]);

  useEffect(() => {
    loadWordData('salita'); // Load default word on initial mount
    
    // Fetch static data like POS and Stats
    const fetchStaticData = async () => {
      setIsLoadingStatistics(true);
      try {
        const [posData, statsData] = await Promise.all([
          getPartsOfSpeech(),
          getStatistics(),
        ]);
        setPartsOfSpeech(posData || []);
        setStatistics(statsData || null);
      } catch (error) {
        console.error("Error fetching static data:", error);
      } finally {
        setIsLoadingStatistics(false);
      }
    };
    fetchStaticData();

    // API Connection Check
    const checkApiConnection = async () => {
      try {
        console.log("Checking API connection...");
        setApiConnected(null);
        // setError(null); // Keep existing errors if any
        const isConnected = await testApiConnection();
        setApiConnected(isConnected);
        if (!isConnected) {
           setError(prev => prev ? prev + "\nAPI Connection Failed." : "API Connection Failed.");
           console.warn("API Connection Failed");
        }
      } catch (e) {
        console.error("Error checking API connection:", e);
        setApiConnected(false);
        setError(prev => prev ? prev + "\nError checking API." : "Error checking API.");
      }
    };
    checkApiConnection();

  }, [loadWordData]); // Only run on mount

  useEffect(() => {
    if (mainWord) {
      console.log(`Depth or breadth changed, refetching network: depth=${depth}, breadth=${breadth}`);
      fetchWordNetworkData(mainWord, depth, breadth);
    }
  }, [depth, breadth, mainWord, fetchWordNetworkData]); // Rerun when controls change

  const debouncedSearch = useMemo(() =>
    debounce(async (query: string) => {
      if (query.length < 2) {
        setSearchResults([]);
        setShowSuggestions(false);
        setIsLoadingSuggestions(false);
        return;
      }
      setIsLoadingSuggestions(true);
      try {
        const results = await searchWords({ q: query, limit: 10, mode: 'prefix' });
        const validResults: SearchWordResult[] = (results.results || [])
         .filter((item): item is SearchWordResult => typeof item.lemma === 'string' && item.lemma.trim() !== '')
         .map(item => ({
           ...item,
           normalized_lemma: item.normalized_lemma || undefined
         }));
        setSearchResults(validResults);
          setShowSuggestions(true);
      } catch (error) {
        console.error('Search suggestion error:', error);
        setSearchResults([]);
        setShowSuggestions(false);
      } finally {
        setIsLoadingSuggestions(false);
      }
    }, 300),
  []); // Empty dependency array

  useEffect(() => {
    debouncedSearch(inputValue);
    // Cleanup function to cancel debounced call if component unmounts or query changes quickly
    return () => {
      debouncedSearch.cancel();
    };
  }, [inputValue, debouncedSearch]);

  const handleSearch = useCallback((query?: string) => {
    const wordToSearch = query ?? inputValue;
    const normalizedInput = normalizeInput(wordToSearch);
    if (!normalizedInput) return;
    
    console.log('[handleSearch] triggered for:', normalizedInput);
    setShowSuggestions(false); // Hide suggestions
    setSearchResults([]); // Clear suggestions
    loadWordData(normalizedInput); // Use combined load function

  }, [inputValue, normalizeInput, loadWordData]);

  const handleNodeClick = useCallback((nodeData: NodeData) => {
    const word = nodeData.lemma; // Extract lemma from NodeData
    if (!word || word === mainWord) return;
    console.log(`Node clicked: ${word}`);
    loadWordData(word); // Use combined load function
  }, [mainWord, loadWordData]); // Depend on mainWord and loadWordData

  const handleWordLinkClick = useCallback((word: string) => { 
    if (!word || word === mainWord) return;
    console.log(`Word link clicked: ${word}`);
    loadWordData(word); // Use combined load function
  }, [mainWord, loadWordData]); // Depend on mainWord and loadWordData

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setInputValue(query);
    // Debounced search is handled by useEffect
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

  const handleSuggestionClick = (word: SearchWordResult) => {
    console.log("Suggestion clicked:", word.lemma);
    setShowSuggestions(false);
    setSearchResults([]);
    setInputValue(word.lemma);
    loadWordData(word.lemma); // Use combined load function
  };

  const handleBack = useCallback(() => {
    if (currentHistoryIndex > 0) {
      const newIndex = currentHistoryIndex - 1;
      setCurrentHistoryIndex(newIndex);
      const previousWord = wordHistory[newIndex];
      console.log(`Navigating back to: ${previousWord} (index ${newIndex})`);
      setInputValue(previousWord); // Update input value as well
      loadWordData(previousWord);
    }
  }, [currentHistoryIndex, wordHistory, loadWordData]);

  const handleForward = useCallback(() => {
    if (currentHistoryIndex < wordHistory.length - 1) {
      const newIndex = currentHistoryIndex + 1;
      setCurrentHistoryIndex(newIndex);
      const nextWord = wordHistory[newIndex];
      console.log(`Navigating forward to: ${nextWord} (index ${newIndex})`);
      setInputValue(nextWord); // Update input value as well
      loadWordData(nextWord);
    }
  }, [currentHistoryIndex, wordHistory, loadWordData]);

  const handleResetCircuitBreaker = () => {
    resetCircuitBreaker();
    setError(null);
    if (mainWord) { // Retry fetch for the current main word
      loadWordData(mainWord);
    }
  };

  const handleTestApiConnection = async () => {
    setError(null);
    setApiConnected(null);
    try {
      console.log("Manually testing API connection...");
      // First try the test endpoint
      const isConnected = await testApiConnection();
      setApiConnected(isConnected);
      
      if (!isConnected) {
        // Try to diagnose the issue
        try {
          // Check if the server is reachable at all
          const response = await fetch('http://localhost:5000/api/v2/test', {
            method: 'GET',
            mode: 'cors',
            headers: { 'Accept': 'application/json' }
          });
          
          if (response.ok) {
            setError("API server is reachable, but there may be CORS issues. Check the browser console for details.");
            console.warn("API server responded, but testApiConnection returned false. Possible CORS issue.");
          } else {
            setError(`API server is reachable but returned status ${response.status}. Check that the backend is running correctly.`);
          }
        } catch (fetchError) {
          // Server is not reachable
          setError("Cannot connect to the API server at http://localhost:5000. Please check that the backend server is running.");
          console.error("Direct connectivity test failed:", fetchError);
        }
      } else {
        setError(null); // Clear error if connection is now successful
        // Optionally re-fetch data if needed
        if (mainWord) loadWordData(mainWord);
      }
    } catch (e) {
      console.error("Error testing API connection:", e);
      setApiConnected(false);
      setError("Error testing API connection. Please try again later.");
    }
  };

  const handleRandomWord = useCallback(async () => {
    setError(null);
    setIsLoading(true);
    setEtymologyTree(null);
    setSelectedWordInfo(null);
    setWordNetwork(null);

    try {
      console.log('Fetching random word (V2)');
      const randomWordData = await getRandomWord();
      console.log('Raw random word data:', randomWordData);

      if (randomWordData && randomWordData.lemma) {
        loadWordData(randomWordData.lemma); // Use loadWordData to handle everything
      } else {
        throw new Error("Could not fetch valid random word data.");
      }
    } catch (error) {
      console.error("Error fetching random word:", error);
      setError(error instanceof Error ? error.message : "Failed to fetch random word.");
    } finally {
      setIsLoading(false);
    }
  }, [loadWordData]);

  const handleDepthChange = useCallback((newDepth: number) => {
    setDepth(newDepth);
  }, []);

  const handleBreadthChange = useCallback((newBreadth: number) => {
    setBreadth(newBreadth);
  }, []);

  const handleCloseDiagnostic = () => {
    setDiagnostic(prev => ({ ...prev, shown: false })); // Use shown: false
  };

  useEffect(() => {
    if (!mainRef.current || !resizerRef.current) return;

    const container = mainRef.current;
    const resizer = resizerRef.current;

    const onMouseDown = (e: MouseEvent) => {
      e.preventDefault();
      setIsResizing(true);
      document.body.style.cursor = 'col-resize';
      document.documentElement.style.userSelect = 'none'; // Prevent text selection
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isResizing || !container) return;
      const containerRect = container.getBoundingClientRect();
      // Calculate position as percentage
      const position = ((e.clientX - containerRect.left) / containerRect.width) * 100;
      // Constrain between 30% and 80%
      const constrainedPosition = Math.max(30, Math.min(80, position));
      setLeftPanelWidth(`${constrainedPosition}%`);
    };

    const onMouseUp = () => {
      if (isResizing) {
        setIsResizing(false);
        document.body.style.cursor = 'default';
        document.documentElement.style.userSelect = 'auto';
      }
    };

    resizer.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);

    return () => {
      resizer.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, [isResizing]);

  return (
    <div className={`word-explorer ${theme}`}>
      <Header
        title="Filipino Lexical Explorer"
        onRandomWord={handleRandomWord}
        onTestApiConnection={handleTestApiConnection}
        apiConnected={apiConnected}
      />
      
      <SearchBar
        query={inputValue}
        setQuery={setInputValue}
        suggestions={searchResults}
        onSearch={handleSearch}
        onSuggestionClick={handleSuggestionClick}
        showSuggestions={showSuggestions}
        isLoading={isLoadingSuggestions}
      />
      
      {error && (
        <div className="api-error-banner">
          <p>{error}</p>
          <button onClick={handleResetCircuitBreaker}>Reset Connection</button>
          <button onClick={handleTestApiConnection}>Test Connection</button>
        </div>
      )}
      
      <main ref={mainRef} className="main-content">
        <div className="panel-left" style={{ width: leftPanelWidth }}>
          {isLoadingNetwork && !wordNetwork ? (
            <Loader message="Loading word connections..." />
          ) : networkError ? (
            <ErrorDisplay message={`Error loading network: ${networkError}`} />
          ) : !wordNetwork || wordNetwork.nodes.length === 0 ? (
            <div className="empty-graph">
              <p>No connections available for this word.</p>
              <p className="empty-suggestion">Try searching for a different word.</p>
            </div>
          ) : (
            <div className="graph-content">
              <WordGraph
                wordNetwork={{
                  nodes: wordNetwork.nodes.map(node => ({
                    id: node.id,
                    lemma: node.lemma || node.label || `Node-${node.id}`, // Ensure lemma is never undefined
                    language_code: node.language_code,
                    main: node.id === selectedWordInfo?.id
                  })),
                  edges: (wordNetwork.edges || wordNetwork.links || []).map(edge => ({
                    id: edge.id || `edge-${typeof edge.source === 'object' ? edge.source.id : edge.source}-${typeof edge.target === 'object' ? edge.target.id : edge.target}`,
                    source: typeof edge.source === 'object' ? edge.source.id : edge.source,
                    target: typeof edge.target === 'object' ? edge.target.id : edge.target,
                    type: edge.type || 'default'
                  })),
                  stats: wordNetwork.stats
                }}
                mainWord={mainWord}
                onNodeClick={handleNodeClick}
                onNetworkChange={() => {}} // No-op function since we handle depth changes elsewhere
                initialDepth={depth}
                initialBreadth={breadth}
              />
            </div>
          )}
          
          <NetworkControls
            depth={depth}
            breadth={breadth}
            onDepthChange={setDepth}
            onBreadthChange={setBreadth}
          />
        </div>
        
        <div className="resizer" ref={resizerRef} />
        
        <div className="panel-right">
          {isLoadingDetails ? (
            <Loader message="Loading word details..." />
          ) : detailsError ? (
            <ErrorDisplay message={`Error loading details: ${detailsError}`} />
          ) : !selectedWordInfo ? (
            <div className="no-word-selected">
              <p>No word selected</p>
              <p className="empty-tip">Search for a word or click a node in the graph.</p>
            </div>
          ) : (
            <WordDetails
              wordInfo={selectedWordInfo}
              etymologyTree={etymologyTree}
              isLoadingEtymology={isLoadingEtymology}
              etymologyError={etymologyError}
              onWordLinkClick={handleWordLinkClick}
              onEtymologyNodeClick={handleNodeClick}
            />
          )}
        </div>
      </main>
      
      <Footer />
      
      {diagnostic.shown && (
        <div className="diagnostic-overlay">
          <div className="diagnostic-content">
            <h3>Diagnostic Info</h3>
            <pre>{JSON.stringify(diagnostic.data, null, 2)}</pre>
            <button onClick={handleCloseDiagnostic}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default WordExplorer;
