import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchResult, SearchOptions, EtymologyTree, Statistics, Definition, WordNetworkOptions, Etymology, SearchWordResult } from "../types";
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
import { useNavigate, useLocation, useParams } from 'react-router-dom';
import { Box, CircularProgress, Typography, TextField, List, ListItem, ListItemButton, ListItemText, Divider, InputAdornment, IconButton } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

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
  const [searchResults, setSearchResults] = useState<SearchWordResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [apiEndpoint, setApiEndpoint] = useState<string | null>(localStorage.getItem('successful_api_endpoint'));
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [isLoadingStatistics, setIsLoadingStatistics] = useState<boolean>(false);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('tl');
  const navigate = useNavigate();
  const location = useLocation();
  const { word: wordFromParams } = useParams<{ word?: string }>();

  const detailsContainerRef = useRef<HTMLDivElement>(null);

  // --- Search State (Local to Component) ---
  const [searchResultsLocal, setSearchResultsLocal] = useState<SearchWordResult[]>([]);
  const [isSearchLoading, setIsSearchLoading] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [offset, setOffset] = useState<number>(0);
  const [limit, setLimit] = useState<number>(20);
  const [hasMore, setHasMore] = useState<boolean>(false);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // *** ADDED: State for toggling metadata in WordDetails ***
  const [showMetadata, setShowMetadata] = useState<boolean>(false);

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
        setSearchResults(searchResultsData.words);
        
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
        setSearchResults([]);
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
        setSearchResults(results.words);
        setShowSuggestions(true);
      } catch (error) {
        console.error("Error fetching search suggestions:", error);
        setSearchResults([]);
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

  const handleGraphParamsChange = useCallback((newDepth: number, newBreadth: number) => {
    console.log("Graph params changed - New Depth:", newDepth, "New Breadth:", newBreadth);
    setDepth(newDepth);
    setBreadth(newBreadth);
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

  useEffect(() => {
    const currentSelected = wordFromParams ? decodeURIComponent(wordFromParams) : null;
    setSelectedWordInfo(null);
    setMainWord("");
    setInputValue("");
    setShowSuggestions(false);
    setSearchTerm(currentSelected || "");
    setDepth(3);
    setBreadth(8);
    setWordNetwork(null);
    setEtymologyTree(null);
    setIsLoadingEtymology(false);
    setEtymologyError(null);
  }, [wordFromParams, setSearchTerm]);

  useEffect(() => {
    if (selectedWordInfo?.id) { 
        const wordId = selectedWordInfo.id;
        const fetchEtymology = async () => {
          setIsLoadingEtymology(true);
          setEtymologyError(null);
          try {
            const fetchedTree = await getEtymologyTree(wordId, 3); 
            setEtymologyTree(fetchedTree);
          } catch (err: any) {
            console.error("Error fetching etymology tree:", err);
            setEtymologyError(err.message || 'Failed to load etymology tree');
            setEtymologyTree(null);
          } finally {
            setIsLoadingEtymology(false);
          }
        };
        fetchEtymology();
      } else {
        setEtymologyTree(null);
        setIsLoadingEtymology(false);
        setEtymologyError(null);
      }
  }, [selectedWordInfo]);

  // --- Handlers ---
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
  };

  const handleWordClick = (word: string) => {
    setSelectedWordInfo(null);
    setMainWord(word);
    setInputValue(word);
    handleSearch(word);
    navigate(`/explore/${encodeURIComponent(word)}`);
  };
  
  // Handler for clicking node in the main word network graph
  const handleGraphNodeClick = (word: string) => {
    handleWordClick(word); // Reuse existing logic
  };
  
  // Handler for clicking node in the etymology tree
  const handleEtymologyNodeClick = (node: any) => {
    console.log('Etymology Node Clicked:', node);
    // Decide action: Maybe fetch details for this word if it's different?
    if (node.label && node.label !== mainWord) {
       handleWordClick(node.label); 
    }
  };
  
  const handleNetworkSettingsChange = (depth: number, breadth: number) => {
    setDepth(depth);
    setBreadth(breadth);
    // The useWordNetwork hook will re-fetch automatically due to dependencies
  };

  // Add handler for search submission (e.g., Enter key)
  const handleSearchSubmit = (event?: React.FormEvent) => {
      if (event) event.preventDefault(); // Prevent default form submission if used
      setShowSuggestions(false); // Hide suggestions on explicit search
      handleSearch(); // Use existing handleSearch logic
  };

  return (
    <Box className={`word-explorer-container ${theme}-theme`} sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* Left Panel: Search Input and Results */}
      <Box sx={{ width: '300px', borderRight: '1px solid', borderColor: 'divider', display: 'flex', flexDirection: 'column', p: 1 }}>
        {/* Replace SearchBar with TextField */}
        <Box component="form" onSubmit={handleSearchSubmit} sx={{ mb: 1 }}>
          <TextField
            fullWidth
            variant="outlined"
            size="small"
            placeholder="Search for a word..."
            value={inputValue} // Bind to inputValue state
            onChange={handleInputChange} // Use existing handler
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton type="submit" size="small" title="Search">
                    <SearchIcon />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
        </Box>

        {/* Replace SearchResultsList with MUI List */}
        <Box sx={{ flexGrow: 1, overflowY: 'auto', position: 'relative' }}>
          {isLoading && !searchResults.length && (
             <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <CircularProgress size={24} />
             </Box> 
          )}
          {error && !searchResults.length && (
             <Typography color="error" sx={{ p: 1 }}>{error}</Typography>
          )}
          {showSuggestions && searchResults.length > 0 && (
            <List dense disablePadding>
              {searchResults.map((result) => (
                <ListItem key={result.id || result.lemma} disablePadding>
                  <ListItemButton 
                    onClick={() => handleWordClick(result.lemma)} 
                    selected={selectedWordInfo?.lemma === result.lemma} // Highlight selected
                  >
                    <ListItemText 
                        primary={result.lemma} 
                        // Optionally add definition preview if available in SearchWordResult
                        // secondary={result.definitions?.[0]?.text.substring(0, 50) + '...'} 
                    />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          )}
          {!isLoading && !error && !showSuggestions && searchResults.length === 0 && !selectedWordInfo && (
            <Typography variant="body2" color="text.secondary" sx={{ p: 1, fontStyle: 'italic' }}>
              Enter a word to search.
            </Typography>
          )}
          {/* Add Load More button or logic here if needed */}
          {/* {hasMore && <Button onClick={handleLoadMore}>Load More</Button>} */}
        </Box>
      </Box>

      {/* Right Panel: Details and Graph */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Details Pane */}
        <Box className="details-pane" sx={{ flexBasis: '50%', overflowY: 'auto', borderBottom: '1px solid', borderColor: 'divider', p: 2 }}>
          {selectedWordInfo ? (
            <WordDetails 
              wordInfo={selectedWordInfo} 
              etymologyTree={etymologyTree}
              isLoadingEtymology={isLoadingEtymology}
              etymologyError={etymologyError}
              onWordLinkClick={handleWordClick}
              onEtymologyNodeClick={handleEtymologyNodeClickWrapper}
              showMetadata={showMetadata}
              setShowMetadata={setShowMetadata}
            />
          ) : isLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <CircularProgress />
            </Box>
          ) : error ? (
            <Typography color="error">Error: {error}</Typography>
          ) : (
            <Typography color="text.secondary">Select a word from the search results or graph.</Typography>
          )}
        </Box>
        {/* Graph Pane */}
        <Box className="graph-pane" sx={{ flexBasis: '50%', position: 'relative' }}>
           {selectedWordInfo && wordNetwork ? (
              <WordGraph
                key={selectedWordInfo.lemma}
                wordNetwork={wordNetwork}
                mainWord={mainWord}
                onNodeClick={handleGraphNodeClick}
                onNetworkChange={handleNetworkSettingsChange}
                initialDepth={depth}
                initialBreadth={breadth}
              />
           ) : selectedWordInfo && isLoading ? (
               <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <CircularProgress /> 
                  <Typography sx={{ml: 1}}>Loading graph...</Typography>
               </Box>
           ) : (
             !isLoading && <Typography color="text.secondary" sx={{ p: 2 }}>Word network will appear here.</Typography>
           )}
           {/* Consider showing graph-specific errors here */}
           {/* {graphError && <Typography color="error">{graphError}</Typography>} */}
        </Box>
      </Box>
    </Box>
  );
};

export default WordExplorer;
