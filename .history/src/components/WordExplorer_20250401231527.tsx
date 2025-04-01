import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchResult, SearchOptions, EtymologyTree, Statistics, Definition } from "../types";
import unidecode from "unidecode";
import { 
  fetchWordNetwork, 
  fetchWordDetails, 
  searchWords, 
  resetCircuitBreaker, 
  testApiConnection,
  getEtymologyTree,
  getPartsOfSpeech,
  getRandomWord,
  getStatistics,
  getBaybayinWords,
  getAffixes,
  getRelations,
  getAllWords
} from "../api/wordApi";
import axios from 'axios';
import DOMPurify from 'dompurify';
import { debounce } from "lodash";

// MUI Core & Layout
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Container from '@mui/material/Container';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip'; // For button hints

// MUI Input & Buttons
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton'; // Keep for structure, might use text inside
import InputAdornment from '@mui/material/InputAdornment';
import Popper from '@mui/material/Popper';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';

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
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [partsOfSpeech, setPartsOfSpeech] = useState<any[]>([]);
  
  // New state variables for additional API data
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [isLoadingStatistics, setIsLoadingStatistics] = useState<boolean>(false);
  const [baybayinWords, setBaybayinWords] = useState<any[]>([]);
  const [isLoadingBaybayin, setIsLoadingBaybayin] = useState<boolean>(false);
  const [affixes, setAffixes] = useState<any[]>([]);
  const [isLoadingAffixes, setIsLoadingAffixes] = useState<boolean>(false);
  const [relations, setRelations] = useState<any[]>([]);
  const [isLoadingRelations, setIsLoadingRelations] = useState<boolean>(false);
  const [allWords, setAllWords] = useState<any[]>([]);
  const [isLoadingAllWords, setIsLoadingAllWords] = useState<boolean>(false);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('tl');

  // Add a new state variable for toggling metadata
  const [showMetadata, setShowMetadata] = useState<boolean>(false);

  const detailsContainerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null); // Ref for suggestions popper anchor

  // Initialize details width from localStorage
  useEffect(() => {
    const savedWidth = localStorage.getItem('wordDetailsWidth');
    if (savedWidth && detailsContainerRef.current) {
      detailsContainerRef.current.style.width = `${savedWidth}px`;
    }
  }, []);

  // Track changes to container width with ResizeObserver
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

  // Function to normalize input
  const normalizeInput = (input: string) => unidecode(input.trim().toLowerCase());

  // DEFINE FETCHERS FIRST
  const fetchWordNetworkData = useCallback(async (word: string, depth: number = 2, breadth: number = 10) => {
    try {
      setIsLoading(true);
      const data = await fetchWordNetwork(word, { 
        depth,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.3
      });
      
      // Add nodes and edges from the fetched data
      if (data) {
        // // Initialize any missing clusters
        // requiredClusters.forEach(cluster => {
        //   if (!data.clusters[cluster]) {
        //     data.clusters[cluster] = [];
        //   }
        // });
        
        // // Merge clusters
        // Object.keys(data.clusters).forEach(key => {
        //   clusters[key] = [...(clusters[key] || []), ...data.clusters[key]];
        // });
      }
      
      setWordNetwork(data);
      return data;
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
      throw error;
    } finally {
      setIsLoading(false); // Ensure loading state is reset
    }
  }, []); // Dependencies: None, as it uses props/constants

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
  }, []); // Dependencies: None

  // NOW DEFINE HANDLERS THAT USE FETCHERS
  const handleSearch = useCallback(async (searchWord?: string) => {
    // --- DEBUG LOGS START ---
    console.log(`handleSearch called. searchWord: ${searchWord}, inputValue: ${inputValue}`);
    // --- DEBUG LOGS END ---
    const wordToSearch = searchWord || inputValue.trim();
    // --- DEBUG LOGS START ---
    console.log(`handleSearch: wordToSearch calculated as: ${wordToSearch}`);
    // --- DEBUG LOGS END ---

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
    setEtymologyTree(null); 

    try {
      console.log('Searching for word:', normalizedInput);
      
      try {
        console.log('Fetching word details for:', normalizedInput);
        const wordData = await fetchWordDetails(normalizedInput);
        console.log('Word details received:', wordData);
        
        if (wordData && wordData.lemma) {
          console.log('Setting selected word info with valid data');
          setSelectedWordInfo(wordData);
          setMainWord(wordData.lemma);
          
          const [networkData] = await Promise.all([
            fetchWordNetworkData(normalizedInput, depth, breadth), // CALL fetcher
            fetchEtymologyTree(wordData.id) // CALL fetcher
          ]);
          
          console.log('Word network data received:', networkData);
          
          setWordNetwork(networkData);
          setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), wordData.lemma]);
          setCurrentHistoryIndex(prevIndex => prevIndex + 1);
          setInputValue(wordData.lemma);
          setShowSuggestions(false);
          return;
        } else {
          console.log('Word details missing lemma or invalid:', wordData);
        }
      } catch (detailsError) {
        console.error('Direct word details fetch failed, trying search:', detailsError);
      }
      
      const searchOptions: SearchOptions = { 
          page: 1, 
          per_page: 20, 
          exclude_baybayin: true,
          language: 'tl', 
          mode: 'all', 
          sort: 'relevance',
          order: 'desc'
        };
      const searchResults = await searchWords(normalizedInput, searchOptions);
      
      console.log('Search results:', searchResults);
      
      if (searchResults && searchResults.words && searchResults.words.length > 0) {
        const firstResult = searchResults.words[0];
        console.log('Using first search result:', firstResult);
        
        const wordData = await fetchWordDetails(firstResult.lemma);
        console.log('Word details for search result:', wordData);
        
        if (wordData && wordData.lemma) {
          setSelectedWordInfo(wordData);
          setMainWord(wordData.lemma);
          
          console.log('Fetching word network and etymology tree for search result:', wordData.lemma);
          const [networkData] = await Promise.all([
            fetchWordNetworkData(wordData.lemma, depth, breadth), // CALL fetcher
            fetchEtymologyTree(wordData.id) // CALL fetcher
          ]);
          console.log('Word network for search result:', networkData);
          
          setWordNetwork(networkData);
          setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), wordData.lemma]);
          setCurrentHistoryIndex(prevIndex => prevIndex + 1);
          setInputValue(wordData.lemma);
          setShowSuggestions(false);
        } else {
          setError(`Could not fetch full details for "${firstResult.lemma}". Displaying basic info.`);
          setMainWord(firstResult.lemma);
          setSelectedWordInfo({
              id: firstResult.id,
              lemma: firstResult.lemma,
              normalized_lemma: firstResult.normalized_lemma || firstResult.lemma,
              language_code: firstResult.language_code || 'tl',
              has_baybayin: firstResult.has_baybayin || false,
              baybayin_form: firstResult.baybayin_form,
              romanized_form: firstResult.romanized_form,
              definitions: (firstResult.definitions || []).map(def => ({ 
                  id: def.id,
                  text: def.definition_text,
                  definition_text: def.definition_text,
                  original_pos: def.part_of_speech,
                  part_of_speech: null, 
                  examples: [],
                  usage_notes: [],
                  tags: [],
                  sources: [],
                  relations: []
              })),
              pronunciations: [],
              etymologies: [],
              tags: null,
          });
          setWordNetwork(null); 
          setEtymologyTree(null); 
        }
      } else {
        setError(`No results found for "${wordToSearch}"`);
        setSelectedWordInfo(null);
        setWordNetwork(null);
      }
    } catch (err: any) {
      console.error("Error during search:", err);
      setError(err.message || "An error occurred during the search.");
      setSelectedWordInfo(null);
      setWordNetwork(null);
    } finally {
      setIsLoading(false);
    }
  // Make sure dependencies include the fetcher functions now defined above
  }, [inputValue, depth, breadth, fetchWordNetworkData, fetchEtymologyTree, currentHistoryIndex]); 

  const handleWordLinkClick = useCallback(async (word: string) => {
    console.log("Word link clicked:", word);
    if (word !== mainWord) {
      await handleSearch(word); // handleSearch is now defined before this
      detailsContainerRef.current?.scrollTo(0, 0);
    }
  // handleSearch is now a valid dependency
  }, [mainWord, handleSearch]); 

  const handleNodeClick = handleWordLinkClick;

  // Debounced search for suggestions
  const debouncedSearch = useCallback(
    debounce(async (query: string) => {
      if (query.length < 2) { // Only search if query is long enough
        setSearchResults([]);
        setShowSuggestions(false);
        return;
      }
      setIsLoading(true); // Indicate loading for suggestions
      try {
        const searchOptions: SearchOptions = { page: 1, per_page: 10, language: 'tl', mode: 'all' };
        const results = await searchWords(query, searchOptions);
        if (results && results.words) {
          const suggestions = results.words.map(word => ({ id: word.id, word: word.lemma })); // Map to simple {id, word}
          setSearchResults(suggestions);
          setShowSuggestions(suggestions.length > 0);
        }
      } catch (error) {
        console.error("Error fetching search suggestions:", error);
        setSearchResults([]);
        setShowSuggestions(false);
      } finally {
        setIsLoading(false);
      }
    }, 300), // Debounce time
    [] // No dependencies needed for debounced function itself
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
    setInputValue(word);
    setShowSuggestions(false);
    handleSearch(word);
  };

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

  const handleBack = useCallback(() => {
    if (currentHistoryIndex > 0) {
      // First update the index
      const newIndex = currentHistoryIndex - 1;
      setCurrentHistoryIndex(newIndex);
      
      // Then get the word at that index
      const previousWord = wordHistory[newIndex];
      console.log(`Navigating back to: ${previousWord} (index ${newIndex})`);
      
      // Fetch the word data without updating history
      setIsLoading(true);
      setError(null);
      
      Promise.all([
        fetchWordDetails(previousWord),
        fetchWordNetworkData(previousWord, depth, breadth)
      ])
      .then(([wordData, networkData]) => {
        setSelectedWordInfo(wordData);
        setWordNetwork(networkData);
        setMainWord(wordData.lemma);
        setInputValue(wordData.lemma);
      })
      .catch(error => {
        console.error("Error navigating back:", error);
        let errorMessage = "Failed to navigate back.";
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        setError(errorMessage);
      })
      .finally(() => {
        setIsLoading(false);
      });
    }
  }, [currentHistoryIndex, wordHistory, depth, breadth, fetchWordNetworkData]);

  const handleForward = useCallback(() => {
    if (currentHistoryIndex < wordHistory.length - 1) {
      // First update the index
      const newIndex = currentHistoryIndex + 1;
      setCurrentHistoryIndex(newIndex);
      
      // Then get the word at that index
      const nextWord = wordHistory[newIndex];
      console.log(`Navigating forward to: ${nextWord} (index ${newIndex})`);
      
      // Fetch the word data without updating history
      setIsLoading(true);
      setError(null);
      
      Promise.all([
        fetchWordDetails(nextWord),
        fetchWordNetworkData(nextWord, depth, breadth)
      ])
      .then(([wordData, networkData]) => {
        setSelectedWordInfo(wordData);
        setWordNetwork(networkData);
        setMainWord(wordData.lemma);
        setInputValue(wordData.lemma);
      })
      .catch(error => {
        console.error("Error navigating forward:", error);
        let errorMessage = "Failed to navigate forward.";
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        setError(errorMessage);
      })
      .finally(() => {
        setIsLoading(false);
      });
    }
  }, [currentHistoryIndex, wordHistory, depth, breadth, fetchWordNetworkData]);

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

  // Fetch parts of speech on mount
  useEffect(() => {
    const fetchPartsOfSpeech = async () => {
      try {
        const data = await getPartsOfSpeech();
        setPartsOfSpeech(data);
      } catch (error) {
        console.error("Error fetching parts of speech:", error);
      }
    };
    
    fetchPartsOfSpeech();
  }, []);

  // Function to handle random word button click
  const handleRandomWord = useCallback(async () => {
    setError(null);
    setIsLoading(true);
    setEtymologyTree(null); // Reset etymology tree
    
    try {
      console.log('Fetching random word');
      
      // Get random word
      const randomWordData = await getRandomWord();
      console.log('Random word data:', randomWordData);
      
      // The random word API might not return fully normalized data, so fetch the detailed word info
      if (randomWordData && randomWordData.lemma) {
        const wordData = await fetchWordDetails(randomWordData.lemma);
        
        setSelectedWordInfo(wordData);
        setMainWord(wordData.lemma);
        
        // Fetch word network and etymology tree
        const networkData = await fetchWordNetworkData(wordData.lemma, depth, breadth);
        console.log('Word network for random word:', networkData);
        
        setWordNetwork(networkData);
        setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), wordData.lemma]);
        setCurrentHistoryIndex(prevIndex => prevIndex + 1);
        setInputValue(wordData.lemma);
        
        // Fetch etymology tree
        fetchEtymologyTree(wordData.id);
      } else {
        throw new Error("Could not fetch random word.");
      }
    } catch (error) {
      console.error("Error fetching random word:", error);
      let errorMessage = "Failed to fetch random word.";
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (axios.isAxiosError(error) && error.response) {
        errorMessage = error.response.data?.error?.message || error.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [depth, breadth, currentHistoryIndex, fetchWordNetworkData, fetchEtymologyTree]);

  // Function to fetch statistics
  const fetchStatistics = useCallback(async () => {
    setIsLoadingStatistics(true);
    try {
      const data = await getStatistics();
      console.log('Statistics data:', data);
      setStatistics(data);
    } catch (error) {
      console.error('Error fetching statistics:', error);
    } finally {
      setIsLoadingStatistics(false);
    }
  }, []);

  // Function to fetch baybayin words
  const fetchBaybayinWords = useCallback(async (page: number = 1, language: string = 'tl') => {
    setIsLoadingBaybayin(true);
    try {
      const data = await getBaybayinWords(page, 20, language);
      console.log('Baybayin words:', data);
      if (data && Array.isArray(data)) {
        setBaybayinWords(data);
      } else if (data && data.data && Array.isArray(data.data)) {
        setBaybayinWords(data.data);
        if (data.meta && data.meta.pages) {
          setTotalPages(data.meta.pages);
        }
      }
    } catch (error) {
      console.error('Error fetching baybayin words:', error);
    } finally {
      setIsLoadingBaybayin(false);
    }
  }, []);

  // Function to fetch affixes
  const fetchAffixes = useCallback(async (language: string = 'tl') => {
    setIsLoadingAffixes(true);
    try {
      const data = await getAffixes(language);
      console.log('Affixes data:', data);
      if (data && Array.isArray(data)) {
        setAffixes(data);
      } else if (data && data.data && Array.isArray(data.data)) {
        setAffixes(data.data);
      }
    } catch (error) {
      console.error('Error fetching affixes:', error);
    } finally {
      setIsLoadingAffixes(false);
    }
  }, []);

  // Function to fetch relations
  const fetchRelations = useCallback(async (language: string = 'tl') => {
    setIsLoadingRelations(true);
    try {
      const data = await getRelations(language);
      console.log('Relations data:', data);
      if (data && Array.isArray(data)) {
        setRelations(data);
      } else if (data && data.data && Array.isArray(data.data)) {
        setRelations(data.data);
      }
    } catch (error) {
      console.error('Error fetching relations:', error);
    } finally {
      setIsLoadingRelations(false);
    }
  }, []);

  // Function to fetch all words
  const fetchAllWords = useCallback(async (page: number = 1, language: string = 'tl') => {
    setIsLoadingAllWords(true);
    try {
      const data = await getAllWords(page, 20, language);
      console.log('All words data:', data);
      if (data && Array.isArray(data)) {
        setAllWords(data);
      } else if (data && data.data && Array.isArray(data.data)) {
        setAllWords(data.data);
        if (data.meta && data.meta.pages) {
          setTotalPages(data.meta.pages);
        }
      }
    } catch (error) {
      console.error('Error fetching all words:', error);
    } finally {
      setIsLoadingAllWords(false);
    }
  }, []);

  // Function to handle language change
  const handleLanguageChange = useCallback((language: string) => {
    setSelectedLanguage(language);
    fetchBaybayinWords(1, language);
    fetchAffixes(language);
    fetchRelations(language);
    fetchAllWords(1, language);
  }, [fetchBaybayinWords, fetchAffixes, fetchRelations, fetchAllWords]);

  // Function to handle page change
  const handlePageChange = useCallback((page: number) => {
    setCurrentPage(page);
    fetchAllWords(page, selectedLanguage);
  }, [fetchAllWords, selectedLanguage]);

  // Fetch additional data on mount
  useEffect(() => {
    // Fetch statistics
    fetchStatistics();
    
    // Fetch baybayin words
    fetchBaybayinWords();
    
    // Fetch affixes
    fetchAffixes();
    
    // Fetch relations
    fetchRelations();
    
    // Fetch all words
    fetchAllWords();
    
  }, [fetchStatistics, fetchBaybayinWords, fetchAffixes, fetchRelations, fetchAllWords]);

  // Render statistics section
  const renderStatistics = () => {
    if (!statistics) return null;

    // Use correct fields from Statistics type
    return (
      <div className="statistics-section">
        <h4>API Statistics</h4>
        <p>Status: {statistics.status}</p>
        <p>API Version: {statistics.api_version}</p>
        <p>DB Connected: {statistics.database.connected ? 'Yes' : 'No'}</p>
        {statistics.database.word_count !== undefined && (
          <p>Total Words: {statistics.database.word_count}</p>
        )}
        {statistics.database.language_count !== undefined && (
          <p>Total Languages: {statistics.database.language_count}</p>
        )}
        {statistics.database.stats_error && (
           <p>Stats Error: {statistics.database.stats_error}</p>
        )}
        <p>Timestamp: {new Date(statistics.timestamp).toLocaleString()}</p>
      </div>
    );
  };

  // Render baybayin words section
  const renderBaybayinWords = () => {
    if (isLoadingBaybayin) {
      return (
        <div className="baybayin-section loading">
          <div className="spinner"></div>
          <p>Loading Baybayin words...</p>
        </div>
      );
    }

    if (!baybayinWords || baybayinWords.length === 0) {
      return null;
    }

    return (
      <div className="baybayin-section">
        <h3>Baybayin Words</h3>
        <div className="baybayin-grid">
          {baybayinWords.map((word, index) => (
            <div key={index} className="baybayin-card" onClick={() => handleSearch(word.word)}>
              <p className="baybayin-text">{word.baybayin}</p>
              <p className="baybayin-latin">{word.word}</p>
            </div>
          ))}
        </div>
        <div className="pagination">
          {Array.from({ length: totalPages }, (_, i) => (
            <button 
              key={i} 
              className={`page-button ${currentPage === i + 1 ? 'active' : ''}`}
              onClick={() => fetchBaybayinWords(i + 1, selectedLanguage)}
            >
              {i + 1}
            </button>
          ))}
        </div>
      </div>
    );
  };

  // Render affixes section
  const renderAffixes = () => {
    if (isLoadingAffixes) {
      return (
        <div className="affixes-section loading">
          <div className="spinner"></div>
          <p>Loading affixes...</p>
        </div>
      );
    }

    if (!affixes || affixes.length === 0) {
      return null;
    }

    // Group affixes by type
    const affixesByType: Record<string, any[]> = {};
    affixes.forEach(affix => {
      const type = affix.type || 'Other';
      if (!affixesByType[type]) {
        affixesByType[type] = [];
      }
      affixesByType[type].push(affix);
    });

    return (
      <div className="affixes-section">
        <h3>Affixes</h3>
        {Object.entries(affixesByType).map(([type, affixList]) => (
          <div key={type} className="affix-group">
            <h4>{type} ({affixList.length})</h4>
            <div className="affix-grid">
              {affixList.map((affix, index) => (
                <div key={index} className="affix-card">
                  <p className="affix-text">{affix.affix}</p>
                  <p className="affix-meaning">{affix.meaning}</p>
                  {affix.examples && affix.examples.length > 0 && (
                    <div className="affix-examples">
                      <h5>Examples:</h5>
                      <ul>
                        {affix.examples.map((example: string, i: number) => (
                          <li key={i}>{example}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  };

  console.log('Search query:', inputValue);
  console.log('Search results:', searchResults);
  console.log('Show suggestions:', showSuggestions);

  // Function to toggle metadata visibility
  const toggleMetadata = useCallback(() => {
    setShowMetadata(prev => !prev);
  }, []);

  const resetDisplay = useCallback(() => {
    setSelectedWordInfo({
        id: 0,
        lemma: 'Loading...',
        definitions: [],
        etymologies: [],
        pronunciations: [],
        credits: [],
        tags: null,
        outgoing_relations: [],
        incoming_relations: [],
        root_affixations: [],
        affixed_affixations: [],
    });
    setWordNetwork(null); 
    setEtymologyTree(null);
    setError(null);
  }, []);

  useEffect(() => {
    // Example: Fetch initial word or reset on mount
    // resetDisplay(); 
  }, [resetDisplay]); // Include resetDisplay if called on mount

  return (
    <Box sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        bgcolor: 'background.default', // Use theme background
        color: 'text.primary'        // Use theme text color
    }}>
      {/* === Top App Bar === */}
      <AppBar position="static" elevation={1} sx={{ bgcolor: 'background.paper' }}>
        <Toolbar variant="dense">
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: 'text.primary' }}>
            Filipino Word Explorer
          </Typography>
          
          {/* API Status Indicator - Text based */}
          <Typography 
             variant="caption" 
             sx={{ 
                 mr: 2, 
                 p: '2px 8px', 
                 borderRadius: 1, 
                 bgcolor: apiConnected === null ? 'action.disabledBackground' : apiConnected ? 'success.light' : 'error.light', 
                 color: apiConnected === null ? 'text.disabled' : apiConnected ? 'success.dark' : 'error.dark' 
             }}
          >
            API: {apiConnected === null ? 'Checking...' : apiConnected ? 'Connected' : 'Disconnected'}
          </Typography>

          <Tooltip title="Explore Random Word">
            <span>
              <Button onClick={handleRandomWord} disabled={isLoading} color="inherit" size="small" startIcon={<>🎲</>}> {/* Emoji Icon */}
                Random
              </Button>
            </span>
          </Tooltip>

          <Tooltip title="Toggle Theme">
            {/* Use IconButton with text/emoji if preferred */}
            <IconButton onClick={toggleTheme} color="inherit" size="small">
              {theme === 'dark' ? <>☀️</> : <>🌙</>} {/* Emojis */} 
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      {/* === Search & Navigation Area === */}
      <Container maxWidth="md" sx={{ pt: 2, pb: 1 }}>
         <Paper elevation={0} sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
             {/* History Buttons */}
             <Tooltip title="Back in History">
                <span>
                   {/* IconButton with text/symbol */}
                   <IconButton onClick={handleBack} disabled={currentHistoryIndex <= 0} size="small">
                     ←
                   </IconButton>
                </span>
             </Tooltip>
             <Tooltip title="Forward in History">
                <span>
                  {/* IconButton with text/symbol */}
                   <IconButton onClick={handleForward} disabled={currentHistoryIndex >= wordHistory.length - 1} size="small">
                     →
                   </IconButton>
                </span>
             </Tooltip>

            {/* Search Input */}
             <TextField
                inputRef={searchInputRef}
                fullWidth
                variant="outlined"
                size="small"
                placeholder="Enter a word..."
                value={inputValue}
                onChange={handleInputChange}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); handleSearch(); }}}
                InputProps={{
                    endAdornment: isLoading ? (
                        <InputAdornment position="end">
                            <CircularProgress size={20} />
                        </InputAdornment>
                    ) : null,
                }}
                sx={{ bgcolor: 'background.default' }}
             />
         </Paper>
         {/* Suggestions Popper */}
          <Popper
            open={showSuggestions && searchResults.length > 0}
            anchorEl={searchInputRef.current}
            placement="bottom-start"
            modifiers={[{ name: 'offset', options: { offset: [0, 4] }}]} // Small offset
            style={{ zIndex: 1200, width: searchInputRef.current?.clientWidth }} // Match input width
          >
            <Paper elevation={3}>
              <List dense disablePadding>
                {searchResults.map((result) => (
                  <ListItemButton key={result.id} onClick={() => handleSuggestionClick(result.word)} dense>
                    <ListItemText primary={result.word} />
                  </ListItemButton>
                ))}
              </List>
            </Paper>
          </Popper>
      </Container>

       {/* === Error Display Area === */}
       {error && (
            <Container maxWidth="lg" sx={{ mt: 1 }}>
                <Alert severity="error" variant="outlined" onClose={() => setError(null)} sx={{ width: '100%' }}>
                    {error}
                    {/* Add specific buttons back if needed */}
                    {(error.includes('Circuit breaker') || error.includes('API connection') || error.includes('Network error') ) && (
                        <Button onClick={handleResetCircuitBreaker} size="small" sx={{ ml: 1 }}>Reset Connection</Button>
                    )}
                    {error.includes('backend server') && (
                         <Button onClick={handleTestApiConnection} size="small" sx={{ ml: 1 }}>Test API</Button>
                    )}
                </Alert>
            </Container>
        )}

      {/* === Main Content Area (Graph + Details) === */}
      <Box sx={{
        flexGrow: 1,
        display: 'flex',
        overflow: 'hidden', // Prevent overall scroll
        p: 1, // Padding around graph/details
        gap: 1 // Gap between graph/details
      }}>

        {/* --- Graph Panel --- */}
        <Paper elevation={1} sx={{
            flexGrow: 1, // Takes up remaining space
            height: '100%', // Fill vertical space
            display: 'flex',
            flexDirection: 'column', // To contain graph and potential overlays
            position: 'relative', // For positioning overlays
            overflow: 'hidden' // Prevent scroll within paper
        }}>
            {/* The .graph-content class used by WordGraph needs to be inside here */} 
            <Box className="graph-content" sx={{ flexGrow: 1, position: 'relative' }}> {/* Ensure WordGraph's container takes full space */} 
                {isLoading && !wordNetwork && ( // Show main loading only if no network exists yet
                   <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', bgcolor: 'rgba(0,0,0,0.1)', zIndex: 5 }}>
                        <CircularProgress />
                        <Typography sx={{ mt: 1 }}>Loading Network...</Typography>
                    </Box>
                )}
                {/* Render WordGraph - it will handle its internal SVG sizing */}
                {/* Pass a reasonable fallback network if needed during initial load or error? */} 
                <WordGraph
                    wordNetwork={wordNetwork} // Can be null
                    mainWord={mainWord}
                    onNodeClick={handleNodeClick}
                    onNetworkChange={handleNetworkChange}
                    initialDepth={depth}
                    initialBreadth={breadth}
                  />
                 {!isLoading && !wordNetwork && !error && (
                    <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'text.secondary' }}>
                        <Typography>Enter a word to explore its network.</Typography>
                    </Box>
                )}
             </Box>
        </Paper>

        {/* --- Details Panel (Resizable) --- */}
        <Paper elevation={1} ref={detailsContainerRef} className="details-resizable-container" sx={{
            // Initial width, user can resize
            width: '450px', // Default width
            minWidth: '300px',
            maxWidth: '70%', // Max % of parent
            height: '100%',
            overflow: 'hidden', // Needed for resize handle and internal scroll
            resize: 'horizontal', // Enable CSS horizontal resize
            position: 'relative', // For potential internal absolute elements
            display: 'flex', // To make WordDetails fill height
            flexDirection: 'column'
        }}>
            {/* Loading state specific to details */} 
            {isLoading && !selectedWordInfo && (
                 <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                    <CircularProgress />
                 </Box>
            )}
            {!isLoading && selectedWordInfo && (
              <WordDetails
                wordInfo={selectedWordInfo}
                etymologyTree={etymologyTree}
                isLoadingEtymology={isLoadingEtymology}
                etymologyError={etymologyError}
                onWordLinkClick={handleWordLinkClick}
                onEtymologyNodeClick={handleNodeClick}
                // Removed show/toggle Metadata
              />
            )}
             {!isLoading && !selectedWordInfo && (
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'text.secondary' }}>
                    <Typography>Select a node or search to view details.</Typography>
                </Box>
            )}
        </Paper>
      </Box>

      {/* === Footer (Optional) === */}
      {/* 
      <Box component="footer" sx={{ p: 1, textAlign: 'center', bgcolor: 'background.paper', borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="caption">© {new Date().getFullYear()} Filipino Root Word Explorer</Typography>
      </Box> 
      */}
    </Box>
  );
};

export default WordExplorer;
