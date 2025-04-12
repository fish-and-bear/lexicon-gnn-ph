import React, { useState, useCallback, useEffect, useRef, FormEvent } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchResult, SearchOptions, EtymologyTree, Statistics, Definition, SearchWordResult } from "../types";
import unidecode from "unidecode";
import { 
  fetchWordNetwork, 
  fetchWordDetails, 
  searchWords, 
  getRandomWord,
  testApiConnection,
  resetCircuitBreaker,
  getPartsOfSpeech,
  getStatistics,
  getBaybayinWords,
  getAffixes,
  getRelations,
  getAllWords,
  getEtymologyTree
} from "../api/wordApi";
import axios from 'axios';
import DOMPurify from 'dompurify';
import { debounce } from "lodash";
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import { styled } from '@mui/material/styles';
import NetworkControls from './NetworkControls';
import { Typography, Button } from "@mui/material";

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
  const [wordHistory, setWordHistory] = useState<Array<string | {id: number | string, text: string}>>([]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState<number>(-1);
  const [searchResults, setSearchResults] = useState<SearchWordResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);
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

  // Add wordData state with other state definitions
  const [wordData, setWordData] = useState<WordInfo | null>(null);

  // Near state declarations, add these new states
  const [randomWordCache, setRandomWordCache] = useState<any[]>([]);
  const [isRefreshingCache, setIsRefreshingCache] = useState<boolean>(false);
  const randomButtonTimeoutRef = useRef<number | null>(null);
  const RANDOM_CACHE_SIZE = 5;
  // Use a ref to track last refresh time to prevent excessive refreshes
  const lastRefreshTimeRef = useRef<number>(0);

  const detailsContainerRef = useRef<HTMLDivElement>(null);

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
      setError(null); // Clear any previous errors
      
      console.log('Fetching word network data for:', word);
      const data = await fetchWordNetwork(word, { 
        depth,
        breadth,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.3
      });
      
      if (data && data.nodes && data.edges) {
        console.log('Word network data received:', data);
        console.log(`Network has ${data.nodes.length} nodes and ${data.edges.length} edges`);
      setWordNetwork(data);
      return data;
      } else {
        console.error('Invalid network data received:', data);
        throw new Error('Invalid network data structure received from API');
      }
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
      
      // Set error message
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError('An unexpected error occurred while fetching word network');
      }
      
      // Clear the word network when there's an error
      setWordNetwork(null);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchEtymologyTree = useCallback(async (wordId: number): Promise<EtymologyTree | null> => {
    if (!wordId) return null;
    
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    
    try {
      console.log("Fetching etymology tree for word ID:", wordId);
      const data = await getEtymologyTree(wordId, 3);
      console.log("Etymology tree data received:", data);
      setEtymologyTree(data);
      return data;
    } catch (error) {
      console.error("Error fetching etymology tree:", error);
      let errorMessage = "Failed to fetch etymology tree.";
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      setEtymologyError(errorMessage);
      setEtymologyTree(null);
      return null;
    } finally {
      setIsLoadingEtymology(false);
    }
  }, [getEtymologyTree]); // Add getEtymologyTree as dependency

  // Move loadWordData before handleSearch and handleSuggestionClick
  const loadWordData = useCallback(async (wordId: number) => {
    setIsLoading(true);
    setError(null);
    try {
      const wordData = await fetchWordDetails(wordId.toString());
      setWordData(wordData);
    } catch (err) {
      console.error('Error loading word data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load word data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Create a ref to hold the actual handleNodeClick function
  const handleNodeClickRef = useRef<(id: string) => Promise<void>>(null as unknown as (id: string) => Promise<void>);
  
  // Define handleNodeClick before handleSearch
  const handleNodeClick = useCallback(async (word: string) => {
    if (!word) {
      console.error("Empty word received in handleNodeClick");
      return;
    }

    console.log(`Node clicked: ${word}`);
    setError(null);
    setIsLoading(true);

    try {
      let wordData: WordInfo;

      try {
        // First try getting the word details directly
        wordData = await fetchWordDetails(word);
      } catch (error: any) {
        // If the word wasn't found, try to search for it as a fallback
        console.warn(`Failed to fetch details for word '${word}', error:`, error.message);
        
        if (error.message.includes('not found') || 
            error.message.includes('Database error')) {
          console.log(`Falling back to search for word: ${word}`);
          
          // Try to extract actual word text if this was an ID format
          const searchText = word.startsWith('id:') ? 
            `id:${word.substring(3)}` : // Keep the ID format
            word; // Use the word directly
          
          const searchResults = await searchWords(searchText, {
            page: 1,
            per_page: 5,
            mode: 'all', // Using 'all' which is a valid search mode
            sort: 'relevance',
            language: 'tl'
          });
          
          if (searchResults.words && searchResults.words.length > 0) {
            console.log(`Search successful, found ${searchResults.words.length} results`);
            // Use the first search result
            const firstResult = searchResults.words[0];
            wordData = await fetchWordDetails(
              firstResult.id.toString().startsWith('id:') ? 
              firstResult.id.toString() : 
              `id:${firstResult.id}`
            );
          } else {
            throw new Error(`Word '${word}' not found by search. Please try a different word.`);
          }
        } else {
          // Rethrow other errors
          throw error;
        }
      }

      // Successfully got the word data
      console.log(`Word data retrieved successfully:`, wordData);
      setSelectedWordInfo(wordData);
      setIsLoading(false);
      
      // Update navigation/history
      const wordId = wordData.id.toString();
      if (!wordHistory.some(w => typeof w === 'object' && 'id' in w && w.id.toString() === wordId)) {
        const newHistory = [{ id: wordData.id, text: wordData.lemma }, ...wordHistory].slice(0, 20);
        setWordHistory(newHistory as any);
      }

      // Update network data if necessary
      const currentNetworkWordId = depth && breadth ? wordData.id : null;
      if (wordData.id !== currentNetworkWordId) {
        setDepth(2); // DEFAULT_NETWORK_DEPTH
        setBreadth(10); // DEFAULT_NETWORK_BREADTH
      }

      // Fetch the etymology tree for the word in the background
      try {
        const etymologyId = wordData.id.toString().startsWith('id:') ? 
          parseInt(wordData.id.toString().substring(3), 10) : 
          wordData.id;
        fetchEtymologyTree(etymologyId)
          .then(tree => {
            setEtymologyTree(tree);
          })
          .catch(err => {
            console.error("Error fetching etymology tree:", err);
          });
      } catch (etymologyError) {
        console.error("Error initiating etymology tree fetch:", etymologyError);
      }
    } catch (error: any) {
      console.error(`Error in handleNodeClick for word '${word}':`, error);
      setIsLoading(false);
      
      // Provide user-friendly error messages
      if (error.message.includes('not found')) {
        setError(`Word '${word}' was found in the graph but its details could not be retrieved. This may be due to a database inconsistency.`);
      } else if (error.message.includes('Circuit breaker')) {
        setError(`Network connection to the backend server is unstable. Please try again in a moment.`);
      } else if (error.message.includes('Network Error')) {
        setError(`Cannot connect to the backend server. Please check your connection or try again later.`);
      } else if (error.message.includes('Database error')) {
        setError(`Database error when retrieving the word '${word}'. Try searching for this word instead.`);
      } else {
        setError(`Error retrieving word details: ${error.message}`);
      }
    }
  }, [wordHistory, depth, breadth, setDepth, setBreadth]);

  // Update the ref whenever handleNodeClick changes
  useEffect(() => {
    handleNodeClickRef.current = handleNodeClick;
  }, [handleNodeClick]);

  // NOW DEFINE HANDLERS THAT USE FETCHERS
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      setShowSuggestions(false);
      return;
    }

    console.log("Starting search for:", query);
    setIsLoadingSuggestions(true);
    setSearchError(null);
    setError(null); // Clear any previous errors

    try {
      const searchOptions: SearchOptions = {
        page: 1,
        per_page: 10,
        mode: 'all',
        sort: 'relevance',
        order: 'desc',
        language: 'tl',
        exclude_baybayin: false
      };
      console.log("Calling searchWords API with query:", query, "and options:", searchOptions);
      const result = await searchWords(query, searchOptions);
      console.log("Search results received:", result);
      setSearchResults(result.words);
      setShowSuggestions(false); // Hide suggestions immediately after search
      
      // Automatically load the first search result if available
      if (result && result.words && result.words.length > 0) {
        console.log("Found search results, selecting first result:", result.words[0]);
        
        setIsLoading(true); // Start loading
        
        try {
          // 1. First get the word details directly
          const firstResult = result.words[0];
          console.log(`Fetching details for word ID: ${firstResult.id}`);
          // Explicitly use the ID number format, not the id: prefix format
          const wordData = await fetchWordDetails(firstResult.id.toString());
          console.log("Word details received:", wordData);
          
          // 2. Update the word data immediately
          setSelectedWordInfo(wordData);
          setMainWord(wordData.lemma);
          setInputValue(wordData.lemma);
          
          // 3. Update history - Remove everything after current index, then add the new word
          const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), wordData.lemma];
          setWordHistory(newHistory);
          setCurrentHistoryIndex(newHistory.length - 1);
          
          // 4. Then load the network in parallel
          console.log(`Fetching network for: ${wordData.lemma}`);
          const networkData = await fetchWordNetworkData(wordData.lemma, depth, breadth);
          console.log("Network data received:", networkData);
          setWordNetwork(networkData);
          
          // 5. Also try to fetch etymology tree if available
          try {
            console.log(`Fetching etymology tree for ID: ${wordData.id}`);
            await fetchEtymologyTree(wordData.id);
            console.log("Etymology tree received");
          } catch (etymErr) {
            console.error("Error fetching etymology tree:", etymErr);
            // Don't throw, we can continue without etymology
          }
          
          // Scroll details container to top
          detailsContainerRef.current?.scrollTo(0, 0);
          console.log("Search loading process completed successfully");
        } catch (dataError) {
          console.error("Error loading word data during search:", dataError);
          let errorMessage = "Error loading word details";
          if (dataError instanceof Error) {
            // Check for common error patterns and provide more helpful messages
            const msg = dataError.message;
            if (msg.includes("Database error")) {
              errorMessage = "Database error occurred. The word exists but there was a problem retrieving its details.";
            } else if (msg.includes("not found")) {
              errorMessage = `Word "${query}" was found in search but its details could not be retrieved.`;
            } else if (msg.includes("Server error")) {
              errorMessage = "Server error occurred. Please try again later.";
            } else {
              errorMessage = dataError.message;
            }
          }
          setError(errorMessage);
        } finally {
          setIsLoading(false);
        }
      } else {
        console.log("No search results found for query:", query);
        setError(`No results found for "${query}"`);
      }
      
      // Return the results for use in the effect
      return result;
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
      let errorMessage = "An error occurred during search";
      if (error instanceof Error) {
        // Check for common error patterns and provide more helpful messages
        const msg = error.message;
        if (msg.includes("Network Error") || msg.includes("Failed to fetch")) {
          errorMessage = "Cannot connect to the backend server. Please ensure the backend server is running.";
        } else if (msg.includes("Circuit breaker")) {
          errorMessage = "Too many failed requests. Please wait a moment and try again.";
        } else {
          errorMessage = error.message;
        }
      }
      setSearchError(errorMessage);
      setError(errorMessage);
      return null;
    } finally {
      setIsLoadingSuggestions(false);
    }
  }, [fetchWordDetails, fetchWordNetworkData, fetchEtymologyTree, wordHistory, currentHistoryIndex, depth, breadth]);

  const handleSuggestionClick = useCallback(async (suggestion: SearchWordResult) => {
    setInputValue(suggestion.lemma);
    setShowSuggestions(false);
    setError(null);
    setIsLoading(true);
    
    try {
      // First load the word details
      await loadWordData(suggestion.id);
      
      // Then try to fetch word network
      try {
        console.log(`Fetching network for ${suggestion.lemma} with ID ${suggestion.id}`);
        await fetchWordNetworkData(suggestion.lemma);
      } catch (networkError) {
        console.error("Error fetching word network:", networkError);
        // Don't rethrow - we want to keep the word data even if network fails
      }
      
      // Finally try to fetch etymology tree
      try {
        await fetchEtymologyTree(suggestion.id);
      } catch (etymologyError) {
        console.error("Error fetching etymology tree:", etymologyError);
        // Don't rethrow - we want to keep the word data even if etymology fails
      }
    } catch (error) {
      console.error("Error in handleSuggestionClick:", error);
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError('An unexpected error occurred while loading word data');
      }
    } finally {
      setIsLoading(false);
    }
  }, [loadWordData, fetchWordNetworkData, fetchEtymologyTree]);

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

  // Function to fetch a batch of random words for the cache without displaying them
  const fetchRandomWordsForCache = useCallback(async (count: number = RANDOM_CACHE_SIZE): Promise<any[]> => {
    if (isRefreshingCache) return []; // Prevent multiple simultaneous refreshes, return empty array
    
    const currentTime = Date.now();
    const MIN_REFRESH_INTERVAL = 10000; // 10 seconds minimum between refreshes
    
    if (currentTime - lastRefreshTimeRef.current < MIN_REFRESH_INTERVAL) {
      console.log('Random word cache refresh throttled (too many requests)');
      return [];
    }
    
    lastRefreshTimeRef.current = currentTime;
    setIsRefreshingCache(true);
    
    try {
      console.log('Refreshing random word cache...');
      const newCache: any[] = [];
      
      // Fetch multiple random words in parallel
      const promises = Array.from({ length: count }, () => 
        getRandomWord().catch(err => {
          console.error('Error fetching random word for cache:', err);
          return null;
        })
      );
      
      const results = await Promise.all(promises);
      
      // Filter out any failed requests and add successful ones to cache
      const validResults = results.filter(result => 
        result !== null && 
        result.lemma && 
        typeof result.lemma === 'string'
      );
      
      if (validResults.length === 0) {
        console.warn('No valid random words found for cache. Will retry later.');
        // Schedule another attempt after a longer delay (30 seconds)
        setTimeout(() => fetchRandomWordsForCache(count), 30000);
        return [];
      } else {
        newCache.push(...validResults);
        // Update the cache
        setRandomWordCache(prev => [...newCache, ...prev].slice(0, RANDOM_CACHE_SIZE * 2));
        console.log(`Added ${validResults.length} words to random word cache (${count - validResults.length} failed)`);
        return newCache;
      }
    } catch (error) {
      console.error('Error refreshing random word cache:', error);
      return [];
    } finally {
      setIsRefreshingCache(false);
    }
  }, [isRefreshingCache]);

  // Add effect to initialize random word cache on mount
  useEffect(() => {
    if (apiConnected) {
      fetchRandomWordsForCache(RANDOM_CACHE_SIZE);
    }
  }, [apiConnected, fetchRandomWordsForCache]);

  // Replace the handleRandomWord function with this improved version
  const handleRandomWord = useCallback(async () => {
    // Disable rapid clicking
    if (isLoading) return;
    
    setIsLoading(true);
    setError('');
    try {
      let randomWord;

      // Check if we have cached words and select one
      if (randomWordCache.length > 0) {
        const randomIndex = Math.floor(Math.random() * randomWordCache.length);
        randomWord = randomWordCache[randomIndex];
        
        // Remove the selected word from cache
        const newCache = [...randomWordCache];
        newCache.splice(randomIndex, 1);
        setRandomWordCache(newCache);
        
        // If cache is getting low, refresh it in background
        if (newCache.length < 5) {
          fetchRandomWordsForCache().then(cachedWords => {
            if (cachedWords.length > 0) {
              setRandomWordCache(prev => [...prev, ...cachedWords]);
            }
          }).catch(err => {
            console.error('Failed to refresh random word cache:', err);
          });
        }
      } else {
        // No cache, fetch directly
        const words = await fetchRandomWordsForCache();
        if (words.length > 0) {
          randomWord = words[0];
          // Store the rest in cache
          setRandomWordCache(words.slice(1));
        } else {
          throw new Error('No random words returned');
        }
      }

      if (randomWord) {
        // Update the currently selected word
        setSelectedWordInfo({
          id: randomWord.id,
          lemma: randomWord.lemma,
          normalized_lemma: randomWord.normalized_lemma || randomWord.lemma,
          language_code: randomWord.language_code || 'tl',
          has_baybayin: randomWord.has_baybayin || false,
          baybayin_form: randomWord.baybayin_form || null,
          romanized_form: randomWord.romanized_form || null,
          definitions: randomWord.definitions || [],
          etymologies: randomWord.etymologies || [],
          pronunciations: randomWord.pronunciations || [],
          credits: randomWord.credits || [],
          outgoing_relations: randomWord.outgoing_relations || [],
          incoming_relations: randomWord.incoming_relations || [],
          root_affixations: randomWord.root_affixations || [],
          affixed_affixations: randomWord.affixed_affixations || [],
          tags: randomWord.tags || null,
          data_completeness: randomWord.data_completeness || null,
          relation_summary: randomWord.relation_summary || null,
          root_word: randomWord.root_word || null,
          derived_words: randomWord.derived_words || [],
        });
        
        // Also set the main word
        setMainWord(randomWord.lemma);
        
        // Update input value to match
        setInputValue(randomWord.lemma);
        
        // Update history
        const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), randomWord.lemma];
        setWordHistory(newHistory);
        setCurrentHistoryIndex(newHistory.length - 1);
        
        // Now fetch the network data for this random word in the background
        fetchWordNetworkData(randomWord.lemma, depth, breadth)
          .catch(networkErr => {
            console.error('Error fetching network for random word:', networkErr);
            // No need to rethrow, we still have the word data
          });
        
        // Also try to fetch the etymology tree in the background
        if (randomWord.id) {
          fetchEtymologyTree(randomWord.id)
            .catch(etymErr => {
              console.error('Error fetching etymology for random word:', etymErr);
              // No need to rethrow, we still have the word data
            });
        }
      }
    } catch (err) {
      console.error('Random word error:', err);
      setError(err instanceof Error ? err.message : 'Error fetching random word');
    } finally {
      setIsLoading(false);
    }
  }, [
    fetchWordNetworkData, 
    fetchEtymologyTree, 
    depth, 
    breadth, 
    randomWordCache, 
    fetchRandomWordsForCache,
    wordHistory,
    currentHistoryIndex
  ]);

  // Add cleanup for timeout ref
  useEffect(() => {
    return () => {
      if (randomButtonTimeoutRef.current !== null) {
        window.clearTimeout(randomButtonTimeoutRef.current);
      }
    };
  }, []);

  // Function to manually test API connection
  const handleTestApiConnection = useCallback(async () => {
        setError(null);
    setApiConnected(null); // Set to checking state
        
    try {
      console.log("Manually testing API connection...");
      
      // First try the testApiConnection function
        const isConnected = await testApiConnection();
        setApiConnected(isConnected);
        
        if (!isConnected) {
        console.log("API connection test failed, showing error...");
        setError(
          "Cannot connect to the API server. Please ensure the backend server is running on port 10000 " +
          "and the /api/v2/test endpoint is accessible. You can start the backend server by running:\n" +
          "1. cd backend\n" +
          "2. python app.py"
        );
      } else {
        console.log("API connection successful!");
        
        // Update API endpoint display
        const savedEndpoint = localStorage.getItem('successful_api_endpoint');
        setApiEndpoint(savedEndpoint);
        
        // If connection is successful, try to use the API
        if (inputValue) {
          console.log("Trying to search with the connected API...");
          handleSearch(inputValue);
                  } else {
          console.log("Fetching a random word to test connection further...");
          handleRandomWord();
        }
      }
    } catch (e) {
      console.error("Error testing API connection:", e);
      setApiConnected(false);
      setError(
        "Error testing API connection. Please ensure the backend server is running and the " +
        "/api/v2/test endpoint is accessible. Check the console for more details."
      );
    }
  }, [inputValue, handleSearch, handleRandomWord]);

  // Test API connectivity on mount
  useEffect(() => {
    console.log("Checking API connection... 2");
    testApiConnection().then(connected => {
      setApiConnected(connected);
      if (connected) {
        // Call the necessary fetch functions after connection is established
        const fetchInitialData = async () => {
          try {
            // Get parts of speech data
            const posData = await getPartsOfSpeech();
            setPartsOfSpeech(posData);
            
            // Get statistics
            fetchStatistics();
            
            // Only prefetch the random word cache if the cache is empty
            // This prevents unnecessary API calls on each page load
            if (randomWordCache.length === 0) {
              console.log("Initial random word cache is empty, prefetching words");
              // Only fetch 2 words initially to reduce load on the backend
              fetchRandomWordsForCache(2);
            }
          } catch (error) {
            console.error("Error fetching initial data:", error);
          }
        };
        
        fetchInitialData();
      }
    }).catch(error => {
      console.error("API connection error:", error);
      setApiConnected(false);
    });
  }, [randomWordCache.length, fetchRandomWordsForCache]); // Added dependencies

  // Implement fetchPartsOfSpeech to use the getPartsOfSpeech function
  const fetchPartsOfSpeech = useCallback(async () => {
    try {
      const data = await getPartsOfSpeech();
      setPartsOfSpeech(data);
    } catch (error) {
      console.error("Error fetching parts of speech:", error);
    }
  }, [getPartsOfSpeech]);

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

  // Debounced search for suggestions
  const handleDebouncedSearch = useCallback(async (query?: string) => {
    if (!query || query.trim().length < 2) {
      setSearchResults([]);
      setShowSuggestions(false);
      return;
    }
    
    setIsLoadingSuggestions(true);
    try {
      const searchOptions: SearchOptions = { 
        page: 1,
        per_page: 10,
        exclude_baybayin: false,
        language: 'tl',
        mode: 'all',
        sort: 'relevance',
        order: 'desc'
      };
      
      const results = await searchWords(query, searchOptions);
      
      if (results && results.words) {
        setSearchResults(results.words);
        setShowSuggestions(results.words.length > 0);
      } else {
        setSearchResults([]);
        setShowSuggestions(false);
      }
    } catch (error) {
      console.error("Error fetching search suggestions:", error);
      setSearchResults([]);
      setShowSuggestions(false);
      if (error instanceof Error) {
        if (error.message.includes('Network Error')) {
          setError('Cannot connect to the backend server. Please ensure the backend server is running on port 10000.');
        } else {
          setError(error.message);
        }
      } else {
        setError('An unexpected error occurred while searching');
      }
    } finally {
      setIsLoadingSuggestions(false);
    }
  }, []);

  // Network change handler
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

  // Render the search bar with navigation buttons
  const renderSearchBar = () => {
    return (
      <div className="search-container">
        <div className="nav-buttons">
          <button 
            className="nav-button back-button" 
            onClick={handleBack}
            disabled={currentHistoryIndex <= 0}
            title="Go back to previous word"
          >
            <span>←</span>
          </button>
          <button 
            className="nav-button forward-button" 
            onClick={handleForward}
            disabled={currentHistoryIndex >= wordHistory.length - 1}
            title="Go forward to next word"
          >
            <span>→</span>
          </button>
        </div>
        
        <div className="search-input-container">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              handleDebouncedSearch(e.target.value);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleSearch(inputValue);
              }
            }}
            placeholder="Search for a word..."
            className="search-input"
          />
          
          {isLoadingSuggestions && (
            <div className="search-loader">
              <CircularProgress size={20} />
            </div>
          )}
          
          {showSuggestions && searchResults.length > 0 && (
            <ul className="search-suggestions">
              {searchResults.map((result) => (
                <li 
                  key={result.id} 
                  onClick={() => {
                    // Use handleNodeClick with ID for more reliable handling
                    handleNodeClick(result.id.toString());
                  }}
                >
                  <strong>{result.lemma}</strong>
                  {result.definitions && result.definitions.length > 0 && (
                    <span className="suggestion-definition">
                      {typeof result.definitions[0] === 'string' 
                        ? result.definitions[0] 
                        : (result.definitions[0] as any)?.text || ''}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
        
        <button 
          className="search-button" 
          onClick={() => handleSearch(inputValue)}
          title="Search for a word"
          disabled={isLoading}
        >
          🔍 Search
        </button>
      </div>
    );
  };

  return (
    <div className={`word-explorer ${theme} ${isLoading ? 'loading' : ''}`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <div className="header-buttons">
          <button
            onClick={() => handleSearch(inputValue)}
            className="search-button"
            title="Search for a word"
            disabled={isLoading}
          >
            🔍 Search
          </button>
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
      {renderSearchBar()}
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
                  <li>Run: <code>python app.py</code></li>
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
                initialBreadth={breadth}
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
              onWordLinkClick={handleNodeClick}
              onEtymologyNodeClick={handleNodeClick}
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
