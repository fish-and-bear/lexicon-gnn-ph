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
  const [wordHistory, setWordHistory] = useState<string[]>([]);
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

  // NOW DEFINE HANDLERS THAT USE FETCHERS
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      setShowSuggestions(false);
      return;
    }

    setIsLoadingSuggestions(true);
    setSearchError(null);

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
      const result = await searchWords(query, searchOptions);
      setSearchResults(result.words);
      setShowSuggestions(true);
    } catch (error) {
      console.error('Search error:', error);
      setSearchError(error instanceof Error ? error.message : 'An error occurred during search');
      setSearchResults([]);
    } finally {
      setIsLoadingSuggestions(false);
    }
  }, []);

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

  const handleNodeClick = useCallback(async (word: string) => {
    console.log("Word link clicked:", word);
    if (word !== mainWord) {
      setIsLoading(true);
      setError(null);
      
      try {
        // First check if the word parameter is numeric (could be a node ID)
        const wordId = parseInt(word, 10);
        let wordData;
        
        if (!isNaN(wordId)) {
          // If it's a numeric ID, try to fetch directly
          console.log(`Treating ${word} as a word ID, fetching directly`);
          try {
            // Fetch by ID
            wordData = await fetchWordDetails(wordId.toString());
          } catch (idError) {
            console.error("Error fetching by ID, will try search:", idError);
            wordData = null; // Will trigger the search fallback
          }
        }
        
        // If we don't have word data yet (not an ID or ID fetch failed), search for it
        if (!wordData) {
          // Search for the word to get its ID
          const searchOptions: SearchOptions = { 
            page: 1,
            per_page: 5,
            exclude_baybayin: false,
            language: 'tl',
            mode: 'all',
            sort: 'relevance',
            order: 'desc'
          };
          
          const searchResult = await searchWords(word, searchOptions);
          if (!searchResult || !searchResult.words || searchResult.words.length === 0) {
            throw new Error(`Word "${word}" not found`);
          }
          
          const wordResult = searchResult.words[0];
          
          // Fetch word details by ID
          wordData = await fetchWordDetails(wordResult.id.toString());
        }
        
        // We now have word data one way or another
        const wordLemma = wordData.lemma;
        
        // Then load the network
        await fetchWordNetworkData(wordLemma, depth, breadth);
        
        // Try to load etymology tree if available
        try {
          await fetchEtymologyTree(wordData.id);
        } catch (etymErr) {
          console.error("Error fetching etymology tree:", etymErr);
          // Don't throw, we can continue without etymology
        }
        
        // Now update the UI state
        setSelectedWordInfo(wordData);
        setMainWord(wordLemma);
        setInputValue(wordLemma);
        
        // Update history - Remove everything after current index, then add the new word
        const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), wordLemma];
        setWordHistory(newHistory);
        setCurrentHistoryIndex(newHistory.length - 1);
        
        detailsContainerRef.current?.scrollTo(0, 0);
      } catch (err) {
        console.error('Error loading word:', err);
        setError(err instanceof Error ? err.message : 'Failed to load word data');
      } finally {
        setIsLoading(false);
      }
    }
  }, [mainWord, fetchWordDetails, fetchWordNetworkData, fetchEtymologyTree, wordHistory, currentHistoryIndex, depth, breadth]);

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
  const fetchRandomWordsForCache = useCallback(async (count: number = RANDOM_CACHE_SIZE) => {
    if (isRefreshingCache) return; // Prevent multiple simultaneous refreshes
    
    // Use a ref to track last refresh time to prevent excessive refreshes
    const lastRefreshTimeRef = useRef<number>(0);
    const currentTime = Date.now();
    const MIN_REFRESH_INTERVAL = 10000; // 10 seconds minimum between refreshes
    
    if (currentTime - lastRefreshTimeRef.current < MIN_REFRESH_INTERVAL) {
      console.log('Random word cache refresh throttled (too many requests)');
      return;
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
      } else {
        newCache.push(...validResults);
        // Update the cache
        setRandomWordCache(prev => [...newCache, ...prev].slice(0, RANDOM_CACHE_SIZE * 2));
        console.log(`Added ${validResults.length} words to random word cache (${count - validResults.length} failed)`);
      }
    } catch (error) {
      console.error('Error refreshing random word cache:', error);
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
    // Prevent rapid clicking
    if (randomButtonTimeoutRef.current !== null) {
      return;
    }
    
    // Set a timeout to prevent multiple rapid clicks
    randomButtonTimeoutRef.current = window.setTimeout(() => {
      randomButtonTimeoutRef.current = null;
    }, 1000);
    
    // Show loading state immediately
    setIsLoading(true);
    setError(null);
    setEtymologyTree(null);
    
    try {
      // Check if we have cached words
      if (randomWordCache.length > 0) {
        // Take the first word from the cache
        const randomWordData = randomWordCache[0];
        
        // Remove it from the cache
        setRandomWordCache(prev => prev.slice(1));
        
        // If cache is getting low, trigger a refresh
        if (randomWordCache.length < RANDOM_CACHE_SIZE / 2) {
          setTimeout(() => fetchRandomWordsForCache(), 100);
        }
        
        // Validate the cached word data
        if (!randomWordData || !randomWordData.lemma) {
          throw new Error('Invalid cached random word data');
        }
        
        // Use the cached word data
        setSelectedWordInfo({
          id: randomWordData.id,
          lemma: randomWordData.lemma,
          normalized_lemma: randomWordData.normalized_lemma || randomWordData.lemma,
          language_code: randomWordData.language_code || 'tl',
          has_baybayin: randomWordData.has_baybayin || false,
          baybayin_form: randomWordData.baybayin_form || null,
          romanized_form: randomWordData.romanized_form || null,
          definitions: randomWordData.definitions || [],
          etymologies: randomWordData.etymologies || [],
          pronunciations: randomWordData.pronunciations || [],
          credits: randomWordData.credits || [],
          outgoing_relations: randomWordData.outgoing_relations || [],
          incoming_relations: randomWordData.incoming_relations || [],
          root_affixations: randomWordData.root_affixations || [],
          affixed_affixations: randomWordData.affixed_affixations || [],
          tags: randomWordData.tags || null,
          data_completeness: randomWordData.data_completeness || null,
          relation_summary: randomWordData.relation_summary || null,
          root_word: randomWordData.root_word || null,
          derived_words: randomWordData.derived_words || [],
        });
        
        // Also set the main word
        setMainWord(randomWordData.lemma);
        
        // Update input value to match
        setInputValue(randomWordData.lemma);
        
        // Update history
        const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), randomWordData.lemma];
        setWordHistory(newHistory);
        setCurrentHistoryIndex(newHistory.length - 1);
        
        // Now fetch the network data for this random word in the background
        fetchWordNetworkData(randomWordData.lemma, depth, breadth)
          .catch(networkErr => {
            console.error('Error fetching network for random word:', networkErr);
            // No need to rethrow, we still have the word data
          });
        
        // Also try to fetch the etymology tree in the background
        if (randomWordData.id) {
          fetchEtymologyTree(randomWordData.id)
            .catch(etymErr => {
              console.error('Error fetching etymology for random word:', etymErr);
              // No need to rethrow, we still have the word data
            });
        }
      } else {
        // If no cache available, fall back to direct API call
        console.log('No cached random words available, fetching directly...');
        
        // Get random word (already returns comprehensive data)
        const randomWordData = await getRandomWord();
        console.log('Random word data:', randomWordData);
        
        // Validate data
        if (!randomWordData || !randomWordData.lemma) {
          throw new Error('Invalid random word data received from API');
        }
        
        // Use the data directly from getRandomWord
        setSelectedWordInfo({
          id: randomWordData.id,
          lemma: randomWordData.lemma,
          normalized_lemma: randomWordData.normalized_lemma || randomWordData.lemma,
          language_code: randomWordData.language_code || 'tl',
          has_baybayin: randomWordData.has_baybayin || false,
          baybayin_form: randomWordData.baybayin_form || null,
          romanized_form: randomWordData.romanized_form || null,
          definitions: randomWordData.definitions || [],
          etymologies: randomWordData.etymologies || [],
          pronunciations: randomWordData.pronunciations || [],
          credits: randomWordData.credits || [],
          outgoing_relations: randomWordData.outgoing_relations || [],
          incoming_relations: randomWordData.incoming_relations || [],
          root_affixations: randomWordData.root_affixations || [],
          affixed_affixations: randomWordData.affixed_affixations || [],
          tags: randomWordData.tags || null,
          data_completeness: randomWordData.data_completeness || null,
          relation_summary: randomWordData.relation_summary || null,
          root_word: randomWordData.root_word || null,
          derived_words: randomWordData.derived_words || [],
        });
        
        // Also set the main word
        setMainWord(randomWordData.lemma);
        
        // Update input value to match
        setInputValue(randomWordData.lemma);
        
        // Update history
        const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), randomWordData.lemma];
        setWordHistory(newHistory);
        setCurrentHistoryIndex(newHistory.length - 1);
        
        // Now try to fetch the network data for this random word in the background
        fetchWordNetworkData(randomWordData.lemma, depth, breadth)
          .catch(networkErr => {
            console.error('Error fetching network for random word:', networkErr);
            // No need to rethrow, we still have the word data
          });
        
        // Also try to fetch the etymology tree in the background
        if (randomWordData.id) {
          fetchEtymologyTree(randomWordData.id)
            .catch(etymErr => {
              console.error('Error fetching etymology for random word:', etymErr);
              // No need to rethrow, we still have the word data
            });
        }
        
        // Refresh the cache in the background
        setTimeout(() => fetchRandomWordsForCache(), 100);
      }
    } catch (error) {
      console.error('Error in handleRandomWord:', error);
      if (error instanceof Error) {
        // Provide user-friendly message for common errors
        if (error.message.includes('baybayin')) {
          setError('Error with Baybayin validation. Please try again.');
        } else if (error.message.includes('Invalid random word data')) {
          setError('The random word data was incomplete. Please try again.');
        } else {
          setError(error.message);
        }
      } else {
        setError('An unexpected error occurred while fetching a random word');
      }
      setSelectedWordInfo(null);
      setWordNetwork(null);
      
      // Try to refresh the cache when there's an error
      setTimeout(() => fetchRandomWordsForCache(), 5000);
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
                    // Use handleNodeClick for consistent history handling
                    handleNodeClick(result.lemma);
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
          className="random-word-button" 
          onClick={handleRandomWord}
          title="Get a random word"
        >
          Random
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
