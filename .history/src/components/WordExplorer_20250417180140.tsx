import React, { useState, useCallback, useEffect, useRef, FormEvent } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchResult, SearchOptions, EtymologyTree, Statistics, Definition, SearchWordResult, Relation } from "../types";
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
  getEtymologyTree,
  fetchWordRelations
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
import { hasMissingOrMinimalRelations } from '../utils/wordUtils';

// Import Resizable Panels components
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import useMediaQuery from '@mui/material/useMediaQuery'; // Reuse useMediaQuery
import { useQuery, useQueryClient } from 'react-query';

// Add a shuffle function utility near the top, outside the component
function shuffleArray(array: any[]) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

// Custom TabPanel component for displaying tab content
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
      style={{
        display: value === index ? 'block' : 'none',
        height: 'calc(100vh - 160px)',
        overflow: 'hidden'
      }}
    >
      {value === index && children}
    </div>
  );
};

const WordExplorer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [selectedWordIdentifier, setSelectedWordIdentifier] = useState<string | null>(null); // Use this to trigger queries
  const { theme, toggleTheme } = useTheme();
  const [inputValue, setInputValue] = useState<string>("");
  const [depth, setDepth] = useState<number>(2);
  const [breadth, setBreadth] = useState<number>(10);
  const [wordHistory, setWordHistory] = useState<string[]>([]); // Store identifiers (id:X or lemma)
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState<number>(-1);
  const [searchResults, setSearchResults] = useState<SearchWordResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [apiEndpoint, setApiEndpoint] = useState<string | null>(localStorage.getItem('successful_api_endpoint'));
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

  // Near state declarations, add these new states
  const [randomWordCache, setRandomWordCache] = useState<any[]>([]);
  const [isRefreshingCache, setIsRefreshingCache] = useState<boolean>(false);
  const randomWordTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const RANDOM_CACHE_SIZE = 20; // Increased from 5 to 20 for better spam clicking support
  const lastRefreshTimeRef = useRef<number>(0);
  const retryCountRef = useRef<number>(0);
  const randomWordCacheRef = useRef<any[]>([]); // Ref for synchronous cache access

  const [isRandomLoading, setIsRandomLoading] = useState<boolean>(false);
  const [selectedTab, setSelectedTab] = useState(0);
  
  // Reference for the details container element
  const detailsContainerRef = useRef<HTMLDivElement>(null);

  const queryClient = useQueryClient();

  // Initialize details width from localStorage (Keep this for initial load before observer runs)
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

  // --- React Query Hooks ---

  // Fetch Word Details using useQuery
  const { data: wordInfo, isLoading: isLoadingDetails, error: detailsError, refetch: refetchDetails } = useQuery<WordInfo, Error>(
    ['wordDetails', selectedWordIdentifier], // Query key includes the identifier
    () => fetchWordDetails(selectedWordIdentifier!), // Fetch function
    {
      enabled: !!selectedWordIdentifier, // Only run query if identifier is set
      staleTime: 5 * 60 * 1000, // Consider data fresh for 5 minutes
      cacheTime: 30 * 60 * 1000, // Keep data in cache for 30 minutes
      retry: 1, // Retry once on failure
      onError: (err) => {
        console.error('Error fetching word details via useQuery:', err);
        // Optionally trigger circuit breaker reset or specific UI update
      },
      onSuccess: (data) => {
          console.log('Successfully fetched word details via useQuery:', data?.lemma);
          // Pre-fetch network data if relations are minimal
          if (data && hasMissingOrMinimalRelations(data)) {
              console.log('Pre-fetching network due to minimal relations for:', data.lemma);
              queryClient.prefetchQuery(
                  ['wordNetwork', data.lemma, depth, breadth], // Use lemma for network key
                  () => fetchWordNetwork(data.lemma!, { depth, breadth })
              );
          }
      }
    }
  );

  // Fetch Word Network using useQuery - Key depends on wordInfo.lemma
  const mainWordLemma = wordInfo?.lemma;
  const { data: wordNetwork, isLoading: isLoadingNetwork, error: networkError, refetch: refetchNetwork } = useQuery<WordNetwork, Error>(
    ['wordNetwork', mainWordLemma, depth, breadth], // Query key
    () => fetchWordNetwork(mainWordLemma!, { depth, breadth }), // Fetch function
    {
      enabled: !!mainWordLemma, // Only run if we have a lemma from wordInfo
      staleTime: 10 * 60 * 1000, // Network data potentially less volatile
      cacheTime: 60 * 60 * 1000,
      retry: 1,
      keepPreviousData: true, // Keep showing old graph while new one loads
      onError: (err) => {
          console.error('Error fetching word network via useQuery:', err);
      },
      onSuccess: (data) => {
         console.log(`Successfully fetched network for ${mainWordLemma}: ${data?.nodes?.length} nodes, ${data?.links?.length} links`);
      }
    }
  );

   // Fetch Etymology Tree using useQuery - depends on wordInfo.id
  const wordIdForEtymology = wordInfo?.id;
  const { data: etymologyTree, isLoading: isLoadingEtymology, error: etymologyError, refetch: refetchEtymology } = useQuery<EtymologyTree, Error>(
      ['etymologyTree', wordIdForEtymology], // Query key
      () => getEtymologyTree(wordIdForEtymology!), // Fetch function
      {
          enabled: !!wordIdForEtymology, // Only run if we have a word ID
          staleTime: Infinity, // Etymology tree unlikely to change often
          cacheTime: 24 * 60 * 60 * 1000, // Cache for 24 hours
          retry: 1,
          onError: (err) => {
            console.error('Error fetching etymology tree via useQuery:', err);
          },
          onSuccess: (data) => {
             console.log(`Successfully fetched etymology tree for ID ${wordIdForEtymology}`);
          }
      }
  );

  // Combined loading state
  const isLoading = isLoadingDetails || (!!mainWordLemma && isLoadingNetwork) || (!!wordIdForEtymology && isLoadingEtymology);
  // Combined error state (simplified, could be more granular)
  const queryError = detailsError || networkError || etymologyError;

  // --- End React Query Hooks ---

  // --- Search Suggestions Logic ---
  const fetchSuggestions = useCallback(debounce(async (currentQuery: string) => {
    if (!currentQuery.trim()) {
      setSearchResults([]);
      setShowSuggestions(false);
      return;
    }

    console.log(`[DEBUG] Starting search for: "${currentQuery}"`);
    setIsLoadingSuggestions(true);
    setSearchError(null);
    setError(null); // Clear any previous errors
    setIsLoading(true); // Start loading indicator immediately

    try {
      const searchOptions: SearchOptions = {
        page: 1,
        per_page: 10,
        mode: 'all', // Try all search modes to increase chances of finding results
        sort: 'relevance',
        order: 'desc',
        language: '', // Empty string to search all languages
        exclude_baybayin: false
      };
      
      console.log(`[DEBUG] Calling searchWords API with query: "${currentQuery}", options:`, searchOptions);
      const result = await searchWords(currentQuery, searchOptions);
      console.log(`[DEBUG] Search API result:`, result);
      
      // Check if there's an error message in the search result
      if (result.error) {
        console.error(`[DEBUG] Search returned an error: ${result.error}`);
        setError(result.error);
        // Clear search results to avoid showing stale data
        setSearchResults([]);
        setShowSuggestions(false);
        return;
      }
      
      if (result && result.words && result.words.length > 0) {
        console.log(`[DEBUG] Found ${result.words.length} search results, first result:`, result.words[0]);
        
        setSearchResults(result.words);
        setShowSuggestions(false); // Hide suggestions immediately after search
        
        // Automatically load the first search result
        console.log(`[DEBUG] Loading details for first result: ${result.words[0].lemma} (ID: ${result.words[0].id})`);
        
        try {
          // 1. First get the word details directly
          const firstResult = result.words[0];
          console.log(`[DEBUG] Fetching details for word ID: ${firstResult.id}`);
          
          // Make sure we're using the right ID format - don't add id: prefix if already has one
          const idString = String(firstResult.id);
          const wordId = idString.startsWith('id:') ? idString : `id:${idString}`;
            
          const wordData = await fetchWordDetails(wordId);
          console.log(`[DEBUG] Word details received:`, wordData);
          
          // *** ADD LOGGING HERE ***
          console.log('[DEBUG] Checking relations in fetched wordData:');
          console.log('[DEBUG] Incoming Relations:', wordData?.incoming_relations);
          console.log('[DEBUG] Outgoing Relations:', wordData?.outgoing_relations);
          console.log(`[DEBUG] Has incoming: ${!!wordData?.incoming_relations?.length}, Has outgoing: ${!!wordData?.outgoing_relations?.length}`);
          // *** END LOGGING ***
          
          // 2. Update the word data immediately
          setSelectedWordInfo(wordData);
          setMainWord(wordData.lemma);
          setInputValue(wordData.lemma);
          
          // 3. Update history
          const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), { id: wordData.id, text: wordData.lemma }];
          setWordHistory(newHistory);
          setCurrentHistoryIndex(newHistory.length - 1);
          
          // 4. Fetch network data for visualization
          const networkData = await fetchWordNetwork(wordData.lemma, { 
            depth,
            breadth,
            include_affixes: true,
            include_etymology: true,
            cluster_threshold: 0.3
          });
          console.log('Network data for visualization:', networkData);
          
          // 5. Check if we need to enhance word data with semantic network
          // If word doesn't have relations data or has minimal relations, fetch semantic network as fallback
          const hasMissingOrMinimalRelations = 
            !wordData.incoming_relations?.length || 
            !wordData.outgoing_relations?.length || 
            (wordData.incoming_relations.length + wordData.outgoing_relations.length < 3);
          
          if (hasMissingOrMinimalRelations && !wordData.semantic_network) {
            console.log('[DEBUG] Word has missing or minimal relations, adding semantic network data');
            
            // Update word data with semantic network from the fetched network data
            if (networkData && networkData.nodes && networkData.edges) {
              // Create a copy of wordData with the semantic network
              const enhancedWordData = {
                ...wordData,
                semantic_network: {
                  nodes: networkData.nodes,
                  links: networkData.edges
                }
              };
              
              console.log('[DEBUG] Enhanced word data with semantic network:', {
                nodeCount: networkData.nodes.length,
                edgeCount: networkData.edges.length
              });
              
              // Update the state with enhanced data
              setSelectedWordInfo(enhancedWordData);
            }
          }
          
          // 6. Fetch etymology tree if available
          if (wordData.id) {
            fetchEtymologyTree(wordData.id).catch(err => {
              console.error("Error fetching etymology tree:", err);
            });
          }
        } catch (error) {
          console.error(`[DEBUG] Error fetching word details:`, error);
          
          // Handle the error - set error message and display search results anyway
          setError(error instanceof Error ? error.message : "Failed to fetch word details");
          
          // We still have search results, so keep them displayed
          setIsLoading(false);
        }
      } else {
        // No results found
        console.log(`[DEBUG] No search results found for "${currentQuery}"`);
        setError(`No results found for "${currentQuery}". Please try a different search term.`);
        setSearchResults([]);
      }
    } catch (error) {
      console.error(`[DEBUG] Search error:`, error);
      setError(error instanceof Error ? error.message : "Search failed");
      setSearchResults([]);
    } finally {
      setIsLoading(false);
      setIsLoadingSuggestions(false);
    }
  }, [depth, breadth, wordHistory, currentHistoryIndex, fetchWordNetwork, fetchEtymologyTree]);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setInputValue(value);
    handleDebouncedSearch(value);
  };

  // Update history
  const updateHistory = useCallback((newIdentifier: string) => {
    setWordHistory((prevHistory) => {
      // Normalize identifier before comparison/adding
      const normalizedNewIdentifier = newIdentifier.toLowerCase();

      const newHistory = prevHistory.slice(0, currentHistoryIndex + 1); // Trim future history

      // Avoid adding duplicates consecutively
      const lastEntry = newHistory[newHistory.length - 1];
      if (!lastEntry || lastEntry.toLowerCase() !== normalizedNewIdentifier) {
          newHistory.push(newIdentifier); // Add the original identifier for display/fetching
      }
 
      // Limit history size (e.g., to 50 entries)
      const limitedHistory = newHistory.slice(-50);
      setCurrentHistoryIndex(limitedHistory.length - 1);
      return limitedHistory;
    });
    // Update the selected word identifier to trigger queries
    setSelectedWordIdentifier(newIdentifier);
  }, [currentHistoryIndex]);

  // Handle click on search suggestion or graph node
  const handleWordClick = useCallback((identifierOrWord: string | {id: number | string, text: string}) => {
      let identifier: string; // This will be id:X or the lemma

      if (typeof identifierOrWord === 'object' && identifierOrWord !== null && identifierOrWord.id) {
          // If it's an object with an ID, use the ID format
          identifier = identifierOrWord.id.toString().startsWith('id:') ? identifierOrWord.id.toString() : `id:${identifierOrWord.id}`; 
      } else {
          // Otherwise, assume it's a word string (lemma)
          identifier = identifierOrWord as string;
      }

      setInputValue(typeof identifierOrWord === 'string' ? identifierOrWord : identifierOrWord.text);
      setSearchTerm(identifier); // Update search term for potential future use?
      setShowSuggestions(false);

      // Trigger data fetching by updating the identifier and history
      updateHistory(identifier); 
  }, [updateHistory]);

  const handleNodeSelect = useCallback((word: string) => {
    // This function might be for highlighting or focusing, not necessarily fetching
    console.log(`Node selected: ${word}`);
  }, []);

  const handleEtymologyNodeClick = useCallback((node: any) => {
      if (node?.data?.word) {
          // If the node data contains a word, use it
          const identifier = node.data.word;
          setInputValue(identifier);
          updateHistory(identifier);
      } else if (node?.data?.id) {
          // If only an ID is available, use the id: format
          const identifier = `id:${node.data.id}`;
          setInputValue(node.data.lemma || identifier); // Show lemma if available
          updateHistory(identifier);
      }
  }, [updateHistory]); // updateHistory is stable due to useCallback

  const handleHistoryNavigation = (direction: 'back' | 'forward') => {
    const newIndex = direction === 'back' ? currentHistoryIndex - 1 : currentHistoryIndex + 1;
    if (newIndex >= 0 && newIndex < wordHistory.length) {
      const identifierToFetch = wordHistory[newIndex];
      setInputValue(identifierToFetch.startsWith('id:') ? 'Word ID: ' + identifierToFetch.substring(3) : identifierToFetch);
      setSelectedWordIdentifier(identifierToFetch);
    }
  };

  const handleSearchSubmit = useCallback((event?: FormEvent) => {
    event?.preventDefault();
    const query = inputValue.trim();
    if (query && query !== selectedWordIdentifier) { // Check against identifier
      console.log(`Search submitted for: ${query}`);
      setSearchTerm(query);
      setShowSuggestions(false);
      updateHistory(query); // Fetch data for the searched lemma
    }
  }, [inputValue, selectedWordIdentifier, updateHistory]);

  // Handle Random Word Button Click
  const handleRandomWord = useCallback(async () => {
    // Clear any existing timeout
    if (randomWordTimeoutRef.current) {
      clearTimeout(randomWordTimeoutRef.current);
      randomWordTimeoutRef.current = null;
    }

    // Show loading state immediately
    setIsRandomLoading(true);
    setError(null); // Clear any existing errors
    
    try {
      let randomWord: any = null;
      
      // Use the ref for immediate access to the current cache
      const currentCache = randomWordCacheRef.current;
      
      // If force refresh is requested or cache is empty, fetch new words immediately
      if (forceRefresh || currentCache.length === 0) {
        console.log("Force refreshing random word cache or cache is empty");
        const words = await fetchRandomWordsForCache(RANDOM_CACHE_SIZE);
        
        if (words.length === 0) {
          throw new Error("Failed to fetch random words");
        }
        
        // Take the first word for immediate display
        randomWord = words[0];
        
        // Update the REF and state with the remaining words for future use
        const remainingWords = words.slice(1);
        randomWordCacheRef.current = remainingWords;
        setRandomWordCache(remainingWords);
        
        console.log(`Using freshly fetched random word: ${randomWord.lemma}`);
      } else {
        // Get current word to avoid showing it again
        const currentWordLemma = selectedWordInfo?.lemma?.toLowerCase();
        
        // Create a filtered cache that excludes the current word
        let filteredCache = [...currentCache];
        if (currentWordLemma && filteredCache.length > 1) {
          filteredCache = filteredCache.filter(word => 
            word.lemma.toLowerCase() !== currentWordLemma
          );
        }
        
        // Choose a random word from filtered cache
        if (filteredCache.length > 0) {
          const randomIndex = Math.floor(Math.random() * filteredCache.length);
          randomWord = filteredCache[randomIndex];
        } else if (currentCache.length > 0) {
          // Fallback: Pick from original cache ref
          console.warn("Could not find a different random word, picking from original cache.");
          const originalCacheIndex = Math.floor(Math.random() * currentCache.length);
          randomWord = currentCache[originalCacheIndex];
        } else {
          console.error("Error: Filtered and original caches (ref) are empty during selection.");
        }
        
        // Now remove this word from the actual cache REF immediately
        if (randomWord) {
          const indexInOriginalCache = currentCache.findIndex(
            word => word.id === randomWord.id
          );
          
          if (indexInOriginalCache !== -1) {
            // Create the new cache based on the ref
            const newCache = [...currentCache];
            newCache.splice(indexInOriginalCache, 1);
            
            // Update the ref immediately for the next click
            randomWordCacheRef.current = newCache;
            // Update the state to trigger re-renders etc.
            setRandomWordCache(newCache);
            
            console.log(`Selected new random word: ${randomWord.lemma}${currentWordLemma ? ` (different from current: ${currentWordLemma})` : ''} - Cache size now: ${newCache.length}`);
          }
        }
      }
      
      // Always refill the cache in the background if it's getting low
      if (randomWordCacheRef.current.length < Math.ceil(RANDOM_CACHE_SIZE * 1.5)) {
        console.log("Cache is getting low, fetching more words in background");
        // Don't await this - let it happen in background
        fetchRandomWordsForCache(RANDOM_CACHE_SIZE).then(cachedWords => {
          if (cachedWords.length > 0) {
            console.log(`Background refill added ${cachedWords.length} new random words to cache`);
          }
        }).catch(err => {
          console.error('Error refilling cache in background:', err);
        });
      }
      
      // Process the chosen random word if one was found
      if (randomWord) {
        // Create a normalized WordInfo object, including semantic_network if available
        const wordInfo: WordInfo = {
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
          // Include semantic_network if available in the random word data
          semantic_network: randomWord.semantic_network || null,
        };
        
        // Update UI with word data immediately
        setSelectedWordInfo(wordInfo);
        setMainWord(randomWord.lemma);
        setInputValue(randomWord.lemma);
        
        // Update history
        const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), { id: randomWord.id, text: randomWord.lemma }];
        setWordHistory(newHistory);
        setCurrentHistoryIndex(newHistory.length - 1);
        
        // Start fetching network data and etymology tree in parallel
        Promise.all([
          // Fetch network data if we have depth and breadth settings
          depth && breadth 
            ? fetchWordNetwork(randomWord.lemma, { 
              depth,
              breadth,
              include_affixes: true,
              include_etymology: true,
              cluster_threshold: 0.3
            })
              .then(networkData => {
                // If word doesn't have relation data, update with semantic network
                if (!wordInfo.outgoing_relations?.length && !wordInfo.incoming_relations?.length) {
                  // Use type-safe update
                  setSelectedWordInfo(prevInfo => {
                    if (!prevInfo) return prevInfo;
                    return {
                      ...prevInfo,
                      semantic_network: {
                        nodes: networkData.nodes || [],
                        links: networkData.edges || []
                      }
                    };
                  });
                }
                setWordNetwork(networkData); // Ensure network is visualized
              })
              .catch(err => {
                console.error("Error fetching network data:", err);
              })
            : Promise.resolve(),
            
          // Fetch etymology tree data
          randomWord.id 
            ? fetchEtymologyTree(randomWord.id)
              .catch(err => {
                console.error("Error fetching etymology tree:", err);
              })
            : Promise.resolve()
        ]);
      } else {
        // This will now only trigger if both cache and direct fetch failed
        setError("Failed to get a random word. Please try again.");
      }
    } catch (error) {
      console.error("Error handling random word:", error);
      setError(error instanceof Error ? error.message : "Failed to get a random word");
    } finally {
      // Set a small delay before allowing another click
      randomWordTimeoutRef.current = setTimeout(() => {
        setIsRandomLoading(false);
      }, 50); // Keep at 50ms for rapid response time
    }
  }, [
    depth, 
    breadth, 
    fetchWordNetwork, 
    fetchEtymologyTree, 
    wordHistory, 
    currentHistoryIndex, 
    selectedWordInfo,
    randomWordCache,
    fetchRandomWordsForCache,
    setWordNetwork,
  ]);

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
          handleSearchSubmit();
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

            // --- BEGIN EDIT ---
            // Comment out potentially problematic/incomplete initial fetches
            // fetchBaybayinWords(1, selectedLanguage); 
            // fetchAffixes(selectedLanguage);
            // fetchRelations(selectedLanguage);
            // fetchAllWords(1, selectedLanguage);
            // --- END EDIT ---
            
            // Only prefetch the random word cache if the cache is empty
            // This prevents unnecessary API calls on each page load
            if (randomWordCache.length === 0) {
              console.log("Initial random word cache is empty, prefetching words for rapid clicking");
              // Fetch a large initial cache to support rapid clicking (multiple words)
              fetchRandomWordsForCache(RANDOM_CACHE_SIZE * 3);
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
  }, [fetchRandomWordsForCache]); // Removed randomWordCache.length from dependencies to prevent infinite loop

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
    // --- BEGIN EDIT ---
    // Comment out potentially problematic/incomplete fetches on language change
    // fetchBaybayinWords(1, language);
    // fetchAffixes(language);
    // fetchRelations(language);
    // fetchAllWords(1, language);
    // --- END EDIT ---
  }, []);

  // Function to handle page change
  const handlePageChange = useCallback((page: number) => {
    setCurrentPage(page);
    fetchAllWords(page, selectedLanguage);
  }, [fetchAllWords, selectedLanguage]);

  // Render statistics section
  const renderStatistics = () => {
    if (isLoadingStatistics) return <p>Loading statistics...</p>; // Added loading state
    if (!statistics) return null;

    // Use correct fields from the Statistics type returned by the API
    return (
      <div className="statistics-section">
        <h4>Dictionary Statistics</h4>
        {/* Display available statistics */}
        {statistics.total_words !== undefined && (
          <p>Total Words: {statistics.total_words.toLocaleString()}</p>
        )}
        {statistics.total_definitions !== undefined && (
          <p>Total Definitions: {statistics.total_definitions.toLocaleString()}</p>
        )}
        {statistics.total_relations !== undefined && (
          <p>Total Relations: {statistics.total_relations.toLocaleString()}</p>
        )}
        {statistics.words_with_baybayin !== undefined && (
          <p>Words with Baybayin: {statistics.words_with_baybayin.toLocaleString()}</p>
        )}
        {statistics.words_with_etymology !== undefined && (
          <p>Words with Etymology: {statistics.words_with_etymology.toLocaleString()}</p>
        )}
        {statistics.words_with_examples !== undefined && (
          <p>Words with Examples: {statistics.words_with_examples.toLocaleString()}</p>
        )}

        {statistics.words_by_language && Object.keys(statistics.words_by_language).length > 0 && (
          <div>
            <h5>Words per Language (Top 5):</h5>
            <ul>
              {Object.entries(statistics.words_by_language)
                .sort(([, countA], [, countB]) => countB - countA)
                .slice(0, 5)
                .map(([lang, count]) => (
                  <li key={lang}>{lang}: {count.toLocaleString()}</li>
              ))}
            </ul>
          </div>
        )}
        {statistics.words_by_pos && Object.keys(statistics.words_by_pos).length > 0 && (
          <div>
            <h5>Words per Part of Speech (Top 5):</h5>
            <ul>
              {Object.entries(statistics.words_by_pos)
                .sort(([, countA], [, countB]) => countB - countA)
                .slice(0, 5)
                .map(([pos, count]) => (
                  <li key={pos}>{pos}: {count.toLocaleString()}</li>
              ))}
            </ul>
          </div>
        )}
        {statistics.timestamp && (
           <p style={{ marginTop: '1em', fontSize: '0.8em', color: '#666' }}>
             Statistics generated on: {new Date(statistics.timestamp).toLocaleString()}
           </p>
        )}
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

  // Create a ref for the debounced search function to ensure it's stable
  const debouncedSearchRef = useRef<Function | null>(null);
  
  // Initialize the debounced search function on component mount
  useEffect(() => {
    debouncedSearchRef.current = debounce(async (query: string) => {
    if (!query || query.trim().length < 2) {
      setSearchResults([]);
      setShowSuggestions(false);
      return;
    }
    
      console.log(`[DEBUG] Debounced search for: "${query}"`);
    setIsLoadingSuggestions(true);
    try {
      const searchOptions: SearchOptions = { 
        page: 1,
        per_page: 10,
        exclude_baybayin: false,
          language: '', // Empty string to search in all languages
        mode: 'all',
        sort: 'relevance',
        order: 'desc'
      };
      
        console.log(`[DEBUG] Making debounced API search call for: "${query}"`);
      const results = await searchWords(query, searchOptions);
        console.log(`[DEBUG] Debounced search results:`, results);
      
        if (results && results.words && results.words.length > 0) {
          // Display all results without filtering
        setSearchResults(results.words);
          setShowSuggestions(true);
          console.log(`[DEBUG] Found ${results.words.length} suggestions for "${query}"`);
      } else {
        setSearchResults([]);
        setShowSuggestions(false);
          console.log(`[DEBUG] No suggestions found for "${query}"`);
      }
    } catch (error) {
        console.error(`[DEBUG] Error fetching search suggestions for "${query}":`, error);
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
    }, 300);
    
    // Cleanup function to cancel debounce on unmount
    return () => {
      if (debouncedSearchRef.current && typeof (debouncedSearchRef.current as any).cancel === 'function') {
        (debouncedSearchRef.current as any).cancel();
      }
    };
  }, []);

  // Wrapper function that calls the debounced search
  const handleDebouncedSearch = useCallback((query: string) => {
    console.log(`[DEBUG] Search input changed: "${query}"`);
    if (debouncedSearchRef.current) {
      debouncedSearchRef.current(query);
    } else {
      console.error('[DEBUG] Debounced search function not initialized');
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

  // Update the existing useEffect for relations
  useEffect(() => {
    // --- BEGIN EDIT ---
    // Simplify relation handling: rely on relations fetched with word details
    // Only log if relations seem missing from the main wordData
    if (wordData && wordData.id) {
      const hasRelations = (
        (wordData.incoming_relations && wordData.incoming_relations.length > 0) || 
        (wordData.outgoing_relations && wordData.outgoing_relations.length > 0)
      );
      
      if (!hasRelations) {
        console.log(`Word ${wordData.lemma} (ID: ${wordData.id}) loaded, but no relations found in the initial details fetch. Relations might be fetched separately by the network graph or may not exist.`);
        // No need to trigger additional fetches here, as the network graph or full detail view might handle it.
      }
    }
    // Removed dependency on fetchWordRelations, mainWord, selectedWordInfo as we are simplifying
    // --- END EDIT ---
  }, [wordData]); // Depend only on wordData

  // Synchronize the ref with the state whenever the state changes
  useEffect(() => {
    randomWordCacheRef.current = randomWordCache;
  }, [randomWordCache]);

  // Render the search bar with navigation buttons
  const renderSearchBar = () => {
    return (
      <div className="search-container">
        <div className="nav-buttons">
          <Button 
            className="nav-button back-button" 
            onClick={handleBack}
            disabled={currentHistoryIndex <= 0}
            title="Go back to previous word"
            variant="contained" // Use MUI Button for consistency
            sx={{
              minWidth: 36, width: 36, height: 36, 
              borderRadius: '50%', 
              p: 0, // Remove padding for icon alignment
              bgcolor: 'var(--button-color)',
              color: 'var(--button-text-color)',
              boxShadow: 'none',
              '&:hover': { bgcolor: 'var(--primary-color)' },
              '&:active': { transform: 'scale(0.9)' },
              '&.Mui-disabled': { 
                 bgcolor: 'var(--card-border-color)', 
                 color: 'var(--text-color)',
                 opacity: 0.6 
              }
            }}
          >
            <span>←</span>
          </Button>
          <Button 
            className="nav-button forward-button" 
            onClick={handleForward}
            disabled={currentHistoryIndex >= wordHistory.length - 1}
            title="Go forward to next word"
            variant="contained"
            sx={{
              minWidth: 36, width: 36, height: 36, 
              borderRadius: '50%', 
              p: 0, 
              bgcolor: 'var(--button-color)',
              color: 'var(--button-text-color)',
              boxShadow: 'none',
              '&:hover': { bgcolor: 'var(--primary-color)' },
              '&:active': { transform: 'scale(0.9)' },
              '&.Mui-disabled': { 
                 bgcolor: 'var(--card-border-color)', 
                 color: 'var(--text-color)',
                 opacity: 0.6 
              }
            }}
          >
            <span>→</span>
          </Button>
        </div>
        
        <div className="search-input-container">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => {
              const newValue = e.target.value;
              setInputValue(newValue);
              
              // Use debounced search for suggestions
              handleDebouncedSearch(newValue);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                console.log(`[DEBUG] Enter key pressed for search: "${inputValue}"`);
                // Execute search on Enter key press
                if (inputValue.trim()) {
                  handleSearch(inputValue);
                  setShowSuggestions(false); // Hide suggestions after search
                }
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
                    const resultId = String(result.id);
                    const wordId = resultId.startsWith('id:') ? resultId : `id:${resultId}`;
                    handleNodeClick(wordId);
                    // Also update the input value to show what was selected
                    setInputValue(result.lemma);
                    // Hide suggestions after selection
                    setShowSuggestions(false);
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
        
        <Button
          variant="contained"
          className="search-button"
          startIcon={isLoading ? <CircularProgress size={16} /> : null}
          onClick={() => inputValue.trim() && handleSearch(inputValue)}
          disabled={isLoading || !inputValue.trim()} 
          title="Search for this word"
          sx={(theme) => ({
            mx: 0.1, 
            whiteSpace: 'nowrap',
            bgcolor: 'var(--button-color)',
            color: 'var(--button-text-color)',
            borderRadius: '8px',
            boxShadow: 'none',
            '&:hover': {
              bgcolor: 'var(--primary-color)',
              boxShadow: 'none'
            }
          })}
        >
          {isLoading ? 'Searching...' : '🔍 Search'}
        </Button>
        
        <Button
          variant="contained"
          className="random-button"
          startIcon={isRandomLoading ? <CircularProgress size={16} /> : null}
          onClick={() => handleRandomWord(false)}
          disabled={isRandomLoading || isLoading} 
          title="Get a random word (long press to refresh cache)"
          onContextMenu={(e) => {
            e.preventDefault();
            handleRandomWord(true); // Force refresh when right-clicked
          }}
          sx={(theme) => ({
            mx: 0.1, 
            whiteSpace: 'nowrap',
            bgcolor: 'var(--accent-color)',
            color: 'var(--primary-color)',
            fontWeight: 'normal',
            borderRadius: '8px',
            boxShadow: 'none',
            '&:hover': {
              bgcolor: 'var(--secondary-color)',
              color: '#ffffff',
              boxShadow: 'none'
            }
          })}
        >
          {isRandomLoading ? '⏳ Loading...' : '🎲 Random Word'}
        </Button>
      </div>
    );
  };

  // Use MUI's useMediaQuery to check screen size (similar to WordDetails)
  // Using 900px as the breakpoint for side-by-side vs stacked
  const isWideLayout = useMediaQuery('(min-width:769px)'); 

  // Add cleanup for timeout ref
  useEffect(() => {
    return () => {
      if (randomWordTimeoutRef.current !== null) {
        clearTimeout(randomWordTimeoutRef.current);
      }
    };
  }, []);

  // Add properly typed button handlers
  const handleRandomClick = useCallback(() => {
    handleRandomWord(false);
  }, [handleRandomWord]);
  
  const handleRandomRightClick = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    handleRandomWord(true); // Force refresh
  }, [handleRandomWord]);

  return (
    <div className={`word-explorer ${theme} ${isLoading ? 'loading' : ''}`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <div className="header-buttons">
          <button
            onClick={handleRandomClick}
            className="random-button"
            title="Get a random word"
            disabled={isRandomLoading || isLoading}
          >
            {isRandomLoading ? '⏳ Loading...' : '🎲 Random Word'}
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
        {isWideLayout ? (
          <div style={{ width: '100%', height: '100%', display: 'flex', overflow: 'hidden' }}>
            <PanelGroup direction="horizontal" autoSaveId="wordExplorerLayout" style={{ width: '100%', height: '100%', display: 'flex' }}>
              <Panel defaultSize={60} minSize={30} style={{ overflow: 'hidden', height: '100%' }}>
                {/* Graph Container Content */}
                <div className="graph-container" style={{ width: '100%', height: '100%' }}>
                  <div className="graph-content" style={{ width: '100%', height: '100%' }}>
                    {isLoading && <div className="loading">Loading Network...</div>}
                    {!isLoading && wordNetwork && (
                      <WordGraph
                        wordNetwork={wordNetwork}
                        mainWord={mainWord}
                        onNodeClick={handleNodeClick}
                        onNodeSelect={handleNodeSelect}
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
              </Panel>
              <PanelResizeHandle style={{ width: '4px', background: 'var(--card-border-color)' }} />
              <Panel defaultSize={40} minSize={25} style={{ overflow: 'hidden', height: '100%', width: '100%' }}>
                {/* Details Container Content */}
                <div ref={detailsContainerRef} className="details-container" style={{ width: '100%', height: '100%', overflow: 'auto' }}>
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
                  {error && <div className="error-message">Error: {error}</div>}
                </div>
              </Panel>
            </PanelGroup>
          </div>
        ) : (
          // Mobile/Stacked Layout
          <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div className="graph-container" style={{ flexBasis: '50%', minHeight: '300px' }}>
              <div className="graph-content">
                {isLoading && <div className="loading">Loading Network...</div>}
                {!isLoading && wordNetwork && (
                  <WordGraph
                    wordNetwork={wordNetwork}
                    mainWord={mainWord}
                    onNodeClick={handleNodeClick}
                    onNodeSelect={handleNodeSelect}
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
            <div ref={detailsContainerRef} className="details-container" style={{ flexBasis: '50%', minHeight: '300px' }}>
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
              {error && <div className="error-message">Error: {error}</div>}
            </div>
          </div>
        )}
      </main>
      
      <footer className="footer">
        © {new Date().getFullYear()} Filipino Root Word Explorer. All Rights Reserved.
      </footer>
    </div>
  );
};

export default WordExplorer;
