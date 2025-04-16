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
  const randomWordTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const RANDOM_CACHE_SIZE = 5;
  // Use a ref to track last refresh time to prevent excessive refreshes
  const lastRefreshTimeRef = useRef<number>(0);

  const detailsContainerRef = useRef<HTMLDivElement>(null);

  // Add a state to track when a random word request is in progress
  const [isRandomLoading, setIsRandomLoading] = useState<boolean>(false);

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
        
        // Sync network data with word details if we have wordData
        if (wordData && wordData.id) {
          // Find the main node (root word)
          const mainNode = data.nodes.find(node => 
            node.type === 'main' || node.word === word || node.label === word
          );
          
          if (mainNode) {
            // Create relations from network data
            const incomingRelations: Relation[] = [];
            const outgoingRelations: Relation[] = [];
            
            // Process each edge to build incoming and outgoing relations
            data.edges.forEach(edge => {
              const sourceNode = data.nodes.find(n => n.id === edge.source);
              const targetNode = data.nodes.find(n => n.id === edge.target);
              
              if (sourceNode && targetNode) {
                // If this edge points to the main node, it's an incoming relation
                if (targetNode.id === mainNode.id) {
                  incomingRelations.push({
                    id: Math.floor(Math.random() * 1000000), // Generate a random ID
                    relation_type: edge.type,
                    source_word: {
                      id: Number(sourceNode.id) || 0,
                      lemma: sourceNode.word || sourceNode.label,
                      has_baybayin: sourceNode.has_baybayin,
                      baybayin_form: sourceNode.baybayin_form
                    }
                  });
                }
                // If this edge comes from the main node, it's an outgoing relation
                else if (sourceNode.id === mainNode.id) {
                  outgoingRelations.push({
                    id: Math.floor(Math.random() * 1000000), // Generate a random ID
                    relation_type: edge.type,
                    target_word: {
                      id: Number(targetNode.id) || 0,
                      lemma: targetNode.word || targetNode.label,
                      has_baybayin: targetNode.has_baybayin,
                      baybayin_form: targetNode.baybayin_form
                    }
                  });
                }
              }
            });
            
            // Update wordData with the relations we extracted
            console.log("Syncing relations:", {
              incoming: incomingRelations.length,
              outgoing: outgoingRelations.length
            });
            
            setWordData(prevData => {
              if (!prevData) return prevData;
              return {
                ...prevData,
                incoming_relations: incomingRelations.length > 0 ? incomingRelations : prevData.incoming_relations,
                outgoing_relations: outgoingRelations.length > 0 ? outgoingRelations : prevData.outgoing_relations
              };
            });
          }
        }
        
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
      
      // Check if we actually got meaningful data
      if (data && Array.isArray(data.nodes) && data.nodes.length > 0) {
        console.log(`Received valid etymology tree with ${data.nodes.length} nodes`);
        setEtymologyTree(data);
        return data;
      } else {
        console.warn("Received empty etymology tree or invalid structure");
        // Return null but don't set error - we'll fall back to basic etymologies
        setEtymologyTree(null);
        return null;
      }
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
      // Initialize wordData with a default empty structure
      let wordData: WordInfo | null = null;
      let fallbackToSearch = false;

      try {
        // First try getting the word details directly
        wordData = await fetchWordDetails(word);
      } catch (error: any) {
        // If there's any error fetching the word, try to search for it as a fallback
        console.warn(`Failed to fetch details for word '${word}', error:`, error.message);
        fallbackToSearch = true;
        
        // Expanded error conditions to be more inclusive
        if (error.message.includes('not found') || 
            error.message.includes('Database error') ||
            error.message.includes('dictionary update sequence') ||
            error.message.includes('Server database error') ||
            error.message.includes('unexpected error')) {
          console.log(`Falling back to search for word: ${word}`);
          
          // Extract search text from the word or ID
          let searchText;
          if (word.startsWith('id:')) {
            // For ID format, we'll try to search by ID but also have a fallback
            const wordId = word.substring(3);
            // Try both searching with the ID and searching with 'id:' prefix removed
            try {
              const idSearchResults = await searchWords(`id:${wordId}`, {
                page: 1,
                per_page: 5,
                mode: 'all',
                sort: 'relevance'
              });
              
              if (idSearchResults.words && idSearchResults.words.length > 0) {
                console.log(`Search by ID successful, found ${idSearchResults.words.length} results`);
                // Use the first search result
                const firstResult = idSearchResults.words[0];
                wordData = await fetchWordDetails(
                  String(firstResult.id).startsWith('id:') ? 
                  String(firstResult.id) : 
                  `id:${firstResult.id}`
                );
                
                // Successfully got data
                fallbackToSearch = false; // No need to continue with search
              }
            } catch (idSearchError) {
              console.warn(`ID-based search failed, trying word search:`, idSearchError);
              // Fall through to regular search
            }
            
            // If ID search failed or wasn't attempted, set searchText
            if (fallbackToSearch) {
              // If ID search failed, try to extract a searchable text 
              // For now we'll just use the ID as is, but could implement smarter extraction
              searchText = wordId;
            }
          } else {
            // Regular word search
            searchText = word;
          }
          
          // Perform general search if we still need to
          if (fallbackToSearch && searchText) {
            const searchResults = await searchWords(searchText, {
              page: 1,
              per_page: 10,
              mode: 'all',
              sort: 'relevance',
              language: ''  // Search in all languages
            });
            
            if (searchResults.words && searchResults.words.length > 0) {
              console.log(`Search successful, found ${searchResults.words.length} results`);
              
              // Show search results in UI
              setSearchResults(searchResults.words);
              
              // Use the first search result
              const firstResult = searchResults.words[0];
              try {
                wordData = await fetchWordDetails(
                  String(firstResult.id).startsWith('id:') ? 
                  String(firstResult.id) : 
                  `id:${firstResult.id}`
                );
                fallbackToSearch = false; // Successfully got the data
              } catch (detailError) {
                console.error(`Failed to fetch details for search result:`, detailError);
                
                // If fetching details for first result also fails, 
                // just display the search results and a helpful message
                setError(`Could not load full details for "${searchText}". Please select one of the search results below.`);
                setIsLoading(false);
                return; // Exit function early, letting user select from search results
              }
            } else {
              throw new Error(`Word '${searchText}' not found. Please try a different word.`);
            }
          }
        } else {
          // Rethrow other errors
          throw error;
        }
      }

      // Successfully got the word data (either directly or via search)
      if (!fallbackToSearch && wordData) {
        console.log(`Word data retrieved successfully:`, wordData);
        setSelectedWordInfo(wordData);
        setMainWord(wordData.lemma);
        setInputValue(wordData.lemma);
        
        // Update navigation/history
        const wordId = String(wordData.id);
        if (!wordHistory.some(w => typeof w === 'object' && 'id' in w && String(w.id) === wordId)) {
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
          const etymologyIdString = String(wordData.id);
          const etymologyId = etymologyIdString.startsWith('id:') ? 
            parseInt(etymologyIdString.substring(3), 10) : 
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
        
        // Update word network for the new main word
        try {
          console.log(`Fetching network for new main word: ${wordData.lemma}`);
          const networkData = await fetchWordNetworkData(wordData.lemma, depth, breadth);
          setWordNetwork(networkData);
        } catch (networkError) {
          console.error("Error fetching word network:", networkError);
          // Don't fail the entire operation if network fetch fails
        }
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
  }, [wordHistory, depth, breadth, setDepth, setBreadth, fetchWordDetails, searchWords, fetchWordNetworkData, fetchEtymologyTree]);

  // New function for handling node selection (single click) without navigation
  const handleNodeSelect = useCallback(async (word: string) => {
    if (!word) {
      console.error("Empty word received in handleNodeSelect");
      return;
    }

    console.log(`Node selected: ${word}`);
    
    try {
      // Don't show loading for the whole UI, just for the details panel
      setError(null);
      
      let wordData: WordInfo | null = null;
      
      try {
        // Try getting the word details directly
        wordData = await fetchWordDetails(word);
      } catch (error: any) {
        console.warn(`Failed to fetch details for selected word '${word}', error:`, error.message);
        // Don't show error in the UI for selection, just log it
        return;
      }
      
      if (wordData) {
        console.log(`Selected word data retrieved:`, wordData);
        // Update selected word info but DON'T change main word or network
        setSelectedWordInfo(wordData);
        
        // Fetch the etymology tree for the selected word in the background
        try {
          const etymologyIdString = String(wordData.id);
          const etymologyId = etymologyIdString.startsWith('id:') ? 
            parseInt(etymologyIdString.substring(3), 10) : 
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
      }
    } catch (error: any) {
      console.error(`Error in handleNodeSelect for word '${word}':`, error);
      // Don't show error in the UI for selection, just log it
    }
  }, [fetchWordDetails, fetchEtymologyTree]);

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

    console.log(`[DEBUG] Starting search for: "${query}"`);
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
      
      console.log(`[DEBUG] Calling searchWords API with query: "${query}", options:`, searchOptions);
      const result = await searchWords(query, searchOptions);
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
          
          // 2. Update the word data immediately
          setSelectedWordInfo(wordData);
          setMainWord(wordData.lemma);
          setInputValue(wordData.lemma);
          
          // 3. Update history
          const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), { id: wordData.id, text: wordData.lemma }];
          setWordHistory(newHistory);
          setCurrentHistoryIndex(newHistory.length - 1);
          
          // 4. Then load the network in parallel
          console.log(`[DEBUG] Fetching network for: ${wordData.lemma}`);
          fetchWordNetworkData(wordData.lemma, depth, breadth)
            .then(networkData => {
              console.log(`[DEBUG] Network data received with ${networkData?.nodes?.length || 0} nodes`);
              setWordNetwork(networkData);
            })
            .catch(networkErr => {
              console.error(`[DEBUG] Error fetching network:`, networkErr);
              // Don't fail the whole search if network fails
            });
          
          // 5. Also try to fetch etymology tree if available
          try {
            console.log(`[DEBUG] Fetching etymology tree for ID: ${wordData.id}`);
            // Convert ID to appropriate format for etymology tree fetch
            const etymologyIdString = String(wordData.id);
            const etymologyId = etymologyIdString.startsWith('id:') 
              ? parseInt(etymologyIdString.substring(3), 10) 
              : wordData.id;
            fetchEtymologyTree(etymologyId)
              .then(tree => {
                console.log(`[DEBUG] Etymology tree received`);
                setEtymologyTree(tree);
              })
              .catch(etymErr => {
                console.error(`[DEBUG] Error fetching etymology tree:`, etymErr);
                // Don't fail the search if etymology fetch fails
              });
          } catch (etymErr) {
            console.error(`[DEBUG] Error initiating etymology tree fetch:`, etymErr);
            // Don't throw, we can continue without etymology
          }
          
          // Scroll details container to top
          detailsContainerRef.current?.scrollTo(0, 0);
          console.log(`[DEBUG] Search loading process completed successfully`);
        } catch (dataError) {
          console.error(`[DEBUG] Error loading word data during search:`, dataError);
          let errorMessage = "Error loading word details";
          
          if (dataError instanceof Error) {
            // Check for common error patterns and provide more helpful messages
            const msg = dataError.message;
            if (msg.includes("dictionary update sequence")) {
              errorMessage = "There was a database error on the server. Try a different search term or try again later.";
            } else if (msg.includes("Database error")) {
              errorMessage = "Database error occurred. The word exists but there was a problem retrieving its details.";
            } else if (msg.includes("not found")) {
              errorMessage = `Word "${query}" was found in search but its details could not be retrieved.`;
            } else if (msg.includes("Server error")) {
              errorMessage = "Server error occurred. Please try again later.";
            } else {
              errorMessage = dataError.message;
            }
          }
          
          console.error(`[DEBUG] Setting error:`, errorMessage);
          setError(errorMessage);
          
          // Even if detail fetching fails, still show search results
          setSearchResults(result.words);
        }
      } else {
        // No results found
        console.log(`[DEBUG] No results found for query: "${query}"`);
        setSearchResults([]);
        setShowSuggestions(false);
        setError(`No results found for "${query}". Please try a different word.`);
      }
    } catch (error) {
      console.error(`[DEBUG] Search error:`, error);
      setSearchResults([]);
      let errorMessage = "An error occurred during search";
      
      if (error instanceof Error) {
        // Check for common error patterns and provide more helpful messages
        const msg = error.message;
        if (msg.includes("dictionary update sequence")) {
          errorMessage = "There was a database error on the server. Try a different search term or try again later.";
        } else if (msg.includes("Network Error") || msg.includes("Failed to fetch")) {
          errorMessage = "Cannot connect to the backend server. Please ensure the backend server is running.";
        } else if (msg.includes("Circuit breaker")) {
          errorMessage = "Too many failed requests. Please wait a moment and try again.";
        } else {
          errorMessage = error.message;
        }
      }
      
      console.error(`[DEBUG] Setting error message:`, errorMessage);
      setSearchError(errorMessage);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
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
      
      // Extract the actual word text based on type
      const wordText = typeof previousWord === 'string' 
        ? previousWord 
        : previousWord.text;
        
      // Extract the ID if it's an object
      const wordId = typeof previousWord === 'string'
        ? previousWord
        : previousWord.id.toString();
      
      // Fetch the word data without updating history
      setIsLoading(true);
      setError(null);
      
      Promise.all([
        fetchWordDetails(wordId),
        fetchWordNetworkData(wordText, depth, breadth)
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
      
      // Extract the actual word text based on type
      const wordText = typeof nextWord === 'string' 
        ? nextWord 
        : nextWord.text;
        
      // Extract the ID if it's an object
      const wordId = typeof nextWord === 'string'
        ? nextWord
        : nextWord.id.toString();
      
      // Fetch the word data without updating history
      setIsLoading(true);
      setError(null);
      
      Promise.all([
        fetchWordDetails(wordId),
        fetchWordNetworkData(wordText, depth, breadth)
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
    // If already loading, don't try to fetch another random word
    if (isRandomLoading) {
      console.log("Random word request already in progress, ignoring click");
      return;
    }

    // Clear any existing timeout
    if (randomWordTimeoutRef.current) {
      clearTimeout(randomWordTimeoutRef.current);
      randomWordTimeoutRef.current = null;
    }

    setIsRandomLoading(true);
    setError(null); // Clear any existing errors
    
    try {
      let randomWord;
      
      // Prefer using the cache if possible
      if (randomWordCache.length > 0) {
        const randomIndex = Math.floor(Math.random() * randomWordCache.length);
        randomWord = randomWordCache[randomIndex];
        
        // Remove the selected word from the cache
        const newCache = [...randomWordCache];
        newCache.splice(randomIndex, 1);
        setRandomWordCache(newCache);
        
        // Refill the cache in the background
        fetchRandomWordsForCache().then(cachedWords => {
          console.log(`Added ${cachedWords.length} new random words to cache`);
          setRandomWordCache(prev => [...prev, ...cachedWords]);
        });
      } else {
        // If cache is empty, fetch directly (this path should be rare)
        console.log("Random word cache empty, fetching directly");
        const words = await fetchRandomWordsForCache();
        
        if (words.length === 0) {
          throw new Error("Failed to fetch random words");
        }
        
        randomWord = words[0];
        setRandomWordCache(words.slice(1));
      }
      
      if (randomWord) {
        // Create a normalized WordInfo object
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
        };
        
        setSelectedWordInfo(wordInfo);
        setMainWord(randomWord.lemma);
        
        // Update the input field
        setInputValue(randomWord.lemma);
        
        // Update history
        const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), randomWord.lemma];
        setWordHistory(newHistory);
        setCurrentHistoryIndex(newHistory.length - 1);
        
        // Fetch network data (if configured)
        if (depth && breadth) {
          fetchWordNetworkData(randomWord.lemma, depth, breadth)
            .catch(err => console.error("Error fetching network data:", err));
        }
        
        // Fetch etymology tree (if available)
        if (randomWord.id) {
          fetchEtymologyTree(randomWord.id)
            .catch(err => console.error("Error fetching etymology tree:", err));
        }
      } else {
        setError("Failed to get a random word. Please try again.");
      }
    } catch (error) {
      console.error("Error handling random word:", error);
      setError(error instanceof Error ? error.message : "Failed to get a random word");
    } finally {
      // Set a small delay before allowing another click
      randomWordTimeoutRef.current = setTimeout(() => {
        setIsRandomLoading(false);
      }, 500); // 500ms delay to prevent rapid clicking
    }
  }, [
    depth, 
    breadth, 
    fetchWordNetworkData, 
    fetchEtymologyTree, 
    wordHistory, 
    currentHistoryIndex, 
    randomWordCache,
    fetchRandomWordsForCache,
    isRandomLoading
  ]);

  // Add cleanup for timeout ref
  useEffect(() => {
    return () => {
      if (randomWordTimeoutRef.current !== null) {
        clearTimeout(randomWordTimeoutRef.current);
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
            borderRadius: '10px',
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
          onClick={handleRandomWord}
          disabled={isRandomLoading || isLoading} 
          title="Get a random word"
          sx={(theme) => ({
            mx: 0.1, 
            whiteSpace: 'nowrap',
            bgcolor: 'var(--button-color)',
            color: 'var(--button-text-color)',
            borderRadius: '15px',
            boxShadow: 'none',
            '&:hover': {
              bgcolor: 'var(--primary-color)',
              boxShadow: 'none'
            }
          })}
        >
          {isRandomLoading ? '⏳ Loading...' : '🎲 Random Word'}
        </Button>
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
        <div className="graph-container">
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
        <div ref={detailsContainerRef} className="details-container">
          {isLoading && <div className="loading-spinner">Loading Details...</div>} 
          {!isLoading && selectedWordInfo && (
            <>
              {/* Add debugging code properly - outside of JSX rendering */}
              <WordDetails 
                wordInfo={
                  // Execute debug log and return the selectedWordInfo
                  (() => {
                    // Create enhanced wordInfo with graph data if available
                    const enhancedWordInfo = {
                      ...selectedWordInfo,
                      // Add semantic network data from the graph if available
                      semantic_network: wordNetwork && wordNetwork.nodes && wordNetwork.edges ? {
                        nodes: wordNetwork.nodes,
                        links: wordNetwork.edges
                      } : null
                    };
                    
                    console.log("DEBUG - Passing data to WordDetails:", {
                      id: enhancedWordInfo.id,
                      lemma: enhancedWordInfo.lemma,
                      hasIncomingRelations: Boolean(enhancedWordInfo.incoming_relations),
                      incomingRelationsCount: enhancedWordInfo.incoming_relations?.length || 0,
                      hasOutgoingRelations: Boolean(enhancedWordInfo.outgoing_relations),
                      outgoingRelationsCount: enhancedWordInfo.outgoing_relations?.length || 0,
                      hasSemanticNetwork: Boolean(enhancedWordInfo.semantic_network),
                      semanticNetworkNodes: enhancedWordInfo.semantic_network?.nodes?.length || 0,
                      semanticNetworkLinks: enhancedWordInfo.semantic_network?.links?.length || 0,
                      sample: {
                        incoming: enhancedWordInfo.incoming_relations?.[0] || null,
                        outgoing: enhancedWordInfo.outgoing_relations?.[0] || null
                      }
                    });
                    
                    return enhancedWordInfo;
                  })()
                } 
                etymologyTree={etymologyTree}
                isLoadingEtymology={isLoadingEtymology}
                etymologyError={etymologyError}
                onWordLinkClick={handleNodeClick}
                onEtymologyNodeClick={handleNodeClick}
              />
            </>
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
