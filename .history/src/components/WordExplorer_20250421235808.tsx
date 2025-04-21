import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchOptions, EtymologyTree, Statistics, SearchWordResult, Relation, WordSuggestion } from "../types";
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
  getAllWords,
  getEtymologyTree,
  fetchSuggestions,
} from "../api/wordApi";
import { Button, Box, Autocomplete, TextField, Chip, CircularProgress, IconButton } from "@mui/material";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import useMediaQuery from '@mui/material/useMediaQuery';
import { Theme, useTheme as useMuiTheme } from '@mui/material/styles';
import CloseIcon from '@mui/icons-material/Close';
import { debounce } from '@mui/material/utils';

const isDevMode = () => {
  // Check if we have a URL parameter for showing debug tools
  if (typeof window !== 'undefined') {
    return window.location.href.includes('debug=true') ||
           window.location.hostname === 'localhost' ||
           window.location.hostname.includes('127.0.0.1');
  }
  return false;
};

const WordExplorer: React.FC = () => {
  const [wordData, setWordData] = useState<WordInfo | null>(null);
  const [wordNetwork, setWordNetwork] = useState<WordNetwork | null>(null);
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState(false);
  const [isRandomLoading, setIsRandomLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const { theme: themeName, toggleTheme } = useTheme();
  const [inputValue, setInputValue] = useState<string>("");
  const [depth, setDepth] = useState<number>(2);
  const [breadth, setBreadth] = useState<number>(10);
  const [wordHistory, setWordHistory] = useState<Array<string | {id: number | string, text: string}>>([]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState<number>(-1);
  const [searchResults, setSearchResults] = useState<SearchWordResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState<boolean>(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  const randomWordTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const detailsContainerRef = useRef<HTMLDivElement>(null);

  const [autocompleteOpen, setAutocompleteOpen] = useState(false);
  const [suggestions, setSuggestions] = useState<WordSuggestion[]>([]);
  const [isSuggestionsLoading, setIsSuggestionsLoading] = useState(false);

  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const [isDrawerExpanded, setIsDrawerExpanded] = useState<boolean>(false);

  const fetchSuggestionsDebounced = React.useMemo(
    () =>
      debounce(
        async (request: { input: string }, callback: (results: WordSuggestion[]) => void) => {
          if (!request.input) {
            callback([]);
            return;
          }
          setIsSuggestionsLoading(true);
          try {
            const results = await fetchSuggestions(request.input);
            callback(results);
          } catch (error) {
            console.error("Error fetching suggestions inside debounce:", error);
            callback([]);
          }
          setIsSuggestionsLoading(false);
        },
        400,
      ),
    [fetchSuggestions],
  );

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

  const fetchWordNetworkData = useCallback(async (word: string, depthParam: number = 2, breadthParam: number = 10) => {
    try {
      setIsLoading(true);
      setError(null);
      
      console.log('Fetching word network data for:', word);
      const data = await fetchWordNetwork(word, { 
        depth: depthParam,
        breadth: breadthParam,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.3
      });
      
      if (data && data.nodes && data.edges) {
        console.log('Word network data received:', data);
        console.log(`Network has ${data.nodes.length} nodes and ${data.edges.length} edges`);
        setWordNetwork(data);
        
        if (wordData && wordData.id) {
          const mainNode = data.nodes.find(node => 
            node.type === 'main' || node.word === word || node.label === word
          );
          
          if (mainNode) {
            const incomingRelations: Relation[] = [];
            const outgoingRelations: Relation[] = [];
            
            data.edges.forEach(edge => {
              const sourceNode = data.nodes.find(n => n.id === edge.source);
              const targetNode = data.nodes.find(n => n.id === edge.target);
              
              if (sourceNode && targetNode) {
                if (targetNode.id === mainNode.id) {
                  incomingRelations.push({
                    id: Math.floor(Math.random() * 1000000),
                    relation_type: edge.type,
                    source_word: {
                      id: Number(sourceNode.id) || 0,
                      lemma: sourceNode.word || sourceNode.label,
                      has_baybayin: sourceNode.has_baybayin,
                      baybayin_form: sourceNode.baybayin_form
                    }
                  });
                }
                else if (sourceNode.id === mainNode.id) {
                  outgoingRelations.push({
                    id: Math.floor(Math.random() * 1000000),
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
      
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError('An unexpected error occurred while fetching word network');
      }
      
      setWordNetwork(null);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [wordData]);

  const fetchEtymologyTree = useCallback(async (wordId: number): Promise<EtymologyTree | null> => {
    if (!wordId) return null;
    
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    
    try {
      console.log("Fetching etymology tree for word ID:", wordId);
      const data = await getEtymologyTree(wordId, 3);
      console.log("Etymology tree data received:", data);
      
      if (data && Array.isArray(data.nodes) && data.nodes.length > 0) {
        console.log(`Received valid etymology tree with ${data.nodes.length} nodes`);
        setEtymologyTree(data);
        return data;
      } else {
        console.warn("Received empty etymology tree or invalid structure");
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
  }, [getEtymologyTree]);

  const handleNodeClick = useCallback(async (identifier: string) => {
    if (!identifier) {
      console.error("Empty identifier received in handleNodeClick");
      return;
    }

    let wordToFetch: string = identifier; // Use the identifier directly by default
    let isIdSearch = false;

    // Check if the identifier is an ID
    if (identifier.startsWith('id:')) {
        console.log(`Node clicked (by ID): ${identifier}`);
        isIdSearch = true;
        // Keep wordToFetch as "id:123"
    } else {
        console.log(`Node clicked (by lemma): ${identifier}`);
        // wordToFetch remains the lemma string
    }

    setError(null);
    setIsLoading(true);

    try {
      let wordData: WordInfo | null = null;
      let fallbackToSearch = false;

      try {
        // Fetch using the determined identifier (lemma or "id:...")
        wordData = await fetchWordDetails(wordToFetch);
      } catch (error: any) {
        console.warn(`Initial fetch failed for '${wordToFetch}', error:`, error.message);
        fallbackToSearch = true;

        // Fallback logic: Only try searching if the initial fetch was NOT by ID
        // or if the ID fetch resulted in a specific known error (like not found)
        if (!isIdSearch || error.message.includes('not found')) { 
            const lemmaToSearch = isIdSearch ? identifier.substring(3) : identifier; // Get lemma for search
            console.log(`Falling back to search for lemma: ${lemmaToSearch}`);
            try {
              const searchResults = await searchWords(lemmaToSearch, {
                page: 1,
                per_page: 1,
                mode: 'exact',
                sort: 'relevance',
                language: '' // Search all languages in fallback
              });

              if (searchResults.words && searchResults.words.length > 0) {
                console.log(`Fallback search successful`);
                const firstResult = searchResults.words[0];
                // Fetch details using the ID from search result
                wordData = await fetchWordDetails(`id:${firstResult.id}`);
                fallbackToSearch = false; // Success
              } else {
                // If fallback search finds nothing, throw the original error
                throw error; 
              }
            } catch(searchError: any) {
               console.error(`Fallback search also failed for '${lemmaToSearch}':`, searchError);
               throw error; // Re-throw original error if search fails
            }
        } else {
            // If it was an ID search and not a 'not found' error, just re-throw
            throw error;
        }
      }

      // --- Proceed with processing wordData if fetch was successful ---
      if (wordData) {
        console.log(`Word data retrieved successfully for '${wordToFetch}':`, wordData);
        setWordData(wordData);
        setSelectedNode(wordData.lemma);

        const wordId = String(wordData.id);
        if (!wordHistory.some(w => typeof w === 'object' && 'id' in w && String(w.id) === wordId)) {
          const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), { id: wordData.id, text: wordData.lemma }];
          setWordHistory(newHistory as any);
          setCurrentHistoryIndex(newHistory.length - 1);
        }
        
        // *** Open the drawer when a node is clicked/data loaded ***
        setIsDrawerOpen(true);
        setIsDrawerExpanded(false); // Start partially open or closed, expand on interaction

        // Fetch related data (Network and Etymology)
        // const confirmedIdentifier = identifier; // Revert: Don't use identifier here
        
        try {
          // Use ID for network fetch if available
          const networkIdentifier = wordData.id ? `id:${wordData.id}` : wordData.lemma;
          console.log(`[handleNodeClick] Fetching network using identifier: ${networkIdentifier}`); 
          fetchWordNetworkData(networkIdentifier, depth, breadth) 
              .then(networkData => setWordNetwork(networkData))
              .catch(networkError => console.error("Error fetching word network:", networkError));
        } catch (networkError) {
          console.error("Error initiating word network fetch:", networkError);
        }
        
        try {
          // Etymology should still use the confirmed ID from wordData if available
          const confirmedId = wordData.id; 
          if (confirmedId) {
            fetchEtymologyTree(confirmedId)
                .then(tree => setEtymologyTree(tree))
                .catch(etymologyError => console.error("Error fetching etymology tree:", etymologyError));
          } else {
             console.warn("No ID available to fetch etymology tree.");
             setEtymologyTree(null);
          }
        } catch (etymologyError) {
          console.error("Error initiating etymology tree fetch:", etymologyError);
        }
      }
    } catch (error: any) {
      console.error(`Error in handleNodeClick processing identifier '${identifier}':`, error);
      const displayTerm = identifier.startsWith('id:') ? `ID ${identifier.substring(3)}` : identifier;
      if (error.message.includes('not found')) {
        setError(`Word with ${displayTerm} was not found or could not be retrieved.`);
      } else if (error.message.includes('Circuit breaker')) {
        setError(`Network connection unstable. Please try again.`);
      } else if (error.message.includes('Network Error')) {
        setError(`Cannot connect to the backend server.`);
      } else if (error.message.includes('Database error')) {
        setError(`Database error retrieving word for ${displayTerm}.`);
      } else {
        setError(`Error retrieving details for ${displayTerm}: ${error.message}`);
      }
      setWordData(null);
      setWordNetwork(null);
    } finally {
      setIsLoading(false);
    }
  }, [
      wordHistory, 
      currentHistoryIndex, 
      depth, 
      breadth, 
      fetchWordDetails, 
      searchWords, 
      fetchWordNetworkData, 
      fetchEtymologyTree, 
      setWordData, 
      setSelectedNode, 
      setWordHistory, 
      setCurrentHistoryIndex, 
      setError, 
      setWordNetwork, 
      setEtymologyTree,
      setIsDrawerOpen, setIsDrawerExpanded
  ]); // Ensure all dependencies are listed

  const handleNodeSelect = useCallback(async (word: string) => {
    if (!word) {
      console.error("Empty word received in handleNodeSelect");
      return;
    }

    console.log(`Node selected: ${word}`);
    
    try {
      setError(null);
      
      let wordData: WordInfo | null = null;
      
      try {
        wordData = await fetchWordDetails(word);
      } catch (error: any) {
        console.warn(`Failed to fetch details for selected word '${word}', error:`, error.message);
      }
      
      if (wordData) {
        console.log(`Selected word data retrieved:`, wordData);
        setSelectedNode(wordData.lemma);
        
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
    }
  }, [fetchEtymologyTree]);

  const handleSearch = useCallback(async (searchTerm: string | WordSuggestion | null) => {
    if (!searchTerm) return;

    let identifierToSearch: string;
    let displayLemma: string;

    if (typeof searchTerm === 'string') {
      // If user types and presses Enter, search by the text
      identifierToSearch = searchTerm.trim();
      displayLemma = identifierToSearch;
      console.log(`Searching by typed text: ${identifierToSearch}`);
    } else {
      // If user selects a suggestion object, use its ID
      // Ensure the suggestion object has an 'id' field from the API
      if (!searchTerm.id) {
        console.error("Selected suggestion is missing an ID:", searchTerm);
        setError("Selected suggestion is invalid. Please try again.");
        return;
      }
      identifierToSearch = `id:${searchTerm.id}`; // Use format "id:123"
      displayLemma = searchTerm.lemma;
      console.log(`Searching by selected ID: ${identifierToSearch}`);
    }

    if (!identifierToSearch) return;

    // Keep the input field showing just the lemma for better UX
    setInputValue(displayLemma);
    setSuggestions([]);
    setAutocompleteOpen(false);

    // Call handleNodeClick directly (which now opens the drawer)
    await handleNodeClick(identifierToSearch);

  }, [handleNodeClick]);

  const handleRandomWord = useCallback(async () => {
    setError(null);
    setIsRandomLoading(true);
    setWordData(null);
    setWordNetwork(null);
    setSelectedNode(null);
    setIsLoading(true);
    
    try {
      console.log("Fetching a single random word...");
      const randomWordResult = await getRandomWord();
      
      if (!randomWordResult || !randomWordResult.lemma) {
        throw new Error("Received invalid random word data from API.");
      }
      
      console.log("Random word received:", randomWordResult);

      const wordInfo: WordInfo = randomWordResult;
      
      setWordData(wordInfo);
      setSelectedNode(wordInfo.lemma);
      
      const historyEntry = { id: wordInfo.id, text: wordInfo.lemma };
      const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), historyEntry];
      setWordHistory(newHistory as any);
      setCurrentHistoryIndex(newHistory.length - 1);
      
      // *** Open the drawer ***
      setIsDrawerOpen(true);
      setIsDrawerExpanded(false);

      // Use ID for network fetch
      const networkIdentifier = `id:${wordInfo.id}`;
      fetchWordNetworkData(networkIdentifier, depth, breadth)
        .catch(err => console.error("Error fetching network data for random word:", err))
        .then(networkData => {
          if (networkData && networkData.nodes && networkData.edges) {
            setWordNetwork(networkData);
          }
        });
      
      if (wordInfo.id) {
        fetchEtymologyTree(wordInfo.id)
          .then(tree => {
            setEtymologyTree(tree);
          })
          .catch(err => console.error("Error fetching etymology tree for random word:", err));
      }
      
    } catch (error) {
      console.error("Error handling random word:", error);
      setError(error instanceof Error ? error.message : "Failed to get a random word");
    } finally {
        setIsRandomLoading(false);
      setIsLoading(false);
    }
  }, [
    depth, 
    breadth, 
    wordHistory, 
    currentHistoryIndex, 
    fetchWordNetworkData,
    fetchEtymologyTree,
    setSelectedNode,
    setWordHistory,
    setCurrentHistoryIndex,
    setError,
    setIsRandomLoading,
    setEtymologyTree,
    setIsDrawerOpen, setIsDrawerExpanded
  ]);

  const handleBack = useCallback(() => {
    if (currentHistoryIndex > 0) {
      const newIndex = currentHistoryIndex - 1;
      setCurrentHistoryIndex(newIndex);
      
      const previousWord = wordHistory[newIndex];
      console.log(`Navigating back to: ${JSON.stringify(previousWord)} (index ${newIndex})`);
      
      const wordText = typeof previousWord === 'string' 
        ? previousWord 
        : previousWord.text;
        
      // Determine the identifier for network fetch: prioritize ID format
      const networkIdentifier = typeof previousWord === 'object' && previousWord.id 
        ? `id:${previousWord.id}`
        : wordText; // Fallback to text/lemma for network fetch

      // Determine the identifier for details fetch (original logic: uses ID or lemma string)
      const detailsIdentifier = typeof previousWord === 'object' && previousWord.id
        ? `id:${previousWord.id}` // Use id: format for details too, as it handles both
        : wordText;
            
      setIsLoading(true);
      setError(null);
      setWordData(null);
      setWordNetwork(null);
      setSelectedNode(null);
      
      Promise.all([
        fetchWordDetails(detailsIdentifier), // Use ID or lemma string for details
        fetchWordNetworkData(networkIdentifier, depth, breadth) // Use ID format or lemma for network
      ])
      .then(([wordData, networkData]) => {
        setSelectedNode(wordData.lemma);
        setWordNetwork(networkData);
        
        // Sync network relations with word data
        if (networkData && networkData.nodes && networkData.edges && wordData) {
          const mainNode = networkData.nodes.find(node => 
            node.type === 'main' || node.word === wordText || node.label === wordText
          );
          
          if (mainNode) {
            const incomingRelations: Relation[] = [];
            const outgoingRelations: Relation[] = [];
            
            networkData.edges.forEach(edge => {
              const sourceNode = networkData.nodes.find(n => n.id === edge.source);
              const targetNode = networkData.nodes.find(n => n.id === edge.target);
              
              if (sourceNode && targetNode) {
                if (targetNode.id === mainNode.id) {
                  incomingRelations.push({
                    id: Math.floor(Math.random() * 1000000),
                    relation_type: edge.type,
                    source_word: {
                      id: Number(sourceNode.id) || 0,
                      lemma: sourceNode.word || sourceNode.label,
                      has_baybayin: sourceNode.has_baybayin,
                      baybayin_form: sourceNode.baybayin_form
                    }
                  });
                }
                else if (sourceNode.id === mainNode.id) {
                  outgoingRelations.push({
                    id: Math.floor(Math.random() * 1000000),
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
            
            console.log("Syncing relations during navigation:", {
              incoming: incomingRelations.length,
              outgoing: outgoingRelations.length
            });
            
            // Update word data with relations from network
            const updatedWordData = {
              ...wordData,
              incoming_relations: incomingRelations.length > 0 ? incomingRelations : wordData.incoming_relations,
              outgoing_relations: outgoingRelations.length > 0 ? outgoingRelations : wordData.outgoing_relations,
              semantic_network: {
                nodes: networkData.nodes || [],
                links: networkData.edges || [],
                mainWord: wordText
              }
            };
            
            setWordData(updatedWordData);
          } else {
            setWordData(wordData);
          }
        } else {
          setWordData(wordData);
        }
        
        // Fetch etymology tree
        if (wordData.id) {
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
        }
        
        setIsLoading(false);
      })
      .catch(error => {
        console.error("Error navigating back:", error);
        let errorMessage = "Failed to navigate back.";
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        setError(errorMessage);
        setIsLoading(false);
      });
    }
  }, [currentHistoryIndex, wordHistory, depth, breadth, fetchWordNetworkData, fetchWordDetails, fetchEtymologyTree]);

  const handleForward = useCallback(() => {
    if (currentHistoryIndex < wordHistory.length - 1) {
      const newIndex = currentHistoryIndex + 1;
      setCurrentHistoryIndex(newIndex);
      
      const nextWord = wordHistory[newIndex];
      console.log(`Navigating forward to: ${JSON.stringify(nextWord)} (index ${newIndex})`);
      
      const wordText = typeof nextWord === 'string' 
        ? nextWord 
        : nextWord.text;
        
      // Determine the identifier for network fetch: prioritize ID format
      const networkIdentifier = typeof nextWord === 'object' && nextWord.id
        ? `id:${nextWord.id}`
        : wordText; // Fallback to text/lemma for network fetch
        
      // Determine the identifier for details fetch (original logic: uses ID or lemma string)
      const detailsIdentifier = typeof nextWord === 'object' && nextWord.id
        ? `id:${nextWord.id}` // Use id: format for details too, as it handles both
        : wordText;

      setIsLoading(true);
      setError(null);
      setWordData(null);
      setWordNetwork(null);
      setSelectedNode(null);
      
      Promise.all([
        fetchWordDetails(detailsIdentifier), // Use ID or lemma string for details
        fetchWordNetworkData(networkIdentifier, depth, breadth) // Use ID format or lemma for network
      ])
      .then(([wordData, networkData]) => {
        setSelectedNode(wordData.lemma);
        setWordNetwork(networkData);
        
        // Sync network relations with word data
        if (networkData && networkData.nodes && networkData.edges && wordData) {
          const mainNode = networkData.nodes.find(node => 
            node.type === 'main' || node.word === wordText || node.label === wordText
          );
          
          if (mainNode) {
            const incomingRelations: Relation[] = [];
            const outgoingRelations: Relation[] = [];
            
            networkData.edges.forEach(edge => {
              const sourceNode = networkData.nodes.find(n => n.id === edge.source);
              const targetNode = networkData.nodes.find(n => n.id === edge.target);
              
              if (sourceNode && targetNode) {
                if (targetNode.id === mainNode.id) {
                  incomingRelations.push({
                    id: Math.floor(Math.random() * 1000000),
                    relation_type: edge.type,
                    source_word: {
                      id: Number(sourceNode.id) || 0,
                      lemma: sourceNode.word || sourceNode.label,
                      has_baybayin: sourceNode.has_baybayin,
                      baybayin_form: sourceNode.baybayin_form
                    }
                  });
                }
                else if (sourceNode.id === mainNode.id) {
                  outgoingRelations.push({
                    id: Math.floor(Math.random() * 1000000),
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
            
            console.log("Syncing relations during navigation:", {
              incoming: incomingRelations.length,
              outgoing: outgoingRelations.length
            });
            
            // Update word data with relations from network
            const updatedWordData = {
              ...wordData,
              incoming_relations: incomingRelations.length > 0 ? incomingRelations : wordData.incoming_relations,
              outgoing_relations: outgoingRelations.length > 0 ? outgoingRelations : wordData.outgoing_relations,
              semantic_network: {
                nodes: networkData.nodes || [],
                links: networkData.edges || [],
                mainWord: wordText
              }
            };
            
            setWordData(updatedWordData);
          } else {
            setWordData(wordData);
          }
        } else {
          setWordData(wordData);
        }
        
        // Fetch etymology tree
        if (wordData.id) {
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
        }
        
        setIsLoading(false);
      })
      .catch(error => {
        console.error("Error navigating forward:", error);
        let errorMessage = "Failed to navigate forward.";
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        setError(errorMessage);
        setIsLoading(false);
      });
    }
  }, [currentHistoryIndex, wordHistory, depth, breadth, fetchWordNetworkData, fetchWordDetails, fetchEtymologyTree]);

  const handleResetCircuitBreaker = () => {
    resetCircuitBreaker();
    setError(null);
    if (inputValue) {
      handleSearch(inputValue);
    }
  };

  const handleTestApiConnection = useCallback(async () => {
        setError(null);
    setApiConnected(null);
        
    try {
      console.log("Manually testing API connection...");
      
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
        
        const savedEndpoint = localStorage.getItem('successful_api_endpoint');
        if (savedEndpoint) {
          console.log('Loaded initial API endpoint from localStorage:', savedEndpoint);
        }
        
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

  // Define fetchStatistics *before* the useEffect that uses it
  const fetchStatistics = useCallback(async () => {
    try {
      const data = await getStatistics();
      console.log('Statistics data:', data);
    } catch (error) {
      console.error('Error fetching statistics:', error);
    }
  }, []); // Removed stable getStatistics dependency

  useEffect(() => {
    console.log("Checking API connection... 2");
    testApiConnection().then(connected => {
      setApiConnected(connected);
      if (connected) {
        const fetchInitialData = async () => {
          try {
            fetchStatistics();
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
  }, [fetchStatistics]); // Now fetchStatistics is declared before this hook

  const handleNetworkChange = useCallback((newDepth: number, newBreadth: number) => {
    setDepth(newDepth);
    setBreadth(newBreadth);
    if (selectedNode) {
      // Use word ID if available, otherwise fall back to selectedNode (lemma)
      const identifier = wordData?.id ? `id:${wordData.id}` : normalizeInput(selectedNode);
      setIsLoading(true);
      fetchWordNetworkData(identifier, newDepth, newBreadth)
        .then(networkData => {
          setWordNetwork(networkData);
        })
        .catch(error => {
          console.error("Error updating network:", error);
          setError("Failed to update network. Please try again.");
        });
    }
  }, [selectedNode, fetchWordNetworkData]);

  useEffect(() => {
    if (wordData && wordData.id) {
      const hasRelations = (
        (wordData.incoming_relations && wordData.incoming_relations.length > 0) || 
        (wordData.outgoing_relations && wordData.outgoing_relations.length > 0)
      );
      
      if (!hasRelations) {
        console.log(`Word ${wordData.lemma} (ID: ${wordData.id}) loaded, but no relations found in the initial details fetch. Relations might be fetched separately by the network graph or may not exist.`);
      }
    }
  }, [wordData]);

  useEffect(() => {
    try {
      const savedEndpoint = localStorage.getItem('successful_api_endpoint');
      if (savedEndpoint) {
        console.log('Loaded initial API endpoint from localStorage:', savedEndpoint);
      }
    } catch (e) {
      console.error('Error reading saved API endpoint from localStorage on mount:', e);
    }
  }, []);

  // Synchronize wordNetwork with wordData
  useEffect(() => {
    if (wordNetwork && wordData) {
      setWordData(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          semantic_network: {
            nodes: wordNetwork.nodes || [],
            links: wordNetwork.edges || [],
            mainWord: prev.lemma
          }
        };
      });
    }
  }, [wordNetwork, wordData?.id]);

  // Add handlers for drawer state
  const handleCloseDrawer = useCallback(() => {
    setIsDrawerOpen(false);
    setIsDrawerExpanded(false);
    setWordData(null); // Optionally clear data when drawer closes
    setSelectedNode(null);
  }, []);

  const handleToggleDrawerExpand = useCallback(() => {
    setIsDrawerExpanded(prev => !prev);
  }, []);

  const muiTheme = useMuiTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md')); // Check for mobile/tablet

  const renderSearchBar = () => {
    const muiTheme = useMuiTheme(); // Get the actual MUI theme object
    return (
      <Box className="search-container" sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%', padding: muiTheme.spacing(1, 2) }}>
         <Box className="nav-buttons" sx={{ display: 'flex', gap: 0.5 }}>
           <Button
             className="nav-button back-button"
             onClick={handleBack}
             disabled={currentHistoryIndex <= 0}
             title="Go back to previous word"
             variant="contained"
             sx={{
               minWidth: 36, width: 36, height: 36,
               borderRadius: '50%', p: 0,
               bgcolor: 'var(--button-color)', color: 'var(--button-text-color)', boxShadow: 'none',
               '&:hover': { bgcolor: 'var(--primary-color)' }, '&:active': { transform: 'scale(0.9)' },
               '&.Mui-disabled': { bgcolor: 'var(--card-border-color)', color: 'var(--text-color)', opacity: 0.6 }
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
               borderRadius: '50%', p: 0,
               bgcolor: 'var(--button-color)', color: 'var(--button-text-color)', boxShadow: 'none',
               '&:hover': { bgcolor: 'var(--primary-color)' }, '&:active': { transform: 'scale(0.9)' },
               '&.Mui-disabled': { bgcolor: 'var(--card-border-color)', color: 'var(--text-color)', opacity: 0.6 }
             }}
           >
             <span>→</span>
           </Button>
         </Box>

         <Box className="search-input-container" sx={{ flexGrow: 1, position: 'relative' }}>
           <Autocomplete<
             WordSuggestion | string,
             false, // Multiple
             false, // DisableClearable
             true   // FreeSolo
           >
             id="word-search-autocomplete"
             sx={{ width: '100%' }}
             open={autocompleteOpen}
             onOpen={() => setAutocompleteOpen(true)}
             onClose={() => setAutocompleteOpen(false)}
             isOptionEqualToValue={(option, value) => 
                typeof option === 'object' && typeof value === 'object' ? option.lemma === value.lemma : option === value
             }
             getOptionLabel={(option) => typeof option === 'string' ? option : option.lemma}
             options={suggestions}
             loading={isSuggestionsLoading}
             filterOptions={(x) => x}
             freeSolo
             autoComplete
             includeInputInList
             value={inputValue}
             onInputChange={(event, newInputValue, reason) => {
               setInputValue(newInputValue);
               if (reason === 'input' && newInputValue) {
                 fetchSuggestionsDebounced({ input: newInputValue }, (results) => {
                   setSuggestions(results);
                 });
               } else if (reason === 'clear' || !newInputValue) {
                 setSuggestions([]);
               }
             }}
             onChange={(event, newValue, reason) => {
               if ((reason === 'selectOption' || reason === 'createOption') && newValue) {
                 handleSearch(newValue);
               }
             }}
             renderInput={(params) => (
               <TextField
                 {...params}
                 placeholder="Search for a word..."
                 variant="outlined"
                 size="small"
                 className="search-input"
                 InputLabelProps={{ shrink: false, style: { display: 'none' } }}
                 InputProps={{
                   ...params.InputProps,
                   notched: false,
                   endAdornment: (
                     <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                       {isSuggestionsLoading && (
                         <CircularProgress color="inherit" size={20} />
                       )}
                       {params.InputProps.endAdornment}
                     </Box>
                   ),
                 }}
               />
             )}
             ListboxProps={{
                className: 'search-suggestions',
                sx: {
                    // Add styles here to mimic .search-suggestions padding, etc.
                    // Example:
                    // padding: '0',
                    // maxHeight: '300px'
                }
             }}
             slotProps={{
                popper: {
                    className: 'search-suggestions-popper',
                    sx: { zIndex: 1300 }
                },
                paper: {
                    className: 'search-suggestions-paper',
                    sx: {
                        // Add styles here to mimic .search-suggestions background, border, etc.
                    }
                },
                clearIndicator: {
                  sx: {
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    '& .MuiSvgIcon-root': {
                      fontSize: '1.1rem' 
                    },
                    marginRight: '6px',
                    marginBottom: '5px'
                  }
                }
             }}
             renderOption={(props, option) => {
               if (typeof option === 'string') {
                 return <li {...props}>{option}</li>; 
               }
               return (
                 <li {...props} key={option.lemma + option.language_code}>
                   <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                     <span>{option.lemma}</span>
                     <Chip 
                       label={option.language_code?.toUpperCase() || 'N/A'}
                       size="small" 
                       variant="outlined"
                       sx={{ 
                         ml: 1, 
                         height: 'auto',
                         fontSize: '0.7rem',
                         lineHeight: 1.2,
                         padding: '1px 4px'
                       }}
                     />
                   </Box>
                 </li>
               );
             }}
           />
         </Box>

         <Button
           variant="contained"
           className="search-button"
           onClick={() => inputValue.trim() && handleSearch(inputValue)}
           disabled={isLoading || !inputValue.trim()}
           title="Search for this word"
           sx={{
             height: '40px',
             mx: 0.1,
             whiteSpace: 'nowrap',
             bgcolor: 'var(--button-color)', color: 'var(--button-text-color)',
             borderRadius: '8px', boxShadow: 'none',
             '&:hover': { bgcolor: 'var(--primary-color)', boxShadow: 'none' }
           }}
         >
           {isLoading ? <CircularProgress size={20} color="inherit"/> : '🔍 Search'}
         </Button>

         <Button
           variant="contained"
           className="random-button"
           onClick={handleRandomWord}
           disabled={isRandomLoading || isLoading}
           title="Get a random word"
           sx={{
             height: '40px',
             mx: 0.1,
             whiteSpace: 'nowrap',
             bgcolor: 'var(--accent-color)', color: 'var(--button-text-color)',
             fontWeight: 'normal', borderRadius: '8px', boxShadow: 'none',
             '&:hover': { bgcolor: 'var(--secondary-color)', color: '#ffffff', boxShadow: 'none' }
           }}
         >
           {isRandomLoading ? <CircularProgress size={20} color="inherit"/> : '🎲 Random Word'}
         </Button>
      </Box>
    );
  };

  return (
    <div className={`word-explorer ${themeName} ${isLoading ? 'loading' : ''}`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <div className="header-buttons">
          {isDevMode() && (
            <>
              <button
                onClick={handleResetCircuitBreaker}
                className="debug-button"
                title="Reset API connection"
              >
                <span>🔄</span> Reset API
              </button>
              <button
                onClick={handleTestApiConnection}
                className="debug-button"
                title="Test API connection"
              >
                <span>🔌</span> Test API
              </button>
              <div className={`api-status ${
                (apiConnected === null ? 'checking' : (apiConnected ? 'connected' : 'disconnected'))
              }`}>
                {apiConnected === null ? <span>⏳</span> : 
                 apiConnected ? <span>✅</span> : <span>❌</span>}
                 API
              </div>
            </>
          )}
          <button
            onClick={toggleTheme}
            className="theme-toggle"
            aria-label="Toggle theme"
          >
            {themeName && themeName === "light" ? "🌙" : "☀️"}
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
      
      <main className="main-content-area">
        <div className="graph-view-container">
          {isLoading && <CircularProgress />}
          {error && <div className="error-message">{error}</div>}
          {!isLoading && !error && wordNetwork && (
            <WordGraph
              wordNetwork={wordNetwork}
              onNodeClick={handleNodeClick}
              mainWord={selectedNode}
              onNodeSelect={handleNodeSelect}
              onNetworkChange={handleNetworkChange}
              initialDepth={depth}
              initialBreadth={breadth}
            />
          )}
          {!isLoading && !wordNetwork && !error && (
            <div className="graph-placeholder">Enter a word or click Random to explore.</div>
          )}
        </div>

        {wordData && (
          <div 
             className={`details-drawer ${isDrawerOpen ? 'open' : ''} ${isDrawerExpanded ? 'expanded' : ''}`}
             ref={detailsContainerRef}
          >
            <div className="drawer-header" onClick={handleToggleDrawerExpand}>
              <div className="drawer-handle"></div>
            {wordData && (
              <div className="details-area-mobile" ref={detailsContainerRef}>
                 <WordDetails
                  wordInfo={wordData}
                  etymologyTree={etymologyTree}
                  isLoadingEtymology={isLoadingEtymology}
                  etymologyError={etymologyError}
                  onWordLinkClick={handleNodeClick}
                  onEtymologyNodeClick={handleNodeClick}
                />
              </div>
            )}
          </div>
        ) : (
          <PanelGroup direction="horizontal" className="explorer-panel-group">
            <Panel defaultSize={65} minSize={30} className="explorer-panel-main">
              <div className="explorer-content">
                {isLoading && <CircularProgress />}
                {error && <div className="error-message">{error}</div>}
                {!isLoading && !error && wordNetwork && (
                   <WordGraph
                     wordNetwork={wordNetwork}
                     onNodeClick={handleNodeClick}
                     mainWord={selectedNode}
                     onNodeSelect={handleNodeSelect}
                     onNetworkChange={handleNetworkChange}
                     initialDepth={depth}
                     initialBreadth={breadth}
                   />
                 )}
                 {!isLoading && !wordNetwork && !error && (
                   <div className="placeholder">Enter a word to explore its network.</div>
                 )}
              </div>
            </Panel>
            <PanelResizeHandle className="resize-handle" />
            <Panel defaultSize={35} minSize={25} className="explorer-panel-details">
               {wordData && (
                  <div className="details-content" ref={detailsContainerRef}>
                     <WordDetails
                       wordInfo={wordData}
                       etymologyTree={etymologyTree}
                       isLoadingEtymology={isLoadingEtymology}
                       etymologyError={etymologyError}
                       onWordLinkClick={handleNodeClick}
                       onEtymologyNodeClick={handleNodeClick}
                     />
                   </div>
               )}
               {!wordData && !isLoading && (
                <div className="details-placeholder">Select a node or search a word to see details.</div>
               )}
            </Panel>
          </PanelGroup>
        )}
      </main>
      
      <footer className="footer">
        © {new Date().getFullYear()} Filipino Root Word Explorer. All Rights Reserved.
      </footer>
    </div>
  );
};

export default WordExplorer;
