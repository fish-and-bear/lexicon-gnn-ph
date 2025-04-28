import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useAppTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchOptions, EtymologyTree, Statistics, SearchResultItem, Relation, WordSuggestion, BasicWord } from "../types";
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
import { Button } from "@mui/material";
import { Panel, PanelGroup, PanelResizeHandle, ImperativePanelHandle } from "react-resizable-panels";
import useMediaQuery from '@mui/material/useMediaQuery';
import CircularProgress from '@mui/material/CircularProgress';
import { Theme, useTheme as useMuiTheme } from '@mui/material/styles';
import { Box } from "@mui/material";
import { Autocomplete, TextField } from "@mui/material";
import { debounce } from '@mui/material/utils';
import Chip from '@mui/material/Chip';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import { alpha } from '@mui/material/styles';

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
  const [isLoadingNetwork, setIsLoadingNetwork] = useState(false);
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState(false);
  const [isRandomLoading, setIsRandomLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const { themeMode, toggleTheme } = useAppTheme();
  const [inputValue, setInputValue] = useState<string>("");
  const [depth, setDepth] = useState<number>(2);
  const [breadth, setBreadth] = useState<number>(10);
  // Store history with associated graph settings
  type HistoryEntry = { identifier: string; lemma: string; depth: number; breadth: number };
  const [wordHistory, setWordHistory] = useState<HistoryEntry[]>([]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState<number>(-1);
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState<boolean>(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  const randomWordTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const detailsContainerRef = useRef<ImperativePanelHandle>(null);

  const [autocompleteOpen, setAutocompleteOpen] = useState(false);
  const [suggestions, setSuggestions] = useState<WordSuggestion[]>([]);
  const [isSuggestionsLoading, setIsSuggestionsLoading] = useState(false);

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
    // Add null check
    if (detailsContainerRef.current) {
      // Check if the ref has the getSize method (likely from ImperativePanelHandle)
      // before trying to access style on a potential underlying element
      if (typeof detailsContainerRef.current.getSize === 'function') {
          // We can't reliably get the underlying DOM node's style this way.
          // If resizing needs to be restored, it should likely use Panel API methods.
          // For now, just log that we have the handle.
          console.log("Panel ref is available, but restoring width via direct style access is unreliable.");
          // const savedWidth = localStorage.getItem('wordDetailsWidth');
          // if (savedWidth && !isNaN(parseFloat(savedWidth))) {
          //   // This won't work reliably: detailsContainerRef.current.style.width = `${savedWidth}px`;
          // }
      }
    }
  }, []);

  useEffect(() => {
    // Keep the null check
    if (!detailsContainerRef.current) return;
    
    // We need the actual DOM element to observe, not the Panel handle.
    // Let's try to find the element using the Panel's ID.
    // Note: This relies on the Panel rendering a DOM element with this ID.
    const panelElement = document.getElementById("details-panel");

    if (!panelElement) {
      console.warn("Could not find details panel element for ResizeObserver");
      return; // Exit if element not found
    }
    
    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        if (entry.target === panelElement) {
          const newWidth = entry.contentRect.width;
          // Persist width in localStorage (using size percentage might be better)
          localStorage.setItem('wordDetailsWidthPercent', detailsContainerRef.current?.getSize().toString() ?? '40'); 
        }
      }
    });
    
    resizeObserver.observe(panelElement);
    
    return () => {
        // Check if panelElement was found before trying to unobserve
        if (panelElement) {
          resizeObserver.unobserve(panelElement);
        }
    };
  }, []); // Dependencies remain empty

  const normalizeInput = (input: string) => unidecode(input.trim().toLowerCase());

  const fetchWordNetworkData = useCallback(async (word: string, depthParam: number = 2, breadthParam: number = 10) => {
    try {
      setIsLoadingNetwork(true);
      setError(null);
      
      console.log('Fetching word network data for:', word);
      const data = await fetchWordNetwork(word, { 
        depth: depthParam,
        breadth: breadthParam,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.3
      });
      
      if (data && data.nodes && data.links) {
        console.log('Word network data received:', data);
        console.log(`Network has ${data.nodes.length} nodes and ${data.links.length} links`);
        setWordNetwork(data);
        
        if (wordData && wordData.id) {
          const mainNode = data.nodes.find(node => 
            node.type === 'main' || node.word === word || node.label === word
          );
          
          if (mainNode) {
            const incomingRelations: Relation[] = [];
            const outgoingRelations: Relation[] = [];
            
            data.links.forEach(link => {
              const sourceNode = data.nodes.find(n => n.id === link.source);
              const targetNode = data.nodes.find(n => n.id === link.target);
              
              if (sourceNode && targetNode) {
                if (targetNode.id === mainNode.id) {
                  incomingRelations.push({
                    id: Math.floor(Math.random() * 1000000),
                    relation_type: link.type,
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
                    relation_type: link.type,
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
      setIsLoadingNetwork(false);
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
    setIsLoadingDetails(true);
    setIsLoadingNetwork(true);

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
        // or if the ID fetch resulted in a specific known error (like not found
        // or the specific database error)
        if (!isIdSearch || error.message.includes('not found') || error.message.includes('Server database error')) { 
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

              if (searchResults.results && searchResults.results.length > 0) {
                console.log(`Fallback search successful`);
                const firstResult = searchResults.results[0];
                // Fetch details using the ID from search result
                wordData = await fetchWordDetails(`id:${firstResult.word_id}`);
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

        // Use ID identifier if available, otherwise lemma
        const historyIdentifier = wordData.id ? `id:${wordData.id}` : wordData.lemma;

        // Check if the *identifier* already exists at the current index + 1 to avoid duplicates on re-click
        const nextHistoryEntry = wordHistory[currentHistoryIndex + 1];
        if (!nextHistoryEntry || nextHistoryEntry.identifier !== historyIdentifier) {
          const newHistoryEntry: HistoryEntry = { 
            identifier: historyIdentifier, 
            lemma: wordData.lemma,
            depth: depth, // Store current depth
            breadth: breadth // Store current breadth
          };
          const newHistory = [
            ...wordHistory.slice(0, currentHistoryIndex + 1), 
            newHistoryEntry
          ];
          setWordHistory(newHistory as any);
          setCurrentHistoryIndex(newHistory.length - 1);
        }
        
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
      setIsLoadingDetails(false);
      setIsLoadingNetwork(false);
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
      setEtymologyTree
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

  const handleSearch = useCallback(async (searchTerm: string | WordSuggestion | BasicWord | null) => {
    if (!searchTerm) return;

    let identifierToSearch: string;
    let displayLemma: string;

    if (typeof searchTerm === 'string') {
      // If user types and presses Enter, search by the text
      identifierToSearch = searchTerm.trim();
      displayLemma = identifierToSearch;
      console.log(`Searching by typed text: ${identifierToSearch}`);
    } else {
      // Prefer ID if available (WordSuggestion or BasicWord might have it)
      if (searchTerm.id) {
          identifierToSearch = `id:${searchTerm.id}`; // Use format "id:123"
          displayLemma = searchTerm.lemma; // Use lemma for display
          console.log(`Searching by selected ID: ${identifierToSearch} (lemma: ${displayLemma})`);
      } else if (searchTerm.lemma) {
          // Fallback to lemma if no ID (less common for BasicWord/WordSuggestion)
          identifierToSearch = searchTerm.lemma;
          displayLemma = searchTerm.lemma;
          console.log(`Searching by selected Lemma (no ID found): ${identifierToSearch}`);
      } else {
         console.error("Selected suggestion/word object is missing ID and Lemma:", searchTerm);
         setError("Selected item is invalid. Please try again.");
         return;
      }
    }

    if (!identifierToSearch) return;

    // Keep the input field showing just the lemma for better UX
    setInputValue(displayLemma);
    setSuggestions([]);
    setAutocompleteOpen(false);

    // Call handleNodeClick directly
    await handleNodeClick(identifierToSearch);

  }, [handleNodeClick]); // Add handleNodeClick to dependency array

  const handleRandomWord = useCallback(async () => {
    setError(null);
    setIsRandomLoading(true);
    setIsLoadingDetails(true); // Start details loading indicator
    // Don't clear wordNetwork immediately, wait for details fetch
    // setWordNetwork(null);
    // setSelectedNode(null); 

    try {
      console.log("Fetching a single random word...");
      // Fetch details first
      const randomWordResult = await getRandomWord();

      if (!randomWordResult || !randomWordResult.lemma) {
        throw new Error("Received invalid random word data from API.");
      }
      console.log("Random word received:", randomWordResult);
      const wordInfo: WordInfo = randomWordResult;

      // Update details and selected node
      setWordData(wordInfo);
      setSelectedNode(wordInfo.lemma);
      setIsLoadingDetails(false); // Stop details loading indicator

      // Update history
      const networkIdentifier = `id:${wordInfo.id}`; // Use ID for history/network fetch
      const historyEntry: HistoryEntry = { 
        identifier: networkIdentifier, 
        lemma: wordInfo.lemma,
        depth: depth, // Store current depth
        breadth: breadth // Store current breadth
      };
      const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), historyEntry];
      setWordHistory(newHistory as any);
      setCurrentHistoryIndex(newHistory.length - 1);

      // NOW start fetching network and etymology in the background
      setIsLoadingNetwork(true); // Start network loading indicator
      setWordNetwork(null); // Clear previous network now
      setEtymologyTree(null);

      // Fetch network data (don't await)
      fetchWordNetworkData(networkIdentifier, depth, breadth)
        .then(networkData => {
          if (networkData && networkData.nodes && networkData.links) {
            setWordNetwork(networkData);
          }
        })
        .catch(err => {
          console.error("Error fetching network data for random word:", err);
          setError("Failed to load word network. Please try again.");
          setWordNetwork(null); // Ensure network is null on error
        })
        .finally(() => {
          setIsLoadingNetwork(false); // Stop network loading indicator regardless of outcome
        });

      // Fetch etymology tree (don't await)
      if (wordInfo.id) {
        setIsLoadingEtymology(true); // Start etymology loading
        fetchEtymologyTree(wordInfo.id)
          .then(tree => {
            setEtymologyTree(tree);
          })
          .catch(err => {
            console.error("Error fetching etymology tree for random word:", err);
            setEtymologyError("Failed to load etymology.");
            setEtymologyTree(null);
          })
          .finally(() => {
            setIsLoadingEtymology(false); // Stop etymology loading
          });
      } else {
        setEtymologyTree(null);
      }

    } catch (error) {
      console.error("Error handling random word:", error);
      setError(error instanceof Error ? error.message : "Failed to get a random word");
      setWordData(null); // Clear data on error
      setWordNetwork(null);
      setSelectedNode(null);
      setIsLoadingDetails(false); // Ensure details loading stops on error
      setIsLoadingNetwork(false); // Ensure network loading stops on error
      setIsLoadingEtymology(false); // Ensure etymology loading stops on error
    } finally {
        setIsRandomLoading(false); // Stop the main random button loading indicator
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
    setEtymologyTree
  ]);

  const handleBack = useCallback(() => {
    if (currentHistoryIndex > 0) {
      const newIndex = currentHistoryIndex - 1;
      setCurrentHistoryIndex(newIndex);

      const previousWord = wordHistory[newIndex];
      // Restore settings from history
      setDepth(previousWord.depth);
      setBreadth(previousWord.breadth);
      // Set selected node *immediately* for better visual sync during load
      setSelectedNode(previousWord.lemma);

      console.log(`Navigating back to: ${JSON.stringify(previousWord)} (index ${newIndex})`);
      
      // Use the stored identifier and settings
      const detailsIdentifier = previousWord.identifier;
      const networkIdentifier = previousWord.identifier;
      const historyDepth = previousWord.depth;
      const historyBreadth = previousWord.breadth;

      setIsLoadingDetails(true);
      setIsLoadingNetwork(true);
      setError(null);
      setWordData(null);
      setWordNetwork(null);
      
      Promise.all([
        fetchWordDetails(detailsIdentifier), // Use ID or lemma string for details
        fetchWordNetworkData(networkIdentifier, historyDepth, historyBreadth) // Use restored settings
      ])
      .then(([wordData, networkData]) => {
        setWordNetwork(networkData);
        
        // Sync network relations with word data
        if (networkData && networkData.nodes && networkData.links && wordData) {
          const mainNode = networkData.nodes.find(node => 
            node.type === 'main' || node.word === detailsIdentifier || node.label === detailsIdentifier
          );
          
          if (mainNode) {
            const incomingRelations: Relation[] = [];
            const outgoingRelations: Relation[] = [];
            
            networkData.links.forEach(link => {
              const sourceNode = networkData.nodes.find(n => n.id === link.source);
              const targetNode = networkData.nodes.find(n => n.id === link.target);
              
              if (sourceNode && targetNode) {
                if (targetNode.id === mainNode.id) {
                  incomingRelations.push({
                    id: Math.floor(Math.random() * 1000000),
                    relation_type: link.type,
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
                    relation_type: link.type,
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
        
        setIsLoadingDetails(false);
        setIsLoadingNetwork(false);
      })
      .catch(error => {
        console.error("Error navigating back:", error);
        let errorMessage = "Failed to navigate back.";
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        setError(errorMessage);
        setIsLoadingDetails(false);
        setIsLoadingNetwork(false);
      });
    }
  }, [currentHistoryIndex, wordHistory, fetchWordNetworkData, fetchWordDetails, fetchEtymologyTree]);

  const handleForward = useCallback(() => {
    if (currentHistoryIndex < wordHistory.length - 1) {
      const newIndex = currentHistoryIndex + 1;
      setCurrentHistoryIndex(newIndex);

      const nextWord = wordHistory[newIndex];
      // Restore settings from history
      setDepth(nextWord.depth);
      setBreadth(nextWord.breadth);
      // Set selected node *immediately* for better visual sync during load
      setSelectedNode(nextWord.lemma);

      console.log(`Navigating forward to: ${JSON.stringify(nextWord)} (index ${newIndex})`);
      
      // Use the stored identifier and settings
      const detailsIdentifier = nextWord.identifier;
      const networkIdentifier = nextWord.identifier;
      const historyDepth = nextWord.depth;
      const historyBreadth = nextWord.breadth;

      setIsLoadingDetails(true);
      setIsLoadingNetwork(true);
      setError(null);
      setWordData(null);
      setWordNetwork(null);
      
      Promise.all([
        fetchWordDetails(detailsIdentifier), // Use ID or lemma string for details
        fetchWordNetworkData(networkIdentifier, historyDepth, historyBreadth) // Use restored settings
      ])
      .then(([wordData, networkData]) => {
        setWordNetwork(networkData);
        
        // Sync network relations with word data
        if (networkData && networkData.nodes && networkData.links && wordData) {
          const mainNode = networkData.nodes.find(node => 
            node.type === 'main' || node.word === detailsIdentifier || node.label === detailsIdentifier
          );
          
          if (mainNode) {
            const incomingRelations: Relation[] = [];
            const outgoingRelations: Relation[] = [];
            
            networkData.links.forEach(link => {
              const sourceNode = networkData.nodes.find(n => n.id === link.source);
              const targetNode = networkData.nodes.find(n => n.id === link.target);
              
              if (sourceNode && targetNode) {
                if (targetNode.id === mainNode.id) {
                  incomingRelations.push({
                    id: Math.floor(Math.random() * 1000000),
                    relation_type: link.type,
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
                    relation_type: link.type,
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
        
        setIsLoadingDetails(false);
        setIsLoadingNetwork(false);
      })
      .catch(error => {
        console.error("Error navigating forward:", error);
        let errorMessage = "Failed to navigate forward.";
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        setError(errorMessage);
        setIsLoadingDetails(false);
        setIsLoadingNetwork(false);
      });
    }
  }, [currentHistoryIndex, wordHistory, fetchWordNetworkData, fetchWordDetails, fetchEtymologyTree]);

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
      setIsLoadingDetails(true);
      setIsLoadingNetwork(true);
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

  const muiTheme = useMuiTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md')); // Check for mobile/tablet

  const renderSearchBar = () => {
    const muiTheme = useMuiTheme(); // Get the actual MUI theme object
    return (
      <Box className="search-container-desktop" sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%', padding: muiTheme.spacing(1, 2), borderBottom: '1px solid var(--card-border-color)' }}>
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
           disabled={isLoadingDetails || isLoadingNetwork || !inputValue.trim()}
           title="Search for this word"
           sx={{
             height: '40px',
             ml: 0.5,
             whiteSpace: 'nowrap',
             bgcolor: 'var(--button-color)', color: 'var(--button-text-color)',
             borderRadius: '8px', boxShadow: 'none',
             '&:hover': { bgcolor: 'var(--primary-color)', boxShadow: 'none' }
           }}
         >
           {(isLoadingDetails || isLoadingNetwork) ? <CircularProgress size={20} color="inherit"/> : '🔍 Search'}
         </Button>

         <Button
           variant="contained"
           className="random-button"
           onClick={handleRandomWord}
           disabled={isRandomLoading || isLoadingDetails || isLoadingNetwork}
           title="Get a random word"
           sx={{
             height: '40px',
             ml: 0.5,
             whiteSpace: 'nowrap',
             bgcolor: 'var(--accent-color)', 
             color: 'var(--button-text-color)',
             fontWeight: 'normal', 
             borderRadius: '8px', 
             boxShadow: 'none',
             transition: 'background-color 0.2s ease-in-out, transform 0.1s ease-out',
             '&:hover': {
               bgcolor: themeMode === 'dark' ? 'var(--secondary-color)' : alpha(muiTheme.palette.warning.dark, 0.9),
               color: '#ffffff',
               boxShadow: 'none' 
             },
             '&:active': {
               transform: 'scale(0.95)'
             },
             '&.Mui-disabled': {
                bgcolor: 'action.disabledBackground',
                color: 'action.disabled'
             }
           }}
         >
           {isRandomLoading ? <CircularProgress size={20} color="inherit"/> : '🎲 Random Word'}
         </Button>
      </Box>
    );
  };

  // New function to render the mobile header using AppBar/Toolbar
  const renderMobileHeader = () => {
    return (
      <AppBar position="static" color="default" elevation={1} sx={{ pt: 0, pb: 0 }}>
        <Toolbar variant="dense" sx={{ minHeight: 48, py: 0.5 }}>
          {/* Optional: Add a title or back button */}
          <Typography variant="h6" sx={{ flexGrow: 1, display: { xs: 'none', sm: 'block' }, fontSize: '1rem' }}>
             Filipino Root Explorer
          </Typography>

          {/* Search Autocomplete */}
          <Box sx={{ flexGrow: 0.9, flexShrink: 1, minWidth: '100px', px: theme => theme.spacing(1) }}> {/* Reduce flexGrow slightly */}
            <Autocomplete<
              WordSuggestion | string,
              false, // Multiple
              false, // DisableClearable
              true   // FreeSolo
            >
              id="word-search-autocomplete-mobile"
              size="small" // Make it smaller for mobile
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
                  placeholder="Search..." // Shorter placeholder
                  variant="outlined"
                  className="search-input-mobile" // Add specific class
                  InputLabelProps={{ shrink: false, style: { display: 'none' } }}
                  InputProps={{
                    ...params.InputProps,
                    notched: false,
                    endAdornment: (
                      <React.Fragment>
                        {isSuggestionsLoading ? <CircularProgress color="inherit" size={20} /> : null}
                        {params.InputProps.endAdornment}
                      </React.Fragment>
                    ),
                  }}
                />
              )}
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
          </Box> {/* Added missing closing tag */}

          {/* Action Buttons */}
          <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 1, flexGrow: 0 }}> {/* Ensure buttons don't grow but CAN shrink */}
            <Button
              variant="contained"
              size="small" // Smaller button
              className="search-button-mobile" // Add specific class
              onClick={() => inputValue.trim() && handleSearch(inputValue)}
              disabled={isLoadingDetails || isLoadingNetwork || !inputValue.trim()}
              title="Search"
              sx={{ 
                minWidth: 'auto', px: 1, // Allow shrinking
                height: 38, // Match TextField height
                bgcolor: 'var(--button-color)', // Use theme variable
                color: 'var(--button-text-color)', // Use theme variable
                '&:hover': { 
                  bgcolor: 'var(--primary-color)' // Use theme variable for hover
                },
                '&.Mui-disabled': { // Keep consistent disabled style
                  bgcolor: themeMode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)', // Use themeMode
                  color: themeMode === 'dark' ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.26)', // Use themeMode
                }
              }}
            >
              {(isLoadingDetails || isLoadingNetwork) ? <CircularProgress size={20} color="inherit" /> : '🔍'} {/* Icon only */}
            </Button>
            <Button
              variant="contained"
              size="small" // Smaller button
              className="random-button-mobile" // Add specific class
              onClick={handleRandomWord}
              disabled={isRandomLoading || isLoadingDetails || isLoadingNetwork}
              title="Random Word"
              sx={{ 
                minWidth: 'auto', px: 1, // Allow shrinking
                height: 38, // Match TextField height
                bgcolor: 'var(--accent-color)', 
                color: 'var(--button-text-color)',
                transition: 'background-color 0.2s ease-in-out, transform 0.1s ease-out',
                '&:hover': { 
                  bgcolor: themeMode === 'dark' ? 'var(--secondary-color)' : alpha(muiTheme.palette.warning.dark, 0.9) // Use themeMode
                },
                '&:active': {
                  transform: 'scale(0.95)'
                },
                '&.Mui-disabled': {
                  bgcolor: 'action.disabledBackground',
                  color: 'action.disabled'
                }
              }}
            >
              {isRandomLoading ? <CircularProgress size={20} color="inherit" /> : '🎲'} {/* Icon only */}
            </Button>
          </Box>
        </Toolbar>
      </AppBar>
    );
  };

  return (
    <div className={`word-explorer ${themeMode} ${(isLoadingDetails || isLoadingNetwork) ? 'loading' : ''}`}>
      <header className="header-content" style={{ padding: isMobile ? '0.5rem 0.8rem' : '1rem 1.5rem' }}>
        <h1 style={{ fontSize: isMobile ? '1.1rem' : '1.5rem' }}>Filipino Root Word Explorer</h1>
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
            title="Toggle theme"
          >
            {themeMode === "light" ? "🌙" : "☀️"}
          </button>
        </div>
      </header>
      
      {/* Conditionally render search bars */}
      {!isMobile && renderSearchBar()} 
      {isMobile && renderMobileHeader()}
      
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
      
      <main style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}> {/* Ensure main takes remaining height and contains flex children */}
        {/* Remove the conditional rendering and the Box-based mobile layout */}
        {/* Always render PanelGroup, adjust direction based on isMobile */}
        <PanelGroup 
          direction={isMobile ? "vertical" : "horizontal"} 
          className="explorer-panel-group" 
          style={{ flexGrow: 1, minHeight: 0 }}
        >
            <Panel minSize={25} className="explorer-panel-main">
              <div className="explorer-content" style={{ height: '100%', width: '100%', overflow: 'hidden' }}> {/* Ensure graph area fills panel */}
                {/* Graph rendering logic - remains the same */}
                {isLoadingNetwork && !wordNetwork && !error && <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}><CircularProgress /></Box>}
                {error && !isLoadingDetails && !isLoadingNetwork && <div className="error-message">{error}</div>}
                {wordNetwork && !error && (
                   <WordGraph
                     wordNetwork={wordNetwork}
                     onNodeClick={handleNodeClick}
                     mainWord={selectedNode}
                     isMobile={isMobile} // Pass isMobile to children if they need it
                     onNetworkChange={handleNetworkChange}
                     initialDepth={depth}
                     initialBreadth={breadth}
                     isLoading={isLoadingNetwork}
                   />
                 )}
                 {/* Placeholders/Loading - remains the same */}
                 {!wordNetwork && !isLoadingDetails && !isLoadingNetwork && !error && (
                  <div className="details-placeholder">Select a node or search a word to see details.</div>
                 )}
                 {isLoadingDetails && !wordData && !error && <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}><CircularProgress /></Box>}
              </div>
            </Panel>
            <PanelResizeHandle className={`resize-handle ${isMobile ? 'vertical' : 'horizontal'}`} /> {/* Add class for potentially different vertical/horizontal handle styling */}
            <Panel 
              minSize={20} 
              id="details-panel" // ID used by ResizeObserver effect
              className="details-panel-container" 
              ref={detailsContainerRef} // Restore ref for Panel handle
              order={2}
              // Adjust default size based on orientation
              defaultSize={isMobile ? 50 : parseInt(localStorage.getItem('wordDetailsWidthPercent') || '40', 10)}
            >
               <div 
                 style={{ 
                   height: '100%', 
                   width: '100%', 
                   overflowY: 'auto', 
                   position: 'relative' 
                 }}
                 className="details-content"
               >
                 {/* WordDetails rendering - remains the same */}
                 {wordData && (
                     <WordDetails
                       wordData={wordData}
                       isLoading={isLoadingDetails}
                       semanticNetworkData={wordNetwork}
                       etymologyTree={etymologyTree}
                       isLoadingEtymology={isLoadingEtymology}
                       etymologyError={etymologyError}
                       onFetchEtymology={fetchEtymologyTree}
                       onWordClick={handleSearch}
                       isMobile={isMobile} // Pass isMobile to children if they need it
                     />
                 )}
                 {/* Placeholders/Loading - remains the same */}
                 {!wordData && !isLoadingDetails && !isLoadingNetwork && !error && (
                  <div className="details-placeholder">Select a node or search a word to see details.</div>
                 )}
                 {isLoadingDetails && !wordData && <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}><CircularProgress /></Box>}
               </div>
            </Panel>
          </PanelGroup>
      </main>
      
      <footer className="footer">
        © {new Date().getFullYear()} Filipino Root Word Explorer. All Rights Reserved.
      </footer>
    </div>
  );
};

export default WordExplorer;
