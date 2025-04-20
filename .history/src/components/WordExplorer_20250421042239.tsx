import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchOptions, EtymologyTree, Statistics, SearchWordResult, Relation } from "../types";
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
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import useMediaQuery from '@mui/material/useMediaQuery';
import CircularProgress from '@mui/material/CircularProgress';
import { Theme, useTheme as useMuiTheme } from '@mui/material/styles';
import { Box } from "@mui/material";
import { Autocomplete, TextField } from "@mui/material";
import { debounce } from '@mui/material/utils';
import Chip from '@mui/material/Chip';

const isDevMode = () => {
  // Check if we have a URL parameter for showing debug tools
  if (typeof window !== 'undefined') {
    return window.location.href.includes('debug=true') ||
           window.location.hostname === 'localhost' ||
           window.location.hostname.includes('127.0.0.1');
  }
  return false;
};

// Define the suggestion object type locally or import if defined elsewhere
interface WordSuggestion {
  lemma: string;
  language_code: string;
}

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

  const handleNodeClickRef = useRef<(id: string) => Promise<void>>(null as unknown as (id: string) => Promise<void>);
  
  const handleNodeClick = useCallback(async (word: string) => {
    if (!word) {
      console.error("Empty word received in handleNodeClick");
      return;
    }

    console.log(`Node clicked: ${word}`);
    setError(null);
    setIsLoading(true);

    try {
      let wordData: WordInfo | null = null;
      let fallbackToSearch = false;

      try {
        wordData = await fetchWordDetails(word);
      } catch (error: any) {
        console.warn(`Failed to fetch details for word '${word}', error:`, error.message);
        fallbackToSearch = true;
        
        if (error.message.includes('not found') || 
            error.message.includes('Database error') ||
            error.message.includes('dictionary update sequence') ||
            error.message.includes('Server database error') ||
            error.message.includes('unexpected error')) {
          console.log(`Falling back to search for word: ${word}`);
          
          let searchText;
          if (word.startsWith('id:')) {
            const wordId = word.substring(3);
            try {
              const idSearchResults = await searchWords(`id:${wordId}`, {
                page: 1,
                per_page: 5,
                mode: 'all',
                sort: 'relevance'
              });
              
              if (idSearchResults.words && idSearchResults.words.length > 0) {
                console.log(`Search by ID successful, found ${idSearchResults.words.length} results`);
                const firstResult = idSearchResults.words[0];
                wordData = await fetchWordDetails(
                  String(firstResult.id).startsWith('id:') ? 
                  String(firstResult.id) : 
                  `id:${firstResult.id}`
                );
                
                fallbackToSearch = false;
              }
            } catch (idSearchError) {
              console.warn(`ID-based search failed, trying word search:`, idSearchError);
            }
            
            if (fallbackToSearch) {
              searchText = wordId;
            }
          } else {
            searchText = word;
          }
          
          if (fallbackToSearch && searchText) {
            const searchResults = await searchWords(searchText, {
              page: 1,
              per_page: 10,
              mode: 'all',
              sort: 'relevance',
              language: ''
            });
            
            if (searchResults.words && searchResults.words.length > 0) {
              console.log(`Search successful, found ${searchResults.words.length} results`);
              
              setSearchResults(searchResults.words);
              
              const firstResult = searchResults.words[0];
              try {
                wordData = await fetchWordDetails(
                  String(firstResult.id).startsWith('id:') ? 
                  String(firstResult.id) : 
                  `id:${firstResult.id}`
                );
                fallbackToSearch = false;
              } catch (detailError) {
                console.error(`Failed to fetch details for search result:`, detailError);
                
                setError(`Could not load full details for "${searchText}". Please select one of the search results below.`);
                setIsLoading(false);
                return;
              }
            } else {
              throw new Error(`Word '${searchText}' not found. Please try a different word.`);
            }
          }
        } else {
          throw error;
        }
      }

      if (!fallbackToSearch && wordData) {
        console.log(`Word data retrieved successfully:`, wordData);
        setWordData(wordData);
        setSelectedNode(wordData.lemma);
        
        const wordId = String(wordData.id);
        if (!wordHistory.some(w => typeof w === 'object' && 'id' in w && String(w.id) === wordId)) {
          const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), { id: wordData.id, text: wordData.lemma }];
          setWordHistory(newHistory as any);
          setCurrentHistoryIndex(newHistory.length - 1);
        }

        const currentNetworkWordId = depth && breadth ? wordData.id : null;
        if (wordData.id !== currentNetworkWordId) {
          setDepth(2);
          setBreadth(10);
        }

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
        
        try {
          console.log(`Fetching network for new main word: ${wordData.lemma}`);
          const networkData = await fetchWordNetworkData(wordData.lemma, depth, breadth);
          setWordNetwork(networkData);
        } catch (networkError) {
          console.error("Error fetching word network:", networkError);
        }
      }
    } catch (error: any) {
      console.error(`Error in handleNodeClick for word '${word}':`, error);
      
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
    } finally {
      setIsLoading(false);
    }
  }, [wordHistory, depth, breadth, setDepth, setBreadth, fetchWordDetails, searchWords, fetchWordNetworkData, fetchEtymologyTree]);

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

  useEffect(() => {
    handleNodeClickRef.current = handleNodeClick;
  }, [handleNodeClick]);

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

    // Call handleNodeClick with the specific identifier (text or "id:...")
    await handleNodeClickRef.current(identifierToSearch);

  }, [handleNodeClickRef]); // Keep dependency minimal

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
      
      fetchWordNetworkData(wordInfo.lemma, depth, breadth)
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
    setEtymologyTree
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
        
      const wordId = typeof previousWord === 'string'
        ? previousWord
        : previousWord.id.toString();
      
      setIsLoading(true);
      setError(null);
      setWordData(null);
      setWordNetwork(null);
      setSelectedNode(null);
      
      Promise.all([
        fetchWordDetails(wordId),
        fetchWordNetworkData(wordText, depth, breadth)
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
        
      const wordId = typeof nextWord === 'string'
        ? nextWord
        : nextWord.id.toString();
      
      setIsLoading(true);
      setError(null);
      setWordData(null);
      setWordNetwork(null);
      setSelectedNode(null);
      
      Promise.all([
        fetchWordDetails(wordId),
        fetchWordNetworkData(wordText, depth, breadth)
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
      setIsLoading(true);
      fetchWordNetworkData(normalizeInput(selectedNode), newDepth, newBreadth)
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

  const isWideLayout = useMediaQuery('(min-width:769px)'); 

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
      
      <main>
        {isWideLayout ? (
          <div style={{ width: '100%', height: '100%', display: 'flex', overflow: 'hidden' }}>
            <PanelGroup direction="horizontal" autoSaveId="wordExplorerLayout" style={{ width: '100%', height: '100%', display: 'flex' }}>
              <Panel defaultSize={60} minSize={30} style={{ overflow: 'hidden', height: '100%' }}>
                <div className="graph-container" style={{ width: '100%', height: '100%' }}>
                  <div className="graph-content" style={{ width: '100%', height: '100%' }}>
                    {isLoading && <div className="loading">Loading Network...</div>}
                    {!isLoading && wordNetwork && (
                      <WordGraph
                        wordNetwork={wordNetwork}
                        mainWord={selectedNode}
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
                <div ref={detailsContainerRef} className="details-container" style={{ width: '100%', height: '100%', overflow: 'auto' }}>
                  {isLoading && <div className="loading-spinner">Loading Details...</div>} 
                  {!isLoading && wordData && (
                    <WordDetails 
                      wordInfo={wordData} 
                      etymologyTree={etymologyTree}
                      isLoadingEtymology={isLoadingEtymology}
                      etymologyError={etymologyError}
                      onWordLinkClick={handleNodeClick}
                      onEtymologyNodeClick={handleNodeClick}
                    />
                  )}
                  {!isLoading && !wordData && (
                    <div className="no-word-selected">Select a word or search to see details.</div>
                  )}
                  {error && <div className="error-message">Error: {error}</div>}
                </div>
              </Panel>
            </PanelGroup>
          </div>
        ) : (
          <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div className="graph-container" style={{ flexBasis: '50%', minHeight: '300px' }}>
              <div className="graph-content">
                {isLoading && <div className="loading">Loading Network...</div>}
                {!isLoading && wordNetwork && (
                  <WordGraph
                    wordNetwork={wordNetwork}
                    mainWord={selectedNode}
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
              {!isLoading && wordData && (
                <WordDetails 
                  wordInfo={wordData} 
                  etymologyTree={etymologyTree}
                  isLoadingEtymology={isLoadingEtymology}
                  etymologyError={etymologyError}
                  onWordLinkClick={handleNodeClick}
                  onEtymologyNodeClick={handleNodeClick}
                />
              )}
              {!isLoading && !wordData && (
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
