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
} from "../api/wordApi";
import { Button } from "@mui/material";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import useMediaQuery from '@mui/material/useMediaQuery';
import CircularProgress from '@mui/material/CircularProgress';
import { Theme } from '@mui/material/styles';
import { debounce } from 'lodash';

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

  const { theme, toggleTheme } = useTheme();
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
        
        // Check if we have semantic_network data
        const hasSemanticNetwork = 
          wordData.semantic_network && 
          wordData.semantic_network.nodes && 
          wordData.semantic_network.nodes.length > 0 &&
          wordData.semantic_network.links && 
          wordData.semantic_network.links.length > 0;
          
        if (!hasSemanticNetwork) {
          console.log("Semantic network data missing, will load from network data");
        }
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
        
        // Store current data first
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
          
          // If we don't have semantic network data in wordData already, 
          // update wordData with the network data for use in relations
          if (!wordData.semantic_network && networkData) {
            console.log("Adding semantic network data from network result");
            const updatedWordData = {
              ...wordData,
              semantic_network: {
                nodes: networkData.nodes || [],
                links: networkData.edges || []
              }
            };
            setWordData(updatedWordData);
          }
          
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

  const handleSearch = useCallback(async (query: string, options?: SearchOptions) => {
    if (!query) return;
    console.log(`handleSearch called with query: ${query}`);
    setError(null);
    setIsLoading(true);
    setWordData(null);
    setWordNetwork(null);
    setSelectedNode(null); 

    try {
      const searchOptions: SearchOptions = {
        page: 1,
        per_page: 10,
        mode: 'all',
        sort: 'relevance',
        order: 'desc',
        language: '',
        exclude_baybayin: false
      };
      
      console.log(`Calling searchWords API with query: "${query}", options:`, searchOptions);
      const result = await searchWords(query, searchOptions);
      console.log(`Search API result:`, result);
      
      if (result.error) {
        console.error(`Search returned an error: ${result.error}`);
        setError(result.error);
        setSearchResults([]);
        setShowSuggestions(false);
        return;
      }
      
      if (result && result.words && result.words.length > 0) {
        console.log(`Found ${result.words.length} search results, first result:`, result.words[0]);
        
        setSearchResults(result.words);
        setShowSuggestions(false);
        
        console.log(`Loading details for first result: ${result.words[0].lemma} (ID: ${result.words[0].id})`);
        
        try {
          const firstResult = result.words[0];
          console.log(`Fetching details for word ID: ${firstResult.id}`);
          
          const idString = String(firstResult.id);
          const wordId = idString.startsWith('id:') ? idString : `id:${idString}`;
            
          const wordData = await fetchWordDetails(wordId);
          console.log(`Word details received:`, wordData);
          
          setSelectedNode(wordData.lemma);
          setWordData(wordData);
          
          const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), { id: wordData.id, text: wordData.lemma }];
          setWordHistory(newHistory);
          setCurrentHistoryIndex(newHistory.length - 1);
          
          // Check if relations data exists
          const hasRelations = 
            (wordData.incoming_relations && wordData.incoming_relations.length > 0) || 
            (wordData.outgoing_relations && wordData.outgoing_relations.length > 0);
          
          if (!hasRelations) {
            console.log("Relations data missing from search result, will try to populate from network");
          }
          
          console.log(`Fetching network for: ${wordData.lemma}`);
          fetchWordNetworkData(wordData.lemma, depth, breadth)
            .then(networkData => {
              console.log(`Network data received with ${networkData?.nodes?.length || 0} nodes`);
              setWordNetwork(networkData);
              
              // Update wordData with relations from network if needed
              if ((!wordData.incoming_relations || wordData.incoming_relations.length === 0) &&
                  (!wordData.outgoing_relations || wordData.outgoing_relations.length === 0) &&
                  networkData && networkData.nodes && networkData.edges) {
                
                console.log("Populating relations from network data");
                const mainNode = networkData.nodes.find(node => 
                  node.type === 'main' || node.word === wordData.lemma);
                
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
                  
                  console.log(`Generated relations from network: ${incomingRelations.length} incoming, ${outgoingRelations.length} outgoing`);
                  
                  // Update wordData with the new relations
                  const updatedWordData = {
                    ...wordData,
                    incoming_relations: incomingRelations.length > 0 ? incomingRelations : wordData.incoming_relations || [],
                    outgoing_relations: outgoingRelations.length > 0 ? outgoingRelations : wordData.outgoing_relations || []
                  };
                  
                  // Update state with enhanced wordData
                  setWordData(updatedWordData);
                }
              }
            })
            .catch(networkErr => {
              console.error(`Error fetching network:`, networkErr);
            });
          
          try {
            console.log(`Fetching etymology tree for ID: ${wordData.id}`);
            const etymologyIdString = String(wordData.id);
            const etymologyId = etymologyIdString.startsWith('id:') 
              ? parseInt(etymologyIdString.substring(3), 10) 
              : wordData.id;
            fetchEtymologyTree(etymologyId)
              .then(tree => {
                console.log(`Etymology tree received`);
                setEtymologyTree(tree);
              })
              .catch(etymErr => {
                console.error(`Error fetching etymology tree:`, etymErr);
              });
          } catch (etymErr) {
            console.error(`Error initiating etymology tree fetch:`, etymErr);
          }
          
          detailsContainerRef.current?.scrollTo(0, 0);
          console.log(`Search loading process completed successfully`);
        } catch (dataError) {
          console.error(`Error loading word data during search:`, dataError);
          let errorMessage = "Error loading word details";
          
          if (dataError instanceof Error) {
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
          
          console.error(`Setting error:`, errorMessage);
          setError(errorMessage);
          
          setSearchResults(result.words);
        }
      } else {
        console.log(`No results found for query: "${query}"`);
        setSearchResults([]);
        setShowSuggestions(false);
        setError(`No results found for "${query}". Please try a different word.`);
      }
    } catch (error) {
      console.error(`Search error:`, error);
      setSearchResults([]);
      let errorMessage = "An error occurred during search";
      
      if (error instanceof Error) {
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
      
      console.error(`Setting error message:`, errorMessage);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      setIsLoadingSuggestions(false);
    }
  }, [fetchWordNetworkData, fetchEtymologyTree, wordHistory, currentHistoryIndex, depth, breadth]);

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
        .catch(err => console.error("Error fetching network data for random word:", err));
      
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
        setWordData(wordData);
        
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
        setWordData(wordData);
        
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

  // Add debounced search function for showing suggestions
  const debouncedSearch = useCallback(
    debounce(async (query: string) => {
      if (!query || query.length < 2) {
        setShowSuggestions(false);
        return;
      }
      
      setIsLoadingSuggestions(true);
      
      try {
        const searchOptions: SearchOptions = {
          page: 1,
          per_page: 10,
          mode: 'all',
          sort: 'relevance',
          order: 'desc',
          language: '',
          exclude_baybayin: false
        };
        
        console.log(`Fetching search suggestions for: "${query}"`);
        const result = await searchWords(query, searchOptions);
        
        if (result && result.words && result.words.length > 0) {
          setSearchResults(result.words);
          setShowSuggestions(true);
        } else {
          setSearchResults([]);
          setShowSuggestions(false);
        }
      } catch (error) {
        console.error("Error fetching search suggestions:", error);
        setSearchResults([]);
        setShowSuggestions(false);
      } finally {
        setIsLoadingSuggestions(false);
      }
    }, 300),
    [searchWords]
  );

  // Helper function to display language info and differentiate similar words
  const formatSearchResult = (result: SearchWordResult) => {
    // Extract language code
    let languageInfo = '';
    
    if (result.language) {
      // Format common language codes
      switch(result.language.toLowerCase()) {
        case 'tgl':
        case 'fil':
          languageInfo = 'Tagalog';
          break;
        case 'ceb':
        case 'ceb-ph':
          languageInfo = 'Cebuano';
          break;
        case 'hil':
          languageInfo = 'Hiligaynon';
          break;
        case 'ilo':
          languageInfo = 'Ilocano';
          break;
        case 'en':
          languageInfo = 'English';
          break;
        case 'es':
          languageInfo = 'Spanish';
          break;
        default:
          languageInfo = result.language;
      }
    }
    
    // Return formatted language info
    return languageInfo;
  };
  
  // Add debounced search on input change
  useEffect(() => {
    if (inputValue.trim().length >= 2) {
      debouncedSearch(inputValue.trim());
    } else {
      setShowSuggestions(false);
    }
    
    return () => {
      debouncedSearch.cancel();
    };
  }, [inputValue, debouncedSearch]);

  // Helper function to group similar words and highlight differences
  const findSimilarWords = (words: SearchWordResult[]) => {
    // Create a map to group similar lemmas
    const similarGroups: {[key: string]: SearchWordResult[]} = {};
    
    words.forEach(word => {
      const normalizedLemma = normalizeInput(word.lemma);
      
      if (!similarGroups[normalizedLemma]) {
        similarGroups[normalizedLemma] = [];
      }
      
      similarGroups[normalizedLemma].push(word);
    });
    
    // Mark words that have similar lemmas
    return words.map(word => {
      const normalizedLemma = normalizeInput(word.lemma);
      const hasSimilar = similarGroups[normalizedLemma].length > 1;
      
      return {
        ...word,
        hasSimilar
      };
    });
  };

  const renderSearchBar = () => {
    // Find similar words to highlight differences
    const processedResults = findSimilarWords(searchResults);
    
    return (
      <div className="search-container">
        <div className="nav-buttons">
          <Button 
            className="nav-button back-button" 
            onClick={handleBack}
            disabled={currentHistoryIndex <= 0}
            title="Go back to previous word"
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
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                console.log(`Enter key pressed for search: "${inputValue}"`);
                if (inputValue.trim()) {
                  handleSearch(inputValue);
                  setShowSuggestions(false);
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
          
          {showSuggestions && processedResults.length > 0 && (
            <ul className="search-suggestions">
              {processedResults.map((result) => {
                const languageInfo = formatSearchResult(result);
                const wordType = result.pos ? ` (${result.pos})` : '';
                
                return (
                  <li 
                    key={result.id} 
                    onClick={() => {
                      const resultId = String(result.id);
                      const wordId = resultId.startsWith('id:') ? resultId : `id:${resultId}`;
                      handleNodeClick(wordId);
                      setInputValue(result.lemma);
                      setShowSuggestions(false);
                    }}
                  >
                    <div className="suggestion-header">
                      <strong>{result.lemma}</strong>
                      {(result.hasSimilar || languageInfo) && (
                        <span className="suggestion-language">
                          {languageInfo}{wordType}
                        </span>
                      )}
                    </div>
                    {result.definitions && result.definitions.length > 0 && (
                      <span className="suggestion-definition">
                        {typeof result.definitions[0] === 'string' 
                          ? result.definitions[0] 
                          : (result.definitions[0] as any)?.text || ''}
                      </span>
                    )}
                  </li>
                );
              })}
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
          sx={(theme: Theme) => ({
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
          onClick={handleRandomWord}
          disabled={isRandomLoading || isLoading} 
          title="Get a random word"
          sx={{
            mx: 0.1, 
            whiteSpace: 'nowrap',
            bgcolor: theme === 'dark' ? 'var(--button-color)' : 'var(--accent-color)',
            color: 'var(--button-text-color)',
            fontWeight: 'normal',
            borderRadius: '8px',
            boxShadow: 'none',
            '&:hover': {
              bgcolor: 'var(--secondary-color)',
              color: '#ffffff',
              boxShadow: 'none'
            }
          }}
        >
          {isRandomLoading ? '⏳ Loading...' : '🎲 Random Word'}
        </Button>
      </div>
    );
  };

  const isWideLayout = useMediaQuery('(min-width:769px)'); 

  return (
    <div className={`word-explorer ${theme} ${isLoading ? 'loading' : ''}`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <div className="header-buttons">
          {process.env.NODE_ENV === 'development' && (
            <>
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
                (apiConnected === null ? 'checking' : (apiConnected ? 'connected' : 'disconnected'))
              }`}>
                API: {apiConnected === null ? 'Checking...' : 
                     apiConnected ? '✅ Connected' : '❌ Disconnected'}
              </div>
            </>
          )}
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
