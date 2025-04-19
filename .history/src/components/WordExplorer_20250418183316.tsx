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
  getStatistics,
  getBaybayinWords,
  getAffixes,
  getRelations,
  getAllWords,
  getEtymologyTree,
} from "../api/wordApi";
import { debounce } from "lodash";
import CircularProgress from '@mui/material/CircularProgress';
import { Typography, Button } from "@mui/material";

// Import Resizable Panels components
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import useMediaQuery from '@mui/material/useMediaQuery';

const WordExplorer: React.FC = () => {
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
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('tl');

  const [wordData, setWordData] = useState<WordInfo | null>(null);

  const randomWordTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isRandomLoading, setIsRandomLoading] = useState<boolean>(false);
  
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

  const fetchWordNetworkData = useCallback(async (word: string, depth: number = 2, breadth: number = 10) => {
    try {
      setIsLoading(true);
      setError(null);
      
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

  const fetchEtymologyTreeData = useCallback(async (wordId: number) => {
    if (!wordId) return;
    
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    
    try {
      console.log("Fetching etymology tree for word ID:", wordId);
      const data = await getEtymologyTree(wordId);
      console.log("Etymology tree data received:", data);
      
      if (data && Array.isArray(data.nodes) && data.nodes.length > 0) {
        console.log(`Received valid etymology tree with ${data.nodes.length} nodes`);
        setEtymologyTree(data);
      } else {
        console.warn("Received empty etymology tree or invalid structure");
        setEtymologyTree(null);
      }
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
  }, [getEtymologyTree]);

  const fetchDetails = useCallback(async (wordIdentifier: string | number) => {
    const identifierString = typeof wordIdentifier === 'number' ? `id:${wordIdentifier}` : wordIdentifier;
    
    if (!identifierString) return;
    
    const normalized = identifierString.startsWith('id:') ? identifierString : normalizeInput(identifierString);
    
    console.log(`Fetching details for identifier: ${normalized}`);
    
    try {
      setIsLoading(true);
      setError(null);
      
      const data = await fetchWordDetails(normalized);
      console.log('Word details received:', data);
      
      setSelectedWordInfo(data);
      setWordData(data);
      
      if (data.id) {
        fetchEtymologyTreeData(data.id);
      } else {
        setEtymologyTree(null);
      }
      
      if (!wordHistory.some(entry => 
          (typeof entry === 'string' && entry === data.lemma) ||
          (typeof entry === 'object' && entry.id === data.id)
      )) {
        const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), { id: data.id, text: data.lemma }];
        setWordHistory(newHistory);
        setCurrentHistoryIndex(newHistory.length - 1);
      }

      setShowSuggestions(false);
      setSearchResults([]);

    } catch (err) {
      console.error("Error fetching word details:", err);
      setError(err instanceof Error ? err.message : 'Failed to fetch word details');
      setSelectedWordInfo(null);
      setWordData(null);
      setEtymologyTree(null);
    } finally {
      setIsLoading(false);
    }
  }, [getEtymologyTreeData, wordHistory, currentHistoryIndex]);

  const debouncedSearch = useCallback(
    debounce(async (query: string) => {
      if (query.length < 2) {
        setSearchResults([]);
        setShowSuggestions(false);
        return;
      }
      
      console.log(`Debounced search triggered for: ${query}`);
      setIsLoadingSuggestions(true);
      
      try {
        const results = await searchWords(query, { per_page: 10 });
        console.log('Search results received:', results);
        
        if (results && results.words) {
          setSearchResults(results.words);
          setShowSuggestions(true);
        } else {
          setSearchResults([]);
          setShowSuggestions(false);
        }
      } catch (err) {
        console.error("Error during search suggestions:", err);
        setSearchResults([]);
        setShowSuggestions(false);
      } finally {
        setIsLoadingSuggestions(false);
      }
    }, 300),
    []
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setInputValue(value);
    debouncedSearch(value);
  };

  const handleSearchSubmit = useCallback((e?: React.FormEvent<HTMLFormElement>) => {
    if (e) e.preventDefault();
    
    const normalized = normalizeInput(inputValue);
    if (!normalized) return;
    
    console.log(`Search submitted for: ${normalized}`);
    
    setMainWord(normalized);
    fetchWordNetworkData(normalized, depth, breadth);
    fetchDetails(normalized);
    
    setShowSuggestions(false);
    setSearchResults([]);
  }, [inputValue, fetchWordNetworkData, depth, breadth, fetchDetails]);

  const handleNodeSelect = useCallback((nodeId: string | number | null) => {
    if (nodeId !== null) {
      console.log(`Node selected: ${nodeId}`);
      
      const selectedNode = wordNetwork?.nodes.find(n => String(n.id) === String(nodeId));
      
      if (selectedNode) {
        console.log('Selected node details:', selectedNode);
        fetchDetails(selectedNode.id || selectedNode.lemma); 
        setInputValue(selectedNode.lemma);
      } else {
        console.warn(`Node with ID ${nodeId} not found in current network data. Fetching by ID.`);
        fetchDetails(nodeId);
      }
      
    } else {
      console.log('Node deselected or click outside nodes.');
    }
  }, [wordNetwork, fetchDetails]);

  const handleGoBack = useCallback(() => {
    if (currentHistoryIndex > 0) {
      const previousIndex = currentHistoryIndex - 1;
      const previousEntry = wordHistory[previousIndex];
      const identifier = typeof previousEntry === 'object' ? previousEntry.id : previousEntry;
      
      console.log('Navigating back to:', identifier);
      setCurrentHistoryIndex(previousIndex);
      fetchDetails(identifier);
      
      const lemma = typeof previousEntry === 'object' ? previousEntry.text : previousEntry;
      setInputValue(lemma);
      setMainWord(lemma);
      fetchWordNetworkData(lemma, depth, breadth);
    }
  }, [currentHistoryIndex, wordHistory, fetchDetails, fetchWordNetworkData, depth, breadth]);

  const handleGoForward = useCallback(() => {
    if (currentHistoryIndex < wordHistory.length - 1) {
      const nextIndex = currentHistoryIndex + 1;
      const nextEntry = wordHistory[nextIndex];
      const identifier = typeof nextEntry === 'object' ? nextEntry.id : nextEntry;
      
      console.log('Navigating forward to:', identifier);
      setCurrentHistoryIndex(nextIndex);
      fetchDetails(identifier);
      
      const lemma = typeof nextEntry === 'object' ? nextEntry.text : nextEntry;
      setInputValue(lemma);
      setMainWord(lemma);
      fetchWordNetworkData(lemma, depth, breadth);
    }
  }, [currentHistoryIndex, wordHistory, fetchDetails, fetchWordNetworkData, depth, breadth]);

  const fetchRandom = useCallback(async () => {
    console.log("Attempting to fetch a random word...");
    if (randomWordTimeoutRef.current) {
      clearTimeout(randomWordTimeoutRef.current);
      randomWordTimeoutRef.current = null;
    }
    
    setIsRandomLoading(true);
    setError(null);
    
    try {
      const wordInfo = await getRandomWord();
      console.log("Random word fetched:", wordInfo);
      
      if (wordInfo && wordInfo.lemma) {
        setInputValue(wordInfo.lemma);
        setMainWord(wordInfo.lemma);
        fetchWordNetworkData(wordInfo.lemma, depth, breadth);
        setSelectedWordInfo(wordInfo);
        setWordData(wordInfo);
        
        const newHistory = [...wordHistory.slice(0, currentHistoryIndex + 1), { id: wordInfo.id, text: wordInfo.lemma }];
        setWordHistory(newHistory);
        setCurrentHistoryIndex(newHistory.length - 1);

        if (wordInfo.id) {
          fetchEtymologyTreeData(wordInfo.id);
        } else {
          setEtymologyTree(null);
        }

      } else {
        throw new Error("Received invalid data for random word.");
      }
    } catch (err) {
      console.error("Error fetching random word:", err);
      setError(err instanceof Error ? err.message : 'Failed to fetch random word');
    } finally {
      randomWordTimeoutRef.current = setTimeout(() => {
        setIsRandomLoading(false);
        randomWordTimeoutRef.current = null;
      }, 1000); 
    }
  }, [depth, breadth, fetchWordNetworkData, wordHistory, currentHistoryIndex, getEtymologyTreeData, isRandomLoading]);

  useEffect(() => {
    const checkConnection = async () => {
      const connected = await testApiConnection();
      setApiConnected(connected);
      if (!connected) {
        setError("Cannot connect to the API server. Please ensure it's running.");
      }
    };
    checkConnection();
  }, []);

  const handleResetCircuitBreaker = () => {
    resetCircuitBreaker();
    setApiConnected(null);
    setError(null);
    setTimeout(async () => {
      const connected = await testApiConnection();
      setApiConnected(connected);
      if (!connected) {
        setError("Still cannot connect to the API server after reset.");
      } else {
        setError(null);
      }
    }, 1000);
  };

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setIsLoadingStatistics(true);
        const data = await getStatistics();
        setStatistics(data);
      } catch (err) {
        console.error("Error fetching statistics:", err);
      } finally {
        setIsLoadingStatistics(false);
      }
    };
    fetchStats();
  }, []);

  const fetchBaybayin = useCallback(async (page = 1) => {
    try {
      setIsLoadingBaybayin(true);
      const data = await getBaybayinWords(page, 20, selectedLanguage);
      setBaybayinWords(data.words || []);
      setTotalPages(Math.ceil((data.total || 0) / 20));
      setCurrentPage(page);
    } catch (err) {
      console.error("Error fetching Baybayin words:", err);
    } finally {
      setIsLoadingBaybayin(false);
    }
  }, [selectedLanguage]);

  const fetchAll = useCallback(async (page = 1) => {
    try {
      setIsLoadingAllWords(true);
      const data = await getAllWords(page, 20, selectedLanguage);
      setAllWords(data.words || []);
      setTotalPages(Math.ceil((data.total || 0) / 20));
      setCurrentPage(page);
    } catch (err) {
      console.error("Error fetching all words:", err);
    } finally {
      setIsLoadingAllWords(false);
    }
  }, [selectedLanguage]);

  const isWideLayout = useMediaQuery('(min-width:769px)'); 

  return (
    <div className={`word-explorer ${theme} ${isLoading ? 'loading' : ''}`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <div className="header-buttons">
          <button
            onClick={fetchRandom}
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
                        mainWord={mainWord}
                        onNodeClick={handleNodeSelect}
                        onNodeSelect={handleNodeSelect}
                        onNetworkChange={(newDepth, newBreadth) => {
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
                        }}
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
                  {!isLoading && selectedWordInfo && (
                    <WordDetails 
                      wordInfo={selectedWordInfo} 
                      etymologyTree={etymologyTree}
                      isLoadingEtymology={isLoadingEtymology}
                      etymologyError={etymologyError}
                      onWordLinkClick={handleNodeSelect}
                      onEtymologyNodeClick={handleNodeSelect}
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
          <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div className="graph-container" style={{ flexBasis: '50%', minHeight: '300px' }}>
              <div className="graph-content">
                {isLoading && <div className="loading">Loading Network...</div>}
                {!isLoading && wordNetwork && (
                  <WordGraph
                    wordNetwork={wordNetwork}
                    mainWord={mainWord}
                    onNodeClick={handleNodeSelect}
                    onNodeSelect={handleNodeSelect}
                    onNetworkChange={(newDepth, newBreadth) => {
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
                    }}
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
                  onWordLinkClick={handleNodeSelect}
                  onEtymologyNodeClick={handleNodeSelect}
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
