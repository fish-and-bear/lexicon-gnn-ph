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
      const data = await fetchWordNetwork(word, { 
        depth,
        breadth,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.3
      });
      
      if (data) {
        setWordNetwork(data);
      }
      return data;
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
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
    await loadWordData(suggestion.id);
  }, [loadWordData]);

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
      if (error instanceof Error && error.message.includes('Network Error')) {
        setError('Cannot connect to the backend server. Please ensure the backend server is running on port 10000.');
      }
    } finally {
      setIsLoadingSuggestions(false);
    }
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setInputValue(query);
    handleDebouncedSearch(query);
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

  const renderDefinitions = useCallback((wordInfo: WordInfo) => {
    if (!wordInfo.definitions || wordInfo.definitions.length === 0) {
      return (
        <div className="definitions-section">
          <div className="definitions-section-header">
            <h3>Definitions</h3>
            <span className="definition-count">0</span>
          </div>
          <p className="no-definitions">No definitions available for this word.</p>
        </div>
      );
    }

    // Group definitions by part of speech
    const definitionsByPos: Record<string, Definition[]> = {};
    wordInfo.definitions.forEach((def) => {
      const posName = def.part_of_speech?.name_en || def.original_pos || 'Other';
      if (!definitionsByPos[posName]) {
        definitionsByPos[posName] = [];
      }
      definitionsByPos[posName].push(def);
    });

    // Sort parts of speech in a logical order
    const sortedPosEntries = Object.entries(definitionsByPos).sort(([posA], [posB]) => {
      // Define priority order for common parts of speech
      const posOrder: { [key: string]: number } = {
        'Noun': 1,
        'Verb': 2,
        'Adjective': 3,
        'Adverb': 4,
        'Pronoun': 5,
        'Preposition': 6,
        'Conjunction': 7,
        'Interjection': 8,
        'Other': 9
      };
      
      // Get priority or default to high number (low priority)
      const priorityA = posOrder[posA] || 10;
      const priorityB = posOrder[posB] || 10;
      
      return priorityA - priorityB;
    });

    return (
      <div className="definitions-section">
        <div className="definitions-section-header">
          <h3>Definitions</h3>
          <span className="definition-count">{wordInfo.definitions.length}</span>
        </div>
        
        {sortedPosEntries.map(([posName, definitions]: [string, any[]]) => (
          <div key={posName} className="pos-group">
            <div className="pos-group-header">
              {posName}
              <span className="pos-count">{definitions.length}</span>
            </div>
            
            <div className="definition-cards-container">
              {definitions.map((definition: Definition, index: number) => {
                // Check for both possible property names for definition text
                const definitionText = definition.definition_text || definition.text || '';
                
                // Pre-process the text to extract any trailing numbers for superscript
                let textPart = definitionText;
                let numberPart = '';
                
                // Check if the text ends with a number
                const match = definitionText.match(/^(.*[^\d])(\d+)$/);
                if (match) {
                  textPart = match[1];
                  numberPart = match[2];
                }
                
                return (
      <div key={index} className="definition-card">
                    <div className="definition-number">{index + 1}</div>
                    <div className="definition-content">
                      <p className="definition-text">
                        {textPart}
                        {numberPart && <sup>{numberPart}</sup>}
                      </p>
                      
        {definition.examples && definition.examples.length > 0 && (
          <div className="examples">
                          <h4>Examples</h4>
                          <ul>
                            {definition.examples.map((example: string, idx: number) => {
                              // Check if example contains a translation (indicated by parentheses or em dash)
                              const hasTranslation = example.includes('(') || example.includes('—') || example.includes(' - ');
                              
                              if (hasTranslation) {
                                // Split the example into the phrase and translation
                                let phrase, translation;
                                
                                if (example.includes('(')) {
                                  [phrase, translation] = example.split(/\s*\(/);
                                  translation = translation ? `(${translation}` : '';
                                } else if (example.includes('—')) {
                                  [phrase, translation] = example.split(/\s*—\s*/);
                                } else if (example.includes(' - ')) {
                                  [phrase, translation] = example.split(/\s*-\s*/);
                                }
                                
                                return (
                                  <li key={idx}>
                                    <em>{phrase}</em>
                                    {translation && <span className="translation">{translation}</span>}
                                  </li>
                                );
                              }
                              
                              return <li key={idx}><em>{example}</em></li>;
                            })}
            </ul>
          </div>
        )}
                      
        {definition.usage_notes && definition.usage_notes.length > 0 && (
          <div className="usage-notes">
                          <h4>Usage Notes</h4>
                          <ul>
                            {definition.usage_notes.map((note: string, idx: number) => {
                              // Check if note is a category tag (enclosed in square brackets)
                              const isCategoryTag = note.match(/^\[(.*?)\]$/);
                              if (isCategoryTag) {
                                return (
                                  <li key={idx}>
                                    <em className="category-tag">{note}</em>
                                  </li>
                                );
                              }
                              
                              // Check if note contains a detail section (indicated by colon)
                              const hasDetail = note.includes(':');
                              if (hasDetail) {
                                const [label, detail] = note.split(/:\s*/);
                                return (
                                  <li key={idx}>
                                    <em>{label}:</em>
                                    <span className="note-detail">{detail}</span>
                                  </li>
                                );
                              }
                              
                              return <li key={idx}>{note}</li>;
                            })}
            </ul>
          </div>
        )}
                      
                      {definition.sources && definition.sources.length > 0 && (
                        <div className="definition-sources">
                          <span className="sources-label">Sources:</span>
                          <div className="source-tags">
                            {definition.sources.map((source: string, idx: number) => (
                              <span key={idx} className="source-tag">{source}</span>
                            ))}
      </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    );
  }, []);

  const renderArraySection = useCallback((title: string, items?: string[]) => {
    if (!items || items.length === 0) return null;
    return (
      <div className={title.toLowerCase().replace(/\s+/g, "-")}>
        <h3>{title}</h3>
        <ul className="word-list">
          {items
            .filter((item) => item.trim() !== "" && item.trim() !== "0")
            .map((item, index) => (
              <li
                key={index}
                onClick={() => handleNodeClick(item)}
                className="clickable-word"
              >
                {item}
              </li>
            ))}
        </ul>
      </div>
    );
  }, [handleNodeClick]);

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
                  const response = await fetch('http://localhost:10000/api/v2/test', {
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
                    localStorage.setItem('successful_api_endpoint', 'http://localhost:10000/api/v2');
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
      
      // Get random word (already returns comprehensive data)
      const randomWordData = await getRandomWord();
      console.log('Random word data:', randomWordData);
      
      // Use the data directly from getRandomWord
      if (randomWordData && randomWordData.lemma) {
        // Use randomWordData instead of wordData
        setSelectedWordInfo({
          id: randomWordData.id,
          lemma: randomWordData.lemma,
          normalized_lemma: randomWordData.normalized_lemma || randomWordData.lemma,
          language_code: randomWordData.language_code || 'tl',
          has_baybayin: randomWordData.has_baybayin || false,
          baybayin_form: randomWordData.baybayin_form,
          romanized_form: randomWordData.romanized_form,
          definitions: randomWordData.definitions || [],
          pronunciations: randomWordData.pronunciations || [],
          etymologies: randomWordData.etymologies || [],
          tags: randomWordData.tags,
          outgoing_relations: randomWordData.outgoing_relations || [],
          incoming_relations: randomWordData.incoming_relations || [],
          root_affixations: randomWordData.root_affixations || [],
          affixed_affixations: randomWordData.affixed_affixations || [],
        });
        setMainWord(randomWordData.lemma);
        
        // Fetch word network and etymology tree using the random word's lemma/id
        const networkData = await fetchWordNetworkData(randomWordData.lemma, depth, breadth);
        console.log('Word network for random word:', networkData);
        
        setWordNetwork(networkData);
        setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), randomWordData.lemma]);
        setCurrentHistoryIndex(prevIndex => prevIndex + 1);
        setInputValue(randomWordData.lemma);
        
        // Fetch etymology tree using the random word's ID
        fetchEtymologyTree(randomWordData.id);
      } else {
        throw new Error("Could not fetch random word or lemma missing."); // Added lemma check message
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
      <div className="search-container">
        <button
          onClick={handleBack}
          disabled={currentHistoryIndex <= 0}
          className="history-button"
          aria-label="Go back"
        >
          ←
        </button>
        <button
          onClick={handleForward}
          disabled={currentHistoryIndex >= wordHistory.length - 1}
          className="history-button"
          aria-label="Go forward"
        >
          →
        </button>
        <div className="search-input-container">
          <input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                console.log("Enter key pressed! Calling handleSearch(). Input value:", inputValue);
                e.preventDefault();
                handleSearch(inputValue);
              }
            }}
            placeholder="Enter a word"
            className="search-input"
            aria-label="Search word"
          />
          {isLoading && <div className="search-loading">Loading...</div>}
          {showSuggestions && searchResults.length > 0 && (
            <div className="suggestions-container">
              {searchResults.map((suggestion) => (
                <div
                  key={suggestion.id}
                  className="suggestion-item"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  {suggestion.lemma}
                  {suggestion.has_baybayin && suggestion.baybayin_form && (
                    <span className="baybayin-text"> ({suggestion.baybayin_form})</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
        <button
          onClick={() => handleSearch(inputValue)}
          disabled={isLoading}
          className="search-button"
        >
          Search
        </button>
      </div>
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
                  <li>Run: <code>python serve.py</code></li>
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
              onWordLinkClick={handleWordLinkClick}
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
