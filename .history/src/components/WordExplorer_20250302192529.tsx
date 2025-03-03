import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchResult, SearchOptions, EtymologyTree, Statistics, Definition } from "../types";
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

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://api.hapinas.net/api/v1';

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
  const [searchResults, setSearchResults] = useState<Array<{ id: number; word: string }>>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
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

  // Add a new state variable for toggling metadata visibility
  const [showMetadata, setShowMetadata] = useState<boolean>(false);

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

  const fetchWordNetworkData = useCallback(async (word: string, depth: number, breadth: number) => {
    try {
      return await fetchWordNetwork(word, { 
        depth,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.3
      });
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
      throw error; // Pass the error through instead of creating a new one
    }
  }, []);

  // Function to fetch etymology tree
  const fetchEtymologyTree = useCallback(async (word: string) => {
    if (!word) return;
    
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    
    try {
      console.log(`Fetching etymology tree for: ${word}`);
      const data = await getEtymologyTree(word, 3, true, true);
      console.log('Etymology tree data:', data);
      setEtymologyTree(data);
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
  }, []);

  const handleSearch = useCallback(async (searchWord?: string) => {
    const wordToSearch = searchWord || inputValue.trim();
    if (!wordToSearch) {
      setError("Please enter a word to search");
      return;
    }
  
    const sanitizedInput = DOMPurify.sanitize(wordToSearch);
    const normalizedInput = normalizeInput(sanitizedInput);

    console.log('Starting search for normalized input:', normalizedInput);
    
    setIsLoading(true);
    setError(null);
    setWordNetwork(null);
    setEtymologyTree(null); // Reset etymology tree

    try {
      console.log('Searching for word:', normalizedInput);
      
      // First try to get word details directly
      try {
        console.log('Fetching word details for:', normalizedInput);
        const detailsData = await fetchWordDetails(normalizedInput);
        console.log('Word details received:', detailsData);
        console.log('Full word details data:', JSON.stringify(detailsData, null, 2));
        
        if (detailsData && detailsData.lemma) {
          console.log('Setting selected word info with valid data');
          setSelectedWordInfo(detailsData);
          setMainWord(detailsData.lemma);
          
          // Fetch word network and etymology tree in parallel
          const [networkData] = await Promise.all([
            fetchWordNetworkData(normalizedInput, depth, breadth),
            fetchEtymologyTree(normalizedInput) // This updates state directly
          ]);
          
          console.log('Word network data received:', networkData);
          
          setWordNetwork(networkData);
          setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), detailsData.lemma]);
          setCurrentHistoryIndex(prevIndex => prevIndex + 1);
          setInputValue(detailsData.lemma);
          setShowSuggestions(false);
          setIsLoading(false);
          return;
        } else {
          console.log('Word details missing lemma or invalid:', detailsData);
        }
      } catch (detailsError) {
        console.error('Direct word details fetch failed, trying search:', detailsError);
        // Continue to search if direct fetch fails
      }
      
      // If direct fetch fails, try search
      const searchResults = await searchWords(normalizedInput, { 
        page: 1, 
        per_page: 20, 
        exclude_baybayin: true,
        language: 'tl',
        mode: 'all',
        sort: 'relevance',
        order: 'desc'
      });
      
      console.log('Search results:', searchResults);
      
      if (searchResults && searchResults.words && searchResults.words.length > 0) {
        // Use the first search result
        const firstResult = searchResults.words[0];
        console.log('Using first search result:', firstResult);
        
        // Fetch details for the first result
        const detailsData = await fetchWordDetails(firstResult.word);
        console.log('Word details for search result:', detailsData);
        
        if (detailsData && detailsData.lemma) {
          setSelectedWordInfo(detailsData);
          setMainWord(detailsData.lemma);
          
          console.log('Fetching word network for search result:', detailsData.lemma);
          let networkData = await fetchWordNetworkData(detailsData.lemma, depth, breadth);
          console.log('Word network for search result:', networkData);
          
          setWordNetwork(networkData);
          setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), detailsData.lemma]);
          setCurrentHistoryIndex(prevIndex => prevIndex + 1);
          setInputValue(detailsData.lemma);
          setShowSuggestions(false);
        } else {
          throw new Error("Could not fetch details for the search result.");
        }
      } else {
        throw new Error("No results found. Try a different search term.");
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      let errorMessage = "Failed to fetch word data.";
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (axios.isAxiosError(error) && error.response) {
        errorMessage = error.response.data?.error?.message || error.message;
      }
      setError(errorMessage);
      setWordNetwork(null);
      setSelectedWordInfo(null);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, depth, breadth, currentHistoryIndex, fetchWordNetworkData, fetchEtymologyTree]);

  const debouncedSearch = useCallback(
    debounce(async (query: string) => {
      if (query.length > 1) {
        setIsLoading(true);
        setError(null);
        console.log('Searching for:', query);
        try {
          const results = await searchWords(query, { 
            page: 1, 
            per_page: 20, 
            exclude_baybayin: true,
            language: 'tl',
            mode: 'all',
            sort: 'relevance',
            order: 'desc',
            is_real_word: true
          });
          console.log('API search response:', results);
          
          if (results && results.words && results.words.length > 0) {
            const searchResults = results.words
              .filter(word => word && word.word && typeof word.word === 'string')
              .map(word => ({
                id: word.id || Math.random(),
                word: word.word.trim()
              }))
              .filter(result => result.word !== '');
            
            console.log('Processed search results:', searchResults);
            
            setSearchResults(searchResults);
            setShowSuggestions(searchResults.length > 0);
            
            if (searchResults.length === 0) {
              setError("No results found. Try a different search term.");
            }
          } else {
            console.error('Invalid API response or no results:', results);
            setSearchResults([]);
            setShowSuggestions(false);
            
            // Try direct word lookup if search returns no results
            try {
              const normalizedQuery = normalizeInput(query);
              const wordDetails = await fetchWordDetails(normalizedQuery);
              
              if (wordDetails && wordDetails.lemma) {
                console.log('Found word via direct lookup:', wordDetails);
                // If we found the word directly, trigger the search with it
                handleSearch(wordDetails.lemma);
                return;
              }
            } catch (directLookupError) {
              console.log('Direct lookup failed:', directLookupError);
              setError("No results found. Try a different search term.");
            }
          }
        } catch (error) {
          console.error("Error fetching search results:", error);
          let errorMessage = "Failed to fetch search results.";
          if (error instanceof Error) {
            errorMessage = error.message;
          } else if (axios.isAxiosError(error) && error.response) {
            errorMessage = error.response.data?.error?.message || error.message;
          }
          setError(errorMessage);
          setSearchResults([]);
          setShowSuggestions(false);
        } finally {
          setIsLoading(false);
        }
      } else {
        setSearchResults([]);
        setShowSuggestions(false);
      }
    }, 300),
    [handleSearch]
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setInputValue(value);
    setError(null);
    if (value.length > 1) {
      debouncedSearch(value);
      setShowSuggestions(true);
    } else {
      setSearchResults([]);
      setShowSuggestions(false);
    }
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

  const handleSuggestionClick = (word: string) => {
    setInputValue(word);
    setShowSuggestions(false);
    handleSearch(word);
  };

  const handleNodeClick = useCallback(async (word: string) => {
    setError(null);
    setIsLoading(true);
    setEtymologyTree(null); // Reset etymology tree
    
    try {
      // Fetch word details, word network, and etymology tree in parallel
      const [detailsData, networkData] = await Promise.all([
        fetchWordDetails(word),
        fetchWordNetworkData(word, depth, breadth)
      ]);
      
      // Update state with the new data
      setSelectedWordInfo(detailsData);
      setWordNetwork(networkData);
      setMainWord(detailsData.lemma);
      
      // Update history
      setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), detailsData.lemma]);
      setCurrentHistoryIndex(prevIndex => prevIndex + 1);
      
      // Update input value
      setInputValue(detailsData.lemma);
      
      // Fetch etymology tree after setting the main data
      fetchEtymologyTree(word);
    } catch (error) {
      console.error("Error fetching node data:", error);
      let errorMessage = "Failed to fetch word data.";
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

  // New function to render idioms
  const renderIdioms = useCallback((wordInfo: WordInfo) => {
    if (!wordInfo.idioms || wordInfo.idioms.length === 0) {
      return null;
    }

    return (
      <div className="idioms-section">
        <h3>Idioms & Phrases</h3>
        <div className="idioms-list">
          {wordInfo.idioms.map((idiom, index) => {
            // Handle both string and object idioms
            if (typeof idiom === 'string') {
              return (
                <div key={index} className="idiom-card">
                  <p className="idiom-text">{idiom}</p>
                </div>
              );
            } else {
              return (
                <div key={index} className="idiom-card">
                  {idiom.phrase && <p className="idiom-phrase">{idiom.phrase}</p>}
                  {idiom.text && <p className="idiom-text">{idiom.text}</p>}
                  {idiom.meaning && (
                    <p className="idiom-meaning">
                      <span className="meaning-label">Meaning:</span> {idiom.meaning}
                    </p>
                  )}
                  {idiom.example && (
                    <p className="idiom-example">
                      <span className="example-label">Example:</span> {idiom.example}
                    </p>
                  )}
                  {idiom.source && <span className="idiom-source">{idiom.source}</span>}
                </div>
              );
            }
          })}
        </div>
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
      .then(([detailsData, networkData]) => {
        setSelectedWordInfo(detailsData);
        setWordNetwork(networkData);
        setMainWord(detailsData.lemma);
        setInputValue(detailsData.lemma);
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
      .then(([detailsData, networkData]) => {
        setSelectedWordInfo(detailsData);
        setWordNetwork(networkData);
        setMainWord(detailsData.lemma);
        setInputValue(detailsData.lemma);
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
  const handleTestApiConnection = async () => {
    setError(null);
    setApiConnected(null); // Set to checking state
    
    try {
      console.log("Manually testing API connection...");
      
      // First try the testApiConnection function
      const isConnected = await testApiConnection();
      setApiConnected(isConnected);
      
      if (!isConnected) {
        console.log("API connection test failed, trying direct fetch...");
        
        // Reset circuit breaker before trying direct fetch
        resetCircuitBreaker();
        
        // Try direct fetch as a fallback
        try {
          const response = await fetch('http://127.0.0.1:10000/', {
            method: 'GET',
            headers: { 'Accept': 'application/json' },
            mode: 'cors',
            cache: 'no-cache'
          });
          
          const directFetchSuccess = response.ok;
          setApiConnected(directFetchSuccess);
          
          if (directFetchSuccess) {
            console.log("API connection successful with direct fetch!");
            setError(null);
            // Update API client with the successful endpoint
            localStorage.setItem('successful_api_endpoint', 'http://127.0.0.1:10000');
            
            // If connection is successful, try to use the API
            if (inputValue) {
              handleSearch(inputValue);
            }
          } else {
            setError("Cannot connect to the API server. Please check that the backend server is running.");
          }
        } catch (e) {
          console.error("Direct fetch attempt failed:", e);
          setApiConnected(false);
          setError("Cannot connect to the API server. Please check that the backend server is running and network settings allow the connection.");
        }
      } else {
        // If connection is successful, try to use the API
        if (inputValue) {
          handleSearch(inputValue);
        }
      }
    } catch (e) {
      console.error("Error testing API connection:", e);
      setApiConnected(false);
      setError("Error testing API connection. Please try again later.");
    }
  };

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
                  const response = await fetch('http://127.0.0.1:10000/', {
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
                    localStorage.setItem('successful_api_endpoint', 'http://127.0.0.1:10000');
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
      
      // Get random word
      const randomWordData = await getRandomWord();
      console.log('Random word data:', randomWordData);
      
      if (randomWordData && randomWordData.lemma) {
        setSelectedWordInfo(randomWordData);
        setMainWord(randomWordData.lemma);
        
        // Fetch word network and etymology tree
        const networkData = await fetchWordNetworkData(randomWordData.lemma, depth, breadth);
        console.log('Word network for random word:', networkData);
        
        setWordNetwork(networkData);
        setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), randomWordData.lemma]);
      setCurrentHistoryIndex(prevIndex => prevIndex + 1);
        setInputValue(randomWordData.lemma);
        
        // Fetch etymology tree
        fetchEtymologyTree(randomWordData.lemma);
      } else {
        throw new Error("Could not fetch random word.");
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
    if (isLoadingStatistics) {
      return (
        <div className="statistics-section loading">
          <div className="spinner"></div>
          <p>Loading statistics...</p>
        </div>
      );
    }

    if (!statistics) {
      return null;
    }

    return (
      <div className="statistics-section">
        <h3>Dictionary Statistics</h3>
        <div className="statistics-grid">
          <div className="stat-card">
            <h4>Words</h4>
            <p className="stat-number">{statistics.words?.total || 0}</p>
          </div>
          <div className="stat-card">
            <h4>Definitions</h4>
            <p className="stat-number">{statistics.definitions?.total || 0}</p>
          </div>
          <div className="stat-card">
            <h4>Relations</h4>
            <p className="stat-number">{statistics.relations?.total || 0}</p>
          </div>
          <div className="stat-card">
            <h4>Sources</h4>
            <p className="stat-number">{statistics.sources?.total || 0}</p>
          </div>
        </div>
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

  return (
    <div className={`word-explorer ${theme}`}>
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
                e.preventDefault();
                handleSearch();
              }
            }}
            placeholder="Enter a word"
            className="search-input"
            aria-label="Search word"
          />
          {isLoading && <div className="search-loading">Loading...</div>}
          {showSuggestions && searchResults.length > 0 && (
            <ul className="search-suggestions">
              {searchResults.map((result) => (
                <li key={result.id} onClick={() => handleSuggestionClick(result.word)}>
                  {result.word}
                </li>
              ))}
            </ul>
          )}
        </div>
        <button
          onClick={() => handleSearch()}
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
            {wordNetwork && mainWord && wordNetwork.nodes && wordNetwork.nodes.length > 0 && (
              <WordGraph
                wordNetwork={wordNetwork}
                mainWord={mainWord}
                onNodeClick={handleNodeClick}
                onNetworkChange={handleNetworkChange}
                initialDepth={depth}
                initialBreadth={breadth}
              />
            )}
          </div>
        </div>
        <div className="word-details-container" ref={detailsContainerRef}>
          {isLoading ? (
            <div className="loading-spinner">Loading word data...</div>
          ) : selectedWordInfo ? (
            <WordDetails 
              wordInfo={selectedWordInfo} 
              onWordClick={handleNodeClick} 
            />
          ) : (
            <div className="no-word-selected">
              <p>Select a word from the graph or search for a word to view details.</p>
            </div>
          )}
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
