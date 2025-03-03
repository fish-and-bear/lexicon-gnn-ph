import React, { useState, useCallback, useEffect } from "react";
import WordGraph from "./WordGraph";
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
        console.log('Word details:', detailsData);
        
        if (detailsData && detailsData.lemma) {
          setSelectedWordInfo(detailsData);
          setMainWord(detailsData.lemma);
          
          // Fetch word network and etymology tree in parallel
          const [networkData] = await Promise.all([
            fetchWordNetworkData(normalizedInput, depth, breadth),
            fetchEtymologyTree(normalizedInput) // This updates state directly
          ]);
          
          console.log('Word network:', networkData);
          
          setWordNetwork(networkData);
          setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), detailsData.lemma]);
          setCurrentHistoryIndex(prevIndex => prevIndex + 1);
          setInputValue(detailsData.lemma);
          setShowSuggestions(false);
          setIsLoading(false);
          return;
        }
      } catch (detailsError) {
        console.log('Direct word details fetch failed, trying search:', detailsError);
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

    return (
      <div className="definitions-section">
        <div className="definitions-section-header">
          <h3>Definitions</h3>
          <span className="definition-count">{wordInfo.definitions.length}</span>
        </div>
        
        {Object.entries(definitionsByPos).map(([posName, definitions]: [string, any[]]) => (
          <div key={posName} className="pos-group">
            <div className="pos-group-header">
              {posName}
              <span className="pos-count">{definitions.length}</span>
            </div>
            
            {definitions.map((definition: Definition, index: number) => (
              <div key={index} className="definition-card">
                <p className="definition-text">{definition.text}</p>
                
                {definition.examples && definition.examples.length > 0 && (
                  <div className="examples">
                    <h4>Examples</h4>
                    <ul>
                      {definition.examples.map((example: string, idx: number) => (
                        <li key={idx}>{example}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {definition.usage_notes && definition.usage_notes.length > 0 && (
                  <div className="usage-notes">
                    <h4>Usage Notes</h4>
                    <ul>
                      {definition.usage_notes.map((note: string, idx: number) => (
                        <li key={idx}>{note}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {definition.sources && definition.sources.length > 0 && (
                  <div className="definition-sources">
                    <span className="sources-label">Sources:</span>
                    {definition.sources.map((source: string, idx: number) => (
                      <span key={idx} className="source-tag">{source}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
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
        <div className="details-container">
          <div className="details-content">
            {isLoading ? (
              <div className="loading-spinner">Loading word data...</div>
            ) : selectedWordInfo ? (
              <div className="word-details">
                <div className="word-header">
                  <h2>{selectedWordInfo.lemma}</h2>
                  <div className="word-meta">
                    {selectedWordInfo.language_code && (
                      <span className="language">{selectedWordInfo.language_code}</span>
                    )}
                    {selectedWordInfo.has_baybayin && selectedWordInfo?.baybayin_form && (
                      <span className="baybayin-badge" title="Has Baybayin script">
                        {selectedWordInfo.baybayin_form}
                      </span>
                    )}
                    {selectedWordInfo?.tags && selectedWordInfo.tags.length > 0 && (
                      <div className="tags-container">
                        {selectedWordInfo.tags.map((tag, index) => (
                          <span key={index} className="tag">{tag}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="word-info-grid">
                  <div className="word-info-column main-info">
                    {selectedWordInfo?.pronunciation?.text && (
                      <div className="pronunciation-section">
                        <h3>Pronunciation</h3>
                        <div className="pronunciation-content">
                          <span className="pronunciation-text">/{selectedWordInfo.pronunciation.text}/</span>
                          {selectedWordInfo.pronunciation.ipa && (
                            <span className="ipa">[{selectedWordInfo.pronunciation.ipa}]</span>
                          )}
                          {selectedWordInfo.pronunciation.audio_url && (
                            <button 
                              className="play-audio"
                              onClick={() => {
                                const audio = new Audio(selectedWordInfo.pronunciation?.audio_url);
                                audio.play().catch(console.error);
                              }}
                            >
                              🔊
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {selectedWordInfo?.etymologies && selectedWordInfo.etymologies.length > 0 && (
                      <div className="etymology-section">
                        <h3>Etymology</h3>
                        <div className="etymology-content">
                          {selectedWordInfo.etymologies.map((etymology, index) => (
                            <div key={index} className="etymology-item">
                              <p className="etymology-text">{etymology.text}</p>
                              {etymology.languages && etymology.languages.length > 0 && (
                                <div className="etymology-languages">
                                  {etymology.languages.map((lang, i) => (
                                    <span key={i} className="language-tag">{lang}</span>
                                  ))}
                                </div>
                              )}
                              {etymology.sources && etymology.sources.length > 0 && (
                                <div className="etymology-sources">
                                  <span className="sources-label">Sources:</span>
                                  {etymology.sources.map((source, i) => (
                                    <span key={i} className="source-tag">{source}</span>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {renderDefinitions(selectedWordInfo!)}
                    
                    {/* Idioms Section - Moved here to be more prominent */}
                    {renderIdioms(selectedWordInfo!)}
                  </div>
                  
                  <div className="word-info-column side-info">
                    {selectedWordInfo?.has_baybayin && selectedWordInfo?.baybayin_form && (
                      <div className="baybayin-section card-section">
                        <h3>Baybayin Script</h3>
                        <div className="baybayin-display">
                          <p className="baybayin-text">{selectedWordInfo.baybayin_form}</p>
                          {selectedWordInfo.romanized_form && (
                            <p className="romanized-text">Romanized: {selectedWordInfo.romanized_form}</p>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {selectedWordInfo?.source_info && Object.keys(selectedWordInfo.source_info).length > 0 && (
                      <div className="sources-section card-section">
                        <h3>Sources</h3>
                        <ul className="sources-list">
                          {Object.entries(selectedWordInfo.source_info).map(([source, info], index) => (
                            <li key={index} className="source-item">
                              <span className="source-name">{source}</span>
                              {typeof info === 'string' ? (
                                <span className="source-info">{info}</span>
                              ) : (
                                <span className="source-info">✓</span>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {/* Metadata toggle button */}
                    <button 
                      className="metadata-toggle"
                      onClick={() => setShowMetadata(!showMetadata)}
                    >
                      {showMetadata ? 'Hide Technical Details' : 'Show Technical Details'}
                    </button>
                    
                    {/* Collapsible metadata section */}
                    {showMetadata && (
                      <div className="technical-details">
                        {selectedWordInfo?.complexity_score && (
                          <div className="complexity-section card-section">
                            <h3>Word Complexity</h3>
                            <div className="complexity-meter">
                              <div 
                                className="complexity-fill" 
                                style={{width: `${Math.min(100, selectedWordInfo.complexity_score)}%`}}
                              ></div>
                              <span className="complexity-value">{selectedWordInfo.complexity_score}/100</span>
                            </div>
                          </div>
                        )}
                        
                        {selectedWordInfo?.data_quality_score && (
                          <div className="data-quality-section card-section">
                            <h3>Data Quality</h3>
                            <div className="quality-meter">
                              <div 
                                className="quality-fill" 
                                style={{width: `${Math.min(100, selectedWordInfo.data_quality_score)}%`}}
                              ></div>
                              <span className="quality-value">{selectedWordInfo.data_quality_score}/100</span>
                            </div>
                          </div>
                        )}
                        
                        {selectedWordInfo?.created_at && (
                          <div className="metadata-section card-section">
                            <h3>Metadata</h3>
                            <div className="metadata-grid">
                              <div className="metadata-item">
                                <span className="metadata-label">Created:</span>
                                <span className="metadata-value">
                                  {new Date(selectedWordInfo.created_at).toLocaleDateString()}
                                </span>
                              </div>
                              {selectedWordInfo.updated_at && (
                                <div className="metadata-item">
                                  <span className="metadata-label">Updated:</span>
                                  <span className="metadata-value">
                                    {new Date(selectedWordInfo.updated_at).toLocaleDateString()}
                                  </span>
                                </div>
                              )}
                              {selectedWordInfo.view_count !== undefined && (
                                <div className="metadata-item">
                                  <span className="metadata-label">Views:</span>
                                  <span className="metadata-value">{selectedWordInfo.view_count}</span>
                                </div>
                              )}
                              {selectedWordInfo.is_verified !== undefined && (
                                <div className="metadata-item">
                                  <span className="metadata-label">Verified:</span>
                                  <span className="metadata-value">
                                    {selectedWordInfo.is_verified ? '✓' : '✗'}
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Etymology Tree Section */}
                {etymologyTree && (
                  <div className="etymology-tree-section">
                    <h3>Etymology Tree</h3>
                    <div className="etymology-tree-container">
                      {isLoadingEtymology ? (
                        <div className="loading-spinner">Loading etymology tree...</div>
                      ) : etymologyError ? (
                        <div className="error-message">
                          <p>{etymologyError}</p>
                        </div>
                      ) : (
                        <div className="etymology-tree-content">
                          <div className="etymology-root">
                            <h4>{etymologyTree.word}</h4>
                            {etymologyTree.etymologies && etymologyTree.etymologies.length > 0 && (
                              <p className="etymology-text">{etymologyTree.etymologies[0].text}</p>
                            )}
                          </div>
                          
                          {etymologyTree.components && etymologyTree.components.length > 0 && (
                            <div className="etymology-components">
                              <h4>Components</h4>
                              <ul className="component-list">
                                {etymologyTree.components.map((component, index) => (
                                  <li key={index} className="component-item">
                                    <span 
                                      className="clickable-word"
                                      onClick={() => handleNodeClick(component)}
                                    >
                                      {component}
                                    </span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {etymologyTree.component_words && etymologyTree.component_words.length > 0 && (
                            <div className="component-words">
                              <h4>Component Words</h4>
                              <div className="component-words-list">
                                {etymologyTree.component_words.map((compWord, index) => (
                                  <div key={index} className="component-word-card">
                                    <h5 
                                      className="clickable-word"
                                      onClick={() => handleNodeClick(compWord.word)}
                                    >
                                      {compWord.word}
                                    </h5>
                                    {compWord.etymologies && compWord.etymologies.length > 0 && (
                                      <p>{compWord.etymologies[0].text}</p>
                                    )}
                                    {compWord.components && compWord.components.length > 0 && (
                                      <div className="sub-components">
                                        <span>Components: </span>
                                        {compWord.components.map((subComp, idx) => (
                                          <span 
                                            key={idx}
                                            className="clickable-word sub-component"
                                            onClick={() => handleNodeClick(subComp)}
                                          >
                                            {subComp}
                                          </span>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Relations Section */}
                {selectedWordInfo?.relations && (
                  <div className="relations-section">
                    <h3>Word Relations</h3>
                    <div className="relations-grid">
                      {selectedWordInfo.relations.synonyms && selectedWordInfo.relations.synonyms.length > 0 && (
                        <div className="relation-group synonyms">
                          <h4>Synonyms</h4>
                          <div className="relation-tags">
                            {selectedWordInfo.relations.synonyms.map((item, index) => (
                              <span 
                                key={index} 
                                className="relation-tag"
                                onClick={() => handleNodeClick(item.word)}
                              >
                                {item.word}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {selectedWordInfo.relations.antonyms && selectedWordInfo.relations.antonyms.length > 0 && (
                        <div className="relation-group antonyms">
                          <h4>Antonyms</h4>
                          <div className="relation-tags">
                            {selectedWordInfo.relations.antonyms.map((item, index) => (
                              <span 
                                key={index} 
                                className="relation-tag"
                                onClick={() => handleNodeClick(item.word)}
                              >
                                {item.word}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {selectedWordInfo.relations.variants && selectedWordInfo.relations.variants.length > 0 && (
                        <div className="relation-group variants">
                          <h4>Variants</h4>
                          <div className="relation-tags">
                            {selectedWordInfo.relations.variants.map((item, index) => (
                              <span 
                                key={index} 
                                className="relation-tag"
                                onClick={() => handleNodeClick(item.word)}
                              >
                                {item.word}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {selectedWordInfo.relations.related && selectedWordInfo.relations.related.length > 0 && (
                        <div className="relation-group related">
                          <h4>Related Words</h4>
                          <div className="relation-tags">
                            {selectedWordInfo.relations.related.map((item, index) => (
                              <span 
                                key={index} 
                                className="relation-tag"
                                onClick={() => handleNodeClick(item.word)}
                              >
                                {item.word}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {selectedWordInfo.relations.derived && selectedWordInfo.relations.derived.length > 0 && (
                        <div className="relation-group derived">
                          <h4>Derived Words</h4>
                          <div className="relation-tags">
                            {selectedWordInfo.relations.derived.map((item, index) => (
                              <span 
                                key={index} 
                                className="relation-tag"
                                onClick={() => handleNodeClick(item.word)}
                              >
                                {item.word}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {selectedWordInfo.relations.root && (
                        <div className="relation-group root">
                          <h4>Root Word</h4>
                          <div className="relation-tags">
                            <span 
                              className="relation-tag root-tag"
                              onClick={() => handleNodeClick(selectedWordInfo.relations.root!.word)}
                            >
                              {selectedWordInfo.relations.root.word}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Affixations Section */}
                {selectedWordInfo?.affixations && (
                  (selectedWordInfo.affixations.as_root.length > 0 || 
                   selectedWordInfo.affixations.as_affixed.length > 0) && (
                    <div className="affixations-section">
                      <h3>Affixations</h3>
                      
                      {selectedWordInfo.affixations.as_root.length > 0 && (
                        <div className="affixation-group">
                          <h4>As Root Word</h4>
                          <div className="affixation-list">
                            {selectedWordInfo.affixations.as_root.map((aff, index) => (
                              <div key={index} className="affixation-card">
                                <div className="affixation-type">{aff.type}</div>
                                <div 
                                  className="affixation-word"
                                  onClick={() => handleNodeClick(aff.affixed_word)}
                                >
                                  {aff.affixed_word}
                                </div>
                                {aff.sources && aff.sources.length > 0 && (
                                  <div className="affixation-sources">
                                    {aff.sources.map((source, i) => (
                                      <span key={i} className="source-tag">{source}</span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {selectedWordInfo.affixations.as_affixed.length > 0 && (
                        <div className="affixation-group">
                          <h4>As Affixed Word</h4>
                          <div className="affixation-list">
                            {selectedWordInfo.affixations.as_affixed.map((aff, index) => (
                              <div key={index} className="affixation-card">
                                <div className="affixation-type">{aff.type}</div>
                                <div 
                                  className="affixation-word"
                                  onClick={() => handleNodeClick(aff.root_word)}
                                >
                                  {aff.root_word}
                                </div>
                                {aff.sources && aff.sources.length > 0 && (
                                  <div className="affixation-sources">
                                    {aff.sources.map((source, i) => (
                                      <span key={i} className="source-tag">{source}</span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )
                )}
              </div>
            ) : (
              <div className="empty-state">
                <p>Search for a word or click "Random Word" to get started.</p>
                <p>Explore the Filipino Dictionary with over {statistics?.words?.total || 'thousands of'} words!</p>
              </div>
            )}
          </div>
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
