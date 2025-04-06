import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import {
  WordNetworkResponse,
  RawWordData,
  WordInfo,
  SearchResponse,
  SearchResultItem,
  SearchOptions,
  EtymologyTree,
  Statistics,
  Definition,
  Etymology,
  Pronunciation,
  PartOfSpeech,
  Language,
  WordForm,
  WordTemplate,
  RawDefinition,
  RawEtymology,
  BasicWord,
  Relation,
  Affixation,
  Credit,
} from "../types";
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
} from "../api/wordApi";
import axios from 'axios';
import DOMPurify from 'dompurify';
import { debounce } from "lodash";

// Helper function to safely parse JSON strings or comma-separated strings
const safeParse = (input: string | Record<string, any> | null | undefined, returnType: 'array' | 'object' = 'array'): any[] | Record<string, any> => {
  if (typeof input === 'object' && input !== null) {
    // If it's already an object, return it directly if object is expected
    // Ensure we return a copy for objects to avoid mutation issues if needed
    return returnType === 'object' ? { ...input } : (Array.isArray(input) && returnType === 'array' ? [...input] : (returnType === 'array' ? [] : {}));
  }
  if (!input || typeof input !== 'string') {
    return returnType === 'array' ? [] : {};
  }
  try {
    // First, try parsing as JSON
    const parsed = JSON.parse(input);
    if (returnType === 'array' && Array.isArray(parsed)) {
      return parsed;
    } else if (returnType === 'object' && typeof parsed === 'object' && !Array.isArray(parsed) && parsed !== null) {
      return parsed;
    } else if (returnType === 'array') {
      // If JSON parse didn't yield an array, treat as single element array if not empty string
      return input.trim() === '' ? [] : [parsed];
    }
    // If expected object but got array/primitive, return empty object
    return {};

  } catch (e) {
    // If JSON parsing fails, assume comma-separated string for arrays
    if (returnType === 'array') {
        // Handle cases like just "tag1" without commas, filter empty strings
        return input.split(',').map(s => s.trim()).filter(Boolean);
    }
    // Cannot parse as object if not valid JSON
    return {};
  }
};

// Helper function to process raw word data into the clean WordInfo type
const processRawWordData = (rawData: RawWordData): WordInfo => {
  // Ensure rawData is not null/undefined
  if (!rawData) {
    console.error("processRawWordData received null or undefined rawData");
    // Return a default/empty WordInfo structure matching the type
    return { id: 0, lemma: 'Error: No Data' }; // Minimal WordInfo
  }

  // Process definitions
  const cleanDefinitions = (rawData.definitions || []).map((def: RawDefinition): Definition => {
     // Simpler handling for related_definitions: Keep them raw for now, or omit if not needed cleaned
     // const relatedDefs = def.related_definitions ? def.related_definitions.filter(r => r != null) : undefined;

    return {
      // Spread raw definition fields first
      ...def,
      // Overwrite/process specific fields
      id: def.id, // Ensure id is present
      definition_text: def.definition_text || '', // Ensure text is string
      examples: safeParse(def.examples, 'array') as string[],
      usage_notes: safeParse(def.usage_notes, 'array') as string[],
      tags: safeParse(def.tags, 'array') as string[],
      sources: safeParse(def.sources, 'array') as string[],
      // Use the nested PartOfSpeech object if available, otherwise null
      part_of_speech: def.standardized_pos || null,
      // Assign raw related definitions (if Definition type allows RawDefinition[]) or omit
      related_definitions: def.related_definitions?.filter(r => r != null) as Definition[] | undefined, // Cast needed? Check Definition type
      // Exclude fields not in the clean Definition type by not including them
    };
  });


  // Process etymologies
  const cleanEtymologies = (rawData.etymologies || []).map((ety: RawEtymology): Etymology => ({
    // Spread raw etymology fields first
    ...ety,
    // Overwrite/process specific fields
    id: ety.id, // Ensure id is present
    etymology_text: ety.etymology_text || '', // Ensure text is string
    components: safeParse(ety.normalized_components, 'array') as string[], // Use normalized_components
    languages: safeParse(ety.language_codes, 'array') as string[],
    sources: safeParse(ety.sources, 'array') as string[],
    // Exclude fields by not including them
  }));

  // Process pronunciations
   const cleanPronunciations = (rawData.pronunciations || []).map((pron): Pronunciation | null => {
    if (!pron) return null;

    const tags = safeParse(pron.tags, 'object') as Record<string, any>;
    const metadata = safeParse(pron.pronunciation_metadata, 'object') as Record<string, any>;
    const sources = safeParse(pron.sources, 'array') as string[];

    const cleaned: Pronunciation = {
        id: pron.id,
        type: pron.type,
        value: pron.value,
        created_at: pron.created_at,
        updated_at: pron.updated_at,
        // Assign cleaned fields, use undefined if empty for cleaner objects
        tags: Object.keys(tags).length > 0 ? tags : undefined,
        pronunciation_metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
        sources: sources.length > 0 ? sources : undefined,
        // Add extracted fields if needed by components, based on type
        ipa: (pron.type?.toLowerCase() === 'ipa') ? pron.value : undefined,
        audio_url: (pron.type?.toLowerCase() === 'audio') ? pron.value : undefined,
    };
    return cleaned;
   }).filter((p): p is Pronunciation => p !== null); // Filter out any nulls from invalid input


  // Process top-level fields requiring parsing
  const cleanTags = safeParse(rawData.tags, 'array') as string[] | null;

  // Process forms - parse tags within each form
  const cleanForms = (rawData.forms || []).map((form: WordForm): WordForm => ({
      ...form,
      tags: safeParse(form.tags, 'object') as Record<string, any> | null,
  }));

  // Process templates - parse args within each template
  const cleanTemplates = (rawData.templates || []).map((template: WordTemplate): WordTemplate => ({
      ...template,
      args: safeParse(template.args, 'object') as Record<string, any> | null,
  }));

    // Process relations - parse metadata and sources
    const processRelation = (rel: Relation): Relation => ({
        ...rel,
        metadata: safeParse(rel.metadata, 'object') as Record<string, any> | null,
        sources: safeParse(rel.sources, 'array') as string[] | null, // Keep as array for now
    });
    const cleanOutgoingRelations = rawData.outgoing_relations?.map(processRelation) || null;
    const cleanIncomingRelations = rawData.incoming_relations?.map(processRelation) || null;

    // Process affixations - parse sources
    const processAffixation = (affix: Affixation): Affixation => ({
        ...affix,
        sources: safeParse(affix.sources, 'array') as string[] | null, // Keep as array for now
    });
    const cleanRootAffixations = rawData.root_affixations?.map(processAffixation) || null;
    const cleanAffixedAffixations = rawData.affixed_affixations?.map(processAffixation) || null;

  // Assemble the cleaned WordInfo object
  const wordInfo: WordInfo = {
    // Core Word fields (copy directly)
    id: rawData.id,
    lemma: rawData.lemma,
    normalized_lemma: rawData.normalized_lemma,
    language_code: rawData.language_code,
    has_baybayin: rawData.has_baybayin,
    baybayin_form: rawData.baybayin_form,
    romanized_form: rawData.romanized_form,
    root_word_id: rawData.root_word_id,
    preferred_spelling: rawData.preferred_spelling,
    data_hash: rawData.data_hash,
    search_text: rawData.search_text,
    created_at: rawData.created_at,
    updated_at: rawData.updated_at,

    // Parsed/Cleaned fields - use null if array is empty after cleaning
    tags: cleanTags && cleanTags.length > 0 ? cleanTags : null,
    definitions: cleanDefinitions.length > 0 ? cleanDefinitions : null,
    etymologies: cleanEtymologies.length > 0 ? cleanEtymologies : null,
    pronunciations: cleanPronunciations.length > 0 ? cleanPronunciations : null,
    forms: cleanForms.length > 0 ? cleanForms : null,
    templates: cleanTemplates.length > 0 ? cleanTemplates : null,
    outgoing_relations: cleanOutgoingRelations && cleanOutgoingRelations.length > 0 ? cleanOutgoingRelations : null,
    incoming_relations: cleanIncomingRelations && cleanIncomingRelations.length > 0 ? cleanIncomingRelations : null,
    root_affixations: cleanRootAffixations && cleanRootAffixations.length > 0 ? cleanRootAffixations : null,
    affixed_affixations: cleanAffixedAffixations && cleanAffixedAffixations.length > 0 ? cleanAffixedAffixations : null,

    // Other related data (copy directly)
    credits: rawData.credits || null,
    root_word: rawData.root_word || null,
    derived_words: rawData.derived_words || null,
    language: rawData.language || null,

    // Metadata - Assign directly, assuming WordInfo type allows the broader RawWordData type
    data_completeness: rawData.data_completeness || null,
    relation_summary: rawData.relation_summary || null,

    // Calculate completeness score
    completeness_score: typeof rawData.data_completeness?.completeness_score === 'number'
                          ? rawData.data_completeness.completeness_score
                          : null,
  };

  return wordInfo;
};

// const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://api.hapinas.net/api/v1'; // KEEP USING V1 if needed for testing?
// OR Switch to V2 if backend is ready
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:10000/api/v2';

const WordExplorer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [wordNetwork, setWordNetwork] = useState<WordNetworkResponse | null>(null);
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
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [apiEndpoint, setApiEndpoint] = useState<string | null>(localStorage.getItem('successful_api_endpoint'));
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [partsOfSpeech, setPartsOfSpeech] = useState<PartOfSpeech[]>([]);
  const [languages, setLanguages] = useState<Language[]>([]);

  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [isLoadingStatistics, setIsLoadingStatistics] = useState<boolean>(false);

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
      const data = await fetchWordNetwork(word, { depth });
      
      setWordNetwork(data);
      return data;
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
      setWordNetwork(null);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchEtymologyTree = useCallback(async (wordId: number) => {
    if (!wordId) return;
    
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    
    try {
      console.log(`Fetching etymology tree for word ID: ${wordId}`);
      const data = await getEtymologyTree(wordId, 3);
      console.log('Etymology tree data:', data);
      setEtymologyTree(data);
    } catch (error) {
      console.error("Error fetching etymology tree:", error);
      let errorMessage = "Failed to fetch etymology tree.";
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (axios.isAxiosError(error)) {
        errorMessage = error.response?.data?.error || error.message;
      }
      setEtymologyError(errorMessage);
      setEtymologyTree(null);
    } finally {
      setIsLoadingEtymology(false);
    }
  }, []);

  const handleSearch = useCallback(async (searchWord?: string) => {
    console.log(`handleSearch called. searchWord: ${searchWord}, inputValue: ${inputValue}`);
    const wordToSearch = searchWord || inputValue.trim();

    console.log(`handleSearch: wordToSearch calculated as: ${wordToSearch}`);

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
    setEtymologyTree(null); 
    setSelectedWordInfo(null);

    try {
      console.log('Attempting to fetch word details directly for:', normalizedInput);
      const rawWordData = await fetchWordDetails(normalizedInput);
      console.log('Raw word details received:', rawWordData);

      if (rawWordData && rawWordData.lemma) {
        console.log('Direct details fetch successful. Processing data...');
        const processedData = processRawWordData(rawWordData);
        console.log('Processed word info:', processedData);

        setSelectedWordInfo(processedData);
        setMainWord(processedData.lemma);

        const [networkData] = await Promise.all([
          fetchWordNetworkData(processedData.lemma, depth),
          fetchEtymologyTree(processedData.id)
        ]);

        console.log('Word network data received:', networkData);
        setWordNetwork(networkData);

        setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), processedData.lemma]);
        setCurrentHistoryIndex(prevIndex => prevIndex + 1);
        setInputValue(processedData.lemma);
        setShowSuggestions(false);
        return;
      } else {
        console.log('Direct word details fetch failed or returned invalid data. Proceeding to search...');
      }
    } catch (detailsError) {
      console.warn('Direct word details fetch threw error (expected if word not exact match), trying search:', detailsError);
    }

    try {
      const searchOptions: SearchOptions = {
        q: normalizedInput,
        limit: 20,
        offset: 0,
        include_full: false
      };
      console.log("Performing search with options:", searchOptions);
      const searchResponse: SearchResponse = await searchWords(searchOptions);
      console.log('Search response received:', searchResponse);

      if (searchResponse && searchResponse.results && searchResponse.results.length > 0) {
        const firstResult = searchResponse.results[0] as SearchResultItem;
        console.log('Using first search result:', firstResult);

        const minimalWordInfo: WordInfo = {
          id: firstResult.id,
          lemma: firstResult.lemma,
          normalized_lemma: firstResult.normalized_lemma,
          language_code: firstResult.language_code,
          has_baybayin: firstResult.has_baybayin,
          baybayin_form: firstResult.baybayin_form,
        };
        setSelectedWordInfo(minimalWordInfo);
        setMainWord(firstResult.lemma);
        setInputValue(firstResult.lemma);
        setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), firstResult.lemma]);
        setCurrentHistoryIndex(prevIndex => prevIndex + 1);
        setShowSuggestions(false);
        setWordNetwork(null);
        setEtymologyTree(null);

        console.log('Fetching full details for search result ID:', firstResult.id);
        fetchWordDetails(firstResult.id.toString())
          .then(rawWordData => {
            if (rawWordData && rawWordData.id === firstResult.id) {
              console.log('Full details received for search result', rawWordData);
              const processedData = processRawWordData(rawWordData);
              setSelectedWordInfo(processedData);
              return Promise.all([
                fetchWordNetworkData(processedData.lemma, depth),
                fetchEtymologyTree(processedData.id)
              ]);
            }
            return [null, null];
          })
          .then(([networkData]) => {
            if (networkData) {
              setWordNetwork(networkData);
            }
          })
          .catch(fullDetailsError => {
            console.error("Error fetching full details for search result:", fullDetailsError);
            setError(`Could not fetch full details for "${firstResult.lemma}". Displaying basic info.`);
          });
      } else {
        setError(`No results found for "${wordToSearch}"`);
        setSelectedWordInfo(null);
        setWordNetwork(null);
        setEtymologyTree(null);
      }
    } catch (searchError: any) {
      console.error("Error during search phase:", searchError);
      setError(searchError.message || "An error occurred during the search.");
      setSelectedWordInfo(null);
      setWordNetwork(null);
      setEtymologyTree(null);
    }
  } catch (err: any) {
    console.error("Unexpected error in handleSearch:", err);
    setError(err.message || "An unexpected error occurred.");
    setSelectedWordInfo(null);
    setWordNetwork(null);
    setEtymologyTree(null);
  } finally {
    setIsLoading(false);
  }
}, [inputValue, depth, fetchWordNetworkData, fetchEtymologyTree, currentHistoryIndex]);

const handleWordLinkClick = useCallback(async (word: string) => {
  console.log("Word link clicked:", word);
  if (word !== mainWord) {
    await handleSearch(word);
    detailsContainerRef.current?.scrollTo(0, 0);
  }
}, [mainWord, handleSearch]);

const handleNodeClick = handleWordLinkClick;

const debouncedSearch = useCallback(
  debounce(async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      setShowSuggestions(false);
      return;
    }
    try {
      const searchOptions: SearchOptions = { q: query, limit: 10, include_full: false };
      const response: SearchResponse = await searchWords(searchOptions);
      if (response && response.results) {
        setSearchResults(response.results as SearchResultItem[]);
        setShowSuggestions(response.results.length > 0);
      } else {
        setSearchResults([]);
        setShowSuggestions(false);
      }
    } catch (error) {
      console.error("Error fetching search suggestions:", error);
      setSearchResults([]);
      setShowSuggestions(false);
    }
  }, 300),
  []
);

const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
  const query = e.target.value;
  setInputValue(query);
  debouncedSearch(query);
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

const handleNetworkChange = useCallback((newDepth: number, newBreadth: number) => {
  setDepth(newDepth);
  if (mainWord) {
    setIsLoading(true);
    fetchWordNetworkData(normalizeInput(mainWord), newDepth)
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

const handleBack = useCallback(() => {
  if (currentHistoryIndex > 0) {
    const newIndex = currentHistoryIndex - 1;
    setCurrentHistoryIndex(newIndex);
    const previousWord = wordHistory[newIndex];
    console.log(`Navigating back to: ${previousWord} (index ${newIndex})`);

    setIsLoading(true);
    setError(null);
    setSelectedWordInfo(null);
    setWordNetwork(null);
    setEtymologyTree(null);

    fetchWordDetails(previousWord)
      .then(rawWordData => {
        if (!rawWordData) throw new Error("Word details not found");
        const processedData = processRawWordData(rawWordData);
        setSelectedWordInfo(processedData);
        setMainWord(processedData.lemma);
        setInputValue(processedData.lemma);
        return Promise.all([
          fetchWordNetworkData(previousWord, depth),
          fetchEtymologyTree(processedData.id)
        ]);
      })
      .then(([networkData]) => {
        setWordNetwork(networkData);
      })
      .catch(error => {
        console.error("Error navigating back:", error);
        setError(`Failed to navigate back to "${previousWord}": ${error.message}`);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }
}, [currentHistoryIndex, wordHistory, depth, fetchWordNetworkData, fetchEtymologyTree]);

const handleForward = useCallback(() => {
  if (currentHistoryIndex < wordHistory.length - 1) {
    const newIndex = currentHistoryIndex + 1;
    setCurrentHistoryIndex(newIndex);
    const nextWord = wordHistory[newIndex];
    console.log(`Navigating forward to: ${nextWord} (index ${newIndex})`);

    setIsLoading(true);
    setError(null);
    setSelectedWordInfo(null);
    setWordNetwork(null);
    setEtymologyTree(null);

    fetchWordDetails(nextWord)
      .then(rawWordData => {
        if (!rawWordData) throw new Error("Word details not found");
        const processedData = processRawWordData(rawWordData);
        setSelectedWordInfo(processedData);
        setMainWord(processedData.lemma);
        setInputValue(processedData.lemma);
        return Promise.all([
          fetchWordNetworkData(nextWord, depth),
          fetchEtymologyTree(processedData.id)
        ]);
      })
      .then(([networkData]) => {
        setWordNetwork(networkData);
      })
      .catch(error => {
        console.error("Error navigating forward:", error);
        setError(`Failed to navigate forward to "${nextWord}": ${error.message}`);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }
}, [currentHistoryIndex, wordHistory, depth, fetchWordNetworkData, fetchEtymologyTree]);

const handleResetCircuitBreaker = () => {
  resetCircuitBreaker();
  setError(null);
  if (inputValue) {
    handleSearch(inputValue);
  }
};

const handleTestApiConnection = async () => {
  setError(null);
  setApiConnected(null);
  
  try {
    console.log("Manually testing API connection...");
    
    const isConnected = await testApiConnection();
    setApiConnected(isConnected);
    
    if (!isConnected) {
      console.log("API connection test failed, trying direct fetch...");
      
      resetCircuitBreaker();
      
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
          localStorage.setItem('successful_api_endpoint', 'http://127.0.0.1:10000');
          
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

useEffect(() => {
  const checkApiConnection = async () => {
    try {
      console.log("Checking API connection...");
      setApiConnected(null);
      setError(null);
      
      const isConnected = await testApiConnection();
      setApiConnected(isConnected);
      
      if (!isConnected) {
        console.log("Initial API connection failed, retrying in 1 second...");
        setError("API connection failed. Retrying...");
        
        resetCircuitBreaker();
        
        setTimeout(async () => {
          console.log("Retrying API connection...");
          setApiConnected(null);
          const retryResult = await testApiConnection();
          setApiConnected(retryResult);
          
          if (!retryResult) {
            setTimeout(async () => {
              console.log("Final API connection retry...");
              setApiConnected(null);
              
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
            setError(null);
            console.log("API connection successful on retry!");
          }
        }, 1000);
      } else {
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

useEffect(() => {
  const fetchInitialData = async () => {
    try {
      const [posData, langData] = await Promise.all([
        getPartsOfSpeech(),
        Promise.resolve([{ code: 'tl', name_en: 'Tagalog' }, { code: 'en', name_en: 'English' }])
      ]);
      setPartsOfSpeech(posData || []);
      setLanguages(langData || []);
    } catch (error) {
      console.error("Error fetching initial POS/Language data:", error);
      setLanguages([{ code: 'tl', name_en: 'Tagalog' }, { code: 'en', name_en: 'English' }]);
    }
  };
  fetchInitialData();
}, []);

const handleRandomWord = useCallback(async () => {
  setError(null);
  setIsLoading(true);
  setEtymologyTree(null);
  setSelectedWordInfo(null);
  setWordNetwork(null);

  try {
    console.log('Fetching random word (V2)');
    const rawWordData = await getRandomWord();
    console.log('Raw random word data:', rawWordData);

    if (rawWordData && rawWordData.lemma) {
      const processedData = processRawWordData(rawWordData);
      setSelectedWordInfo(processedData);
      setMainWord(processedData.lemma);
      setInputValue(processedData.lemma);
      setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), processedData.lemma]);
      setCurrentHistoryIndex(prevIndex => prevIndex + 1);

      await Promise.all([
        fetchWordNetworkData(processedData.lemma, depth),
        fetchEtymologyTree(processedData.id)
      ]);
    } else {
      throw new Error("Could not fetch valid random word data.");
    }
  } catch (error) {
    console.error("Error fetching random word:", error);
    let errorMessage = "Failed to fetch random word.";
    if (error instanceof Error) {
      errorMessage = error.message;
    } else if (axios.isAxiosError(error)) {
      errorMessage = error.response?.data?.error || error.message;
    }
    setError(errorMessage);
  } finally {
    setIsLoading(false);
  }
}, [depth, currentHistoryIndex, fetchWordNetworkData, fetchEtymologyTree]);

useEffect(() => {
  const fetchStaticData = async () => {
    setIsLoadingStatistics(true);
    try {
      const statsData = await getStatistics();
      setStatistics(statsData);
    } catch (error) {
      console.error("Error fetching statistics:", error);
    } finally {
      setIsLoadingStatistics(false);
    }
  };
  fetchStaticData();
}, []);

console.log('Search query:', inputValue);
console.log('Search results:', searchResults);
console.log('Show suggestions:', showSuggestions);

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
}, []);

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
              handleSearch();
            }
          }}
          placeholder="Enter a word"
          className="search-input"
          aria-label="Search word"
        />
        {showSuggestions && searchResults.length > 0 && (
          <ul className="search-suggestions">
            {searchResults.map((result: SearchResultItem) => (
              <li key={result.id} onClick={() => handleSuggestionClick(result.lemma)}>
                {result.lemma}
                {result.language_code && ` (${result.language_code})`}
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
          {isLoading && <div className="loading">Loading Network...</div>}
          {!isLoading && wordNetwork && (
            <WordGraph
              wordNetwork={wordNetwork}
              mainWord={mainWord}
              onNodeClick={handleNodeClick}
              onNetworkChange={handleNetworkChange}
              initialDepth={depth}
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
