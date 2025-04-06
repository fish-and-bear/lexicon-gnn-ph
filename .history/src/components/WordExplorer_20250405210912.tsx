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
    return returnType === 'object' ? { ...input } : (Array.isArray(input) && returnType === 'array' ? [...input] : (returnType === 'array' ? [] : {}));
  }
  if (!input || typeof input !== 'string') {
    return returnType === 'array' ? [] : {};
  }
  try {
    const parsed = JSON.parse(input);
    if (returnType === 'array' && Array.isArray(parsed)) {
      return parsed;
    } else if (returnType === 'object' && typeof parsed === 'object' && !Array.isArray(parsed) && parsed !== null) {
      return parsed;
    } else if (returnType === 'array') {
      return input.trim() === '' ? [] : [parsed];
    }
    return {};
  } catch (e) {
    if (returnType === 'array') {
        return input.split(',').map(s => s.trim()).filter(Boolean);
    }
    return {};
  }
};

// Helper function to process raw word data into the clean WordInfo type
const processRawWordData = (rawData: RawWordData): WordInfo => {
  if (!rawData) {
    console.error("processRawWordData received null or undefined rawData");
    return { id: 0, lemma: 'Error: No Data' };
  }
  const cleanDefinitions = (rawData.definitions || []).map((def: RawDefinition): Definition => {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { related_definitions, standardized_pos_id, original_pos, ...restOfDef } = def;
    return {
      ...restOfDef,
      id: def.id,
      definition_text: def.definition_text || '',
      examples: safeParse(def.examples, 'array') as string[],
      usage_notes: safeParse(def.usage_notes, 'array') as string[],
      tags: safeParse(def.tags, 'array') as string[],
      sources: safeParse(def.sources, 'array') as string[],
      part_of_speech: def.standardized_pos || null,
    };
  });
  const cleanEtymologies = (rawData.etymologies || []).map((ety: RawEtymology): Etymology => {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { normalized_components, etymology_structure, language_codes, ...restOfEty } = ety;
    return {
      ...restOfEty,
      id: ety.id,
      etymology_text: ety.etymology_text || '',
      components: safeParse(ety.normalized_components, 'array') as string[],
      languages: safeParse(ety.language_codes, 'array') as string[],
      sources: safeParse(ety.sources, 'array') as string[],
    };
  });
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
        tags: Object.keys(tags).length > 0 ? tags : undefined,
        pronunciation_metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
        sources: sources.length > 0 ? sources : undefined,
        ipa: (pron.type?.toLowerCase() === 'ipa') ? pron.value : undefined,
        audio_url: (pron.type?.toLowerCase() === 'audio') ? pron.value : undefined,
    };
    return cleaned;
   }).filter((p): p is Pronunciation => p !== null);
  const cleanTags = safeParse(rawData.tags, 'array') as string[] | null;
  const cleanForms = (rawData.forms || []).map((form: WordForm): WordForm => ({
      ...form,
      tags: safeParse(form.tags, 'object') as Record<string, any> | null,
  }));
  const cleanTemplates = (rawData.templates || []).map((template: WordTemplate): WordTemplate => ({
      ...template,
      args: safeParse(template.args, 'object') as Record<string, any> | null,
  }));
    const processRelation = (rel: Relation): Relation => ({
        ...rel,
        metadata: safeParse(rel.metadata, 'object') as Record<string, any> | null,
        sources: safeParse(rel.sources, 'array') as any, // TEMP CAST
    });
    const cleanOutgoingRelations = rawData.outgoing_relations?.map(processRelation) || null;
    const cleanIncomingRelations = rawData.incoming_relations?.map(processRelation) || null;
    const processAffixation = (affix: Affixation): Affixation => ({
        ...affix,
        sources: safeParse(affix.sources, 'array') as any, // TEMP CAST
    });
    const cleanRootAffixations = rawData.root_affixations?.map(processAffixation) || null;
    const cleanAffixedAffixations = rawData.affixed_affixations?.map(processAffixation) || null;
  const wordInfo: WordInfo = {
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
    tags: cleanTags && cleanTags.length > 0 ? (cleanTags as any) : null, // TEMP CAST
    definitions: cleanDefinitions.length > 0 ? cleanDefinitions : null,
    etymologies: cleanEtymologies.length > 0 ? cleanEtymologies : null,
    pronunciations: cleanPronunciations.length > 0 ? cleanPronunciations : null,
    forms: cleanForms.length > 0 ? cleanForms : null,
    templates: cleanTemplates.length > 0 ? cleanTemplates : null,
    outgoing_relations: cleanOutgoingRelations && cleanOutgoingRelations.length > 0 ? cleanOutgoingRelations : null,
    incoming_relations: cleanIncomingRelations && cleanIncomingRelations.length > 0 ? cleanIncomingRelations : null,
    root_affixations: cleanRootAffixations && cleanRootAffixations.length > 0 ? cleanRootAffixations : null,
    affixed_affixations: cleanAffixedAffixations && cleanAffixedAffixations.length > 0 ? cleanAffixedAffixations : null,
    credits: rawData.credits || null,
    root_word: rawData.root_word || null,
    derived_words: rawData.derived_words || null,
    language: rawData.language || null,
    data_completeness: rawData.data_completeness as any || null, // TEMP CAST
    relation_summary: rawData.relation_summary || null,
    completeness_score: typeof rawData.data_completeness?.completeness_score === 'number'
                          ? rawData.data_completeness.completeness_score
                          : null,
  };
  return wordInfo;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:10000/api/v2';

const WordExplorer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [wordNetwork, setWordNetwork] = useState<WordNetworkResponse | null>(null);
  const [selectedWordInfo, setSelectedWordInfo] = useState<WordInfo | null>(null);
  const [mainWord, setMainWord] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [depth, setDepth] = useState<number>(2);
  const [inputValue, setInputValue] = useState<string>("");
  const [wordHistory, setWordHistory] = useState<string[]>([]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState<number>(-1);
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);
  const [partsOfSpeech, setPartsOfSpeech] = useState<PartOfSpeech[]>([]);
  const [languages, setLanguages] = useState<Language[]>([]);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [isLoadingStatistics, setIsLoadingStatistics] = useState<boolean>(false);

  const detailsContainerRef = useRef<HTMLDivElement>(null);
  const searchContainerRef = useRef<HTMLDivElement>(null);

  const { theme, toggleTheme } = useTheme();
  const [showMetadata, setShowMetadata] = useState(false);

  const normalizeInput = (input: string): string => {
    return unidecode(input.toLowerCase().trim());
  };

  const fetchWordNetworkData = useCallback(async (word: string, currentDepth: number = 2) => {
    try {
      setIsLoading(true);
      const data = await fetchWordNetwork(word, { depth: currentDepth });
      setWordNetwork(data);
      return data;
    } catch (fetchError) {
      console.error("Error in fetchWordNetworkData:", fetchError);
      setWordNetwork(null);
      throw fetchError;
    } finally {
    }
  }, []);

  const fetchEtymologyTree = useCallback(async (wordId: number) => {
    if (!wordId) return;
    setIsLoadingEtymology(true);
    setEtymologyError(null);
    try {
      const data = await getEtymologyTree(wordId, 3);
      setEtymologyTree(data);
    } catch (fetchError) {
      console.error("Error fetching etymology tree:", fetchError);
      let errorMessage = "Failed to fetch etymology tree.";
      if (fetchError instanceof Error) errorMessage = fetchError.message;
      else if (axios.isAxiosError(fetchError)) errorMessage = fetchError.response?.data?.error || fetchError.message;
      setEtymologyError(errorMessage);
      setEtymologyTree(null);
    } finally {
      setIsLoadingEtymology(false);
    }
  }, []);

  const handleSearch = useCallback(async (wordToSearchParam?: string) => {
    const wordToSearch = wordToSearchParam ?? inputValue;
    const normalizedInput = normalizeInput(wordToSearch);
    if (!normalizedInput) return;

    console.log('Starting search for normalized input:', normalizedInput);
    setIsLoading(true);
    setError(null);
    setWordNetwork(null);
    setEtymologyTree(null);
    setSelectedWordInfo(null);
    setMainWord(normalizedInput);

    try {
      console.log('Attempting direct fetch for:', normalizedInput);
      let rawWordData: RawWordData | null = null;
      try {
          rawWordData = await fetchWordDetails(normalizedInput);
      } catch (detailsError) {
          console.warn('Direct fetch failed (may be expected):', detailsError);
      }

      if (rawWordData && rawWordData.lemma) {
        console.log('Direct fetch successful:', rawWordData);
        const processedData: WordInfo = processRawWordData(rawWordData);
        setSelectedWordInfo(processedData);
        setMainWord(processedData.lemma);
        setInputValue(processedData.lemma);

        setWordHistory(prev => [...prev.slice(0, currentHistoryIndex + 1), processedData.lemma]);
        setCurrentHistoryIndex(prev => prev + 1);
        setShowSuggestions(false);

        await Promise.all([
          fetchWordNetworkData(processedData.lemma, depth),
          fetchEtymologyTree(processedData.id)
        ]);
      } else {
        console.log('Direct fetch failed, performing search...');
        const searchOptions: SearchOptions = {
            q: normalizedInput,
            limit: 20,
            offset: 0,
            include_full: false
        };
        const searchResponse: SearchResponse = await searchWords(searchOptions);
        console.log('Search response:', searchResponse);

        if (searchResponse && searchResponse.results && searchResponse.results.length > 0) {
            const firstResult = searchResponse.results[0] as SearchResultItem;
            console.log('Using first search result:', firstResult);

            const minimalWordInfo: WordInfo = {
                id: firstResult.id,
                lemma: firstResult.lemma,
            };
            setSelectedWordInfo(minimalWordInfo);
            setMainWord(firstResult.lemma);
            setInputValue(firstResult.lemma);
            setWordHistory(prev => [...prev.slice(0, currentHistoryIndex + 1), firstResult.lemma]);
            setCurrentHistoryIndex(prev => prev + 1);
            setInputValue(firstResult.lemma);
            setShowSuggestions(false);

            console.log('Fetching full details for search result ID:', firstResult.id);
            fetchWordDetails(firstResult.id.toString())
              .then(fullRawData => {
                if (fullRawData && selectedWordInfo && fullRawData.id === selectedWordInfo.id) {
                  console.log('Full details received for search result', fullRawData);
                  const fullProcessedData: WordInfo = processRawWordData(fullRawData);
                  setSelectedWordInfo(fullProcessedData);
                  return Promise.all([
                    fetchWordNetworkData(fullProcessedData.lemma, depth),
                    fetchEtymologyTree(fullProcessedData.id)
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
      }
    } catch (searchError: any) {
      console.error("Error during search phase:", searchError);
      setError(searchError.message || "An error occurred during the search.");
      setSelectedWordInfo(null);
      setWordNetwork(null);
      setEtymologyTree(null);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, depth, fetchWordNetworkData, fetchEtymologyTree, currentHistoryIndex]);

  const handleWordLinkClick = useCallback((word: string) => {
    console.log(`Word link clicked: ${word}`);
    const normalizedWord = normalizeInput(word);
    handleSearch(normalizedWord);
  }, [handleSearch]);

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

  const handleNetworkChange = useCallback((newDepth: number) => {
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
