import React, { useState, useCallback, useEffect, useRef } from "react";
import WordGraph from "./WordGraph";
import WordDetails from "./WordDetails";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchResultItem, SearchOptions, EtymologyTree, Statistics, Definition, WordNetworkOptions, Etymology } from "../types";
import unidecode from "unidecode";
import { 
  fetchWordNetwork, 
  fetchWordComprehensive,
  searchWords, 
  resetCircuitBreaker, 
  testApiConnection,
  getEtymologyTree,
  getRandomWord,
  getStatistics,
  getEtymologyTreeData,
} from "../api/wordApi";
import axios from 'axios';
import DOMPurify from 'dompurify';
import { debounce } from "lodash";
import { useWordDetails } from '../hooks/useWordDetails';
import { useWordNetwork } from '../hooks/useWordNetwork';
import { useNavigate, useLocation, useParams } from 'react-router-dom';
import { Box, CircularProgress, Typography } from '@mui/material';
import SearchBar from './SearchBar';
import SearchResultsList from './SearchResultsList';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://api.hapinas.net/api/v1';

const WordExplorer: React.FC = () => {
  const { theme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const { word: wordFromParams } = useParams<{ word?: string }>();

  // --- State Management ---

  // Search State (using local state, not useWordSearch hook)
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]); // Use specific item type
  const [isSearchLoading, setIsSearchLoading] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [offset, setOffset] = useState<number>(0);
  const [limit, setLimit] = useState<number>(20);
  const [hasMore, setHasMore] = useState<boolean>(false);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Selected Word State (driven by URL)
  const [selectedWord, setSelectedWord] = useState<string | null>(null);
  
  // Data Fetching Hooks (driven by selectedWord)
  const { wordInfo: selectedWordInfo, isLoading: isLoadingDetails, error: detailsError } = useWordDetails(selectedWord);
  const [graphDepth, setGraphDepth] = useState(3);
  const [graphBreadth, setGraphBreadth] = useState(8);
  const { wordNetwork, isLoading: isLoadingNetwork, error: networkError } = useWordNetwork(selectedWord, graphDepth, graphBreadth);

  // Etymology State
  const [etymologyTree, setEtymologyTree] = useState<EtymologyTree | null>(null);
  const [isLoadingEtymology, setIsLoadingEtymology] = useState<boolean>(false);
  const [etymologyError, setEtymologyError] = useState<string | null>(null);

  // Metadata Toggle State
  const [showMetadata, setShowMetadata] = useState<boolean>(false);

  // --- Effects ---

  // Effect 1: Sync selectedWord state with URL parameter
  useEffect(() => {
    const currentWord = wordFromParams ? decodeURIComponent(wordFromParams) : null;
    // Only update if the word from params is different from the current selected word
    if (currentWord !== selectedWord) {
        setSelectedWord(currentWord);
        setSearchTerm(currentWord || ''); // Update search bar text
        setSearchResults([]); // Clear search results when navigating directly
        setOffset(0);
        setHasMore(false);
        setShowMetadata(false); // Reset metadata view on word change
        // Data fetching is handled by hooks reacting to selectedWord change
    }
  }, [wordFromParams, selectedWord]); // Depend on URL param and local selectedWord

  // Effect 2: Perform debounced search when searchTerm changes
  const performSearch = useCallback(async (query: string, currentOffset: number, replace = false) => {
      if (!query) {
          setSearchResults([]); setHasMore(false); setSearchError(null);
          return;
      }
      setIsSearchLoading(true); setSearchError(null);
      try {
          const options = { query, offset: currentOffset, limit };
          // Assuming searchWords returns { words: SearchResultItem[], pagination: { has_more: boolean } }
          const data = await searchWords(options);
          setSearchResults(prev => replace ? (data.words || []) : [...prev, ...(data.words || [])]);
          setHasMore(data.pagination ? data.pagination.has_more : (data.words || []).length === limit);
          setOffset(currentOffset + (data.words || []).length);
      } catch (err: any) {
          console.error("Search error:", err);
          setSearchError(err.message || 'Failed search');
          setSearchResults([]); setHasMore(false);
      } finally { setIsSearchLoading(false); }
  }, [limit]);

  useEffect(() => {
     if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
      debounceTimerRef.current = setTimeout(() => {
          // Only perform search if the term is not the same as the currently selected word
          // (prevents re-searching when navigating to a word)
          if (searchTerm && searchTerm !== selectedWord) {
              setOffset(0); 
              performSearch(searchTerm, 0, true);
          } else if (!searchTerm) {
              setSearchResults([]); setHasMore(false); setOffset(0); setSearchError(null);
          }
      }, 300);
      return () => { if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current); };
  }, [searchTerm, selectedWord, performSearch]); // Depend on searchTerm and selectedWord

  // Effect 3: Fetch Etymology Tree when selected word details (and ID) are available
  useEffect(() => {
    const wordId = selectedWordInfo?.id; // Get ID from the loaded details

    if (wordId) {
        const fetchEtymology = async () => {
          setIsLoadingEtymology(true); setEtymologyError(null);
          try {
            // Assuming getEtymologyTree takes wordId (number) and depth (optional)
            const fetchedTree = await getEtymologyTree(wordId, 3); // Fetch with default depth
            setEtymologyTree(fetchedTree);
          } catch (err: any) {
            console.error("Etymology fetch error:", err);
            setEtymologyError(err.message || 'Failed load');
            setEtymologyTree(null);
          } finally { setIsLoadingEtymology(false); }
        };
        fetchEtymology();
    } else {
      // Clear tree if no wordId is available (e.g., details not loaded or word has no ID)
      setEtymologyTree(null);
      setIsLoadingEtymology(false);
      setEtymologyError(null);
    }
  }, [selectedWordInfo]); // Trigger when selectedWordInfo changes

  // --- Handlers ---

  // Update search term state (triggers debounced search effect)
  const handleSearchChange = (newQuery: string) => {
    setSearchTerm(newQuery);
  };

  // Load more search results
  const loadMoreResults = () => {
    if (hasMore && !isSearchLoading) {
      performSearch(searchTerm, offset, false); // Append results
    }
  };

  // Navigate to a word's page (triggers URL change -> effect 1 -> hook updates)
  const handleWordClick = (word: string) => {
    if (word && word !== selectedWord) {
      navigate(`/explore/${encodeURIComponent(word)}`);
    }
  };

  // Handler for graph node clicks (same as general word click)
  const handleGraphNodeClick = (word: string) => {
    handleWordClick(word);
  };

  // Handler for etymology node clicks
  const handleEtymologyNodeClick = (node: any) => { 
    // Extract word label from node data (structure might vary)
    const wordLabel = node?.label || node?.data?.label || node?.data?.word;
    if (wordLabel && typeof wordLabel === 'string') {
        handleWordClick(wordLabel);
    } else {
        console.warn("Could not extract word label from etymology node:", node);
    }
  };

  // Update graph parameters (triggers useWordNetwork hook update)
  const handleNetworkSettingsChange = (depth: number, breadth: number) => {
    setGraphDepth(depth);
    setGraphBreadth(breadth);
  };

  // --- Render Logic ---
  return (
    <Box className={`word-explorer-container ${theme}-theme`} sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* Search Column */}
      <Box sx={{ width: '300px', borderRight: '1px solid', borderColor: 'divider', display: 'flex', flexDirection: 'column' }}>
        <SearchBar initialQuery={searchTerm} onSearchChange={handleSearchChange} /> 
        <SearchResultsList
          results={searchResults} // Pass search results state
          isLoading={isSearchLoading} // Pass search loading state
          error={searchError} // Pass search error state
          hasMore={hasMore} // Pass pagination state
          onLoadMore={loadMoreResults} // Pass load more handler
          onWordClick={handleWordClick} // Pass word click handler
          selectedWord={selectedWord} // Pass currently selected word for highlighting
        />
      </Box>

      {/* Main Content Area */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Details Pane */}
        <Box className="details-pane" sx={{ flexBasis: '50%', overflowY: 'auto', borderBottom: '1px solid', borderColor: 'divider', p: 2 }}>
          {selectedWord ? (
              <>
                {/* Loading/Error state from useWordDetails hook */} 
                {isLoadingDetails && <CircularProgress size={24} />}
                {detailsError && <Typography color="error">Error loading details: {detailsError}</Typography>}
                {/* Render WordDetails when info is loaded successfully */} 
                {!isLoadingDetails && selectedWordInfo && (
                  <WordDetails
                    wordInfo={selectedWordInfo}
                    etymologyTree={etymologyTree} // Pass etymology state
                    isLoadingEtymology={isLoadingEtymology} // Pass etymology loading state
                    etymologyError={etymologyError} // Pass etymology error state
                    onWordLinkClick={handleWordClick} // Handler for links within details
                    onEtymologyNodeClick={handleEtymologyNodeClick} // Handler for etymology graph nodes
                    // ** Pass Metadata State and Setter **
                    showMetadata={showMetadata}
                    setShowMetadata={setShowMetadata}
                  />
                )}
                {/* Placeholder while loading or if info is missing after attempt */} 
                {!isLoadingDetails && !selectedWordInfo && selectedWord && !detailsError && (
                     <Typography color="text.secondary">Loading details for "{selectedWord}"...</Typography>
                )}
              </>
            ) : (
                 // Placeholder when no word is selected
                 <Typography color="text.secondary">Select a word to see details.</Typography>
            )}
        </Box>

        {/* Graph Pane */}
        <Box className="graph-pane" sx={{ flexBasis: '50%', position: 'relative' }}>
            {/* Loading/Error state from useWordNetwork hook */} 
            {isLoadingNetwork && <CircularProgress size={24} sx={{ position: 'absolute', top: 16, left: 16 }} />}
            {networkError && <Typography color="error" sx={{ p: 2 }}>Error loading graph: {networkError}</Typography>}
            {/* Render WordGraph when a word is selected */} 
           {selectedWord ? (
             <WordGraph
               key={selectedWord} // Re-mount graph when word changes
               wordNetwork={wordNetwork} // Pass network data from hook
               mainWord={selectedWord} // Pass selected word 
               onNodeClick={handleGraphNodeClick} // Pass node click handler
               onNetworkChange={handleNetworkSettingsChange} // Pass settings change handler
               initialDepth={graphDepth} // Pass depth state
               initialBreadth={graphBreadth} // Pass breadth state
             />
           ) : (
             // Placeholder when no word is selected (and not searching)
             !isSearchLoading && !isLoadingNetwork && !networkError && (
                <Typography color="text.secondary" sx={{ p: 2 }}>Word network will appear here.</Typography>
             )
           )}
        </Box>
      </Box>
    </Box>
  );
};

export default WordExplorer;
