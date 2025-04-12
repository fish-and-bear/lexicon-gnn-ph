import React, { useState, useCallback, useEffect } from 'react';
import { SearchBar } from './components/SearchBar';
import { WordInfo } from './components/WordInfo';
import WordGraphComponent from './components/WordGraphComponent';
import { searchWords, getWordDetails, getSemanticNetwork } from './api/wordApi';
import { Word, SemanticNetworkData } from './types';
import './App.css';

function App() {
  const [searchTerm, setSearchTerm] = useState<string>('mabuti');
  const [wordData, setWordData] = useState<Word | null>(null);
  const [networkData, setNetworkData] = useState<SemanticNetworkData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>(['mabuti']); // Initialize history
  const [historyIndex, setHistoryIndex] = useState<number>(0); // Initialize history index
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null); // State for selected node

  const executeSearch = useCallback(async (term: string) => {
    if (!term) return;
    
    console.log(`Executing search for: ${term}`);
    setIsLoading(true);
    setError(null);
    setWordData(null);
    setNetworkData(null);
    setSelectedNodeId(null); // Reset selected node on new search

    try {
      const details = await getWordDetails(term);
      console.log('Word details fetched:', details);
      setWordData(details);
      
      // Always fetch network data after details
      try {
          const network = await getSemanticNetwork(term);
          console.log('Network data fetched:', network);
          setNetworkData(network);
      } catch (networkError) {
          console.error('Failed to fetch network data:', networkError);
          // Keep word details even if network fails, but show an error
          setError('Failed to load word connections. Please try again.'); 
      }

    } catch (err) {
      console.error('Search failed:', err);
      setError('Failed to load word details. Please try again.');
      setWordData(null);
      setNetworkData(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Effect to run initial search on mount
  useEffect(() => {
    executeSearch(searchTerm);
  }, [executeSearch]); // Removed searchTerm from dependency array to prevent re-triggering on history navigation

  const handleSearch = useCallback((term: string) => {
    if (term === searchTerm) return; // Avoid redundant searches

    setSearchTerm(term);
    executeSearch(term);

    // Update history
    setHistory(prevHistory => {
      const newHistory = prevHistory.slice(0, historyIndex + 1); // Trim future history
      if (newHistory[newHistory.length - 1] !== term) {
        newHistory.push(term);
      }
      setHistoryIndex(newHistory.length - 1); // Move index to the end
      return newHistory;
    });
  }, [executeSearch, historyIndex, searchTerm]);

  const handleBack = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      const term = history[newIndex];
      setSearchTerm(term);
      executeSearch(term);
    }
  }, [history, historyIndex, executeSearch]);

  const handleForward = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      const term = history[newIndex];
      setSearchTerm(term);
      executeSearch(term);
    }
  }, [history, historyIndex, executeSearch]);


  return (
    <div className="App">
      <header className="App-header">
        <h1>Filipino Word Relationships</h1>
        <SearchBar 
          onSearch={handleSearch} 
          initialValue={searchTerm} 
          onBack={handleBack}
          onForward={handleForward}
          canGoBack={historyIndex > 0}
          canGoForward={historyIndex < history.length - 1}
        />
      </header>
      <main className="App-content">
        {isLoading && <div className="loading">Loading...</div>}
        {error && <div className="error">{error}</div>}
        <div className="data-container">
          <div className="word-info-panel">
            {wordData && <WordInfo word={wordData} />}
          </div>
          <div className="word-graph-panel">
            {networkData && (
              <WordGraphComponent 
                graphData={networkData} 
                selectedNodeId={selectedNodeId}
                setSelectedNodeId={setSelectedNodeId}
                onNodeDoubleClick={handleSearch} // Pass handleSearch for double-click
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
