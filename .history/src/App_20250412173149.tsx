import React, { useState, useCallback, useEffect } from 'react';
import SearchBar from './components/SearchBar';
import WordDetails from './components/WordDetails';
import WordGraphComponent from './components/WordGraphComponent';
import { fetchWordDetails, fetchWordNetwork } from './api/wordApi';
import { WordInfo, WordNetwork } from './types';
import './App.css';

function App() {
  const [searchTerm, setSearchTerm] = useState<string>('mabuti');
  const [wordData, setWordData] = useState<WordInfo | null>(null);
  const [networkData, setNetworkData] = useState<WordNetwork | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>(['mabuti']);
  const [historyIndex, setHistoryIndex] = useState<number>(0);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const executeSearch = useCallback(async (term: string) => {
    if (!term) return;
    
    console.log(`Executing search for: ${term}`);
    setIsLoading(true);
    setError(null);
    setWordData(null);
    setNetworkData(null);
    setSelectedNodeId(null);

    try {
      const details = await fetchWordDetails(term);
      console.log('Word details fetched:', details);
      setWordData(details);
      
      try {
          const network = await fetchWordNetwork(term);
          console.log('Network data fetched:', network);
          setNetworkData(network);
      } catch (networkError) {
          console.error('Failed to fetch network data:', networkError);
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

  useEffect(() => {
    executeSearch(searchTerm);
  }, [executeSearch]);

  const handleSearch = useCallback((term: string) => {
    if (term === searchTerm && !isLoading) return;

    const trimmedTerm = term.trim();
    if (!trimmedTerm) return;

    setSearchTerm(trimmedTerm);
    executeSearch(trimmedTerm);

    setHistory(prevHistory => {
      const newHistory = prevHistory.slice(0, historyIndex + 1);
      if (newHistory[newHistory.length - 1] !== trimmedTerm) {
        newHistory.push(trimmedTerm);
      }
      setHistoryIndex(newHistory.length - 1);
      return newHistory;
    });
  }, [executeSearch, historyIndex, searchTerm, isLoading]);

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
            {wordData && <WordDetails word={wordData} />} 
          </div>
          <div className="word-graph-panel">
            {networkData && wordData && (
              <WordGraphComponent 
                wordNetwork={networkData}
                mainWord={wordData.lemma}
                onNodeClick={(nodeId) => setSelectedNodeId(nodeId)}
                onNodeDoubleClick={handleSearch}
                selectedNodeId={selectedNodeId}
                onNetworkChange={async (depth, breadth) => {
                    if (networkData) {
                        console.log(`Network change requested: Depth=${depth}, Breadth=${breadth}`)
                        setIsLoading(true);
                        try {
                            const newNetwork = await fetchWordNetwork(wordData.lemma, { depth, breadth });
                            setNetworkData(newNetwork);
                        } catch (err) {
                            console.error("Failed to update network:", err);
                            setError("Failed to update network connections.");
                        } finally {
                            setIsLoading(false);
                        }
                    }
                }}
                initialDepth={networkData.metadata?.depth || 2}
                initialBreadth={networkData.metadata?.filters_applied?.breadth || 10}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
