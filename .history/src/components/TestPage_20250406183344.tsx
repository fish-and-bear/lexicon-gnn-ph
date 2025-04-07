import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import SimpleWordGraph from './SimpleWordGraph';
import NetworkControls from './NetworkControls';
import './TestPage.css';

const TestPage: React.FC = () => {
  const [mainWord, setMainWord] = useState<string>("");
  const [wordNetwork, setWordNetwork] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [depth, setDepth] = useState<number>(1);
  const [breadth, setBreadth] = useState<number>(15);
  const [inputValue, setInputValue] = useState<string>("");

  // Fetch word network data
  const fetchWordNetworkData = useCallback(async (word: string) => {
    if (!word) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      console.log(`Fetching network for ${word} with depth=${depth}, breadth=${breadth}`);
      const response = await axios.get(`/api/semantic_network/${word}`, {
        params: { depth, breadth }
      });
      
      console.log("API Response:", response.data);
      
      if (response.data) {
        setWordNetwork(response.data);
      }
    } catch (err) {
      console.error("Error fetching word network:", err);
      setError("Failed to load word network");
      setWordNetwork(null);
    } finally {
      setIsLoading(false);
    }
  }, [depth, breadth]);

  // Handle search submission
  const handleSearch = useCallback(() => {
    const word = inputValue.trim().toLowerCase();
    if (word && word !== mainWord) {
      setMainWord(word);
      fetchWordNetworkData(word);
    }
  }, [inputValue, mainWord, fetchWordNetworkData]);

  // Handle clicking on a network node
  const handleNodeClick = useCallback((word: string) => {
    setInputValue(word);
    setMainWord(word);
    fetchWordNetworkData(word);
  }, [fetchWordNetworkData]);

  // Handle depth change
  const handleDepthChange = useCallback((newDepth: number) => {
    setDepth(newDepth);
    if (mainWord) {
      fetchWordNetworkData(mainWord);
    }
  }, [mainWord, fetchWordNetworkData]);

  // Handle breadth change
  const handleBreadthChange = useCallback((newBreadth: number) => {
    setBreadth(newBreadth);
    if (mainWord) {
      fetchWordNetworkData(mainWord);
    }
  }, [mainWord, fetchWordNetworkData]);

  // Generate sample data for testing when no API is available
  const generateSampleData = useCallback(() => {
    const sampleWord = inputValue.trim() || "sample";
    
    // Create sample nodes
    const nodes = [
      { id: "1", lemma: sampleWord, group: "main" },
      { id: "2", lemma: `${sampleWord}_related1`, group: "synonym" },
      { id: "3", lemma: `${sampleWord}_related2`, group: "antonym" },
      { id: "4", lemma: `${sampleWord}_related3`, group: "hypernym" },
      { id: "5", lemma: `${sampleWord}_related4`, group: "hyponym" }
    ];
    
    // Create sample edges
    const edges = [
      { source: "1", target: "2", type: "synonym" },
      { source: "1", target: "3", type: "antonym" },
      { source: "1", target: "4", type: "hypernym" },
      { source: "1", target: "5", type: "hyponym" }
    ];
    
    setMainWord(sampleWord);
    setWordNetwork({
      nodes,
      edges,
      stats: {
        node_count: nodes.length,
        edge_count: edges.length,
        execution_time: 0.123,
        depth,
        breadth
      }
    });
  }, [inputValue, depth, breadth]);

  return (
    <div className="test-page">
      <header className="test-header">
        <h1>Graph Component Test Page</h1>
      </header>
      
      <div className="search-container">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="Enter a word..."
          className="search-input"
        />
        <button onClick={handleSearch} className="search-button">Search API</button>
        <button onClick={generateSampleData} className="sample-button">Generate Sample Data</button>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="content-container">
        <div className="controls-panel">
          <NetworkControls 
            depth={depth}
            breadth={breadth}
            onDepthChange={handleDepthChange}
            onBreadthChange={handleBreadthChange}
          />
          
          {mainWord && (
            <div className="word-info">
              <h2>Current Word: {mainWord}</h2>
              {wordNetwork && (
                <div className="stats">
                  <p>Nodes: {wordNetwork.nodes?.length || 0}</p>
                  <p>Edges: {wordNetwork.edges?.length || 0}</p>
                </div>
              )}
            </div>
          )}
        </div>
        
        <div className="graph-panel">
          {isLoading ? (
            <div className="loading-message">Loading word network...</div>
          ) : (
            <>
              {wordNetwork ? (
                <div style={{ height: '600px', width: '100%' }}>
                  <SimpleWordGraph 
                    wordNetwork={wordNetwork}
                    mainWord={mainWord}
                    onNodeClick={handleNodeClick}
                  />
                </div>
              ) : (
                <div className="no-data">
                  {mainWord ? (
                    <p>No network data found for "{mainWord}"</p>
                  ) : (
                    <p>Enter a word to see its network</p>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default TestPage; 