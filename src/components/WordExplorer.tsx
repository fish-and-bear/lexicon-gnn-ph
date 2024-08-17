import React, { useState, ChangeEvent, useMemo } from 'react';
import WordGraph from './WordGraph';
import { useTheme } from '../contexts/ThemeContext';
import './WordExplorer.css';
import { WordNetwork, WordInfo, Definition } from '../types';

const WordExplorer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [wordNetwork, setWordNetwork] = useState<WordNetwork | null>(null);
  const [mainWord, setMainWord] = useState<string>('');
  const [selectedWordInfo, setSelectedWordInfo] = useState<WordInfo | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const { theme, toggleTheme } = useTheme();

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      setError('Please enter a word to search');
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:5000/api/words/${searchTerm}`);
      if (!response.ok) {
        throw new Error('Word not found');
      }
      const data: WordNetwork = await response.json();
      setWordNetwork(data);
      setMainWord(searchTerm);
      setSelectedWordInfo(null);
    } catch (error) {
      console.error('Error fetching word data:', error);
      setError('Failed to fetch word data. Please try again.');
      setWordNetwork(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeClick = (wordInfo: WordInfo) => {
    setSelectedWordInfo(wordInfo);
  };

  const renderDefinitions = (wordInfo: WordInfo) => {
    if (!wordInfo.definitions) return null;

    return wordInfo.definitions.map((definition: Definition, index: number) => (
      <div key={index} className="definition-card">
        <h3>{definition.part_of_speech}</h3>
        <ol>
          {definition.meanings.map((meaning: string, idx: number) => (
            <li key={idx}>{meaning}</li>
          ))}
        </ol>
        {definition.sources && definition.sources.length > 0 && (
          <p className="sources">Sources: {definition.sources.join(', ')}</p>
        )}
      </div>
    ));
  };

  // Memoize the theme class to avoid unnecessary re-renders
  const themeClass = useMemo(() => `word-explorer ${theme}`, [theme]);

  return (
    <div className={themeClass}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <button onClick={toggleTheme} className="theme-toggle">
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </header>
      <div className="search-container">
        <input
          type="text"
          value={searchTerm}
          onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
          placeholder="Enter a word"
          className="search-input"
        />
        <button onClick={handleSearch} disabled={isLoading} className="search-button">
          {isLoading ? 'Loading...' : 'Explore'}
        </button>
      </div>
      {error && <p className="error-message">{error}</p>}
      <main>
        <div className="graph-container">
          <div className="graph-content">
            {wordNetwork && mainWord && (
              <WordGraph wordNetwork={wordNetwork} mainWord={mainWord} onNodeClick={handleNodeClick} />
            )}
          </div>
        </div>
        <div className="details-container">
          <div className="details-content">
            {selectedWordInfo ? (
              <div className="word-details">
                <h2>{selectedWordInfo.word}</h2>
                {selectedWordInfo.pronunciation && (
                  <p><strong>Pronunciation:</strong> {selectedWordInfo.pronunciation}</p>
                )}
                {selectedWordInfo.etymology && selectedWordInfo.etymology.length > 2 && (
                  <p><strong>Etymology:</strong> {selectedWordInfo.etymology}</p>
                )}
                {selectedWordInfo.language_codes && (
                  <p><strong>Language Codes:</strong> {selectedWordInfo.language_codes}</p>
                )}
                {renderDefinitions(selectedWordInfo)}
                {selectedWordInfo.derivatives && Object.keys(selectedWordInfo.derivatives).length > 0 && (
                  <div className="derivatives">
                    <h3>Derivatives:</h3>
                    <ul>
                      {Object.keys(selectedWordInfo.derivatives).map((derivative, index) => (
                        <li key={index}>{derivative}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {selectedWordInfo.associated_words && selectedWordInfo.associated_words.length > 0 && (
                  <div className="associated-words">
                    <h3>Associated Words:</h3>
                    <ul>
                      {selectedWordInfo.associated_words.map((associatedWord, index) => (
                        <li key={index}>{associatedWord}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {selectedWordInfo.root_words && selectedWordInfo.root_words.length > 0 && (
                  <div className="root-words">
                    <h3>Root Words:</h3>
                    <ul>
                      {selectedWordInfo.root_words.map((rootWord, index) => (
                        <li key={index}>{rootWord}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {selectedWordInfo.root_word && (
                  <p><strong>Root Word:</strong> {selectedWordInfo.root_word}</p>
                )}
              </div>
            ) : (
              <p>Click on a node to see the details.</p>
            )}
          </div>
        </div>
      </main>
      <footer className="footer">
        © 2024 Filipino Root Word Explorer. All Rights Reserved.
      </footer>
    </div>
  );
};

export default WordExplorer;
