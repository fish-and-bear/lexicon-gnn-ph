import React, { useState } from 'react';
import WordGraph from '../../frontend/src/components/WordGraph';
import { WordData } from '../src/types/wordTypes';
import { mockWordData } from '../src/data/mockWordData';
import { useTheme } from '../src/contexts/ThemeContext';
import '../styles/WordExplorer.css';

const WordExplorer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedWord, setSelectedWord] = useState<string | null>(null);
  const { theme, toggleTheme } = useTheme();

  const handleSearch = () => {
    if (mockWordData[searchTerm.toLowerCase()]) {
      setSelectedWord(searchTerm.toLowerCase());
    } else {
      alert('Word not found');
    }
  };

  const handleWordSelect = (word: string) => {
    setSelectedWord(word);
  };

  return (
    <div className={`word-explorer ${theme}`}>
      <header>
        <div className="header-content">
          <h1>Word Explorer</h1>
          <button onClick={toggleTheme} className="theme-toggle">
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
        </div>
        <div className="search-container">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Enter a word"
            className="search-input"
          />
          <button onClick={handleSearch} className="search-button">Explore</button>
        </div>
      </header>
      <main>
        <WordGraph selectedWord={selectedWord} onWordSelect={handleWordSelect} />
        {selectedWord && mockWordData[selectedWord] && (
          <div className="word-details">
            <h2>{mockWordData[selectedWord].word}</h2>
            <p className="word-type">{mockWordData[selectedWord].type}</p>
            <p className="word-definition">{mockWordData[selectedWord].definition}</p>
            <p className="word-etymology"><strong>Etymology:</strong> {mockWordData[selectedWord].etymology}</p>
            <div className="related-words">
              <h3>Related Words</h3>
              <ul>
                {mockWordData[selectedWord].relatedWords.map((word) => (
                  <li key={word} onClick={() => setSelectedWord(word)}>{word}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default WordExplorer;