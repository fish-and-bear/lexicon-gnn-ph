import React from 'react';
import './SearchBar.css';
import { SearchWordResult } from '../types';

interface SearchBarProps {
  query: string;
  setQuery: (query: string) => void;
  suggestions: SearchWordResult[];
  onSearch: (query: string) => void;
  onSuggestionClick: (suggestion: SearchWordResult) => void;
  showSuggestions: boolean;
  isLoading: boolean;
}

const SearchBar: React.FC<SearchBarProps> = ({
  query,
  setQuery,
  suggestions,
  onSearch,
  onSuggestionClick,
  showSuggestions,
  isLoading
}) => {
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && query.trim()) {
      e.preventDefault();
      onSearch(query);
    }
  };

  return (
    <div className="search-container">
      <div className="search-input-container">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Enter a Filipino word"
          className="search-input"
          aria-label="Search word"
        />
        {isLoading && (
          <div className="search-loading">
            <div className="loading-spinner-small"></div>
          </div>
        )}
        {showSuggestions && suggestions.length > 0 && (
          <ul className="search-suggestions">
            {suggestions.map((result) => (
              <li 
                key={result.id} 
                onClick={() => onSuggestionClick(result)}
              >
                {result.lemma}
              </li>
            ))}
          </ul>
        )}
      </div>
      <button
        onClick={() => query.trim() && onSearch(query)}
        disabled={!query.trim() || isLoading}
        className="search-button"
      >
        Search
      </button>
    </div>
  );
};

export default SearchBar; 