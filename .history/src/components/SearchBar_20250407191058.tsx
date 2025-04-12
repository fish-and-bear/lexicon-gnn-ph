import React, { useState, useRef, useEffect } from 'react';
import { SearchWordResult } from '../types';

interface SearchBarProps {
  query: string;
  setQuery: (query: string) => void;
  suggestions: SearchWordResult[];
  onSearch: (e?: React.FormEvent) => void;
  onSuggestionClick: (word: SearchWordResult) => void;
  showSuggestions: boolean;
  isLoading: boolean;
  error?: string | null;
}

const SearchBar: React.FC<SearchBarProps> = ({ 
  query, 
  setQuery, 
  suggestions, 
  onSearch, 
  onSuggestionClick,
  showSuggestions,
  isLoading,
  error
}) => {
  const [focused, setFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionRef = useRef<HTMLUListElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputRef.current) {
      inputRef.current.blur();
    }
    onSearch(e);
  };

  const handleClickOutside = (e: MouseEvent) => {
    if (
      suggestionRef.current && 
      !suggestionRef.current.contains(e.target as Node) &&
      inputRef.current && 
      !inputRef.current.contains(e.target as Node)
    ) {
      setFocused(false);
    }
  };

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit} className="search-form">
        <div className="search-input-container">
          <input
            ref={inputRef}
            type="text"
            className="search-input"
            placeholder="Search for a Filipino word..."
            value={query}
            onChange={handleChange}
            onFocus={() => setFocused(true)}
            aria-label="Search for a Filipino word"
          />
          {isLoading && (
            <div className="search-loading">
              <div className="loading-indicator"></div>
            </div>
          )}
          {showSuggestions && focused && suggestions.length > 0 && (
            <ul ref={suggestionRef} className="search-suggestions">
              {suggestions.map((suggestion, index) => (
                <li 
                  key={`${suggestion.id || index}-${suggestion.lemma}`} 
                  onClick={() => {
                    onSuggestionClick(suggestion);
                    setFocused(false);
                  }}
                >
                  {suggestion.lemma}
                  {suggestion.has_baybayin && suggestion.baybayin_form && (
                    <span className="baybayin-indicator">{suggestion.baybayin_form}</span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {error && <div className="search-error">{error}</div>}
        </div>
        <button 
          type="submit" 
          className="search-button"
          aria-label="Search"
        >
          Search
        </button>
      </form>
    </div>
  );
};

export default SearchBar; 