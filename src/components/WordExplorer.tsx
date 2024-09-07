import React, { useState, useCallback, useEffect } from "react";
import WordGraph from "./WordGraph";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo, SearchResult, SearchOptions } from "../types";
import unidecode from "unidecode";
import { fetchWordNetwork, fetchWordDetails, searchWords } from "../api/wordApi";
import axios from 'axios';
import DOMPurify from 'dompurify';
import { debounce } from "lodash";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://api.hapinas.net/api/v1';

const WordExplorer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [wordNetwork, setWordNetwork] = useState<WordNetwork | null>(null);
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
  const [searchResults, setSearchResults] = useState<Array<{ id: number; word: string }>>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Function to normalize input
  const normalizeInput = (input: string) => unidecode(input.trim().toLowerCase());

  const fetchWordNetworkData = useCallback(async (word: string, depth: number, breadth: number) => {
    try {
      return await fetchWordNetwork(word, depth, breadth);
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
      throw error; // Pass the error through instead of creating a new one
    }
  }, []);

  const debouncedSearch = useCallback(
    debounce(async (query: string) => {
      if (query.length > 1) {
        setIsLoading(true);
        setError(null);
        console.log('Searching for:', query);
        try {
          const results = await searchWords(query, { page: 1, per_page: 10, exclude_baybayin: true });
          console.log('API response:', results);
          
          const searchResults = results.words.map((word: { id: number; word: string }) => ({
            id: word.id,
            word: word.word.trim()
          })).filter((result) => result.word !== '');
          
          console.log('Processed search results:', searchResults);
          
          setSearchResults(searchResults);
          setShowSuggestions(searchResults.length > 0);
        } catch (error) {
          console.error("Error fetching search results:", error);
          setError("Failed to fetch search results. Please try again.");
          setSearchResults([]);
          setShowSuggestions(false);
        } finally {
          setIsLoading(false);
        }
      } else {
        setSearchResults([]);
        setShowSuggestions(false);
      }
    }, 300),
    []
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setInputValue(value);
    setError(null);
    debouncedSearch(value);
    console.log('Input changed:', value); // Add this line
    if (value.length > 0) {
      setShowSuggestions(true);
    } else {
      setShowSuggestions(false);
    }
  };

  const handleSuggestionClick = (word: string) => {
    setInputValue(word);
    setShowSuggestions(false);
    handleSearch(word);
  };

  const handleSearch = useCallback(async (searchWord?: string) => {
    const wordToSearch = searchWord || inputValue.trim();
    if (!wordToSearch) {
      setError("Please enter a word to search");
      return;
    }
  
    const sanitizedInput = DOMPurify.sanitize(wordToSearch);
    const normalizedInput = normalizeInput(sanitizedInput);

    setIsLoading(true);
    setError(null);
    setWordNetwork(null);

    try {
      const detailsData = await fetchWordDetails(normalizedInput);
      
      if (!detailsData.data || !detailsData.data.definitions || detailsData.data.definitions.length === 0) {
        throw new Error("No definitions found for this word.");
      }

      setSelectedWordInfo(detailsData);
      setMainWord(detailsData.data.word);
      
      let networkData = await fetchWordNetworkData(normalizedInput, depth, breadth);
      setWordNetwork(networkData);

      setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), detailsData.data.word]);
      setCurrentHistoryIndex(prevIndex => prevIndex + 1);
    } catch (error) {
      console.error("Error fetching data:", error);
      let errorMessage = "Failed to fetch word data. Please try again.";
      if (error instanceof Error) {
        console.error("Error details:", error.message);
        errorMessage = `Failed to fetch word data: ${error.message}`;
      }
      setError(errorMessage);
      setWordNetwork(null);
      setSelectedWordInfo(null);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, fetchWordNetworkData, fetchWordDetails, depth, breadth, currentHistoryIndex]);

  const handleNodeClick = useCallback(async (word: string) => {
    setError(null);
    setIsLoading(true);
    try {
      const normalizedWord = normalizeInput(word);
      const detailsData = await fetchWordDetails(normalizedWord);
      setSelectedWordInfo(detailsData);
      setMainWord(detailsData.data.word);
      setWordNetwork(null);
      const networkData = await fetchWordNetworkData(normalizedWord, depth, breadth);
      setWordNetwork(networkData);

      setWordHistory(prevHistory => [...prevHistory.slice(0, currentHistoryIndex + 1), detailsData.data.word]);
      setCurrentHistoryIndex(prevIndex => prevIndex + 1);
    } catch (error) {
      console.error("Error fetching word details:", error);
      setError(error instanceof Error ? error.message : "Failed to fetch word details. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [fetchWordDetails, fetchWordNetworkData, depth, breadth, currentHistoryIndex]);

  const handleNetworkChange = useCallback((newDepth: number, newBreadth: number) => {
    setDepth(newDepth);
    setBreadth(newBreadth);
    if (mainWord) {
      setIsLoading(true);
      fetchWordNetworkData(normalizeInput(mainWord), newDepth, newBreadth)
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

  const renderDefinitions = useCallback((wordInfo: WordInfo) => {
    if (!wordInfo.data.definitions) return null;

    return wordInfo.data.definitions.map((definition, index) => (
      <div key={index} className="definition-card">
        {definition.partOfSpeech && <h3>{definition.partOfSpeech}</h3>}
        <ol>
          {definition.meanings
            ?.filter((meaning) => meaning.definition && meaning.definition.trim() !== "0")
            .map((meaning, idx) => (
              <li key={idx}>
                {meaning.definition}
                {meaning.source && (
                  <span className="source">Source: {meaning.source}</span>
                )}
              </li>
            ))}
        </ol>
        {definition.usageNotes && definition.usageNotes.length > 0 && (
          <p className="usage-notes">
            <strong>Usage notes:</strong> {definition.usageNotes.join(", ")}
          </p>
        )}
        {definition.examples && definition.examples.length > 0 && (
          <p className="examples">
            <strong>Examples:</strong> {definition.examples.join("; ")}
          </p>
        )}
      </div>
    ));
  }, []);

  const renderArraySection = useCallback((title: string, items?: string[]) => {
    if (!items || items.length === 0) return null;
    return (
      <div className={title.toLowerCase().replace(/\s+/g, "-")}>
        <h3>{title}:</h3>
        <ul className="word-list">
          {items
            .filter((item) => item.trim() !== "" && item.trim() !== "0")
            .map((item, index) => (
              <li
                key={index}
                onClick={() => handleNodeClick(item)}
                className="clickable-word"
              >
                {item}
              </li>
            ))}
        </ul>
      </div>
    );
  }, [handleNodeClick]);

  const handleBack = useCallback(() => {
    if (currentHistoryIndex > 0) {
      setCurrentHistoryIndex(prevIndex => prevIndex - 1);
      const previousWord = wordHistory[currentHistoryIndex - 1];
      handleNodeClick(previousWord);
    }
  }, [currentHistoryIndex, wordHistory, handleNodeClick]);

  const handleForward = useCallback(() => {
    if (currentHistoryIndex < wordHistory.length - 1) {
      setCurrentHistoryIndex(prevIndex => prevIndex + 1);
      const nextWord = wordHistory[currentHistoryIndex + 1];
      handleNodeClick(nextWord);
    }
  }, [currentHistoryIndex, wordHistory, handleNodeClick]);

  console.log('Search query:', inputValue);
  console.log('Search results:', searchResults);
  console.log('Show suggestions:', showSuggestions);

  return (
    <div className={`word-explorer ${theme}`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer</h1>
        <button
          onClick={toggleTheme}
          className="theme-toggle"
          aria-label="Toggle theme"
        >
          {theme === "light" ? "🌙" : "☀️"}
        </button>
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
            onKeyPress={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleSearch();
              }
            }}
            placeholder="Enter a word"
            className="search-input"
            aria-label="Search word"
          />
          {isLoading && <div className="search-loading">Loading...</div>}
          {showSuggestions && searchResults.length > 0 && (
            <ul className="search-suggestions">
              {searchResults.map((result) => (
                <li key={result.id} onClick={() => handleSuggestionClick(result.word)}>
                  {result.word}
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
          Explore
        </button>
      </div>
      {error && <p className="error-message">{error}</p>}
      <main>
        <div className="graph-container">
          <div className="graph-content">
            {wordNetwork && mainWord && Object.keys(wordNetwork).length > 0 && (
              <WordGraph
                wordNetwork={wordNetwork}
                mainWord={mainWord}
                onNodeClick={handleNodeClick}
                onNetworkChange={handleNetworkChange}
                initialDepth={depth}
                initialBreadth={breadth}
              />
            )}
          </div>
        </div>
        <div className="details-container">
          <div className="details-content">
            {isLoading ? (
              <div className="loading-spinner">Loading...</div>
            ) : selectedWordInfo ? (
              <div className="word-details">
                <h2>{selectedWordInfo.data.word}</h2>
                {selectedWordInfo.data.pronunciation?.text && (
                  <p className="pronunciation">
                    <strong>Pronunciation:</strong>{" "}
                    {selectedWordInfo.data.pronunciation.text}
                  </p>
                )}
                {selectedWordInfo.data.etymology?.text &&
                  selectedWordInfo.data.etymology.text.length > 0 && (
                    <p>
                      <strong>Etymology:</strong>{" "}
                      {selectedWordInfo.data.etymology.text}
                    </p>
                  )}
                {selectedWordInfo.data.languages &&
                  selectedWordInfo.data.languages.length > 0 && (
                    <p>
                      <strong>Language Codes:</strong>{" "}
                      {selectedWordInfo.data.languages.join(", ")}
                    </p>
                  )}
                {renderDefinitions(selectedWordInfo)}
                {renderArraySection(
                  "Synonyms",
                  selectedWordInfo.data.relationships?.synonyms
                )}
                {renderArraySection(
                  "Antonyms",
                  selectedWordInfo.data.relationships?.antonyms
                )}
                {renderArraySection(
                  "Associated Words",
                  selectedWordInfo.data.relationships?.associatedWords
                )}
                {renderArraySection(
                  "Derivatives",
                  selectedWordInfo.data.relationships?.derivatives
                )}
                {selectedWordInfo.data.relationships?.rootWord && (
                  <p>
                    <strong>Root Word:</strong>{" "}
                    <span
                      className="clickable-word"
                      onClick={() =>
                        handleNodeClick(
                          selectedWordInfo.data.relationships.rootWord!
                        )
                      }
                    >
                      {selectedWordInfo.data.relationships.rootWord}
                    </span>
                  </p>
                )}
              </div>
            ) : (
              <p>
                Enter a word to explore or click on a node in the graph to see
                details.
              </p>
            )}
          </div>
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
