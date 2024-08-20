import React, { useState, useCallback, useEffect } from "react";
import WordGraph from "./WordGraph";
import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
import { WordNetwork, WordInfo } from "../types";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:10000/api/v1';

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

  const fetchWordNetwork = useCallback(async (word: string, depth: number, breadth: number) => {
    const response = await fetch(`${API_BASE_URL}/word_network/${word}?depth=${depth}&breadth=${breadth}`);
    const contentType = response.headers.get("content-type");
    if (!response.ok) {
        if (response.status === 404) {
            throw new Error("Word not found");
        }
        throw new Error("Network response was not ok");
    }
    if (!contentType || !contentType.includes("application/json")) {
        throw new Error("Unexpected response format, expected JSON");
    }
    return await response.json();
}, []);

  

  const fetchWordDetails = useCallback(async (word: string) => {
    const response = await fetch(`${API_BASE_URL}/words/${word}`);
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error("Word not found");
      }
      throw new Error("Network response was not ok");
    }
    return await response.json();
  }, []);

  const handleSearch = useCallback(async () => {
    if (!inputValue.trim()) {
      setError("Please enter a word to search");
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const [networkData, detailsData] = await Promise.all([
        fetchWordNetwork(inputValue, depth, breadth),
        fetchWordDetails(inputValue),
      ]);

      if (!detailsData.data.definitions || detailsData.data.definitions.length === 0) {
        setError("No definitions found for this word.");
        setSelectedWordInfo(null);
      } else {
        setWordNetwork(networkData);
        setMainWord(inputValue);
        setSelectedWordInfo(detailsData);
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      setError(error instanceof Error ? error.message : "Failed to fetch word data. Please try again.");
      setWordNetwork(null);
      setSelectedWordInfo(null);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, fetchWordNetwork, fetchWordDetails, depth, breadth]);

  const handleNodeClick = useCallback(async (word: string) => {
    setError(null);
    setIsLoading(true);
    try {
      const detailsData = await fetchWordDetails(word);
      setSelectedWordInfo(detailsData);
    } catch (error) {
      console.error("Error fetching word details:", error);
      setError(error instanceof Error ? error.message : "Failed to fetch word details. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [fetchWordDetails]);

  const handleNetworkChange = useCallback((newDepth: number, newBreadth: number) => {
    setDepth(newDepth);
    setBreadth(newBreadth);
    if (mainWord) {
      setIsLoading(true);
      fetchWordNetwork(mainWord, newDepth, newBreadth)
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
  }, [mainWord, fetchWordNetwork]);

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

  useEffect(() => {
    if (searchTerm) {
      handleSearch();
    }
  }, [searchTerm, handleSearch]);

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
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Enter a word"
          className="search-input"
          aria-label="Search word"
        />
        <button
          onClick={handleSearch}
          disabled={isLoading}
          className="search-button"
        >
          {isLoading ? "Loading..." : "Explore"}
        </button>
      </div>
      {error && <p className="error-message">{error}</p>}
      <main>
        <div className="graph-container">
          <div className="graph-content">
            {wordNetwork && mainWord && (
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