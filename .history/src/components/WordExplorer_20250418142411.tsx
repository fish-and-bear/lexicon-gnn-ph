import React, { useEffect, useState } from 'react'; // Added useState
// import { useTheme } from "../contexts/ThemeContext"; // Commented out for testing
import { testApiConnection } from "../api/wordApi"; // Added testApiConnection import
import { WordInfo } from '../types'; // Added WordInfo type import

const WordExplorer: React.FC = () => {
  // const { theme, toggleTheme } = useTheme(); // Commented out for testing
  const theme = 'light'; // Provide a default theme for testing
  const toggleTheme = () => {}; // Provide a dummy function for testing
  
  // Basic state variables
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null); // Start as null
  const [inputValue, setInputValue] = useState<string>("");
  const [mainWord, setMainWord] = useState<string>("");
  const [selectedWordInfo, setSelectedWordInfo] = useState<WordInfo | null>(null);

  // Updated useEffect for API connection test
  useEffect(() => {
    console.log("Testing API connection...");
    setIsLoading(true); // Set loading state
    setError(null); // Clear previous errors
    testApiConnection()
      .then(connected => {
        console.log("API Connection:", connected);
        setApiConnected(connected);
        if (!connected) {
          setError("API connection failed. Please check the backend server.");
        }
      })
      .catch(error => {
        console.error("API Error:", error);
        setError(error.message || "An unknown error occurred during API connection test.");
        setApiConnected(false);
      })
      .finally(() => {
        setIsLoading(false); // Clear loading state
      });
  }, []);
  
  return (
    <div className={`word-explorer ${theme}`}>
      <header>
        <h1>Filipino Root Word Explorer</h1>
        <button onClick={toggleTheme}>
          {theme === "dark" ? "☀️" : "🌙"}
        </button>
        {/* Display API Connection Status */}
        <div>
          API Status: {apiConnected === null ? 'Checking...' : apiConnected ? 'Connected' : 'Disconnected'}
        </div>
      </header>
      <main>
        {/* Display Loading State */}
        {isLoading && <div>Loading...</div>}
        {/* Display Error Message */}
        {error && <div style={{ color: 'red' }}>Error: {error}</div>}
        {/* Main content - still minimal for now */}
        {!isLoading && !error && <div>Hello world! (API Checked)</div>}
      </main>
    </div>
  );
};

export default WordExplorer;
