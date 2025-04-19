import React, { useEffect } from 'react'; // Added useEffect
import { useTheme } from "../contexts/ThemeContext";
import { testApiConnection } from "../api/wordApi"; // Added testApiConnection import

const WordExplorer: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  // Added useEffect for API connection test
  useEffect(() => {
    console.log("Testing API connection...");
    testApiConnection()
      .then(connected => console.log("API Connection:", connected))
      .catch(error => console.error("API Error:", error));
  }, []);
  
  return (
    <div className={`word-explorer ${theme}`}>
      <header>
        <h1>Filipino Root Word Explorer</h1>
        <button onClick={toggleTheme}>
          {theme === "dark" ? "☀️" : "🌙"}
        </button>
      </header>
      <main>
        <div>Hello world!</div>
      </main>
    </div>
  );
};

export default WordExplorer;
