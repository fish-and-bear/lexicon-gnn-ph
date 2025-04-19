import React from 'react';
import { useTheme } from "../contexts/ThemeContext";

const WordExplorer: React.FC = () => {
  const { theme, toggleTheme } = useTheme();
  
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
