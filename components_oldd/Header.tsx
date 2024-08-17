import React from 'react';
import AlphabetNav from './AlphabetNav';

interface HeaderProps {
  searchTerm: string;
  setSearchTerm: (term: string) => void;
  handleSearch: () => void;
}

const Header: React.FC<HeaderProps> = ({ searchTerm, setSearchTerm, handleSearch }) => {
  return (
    <header>
      <h1>Sideways Dictionary</h1>
      <div className="search-container">
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search for a word"
        />
        <button onClick={handleSearch}>Search</button>
      </div>
      <AlphabetNav />
    </header>
  );
};

export default Header;