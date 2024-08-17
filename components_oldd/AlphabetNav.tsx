import React from 'react';

const AlphabetNav: React.FC = () => {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

  return (
    <nav className="alphabet-nav">
      {alphabet.map(letter => (
        <button key={letter} className="letter-nav">{letter}</button>
      ))}
    </nav>
  );
};

export default AlphabetNav;