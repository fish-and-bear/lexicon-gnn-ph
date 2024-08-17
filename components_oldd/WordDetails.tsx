import React from 'react';
import { WordData } from '../src/data/mockDictionary';

interface WordDetailsProps {
  word: string;
  details: WordData;
}

const WordDetails: React.FC<WordDetailsProps> = ({ word, details }) => {
  return (
    <div className="word-details">
      <h2>{word}</h2>
      <span className="word-type">{details.type}</span>
      <div className="pronunciation">
        <div>
          <span>UK</span>
          <button className="play-button">▶ {details.pronunciationUK}</button>
        </div>
        <div>
          <span>US</span>
          <button className="play-button">▶ {details.pronunciationUS}</button>
        </div>
      </div>
      {details.definitions.map((def, index) => (
        <div key={index} className="definition">
          <h3>{def.category}</h3>
          <p>{index + 1}. {def.meaning}</p>
          <p className="example">"{def.example}"</p>
          {def.opposites && (
            <p><strong>Opposite:</strong> {def.opposites.join(', ')}</p>
          )}
          <p><strong>Related words:</strong> {def.relatedWords.join(', ')}</p>
        </div>
      ))}
    </div>
  );
};

export default WordDetails;