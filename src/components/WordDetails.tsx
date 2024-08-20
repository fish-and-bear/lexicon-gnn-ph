import React from 'react';
import { WordInfo } from '../types';

interface WordDetailsProps {
  wordInfo: WordInfo;
  onWordClick: (word: string) => void;
}

const WordDetails: React.FC<WordDetailsProps> = React.memo(({ wordInfo, onWordClick }) => {
  const renderDefinitions = (wordInfo: WordInfo) => {
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
  };

  const renderArraySection = (title: string, items?: string[]) => {
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
                onClick={() => onWordClick(item)}
                className="clickable-word"
              >
                {item}
              </li>
            ))}
        </ul>
      </div>
    );
  };

  return (
    <div className="word-details">
      <h2>{wordInfo.data.word}</h2>
      {wordInfo.data.pronunciation?.text && (
        <p className="pronunciation">
          <strong>Pronunciation:</strong> {wordInfo.data.pronunciation.text}
        </p>
      )}
      {wordInfo.data.etymology?.text &&
        wordInfo.data.etymology.text.length > 0 && (
          <p>
            <strong>Etymology:</strong> {wordInfo.data.etymology.text}
          </p>
        )}
      {wordInfo.data.languages &&
        wordInfo.data.languages.length > 0 && (
          <p>
            <strong>Language Codes:</strong> {wordInfo.data.languages.join(", ")}
          </p>
        )}
      {renderDefinitions(wordInfo)}
      {renderArraySection("Synonyms", wordInfo.data.relationships?.synonyms)}
      {renderArraySection("Antonyms", wordInfo.data.relationships?.antonyms)}
      {renderArraySection("Associated Words", wordInfo.data.relationships?.associatedWords)}
      {renderArraySection("Derivatives", wordInfo.data.relationships?.derivatives)}
      {wordInfo.data.relationships?.rootWord && (
        <p>
          <strong>Root Word:</strong>{" "}
          <span
            className="clickable-word"
            onClick={() => onWordClick(wordInfo.data.relationships.rootWord!)}
          >
            {wordInfo.data.relationships.rootWord}
          </span>
        </p>
      )}
    </div>
  );
});

export default WordDetails;