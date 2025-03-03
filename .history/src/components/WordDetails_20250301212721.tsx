import React, { useCallback } from 'react';
import { WordInfo, WordNetwork } from '../types';
import { fetchWordNetwork } from '../api/wordApi';

interface WordDetailsProps {
  wordInfo: WordInfo;
  onWordClick: (word: string) => void;
}

const WordDetails: React.FC<WordDetailsProps> = React.memo(({ wordInfo, onWordClick }) => {
  const fetchWordNetworkData = useCallback(async (word: string, depth: number = 2) => {
    try {
      return await fetchWordNetwork(word, { 
        depth,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.3
      });
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
      throw error;
    }
  }, []);

  if (!wordInfo) return null;

  const renderDefinitions = (word: WordInfo) => {
    if (!word.definitions) return null;

    return word.definitions.map((definition, index) => (
      <div key={index} className="definition-card">
        {definition.part_of_speech && (
          <h3>{definition.part_of_speech.name_en}</h3>
        )}
        <p className="definition-text">{definition.text}</p>
        {definition.examples && definition.examples.length > 0 && (
          <div className="examples">
            <h4>Examples:</h4>
            <ul>
              {definition.examples.map((example, idx) => (
                <li key={idx}>{example}</li>
              ))}
            </ul>
          </div>
        )}
        {definition.usage_notes && definition.usage_notes.length > 0 && (
          <div className="usage-notes">
            <h4>Usage Notes:</h4>
            <ul>
              {definition.usage_notes.map((note, idx) => (
                <li key={idx}>{note}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    ));
  };

  const renderRelatedWords = (title: string, words: string[] | undefined) => {
    if (!words || words.length === 0) return null;
    return (
      <div className={title.toLowerCase().replace(/\s+/g, "-")}>
        <h3>{title}:</h3>
        <ul className="word-list">
          {words.map((word, index) => (
            <li
              key={index}
              onClick={() => onWordClick(word)}
              className="clickable-word"
            >
              {word}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  // Extract related words from the new relations structure
  const synonyms = wordInfo.relations.synonyms?.map(item => item.word) || [];
  const antonyms = wordInfo.relations.antonyms?.map(item => item.word) || [];
  const derivatives = wordInfo.relations.derived?.map(item => item.word) || [];
  const rootWord = wordInfo.relations.root?.word;

  return (
    <div className="word-details">
      <div className="word-header">
        <h2>{wordInfo.lemma}</h2>
        {wordInfo.language_code && (
          <span className="language">{wordInfo.language_code}</span>
        )}
      </div>
      
      {wordInfo.pronunciation && (
        <p className="pronunciation">
          <strong>Pronunciation:</strong> {wordInfo.pronunciation.text}
          {wordInfo.pronunciation.ipa && (
            <span className="ipa"> [{wordInfo.pronunciation.ipa}]</span>
          )}
          {wordInfo.pronunciation.audio_url && (
            <button 
              className="play-audio"
              onClick={() => {
                const audio = new Audio(wordInfo.pronunciation?.audio_url);
                audio.play().catch(console.error);
              }}
            >
              🔊
            </button>
          )}
        </p>
      )}
      
      {wordInfo.etymologies?.[0]?.text && (
        <div className="etymology-section">
          <h3>Etymology</h3>
          <p>{wordInfo.etymologies[0].text}</p>
        </div>
      )}
      
      {renderDefinitions(wordInfo)}
      
      <div className="relations-section">
        {renderRelatedWords("Synonyms", synonyms)}
        {renderRelatedWords("Antonyms", antonyms)}
        {renderRelatedWords("Derivatives", derivatives)}
        {rootWord && (
          <div className="root-word">
            <h3>Root Word:</h3>
            <span
              className="clickable-word"
              onClick={() => onWordClick(rootWord)}
            >
              {rootWord}
            </span>
          </div>
        )}
      </div>
    </div>
  );
});

export default WordDetails;