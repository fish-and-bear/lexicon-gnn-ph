import React, { useCallback, useState, useEffect } from 'react';
import { WordInfo, WordNetwork } from '../types';
import { fetchWordNetwork } from '../api/wordApi';

interface WordDetailsProps {
  wordInfo: WordInfo;
  onWordClick: (word: string) => void;
}

const WordDetails: React.FC<WordDetailsProps> = React.memo(({ wordInfo, onWordClick }) => {
  const [activeTab, setActiveTab] = useState<'definitions' | 'relations' | 'etymology'>('definitions');
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const [expandedDefinitions, setExpandedDefinitions] = useState<Record<string, boolean>>({});
  const [networkData, setNetworkData] = useState<WordNetwork | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Reset states when word changes
    setActiveTab('definitions');
    setExpandedDefinitions({});
    setNetworkData(null);
    setIsAudioPlaying(false);
    
    // Create audio element if pronunciation available
    if (wordInfo?.pronunciation?.audio_url) {
      const audio = new Audio(wordInfo.pronunciation.audio_url);
      audio.addEventListener('ended', () => setIsAudioPlaying(false));
      setAudioElement(audio);
      return () => {
        audio.pause();
        audio.removeEventListener('ended', () => setIsAudioPlaying(false));
      };
    }
  }, [wordInfo?.lemma]);

  const fetchWordNetworkData = useCallback(async (word: string, depth: number = 2) => {
    try {
      setIsLoading(true);
      const data = await fetchWordNetwork(word, { 
        depth,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.3
      });
      setNetworkData(data);
      return data;
    } catch (error) {
      console.error("Error in fetchWordNetworkData:", error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const toggleDefinitionExpand = (id: string) => {
    setExpandedDefinitions(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const playAudio = () => {
    if (!audioElement) return;
    
    if (isAudioPlaying) {
      audioElement.pause();
      audioElement.currentTime = 0;
      setIsAudioPlaying(false);
    } else {
      audioElement.play().catch(console.error);
      setIsAudioPlaying(true);
    }
  };

  if (!wordInfo) return null;

  const renderDefinitions = (word: WordInfo) => {
    if (!word.definitions || word.definitions.length === 0) {
      return (
        <div className="empty-state">
          <p>No definitions available for this word.</p>
        </div>
      );
    }

    // Group definitions by part of speech
    const definitionsByPos: Record<string, any[]> = {};
    
    word.definitions.forEach(definition => {
      const posName = definition.part_of_speech?.name_en || 'Other';
      if (!definitionsByPos[posName]) {
        definitionsByPos[posName] = [];
      }
      definitionsByPos[posName].push(definition);
    });

    return (
      <div className="definitions-section">
        <div className="definitions-section-header">
          <h3>Definitions</h3>
          <span className="definition-count">{word.definitions.length}</span>
        </div>
        
        <div className="definition-cards-container">
          {Object.entries(definitionsByPos).map(([posName, definitions]) => (
            <div key={posName} className="pos-group">
              <div className="pos-group-header">
                {posName}
                <span className="pos-count">{definitions.length}</span>
              </div>
              
              {definitions.map((definition, index) => {
                const defId = `${posName}-${index}`;
                const isExpanded = expandedDefinitions[defId] || false;
                
                return (
                  <div 
                    key={index} 
                    className={`definition-card ${isExpanded ? 'expanded' : ''}`}
                    onClick={() => toggleDefinitionExpand(defId)}
                  >
                    <span className="definition-number">{index + 1}</span>
                    <div className="definition-content">
                      <p className="definition-text">{definition.text}</p>
                      
                      {(definition.examples?.length > 0 || definition.usage_notes?.length > 0) && (
                        <div className={`definition-details ${isExpanded ? 'visible' : ''}`}>
                          {definition.examples && definition.examples.length > 0 && (
                            <div className="examples">
                              <h4>Examples</h4>
                              <ul>
                                {definition.examples.map((example: string, idx: number) => (
                                  <li key={idx}>
                                    {example}
                                    {definition.example_translations?.[idx] && (
                                      <span className="translation">{definition.example_translations[idx]}</span>
                                    )}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {definition.usage_notes && definition.usage_notes.length > 0 && (
                            <div className="usage-notes">
                              <h4>Usage Notes</h4>
                              <ul>
                                {definition.usage_notes.map((note: string, idx: number) => (
                                  <li key={idx}>{note}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {(definition.examples?.length > 0 || definition.usage_notes?.length > 0) && (
                        <div className="expand-indicator">
                          <span>{isExpanded ? 'Show less' : 'Show more'}</span>
                          <span className="expand-icon">{isExpanded ? '▲' : '▼'}</span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderEtymology = (word: WordInfo) => {
    if (!word.etymologies || word.etymologies.length === 0) {
      return (
        <div className="empty-state">
          <p>No etymology information available for this word.</p>
        </div>
      );
    }

    // Extract all unique components from etymologies
    const allComponents: string[] = [];
    word.etymologies.forEach(etymology => {
      if (etymology.components) {
        etymology.components.forEach(component => {
          if (!allComponents.includes(component)) {
            allComponents.push(component);
          }
        });
      }
    });

    const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
    const [showAllEtymologies, setShowAllEtymologies] = useState(true);

    // Filter etymologies based on selected component
    const filteredEtymologies = selectedComponent 
      ? word.etymologies.filter(etym => 
          etym.components && etym.components.includes(selectedComponent))
      : word.etymologies;

    return (
      <div className="etymology-section">
        {allComponents.length > 0 && (
          <div className="etymology-filter">
            <div className="filter-label">Filter by component:</div>
            <div className="component-filter-tags">
              <button 
                className={`component-filter-tag ${!selectedComponent ? 'active' : ''}`}
                onClick={() => {
                  setSelectedComponent(null);
                  setShowAllEtymologies(true);
                }}
              >
                All
              </button>
              {allComponents.map((component, idx) => (
                <button 
                  key={idx} 
                  className={`component-filter-tag ${selectedComponent === component ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedComponent(component);
                    setShowAllEtymologies(false);
                  }}
                >
                  {component}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="etymology-content">
          {filteredEtymologies.map((etymology, index) => (
            <div key={index} className="etymology-item">
              <p className="etymology-text">{etymology.text}</p>
              
              {etymology.languages && etymology.languages.length > 0 && (
                <div className="etymology-languages">
                  {etymology.languages.map((lang, idx) => (
                    <span key={idx} className="language-tag">{lang}</span>
                  ))}
                </div>
              )}
              
              {etymology.components && etymology.components.length > 0 && (
                <div className="etymology-components-list">
                  <span className="components-label">Components:</span>
                  <div className="component-tags">
                    {etymology.components.map((component, idx) => (
                      <span 
                        key={idx} 
                        className="component-tag"
                        onClick={() => onWordClick(component)}
                      >
                        {component}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {etymology.sources && etymology.sources.length > 0 && (
                <div className="etymology-sources">
                  <span className="sources-label">Sources:</span>
                  <div className="source-tags">
                    {etymology.sources.map((source, idx) => (
                      <span key={idx} className="source-tag">{source}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {word.etymologies.length > 1 && (
          <div className="etymology-summary">
            <h4>Components Summary</h4>
            <div className="components-summary-list">
              {allComponents.map((component, idx) => (
                <div key={idx} className="component-summary-item">
                  <span 
                    className="component-tag clickable"
                    onClick={() => onWordClick(component)}
                  >
                    {component}
                  </span>
                  <span className="component-count">
                    {word.etymologies.filter(etym => 
                      etym.components && etym.components.includes(component)
                    ).length} references
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderRelatedWords = () => {
    // Extract related words from the relations structure
    const synonyms = wordInfo.relations.synonyms?.map(item => item.word) || [];
    const antonyms = wordInfo.relations.antonyms?.map(item => item.word) || [];
    const derivatives = wordInfo.relations.derived?.map(item => item.word) || [];
    const variants = wordInfo.relations.variants?.map(item => item.word) || [];
    const related = wordInfo.relations.related?.map(item => item.word) || [];
    const rootWord = wordInfo.relations.root?.word;

    const hasRelations = synonyms.length > 0 || antonyms.length > 0 || 
                         derivatives.length > 0 || variants.length > 0 || 
                         related.length > 0 || rootWord;

    // Group all related words by type for filtering
    const relationGroups = [
      { type: 'synonyms', label: 'Synonyms', words: synonyms, icon: '🔄' },
      { type: 'antonyms', label: 'Antonyms', words: antonyms, icon: '⚔️' },
      { type: 'derived', label: 'Derivatives', words: derivatives, icon: '🌱' },
      { type: 'variants', label: 'Variants', words: variants, icon: '🔀' },
      { type: 'related', label: 'Related Words', words: related, icon: '🔗' },
    ].filter(group => group.words.length > 0);

    const [selectedRelationType, setSelectedRelationType] = useState<string | null>(null);

    // Filter relation groups based on selection
    const filteredGroups = selectedRelationType
      ? relationGroups.filter(group => group.type === selectedRelationType)
      : relationGroups;

    if (!hasRelations) {
      return (
        <div className="empty-state">
          <p>No related words available.</p>
          <button 
            className="fetch-network-button"
            onClick={() => fetchWordNetworkData(wordInfo.lemma)}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Discover Word Network'}
          </button>
        </div>
      );
    }

    return (
      <>
        {relationGroups.length > 1 && (
          <div className="relation-filter">
            <div className="filter-label">Filter by type:</div>
            <div className="relation-filter-tags">
              <button 
                className={`relation-filter-tag ${!selectedRelationType ? 'active' : ''}`}
                onClick={() => setSelectedRelationType(null)}
              >
                All
              </button>
              {relationGroups.map((group, idx) => (
                <button 
                  key={idx} 
                  className={`relation-filter-tag ${selectedRelationType === group.type ? 'active' : ''}`}
                  onClick={() => setSelectedRelationType(group.type)}
                >
                  {group.icon} {group.label}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="relations-grid">
          {filteredGroups.map((group, idx) => (
            <div key={idx} className={`relation-group ${group.type}`}>
              <h4>{group.label}</h4>
              <div className="relation-tags">
                {group.words.map((word, index) => (
                  <span
                    key={index}
                    onClick={() => onWordClick(word)}
                    className="relation-tag"
                  >
                    {word}
                  </span>
                ))}
              </div>
            </div>
          ))}
          
          {rootWord && !selectedRelationType && (
            <div className="relation-group root">
              <h4>Root Word</h4>
              <div className="relation-tags">
                <span
                  className="relation-tag root-tag"
                  onClick={() => onWordClick(rootWord)}
                >
                  {rootWord}
                </span>
              </div>
            </div>
          )}
        </div>
      </>
    );
  };

  return (
    <div className="word-details">
      <div className="word-header">
        <h2>{wordInfo.lemma}</h2>
        <div className="word-meta">
          {wordInfo.language_code && (
            <span className="language">{wordInfo.language_code}</span>
          )}
          {wordInfo.tags && wordInfo.tags.map((tag, index) => (
            <span key={index} className="tag">{tag}</span>
          ))}
        </div>
      </div>
      
      {wordInfo.pronunciation && (
        <div className="pronunciation-section">
          <h3>Pronunciation</h3>
          <div className="pronunciation-content">
            <span className="pronunciation-text">{wordInfo.pronunciation.text}</span>
            {wordInfo.pronunciation.ipa && (
              <span className="ipa">{wordInfo.pronunciation.ipa}</span>
            )}
            {wordInfo.pronunciation.audio_url && (
              <button 
                className={`play-audio ${isAudioPlaying ? 'playing' : ''}`}
                onClick={playAudio}
                aria-label={isAudioPlaying ? "Pause pronunciation" : "Play pronunciation"}
              >
                {isAudioPlaying ? '⏸' : '▶'}
              </button>
            )}
          </div>
        </div>
      )}
      
      <div className="word-content-tabs">
        <button 
          className={`tab-button ${activeTab === 'definitions' ? 'active' : ''}`}
          onClick={() => setActiveTab('definitions')}
        >
          Definitions
        </button>
        <button 
          className={`tab-button ${activeTab === 'etymology' ? 'active' : ''}`}
          onClick={() => setActiveTab('etymology')}
        >
          Etymology
        </button>
        <button 
          className={`tab-button ${activeTab === 'relations' ? 'active' : ''}`}
          onClick={() => setActiveTab('relations')}
        >
          Related Words
        </button>
      </div>
      
      <div className="word-content">
        {activeTab === 'definitions' && renderDefinitions(wordInfo)}
        {activeTab === 'etymology' && renderEtymology(wordInfo)}
        {activeTab === 'relations' && renderRelatedWords()}
      </div>
    </div>
  );
});

export default WordDetails;