import React, { useCallback, useState, useEffect, useRef } from 'react';
import { WordInfo, WordNetwork } from '../types';
import { fetchWordNetwork } from '../api/wordApi';
import './WordDetails.css';
import './Tabs.css';

interface WordDetailsProps {
  wordInfo: WordInfo;
  onWordClick: (word: string) => void;
}

const WordDetails: React.FC<WordDetailsProps> = React.memo(({ wordInfo, onWordClick }) => {
  // Simple debug log for development
  console.log("WordDetails component rendering with wordInfo:", wordInfo);
  
  const [activeTab, setActiveTab] = useState<'definitions' | 'relations' | 'etymology' | 'baybayin' | 'stats' | 'idioms'>('definitions');
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const [expandedDefinitions, setExpandedDefinitions] = useState<Record<string, boolean>>({});
  const [networkData, setNetworkData] = useState<WordNetwork | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [showMetadata, setShowMetadata] = useState(false);

  useEffect(() => {
    // Reset states when word changes
    setActiveTab('definitions');
    setExpandedDefinitions({});
    setNetworkData(null);
    setIsAudioPlaying(false);
    setShowMetadata(false);
    
    // Initialize expandedGroups for relation types
    if (wordInfo?.relations) {
      const types = ['synonyms', 'antonyms', 'variants', 'derived', 'related'];
      const initialGroups = types.reduce((acc, type) => {
        acc[type] = true;
        return acc;
      }, {} as Record<string, boolean>);
      setExpandedGroups(initialGroups);
    }
    
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

  const toggleGroup = (type: string) => {
    setExpandedGroups(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };

  const toggleMetadata = () => {
    setShowMetadata(prev => !prev);
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

  // Format date for display
  const formatDate = (dateString?: string | null) => {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }).format(date);
    } catch (e) {
      return dateString;
    }
  };

  if (!wordInfo) {
    return (
      <div className="word-details-container">
        <div className="word-details">
          <div className="empty-state">
            <p>No word information available.</p>
          </div>
        </div>
      </div>
    );
  }

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
      const posName = definition.part_of_speech?.name_en || definition.original_pos || 'Other';
      
      if (!definitionsByPos[posName]) {
        definitionsByPos[posName] = [];
      }
      
      definitionsByPos[posName].push(definition);
    });

    return (
      <div className="definitions-section">
        <div className="section-title">
          <h3>Definitions</h3>
          <span className="definition-count">{word.definitions.length}</span>
        </div>
        
        <div className="definition-groups">
          {Object.entries(definitionsByPos).map(([posName, definitions]) => (
            <div key={posName} className="pos-group">
              <div className="pos-group-header">
                {posName}
                <span className="pos-count">{definitions.length}</span>
              </div>
              
              <div className="definition-cards-container">
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
                        <p className="definition-text">
                          {definition.text || definition.definition_text}
                        </p>
                        
                        {/* Display sources for this definition */}
                        {definition.sources && definition.sources.length > 0 && (
                          <div className="definition-sources">
                            <span className="sources-label">Sources:</span>
                            {definition.sources.map((source: string, idx: number) => (
                              <span key={idx} className="source-tag">{source}</span>
                            ))}
                          </div>
                        )}
                        
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
          <button 
            className="fetch-network-button"
            onClick={() => fetchWordNetworkData(wordInfo.lemma)}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Discover Etymology Connections'}
          </button>
        </div>
      );
    }

    return (
      <div className="etymology-section">
        <div className="section-title">
          <h3>Etymology</h3>
          <span className="definition-count">{word.etymologies.length}</span>
        </div>
        
        {word.etymologies.map((etymology, index) => (
          <div key={index} className="etymology-item">
            <p className="etymology-text">{etymology.text || etymology.etymology_text}</p>
            
            {etymology.components && etymology.components.length > 0 && (
              <div className="etymology-components-list">
                <span className="components-label">Components:</span>
                <div className="component-tags">
                  {etymology.components.map((component, idx) => (
                    <span key={idx} className="component-tag" onClick={() => onWordClick(component)}>
                      {component}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {etymology.languages && etymology.languages.length > 0 && (
              <div className="etymology-languages">
                <span className="languages-label">Languages:</span>
                {etymology.languages.map((language, idx) => (
                  <span key={idx} className="language-tag">{language}</span>
                ))}
              </div>
            )}
            
            {etymology.sources && etymology.sources.length > 0 && (
              <div className="etymology-sources">
                <span className="sources-label">Sources:</span>
                {etymology.sources.map((source, idx) => (
                  <span key={idx} className="source-tag">{source}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderRelatedWords = () => {
    if (!wordInfo.relations) {
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

    // Define relation groups with their properties
    const relationGroups = [
      { 
        type: 'synonyms', 
        label: 'Synonyms', 
        words: synonyms, 
        icon: '🔄',
        description: 'Words with the same or similar meaning'
      },
      { 
        type: 'antonyms', 
        label: 'Antonyms', 
        words: antonyms, 
        icon: '⚔️',
        description: 'Words with opposite meanings'
      },
      { 
        type: 'derived', 
        label: 'Derivatives', 
        words: derivatives, 
        icon: '🌱',
        description: 'Words derived from this word'
      },
      { 
        type: 'variants', 
        label: 'Variants', 
        words: variants, 
        icon: '🔀',
        description: 'Alternative forms of this word'
      },
      { 
        type: 'related', 
        label: 'Related Words', 
        words: related, 
        icon: '🔗',
        description: 'Words semantically related to this word'
      },
    ].filter(group => group.words.length > 0);

    return (
      <div className="relations-container">
        <div className="section-title">
          <h3>Related Words</h3>
        </div>
        
        {rootWord && (
          <div className="root-word-section">
            <h4>Root Word</h4>
            <div className="root-word-tag-container">
              <span
                className="relation-tag root-tag"
                onClick={() => onWordClick(rootWord)}
              >
                {rootWord}
              </span>
              <p className="root-word-description">
                This word derives from the root word shown above
              </p>
            </div>
          </div>
        )}

        <div className="relation-groups">
          {relationGroups.map((group, idx) => (
            <div key={idx} className={`relation-group ${group.type}`}>
              <div 
                className={`relation-group-header ${expandedGroups[group.type] ? 'expanded' : ''}`}
                onClick={() => toggleGroup(group.type)}
              >
                <span className="relation-icon">{group.icon}</span>
                <h4>{group.label}</h4>
                <span className="relation-count">{group.words.length}</span>
                <span className="expand-icon">{expandedGroups[group.type] ? '▼' : '▶'}</span>
              </div>
              
              {expandedGroups[group.type] && (
                <div className="relation-group-content">
                  <p className="relation-description">{group.description}</p>
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
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderBaybayin = () => {
    if (!wordInfo.baybayin_form) {
      return (
        <div className="empty-state">
          <p>No Baybayin form available for this word.</p>
        </div>
      );
    }

    return (
      <div className="baybayin-section">
        <div className="section-title">
          <h3>Baybayin Script</h3>
        </div>
        
        <div className="baybayin-display">
          <p className="baybayin-text">{wordInfo.baybayin_form}</p>
          <p className="romanized-text">{wordInfo.romanized_form || wordInfo.lemma}</p>
        </div>
        
        <div className="baybayin-info">
          <h4>About Baybayin</h4>
          <p>
            Baybayin is a pre-Spanish Philippine writing system. It is an abugida, 
            or alphasyllabary, which was used in the Philippines prior to Spanish colonization.
            Each character represents a consonant with an inherent a vowel sound, while other 
            vowel sounds are indicated by marks above or below the character.
          </p>
          <div className="baybayin-character-guide">
            <h4>Common Characters</h4>
            <div className="character-row">
              {['ᜀ', 'ᜁ', 'ᜂ', 'ᜃ', 'ᜄ', 'ᜅ', 'ᜆ', 'ᜇ', 'ᜈ'].map(char => (
                <div className="baybayin-character" key={char}>
                  <span className="character">{char}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderWordStats = () => {
    return (
      <div className="word-statistics">
        <div className="section-title">
          <h3>Word Statistics</h3>
        </div>

        <div className="statistics-grid">
          <div className="stat-card">
            <h4>Complexity</h4>
            <p className="stat-number">{wordInfo.complexity_score.toFixed(1)}</p>
          </div>
          <div className="stat-card">
            <h4>Usage Frequency</h4>
            <p className="stat-number">{wordInfo.usage_frequency.toFixed(1)}</p>
          </div>
          <div className="stat-card">
            <h4>View Count</h4>
            <p className="stat-number">{wordInfo.view_count}</p>
          </div>
          <div className="stat-card">
            <h4>Data Quality</h4>
            <p className="stat-number">{(wordInfo.data_quality_score * 100).toFixed(0)}%</p>
          </div>
        </div>

        <div className="metadata-section">
          <button className="metadata-toggle" onClick={toggleMetadata}>
            {showMetadata ? 'Hide Metadata' : 'Show Metadata'}
          </button>
          
          <div className={`metadata-content ${showMetadata ? 'visible' : ''}`}>
            <div className="metadata-item">
              <div className="metadata-label">Created</div>
              <div className="metadata-value">{formatDate(wordInfo.created_at)}</div>
            </div>
            <div className="metadata-item">
              <div className="metadata-label">Last Updated</div>
              <div className="metadata-value">{formatDate(wordInfo.updated_at)}</div>
            </div>
            <div className="metadata-item">
              <div className="metadata-label">Last Lookup</div>
              <div className="metadata-value">{formatDate(wordInfo.last_lookup_at)}</div>
            </div>
            <div className="metadata-item">
              <div className="metadata-label">Last Viewed</div>
              <div className="metadata-value">{formatDate(wordInfo.last_viewed_at)}</div>
            </div>
            <div className="metadata-item">
              <div className="metadata-label">Verification Status</div>
              <div className="metadata-value">{wordInfo.is_verified ? 'Verified' : 'Unverified'}</div>
            </div>
            {wordInfo.verification_notes && (
              <div className="metadata-item">
                <div className="metadata-label">Verification Notes</div>
                <div className="metadata-value">{wordInfo.verification_notes}</div>
              </div>
            )}
            <div className="metadata-item">
              <div className="metadata-label">Data Hash</div>
              <div className="metadata-value">{wordInfo.data_hash?.substring(0, 8) || 'N/A'}</div>
            </div>
          </div>
        </div>

        {Object.keys(wordInfo.source_info || {}).length > 0 && (
          <div className="additional-data-section">
            <div className="data-card">
              <h3 className="data-card-title">Source Information</h3>
              <ul className="data-list">
                {Object.entries(wordInfo.source_info).map(([key, value], index) => (
                  <li key={index}>
                    <strong>{key}:</strong> {typeof value === 'string' ? value : JSON.stringify(value)}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderIdioms = () => {
    if (!wordInfo.idioms || !Array.isArray(wordInfo.idioms) || wordInfo.idioms.length === 0) {
      return (
        <div className="empty-state">
          <p>No idiomatic expressions available for this word.</p>
        </div>
      );
    }

    return (
      <div className="idioms-section">
        <div className="section-title">
          <h3>Idioms & Expressions</h3>
          <span className="definition-count">{wordInfo.idioms.length}</span>
        </div>

        <div className="idioms-list">
          {wordInfo.idioms.map((idiom, index) => {
            if (typeof idiom === 'string') {
              return (
                <div key={index} className="idiom-card">
                  <p className="idiom-phrase">{idiom}</p>
                </div>
              );
            }
            
            return (
              <div key={index} className="idiom-card">
                {idiom.phrase && <p className="idiom-phrase">{idiom.phrase}</p>}
                {idiom.text && (!idiom.phrase || idiom.text !== idiom.phrase) && (
                  <p className="idiom-phrase">{idiom.text}</p>
                )}
                {idiom.meaning && (
                  <p className="idiom-meaning">
                    <strong>Meaning:</strong> {idiom.meaning}
                  </p>
                )}
                {idiom.example && (
                  <p className="idiom-example">{idiom.example}</p>
                )}
                {idiom.source && <span className="idiom-source">Source: {idiom.source}</span>}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="word-details-container">
      <div className="word-details">
        <div className="word-header">
          <div className="word-header-main">
            <div className="word-title-pronunciation">
              <h2>
                {wordInfo.lemma}
                {wordInfo.baybayin_form && (
                  <span className="word-baybayin">{wordInfo.baybayin_form}</span>
                )}
              </h2>
              
              <div className="word-meta">
                {wordInfo.language_code && (
                  <span className="language">{wordInfo.language_code.toUpperCase()}</span>
                )}
                {wordInfo.preferred_spelling && wordInfo.preferred_spelling !== wordInfo.lemma && (
                  <span className="preferred-spelling">Preferred: {wordInfo.preferred_spelling}</span>
                )}
                {(wordInfo as any).is_root_word && (
                  <span className="root-word-badge">Root Word</span>
                )}
                {wordInfo.tags && wordInfo.tags.map((tag, idx) => (
                  <span key={idx} className="tag">{tag}</span>
                ))}
              </div>
            </div>
          </div>
          
          {wordInfo.pronunciation && (
            <div className="pronunciation-section">
              <div className="pronunciation-content">
                {wordInfo.pronunciation.text && (
                  <span className="pronunciation-text">{wordInfo.pronunciation.text}</span>
                )}
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
        </div>
        
        <div className="word-content-tabs">
          <div 
            className={`tab-item ${activeTab === 'definitions' ? 'active' : ''}`}
            onClick={() => setActiveTab('definitions')}
          >
            <span className="tab-icon">📚</span> Definitions {wordInfo.definitions?.length > 0 && `(${wordInfo.definitions.length})`}
          </div>
          {(wordInfo.etymologies && wordInfo.etymologies.length > 0) && (
            <div 
              className={`tab-item ${activeTab === 'etymology' ? 'active' : ''}`}
              onClick={() => setActiveTab('etymology')}
            >
              <span className="tab-icon">🔍</span> Etymology {wordInfo.etymologies.length > 0 && `(${wordInfo.etymologies.length})`}
            </div>
          )}
          <div 
            className={`tab-item ${activeTab === 'relations' ? 'active' : ''}`}
            onClick={() => setActiveTab('relations')}
          >
            <span className="tab-icon">🔗</span> Related Words
          </div>
          {wordInfo.idioms && wordInfo.idioms.length > 0 && (
            <div 
              className={`tab-item ${activeTab === 'idioms' ? 'active' : ''}`}
              onClick={() => setActiveTab('idioms')}
            >
              <span className="tab-icon">💬</span> Idioms {wordInfo.idioms.length > 0 && `(${wordInfo.idioms.length})`}
            </div>
          )}
          {wordInfo.baybayin_form && (
            <div 
              className={`tab-item ${activeTab === 'baybayin' ? 'active' : ''}`}
              onClick={() => setActiveTab('baybayin')}
            >
              <span className="tab-icon">𑁋</span> Baybayin
            </div>
          )}
          <div 
            className={`tab-item ${activeTab === 'stats' ? 'active' : ''}`}
            onClick={() => setActiveTab('stats')}
          >
            <span className="tab-icon">📊</span> Stats
          </div>
        </div>
        
        <div className="word-content">
          {activeTab === 'definitions' && renderDefinitions(wordInfo)}
          {activeTab === 'etymology' && renderEtymology(wordInfo)}
          {activeTab === 'relations' && renderRelatedWords()}
          {activeTab === 'baybayin' && renderBaybayin()}
          {activeTab === 'stats' && renderWordStats()}
          {activeTab === 'idioms' && renderIdioms()}
        </div>
      </div>
    </div>
  );
});

export default WordDetails;