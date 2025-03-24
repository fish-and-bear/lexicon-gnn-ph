import React, { useCallback, useState, useEffect, useRef } from 'react';
import { WordInfo, WordNetwork, RelatedWord } from '../types';
import { fetchWordNetwork } from '../api/wordApi';
import './WordDetails.css';
import './Tabs.css';
import { useTheme } from '../contexts/ThemeContext';

interface WordDetailsProps {
  wordInfo: WordInfo;
  onWordClick: (word: string) => void;
}

const WordDetails: React.FC<WordDetailsProps> = React.memo(({ wordInfo, onWordClick }) => {
  // Simple debug log for development
  console.log("WordDetails component rendering with wordInfo:", wordInfo);
  
  const { theme } = useTheme();
  const [activeTab, setActiveTab] = useState<'definitions' | 'relations' | 'etymology' | 'baybayin' | 'idioms'>('definitions');
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
      const types = [
        'main', 'synonyms', 'antonyms', 'variants', 'related', 'kaugnay',
        'derived', 'derived_from', 'root_of', 'component_of', 'cognate',
        'etymology', 'derivative', 'associated', 'other'
      ];
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

  const fetchWordNetworkData = async (word: string) => {
    try {
      setIsLoading(true);
      const response = await fetch(`/api/word-network/${word}`);
      const data = await response.json();
      
      // Transform the data to match our WordNetwork type
      const transformedData: WordNetwork = {
        main_words: data.main_words || [],
        root_words: data.root_words || [],
        antonyms: data.antonyms || [],
        derived_words: data.derived_words || [],
        related_words: data.related_words || [],
        synonyms: data.synonyms || [],
        kaugnay: data.kaugnay || [],
        other: data.other || []
      };
      
      console.log('Network data transformed:', transformedData);
      setNetworkData(transformedData);
    } catch (error) {
      console.error('Error fetching word network:', error);
      // Set empty data structure on error
      setNetworkData({
        main_words: [],
        root_words: [],
        antonyms: [],
        derived_words: [],
        related_words: [],
        synonyms: [],
        kaugnay: [],
        other: []
      });
    } finally {
      setIsLoading(false);
    }
  };

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

  const processNetworkData = (data: any) => {
    const processedData: Record<string, string[]> = {
      main: [],
      root_words: [],
      antonyms: [],
      derived: [],
      related: [],
      other: [],
      synonyms: [],
      kaugnay: []
    };

    if (!data) return processedData;

    // Process main words
    if (data.main_words && Array.isArray(data.main_words)) {
      processedData.main = data.main_words;
    }

    // Process root words
    if (data.root_words && Array.isArray(data.root_words)) {
      processedData.root_words = data.root_words;
    }

    // Process antonyms
    if (data.antonyms && Array.isArray(data.antonyms)) {
      processedData.antonyms = data.antonyms;
    }

    // Process derived words
    if (data.derived_words && Array.isArray(data.derived_words)) {
      processedData.derived = data.derived_words;
    }

    // Process related words
    if (data.related_words && Array.isArray(data.related_words)) {
      processedData.related = data.related_words;
    }

    // Process synonyms
    if (data.synonyms && Array.isArray(data.synonyms)) {
      processedData.synonyms = data.synonyms;
    }

    // Process kaugnay (Filipino-specific related words)
    if (data.kaugnay && Array.isArray(data.kaugnay)) {
      processedData.kaugnay = data.kaugnay;
    }

    // Process other relations
    if (data.other && Array.isArray(data.other)) {
      processedData.other = data.other;
    }

    return processedData;
  };

  if (!wordInfo || !wordInfo.lemma) {
    return (
      <div className={`word-details-container ${theme}`}>
        <div className={`word-details no-word-selected ${theme}`}>
          <div className="empty-state">
            <h3>No Word Selected</h3>
            <p>Please select a word from the search results to view its details.</p>
          </div>
        </div>
      </div>
    );
  }

  const renderDefinitions = (word: WordInfo) => {
    if (!word.definitions || word.definitions.length === 0) {
      return (
        <div className={`empty-state ${theme}`}>
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
      <div className={`definitions-container ${theme}`}>
        <div className="section-title">
          <h3>Definitions</h3>
          <span className={`definition-count ${theme}`}>{word.definitions.length}</span>
        </div>
        
        <div className={`definition-groups ${theme}`}>
        {Object.entries(definitionsByPos).map(([posName, definitions]) => (
          <div key={posName} className="pos-group">
              <div className={`pos-group-header ${theme}`}>
              {posName}
                <span className={`pos-count ${theme}`}>{definitions.length}</span>
            </div>
            
              <div className="definition-cards-container">
                {definitions.map((definition, index) => {
                  const defId = `${posName}-${index}`;
                  const isExpanded = expandedDefinitions[defId] || false;
                  
                  return (
                    <div 
                      key={index} 
                      className={`definition-card ${isExpanded ? 'expanded' : ''} ${theme}`}
                      onClick={() => toggleDefinitionExpand(defId)}
                    >
                      <span className={`definition-number ${theme}`}>{index + 1}</span>
                      <div className={`definition-content ${theme}`}>
                        <p className={`definition-text ${theme}`}>
                          {definition.text || definition.definition_text}
                        </p>
                        
                        {/* Display sources for this definition */}
                        {definition.sources && definition.sources.length > 0 && (
                          <div className={`definition-sources ${theme}`}>
                            <span className={`sources-label ${theme}`}>Sources:</span>
                            <div className={`source-tags ${theme}`}>
                              {definition.sources.map((source: string, idx: number) => (
                                <span key={idx} className={`source-tag ${theme}`}>{source}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {(definition.examples?.length > 0 || definition.usage_notes?.length > 0) && (
                          <div className={`definition-details ${isExpanded ? 'visible' : ''} ${theme}`}>
                {definition.examples && definition.examples.length > 0 && (
                              <div className={`examples ${theme}`}>
                                <h4 className={theme}>Examples</h4>
                                <ul className={theme}>
                      {definition.examples.map((example: string, idx: number) => (
                                    <li key={idx} className={theme}>
                                      {example}
                                      {definition.example_translations?.[idx] && (
                                        <span className={`translation ${theme}`}>{definition.example_translations[idx]}</span>
                                      )}
                                    </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {definition.usage_notes && definition.usage_notes.length > 0 && (
                              <div className={`usage-notes ${theme}`}>
                                <h4 className={theme}>Usage Notes</h4>
                                <ul className={theme}>
                      {definition.usage_notes.map((note: string, idx: number) => (
                                    <li key={idx} className={theme}>{note}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
                        )}
                        
                        {(definition.examples?.length > 0 || definition.usage_notes?.length > 0) && (
                          <div className={`expand-indicator ${theme}`}>
                            <span className={theme}>{isExpanded ? 'Show less' : 'Show more'}</span>
                            <span className={`expand-icon ${theme}`}>{isExpanded ? '▲' : '▼'}</span>
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
        <div className={`empty-state ${theme}`}>
          <p className={theme}>No etymology information available for this word.</p>
          <button 
            className={`fetch-network-button ${theme}`}
            onClick={() => fetchWordNetworkData(wordInfo.lemma)}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Discover Etymology Connections'}
          </button>
        </div>
      );
    }

    return (
      <div className={`etymology-section ${theme}`}>
        <div className={`section-title ${theme}`}>
          <h3 className={theme}>Etymology</h3>
          <span className={`definition-count ${theme}`}>{word.etymologies.length}</span>
        </div>
        
        {word.etymologies.map((etymology, index) => (
          <div 
            key={index} 
            className={`etymology-item ${theme}`}
          >
            <div className={`etymology-text ${theme}`}>
              {etymology.text}
            </div>
            
            {etymology.components && etymology.components.length > 0 && (
              <div className={`etymology-components-list ${theme}`}>
                <span className={`components-label ${theme}`}>Components:</span>
                <div className={`component-tags ${theme}`}>
                  {etymology.components.map((component, idx) => (
                    <span key={idx} className={`component-tag ${theme}`} onClick={() => onWordClick(component)}>
                      {component}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {etymology.languages && etymology.languages.length > 0 && (
              <div className={`etymology-languages ${theme}`}>
                <span className={`languages-label ${theme}`}>Languages:</span>
                <div className={`language-tags ${theme}`}>
                  {etymology.languages.map((language, idx) => (
                    <span key={idx} className={`language-tag ${theme}`}>{language}</span>
                  ))}
                </div>
              </div>
            )}
            
            {etymology.sources && etymology.sources.length > 0 && (
              <div className={`etymology-sources ${theme}`}>
                <span className={`sources-label ${theme}`}>Sources:</span>
                <div className={`source-tags ${theme}`}>
                  {etymology.sources.map((source, idx) => (
                    <span key={idx} className={`source-tag ${theme}`}>{source}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderRelatedWords = () => {
    const allRelationData = processNetworkData(networkData);
    
    const hasRelations = Object.values(allRelationData).some(arr => arr.length > 0);
    
    if (!hasRelations) {
      return (
        <div className={`empty-state ${theme}`}>
          <p className={theme}>No related words available.</p>
          <button 
            className={`fetch-network-button ${theme}`}
            onClick={() => fetchWordNetworkData(wordInfo.lemma)}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Discover Word Network'}
          </button>
        </div>
      );
    }

    // Define all possible relation groups with their properties
    const relationGroups = [
      { 
        type: 'main', 
        label: 'Main Words', 
        words: allRelationData['main'] || [], 
        icon: '📌',
        description: 'Primary words related to this term'
      },
      { 
        type: 'root_words', 
        label: 'Root Words', 
        words: allRelationData['root_words'] || [], 
        icon: '🌳',
        description: 'Base words from which this word is derived'
      },
      { 
        type: 'antonyms', 
        label: 'Antonyms', 
        words: allRelationData['antonyms'] || [], 
        icon: '⚔️',
        description: 'Words with opposite meanings'
      },
      { 
        type: 'derived', 
        label: 'Derived Words', 
        words: allRelationData['derived'] || [], 
        icon: '🌱',
        description: 'Words derived from this term'
      },
      { 
        type: 'synonyms', 
        label: 'Synonyms', 
        words: allRelationData['synonyms'] || [], 
        icon: '🔄',
        description: 'Words with similar meanings'
      },
      { 
        type: 'kaugnay', 
        label: 'Kaugnay', 
        words: allRelationData['kaugnay'] || [], 
        icon: '🔗',
        description: 'Filipino-specific related words'
      },
      { 
        type: 'related', 
        label: 'Related Words', 
        words: allRelationData['related'] || [], 
        icon: '🤝',
        description: 'Words semantically related to this term'
      },
      { 
        type: 'other', 
        label: 'Other Relations', 
        words: allRelationData['other'] || [], 
        icon: '📎',
        description: 'Other word relationships'
      }
    ];

    return (
      <div className={`relations-section ${theme}`}>
        <div className={`section-title ${theme}`}>
          <h3 className={theme}>Word Relations</h3>
        </div>
        
        <div className={`relation-groups ${theme}`}>
          {relationGroups
            .filter(group => group.words && group.words.length > 0)
            .map((group, index) => (
              <div key={index} className={`relation-group ${group.type} ${theme}`}>
                <div 
                  className={`relation-group-header ${expandedGroups[group.type] ? 'expanded' : ''} ${theme}`}
                  onClick={() => toggleGroup(group.type)}
                >
                  <span className={`relation-icon ${theme}`}>{group.icon}</span>
                  <h4 className={theme}>{group.label}</h4>
                  <span className={`relation-count ${theme}`}>{group.words.length}</span>
                  <span className={`expand-icon ${theme}`}>
                    {expandedGroups[group.type] ? '▼' : '▶'}
                  </span>
                </div>
                
                {expandedGroups[group.type] && (
                  <div className={`relation-group-content ${theme}`}>
                    <p className={`relation-description ${theme}`}>{group.description}</p>
                    <div className={`relation-tags ${theme}`}>
                      {group.words.map((word, idx) => (
                        <span 
                          key={idx} 
                          className={`relation-tag ${theme}`}
                          onClick={() => onWordClick(word)}
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
        <div className={`empty-state ${theme}`}>
          <p className={theme}>No Baybayin form available for this word.</p>
        </div>
      );
    }

    return (
      <div 
        className={`baybayin-section ${theme}`} 
        style={{
          backgroundColor: theme === 'dark' ? '#121828' : '',
          backgroundImage: theme === 'dark' ? 'linear-gradient(135deg, rgba(25, 31, 52, 0.7) 0%, rgba(18, 24, 43, 0.6) 100%)' : '',
          borderLeft: theme === 'dark' ? '3px solid #ffd166' : '',
          boxShadow: theme === 'dark' ? '0 4px 16px rgba(0, 0, 0, 0.25)' : '',
          marginBottom: '24px',
          padding: '20px',
          borderRadius: '8px',
          overflow: 'hidden'
        }}
      >
        <div 
          className={`section-title ${theme}`}
          style={{
            marginBottom: '16px',
            borderBottom: theme === 'dark' ? '1px solid rgba(75, 86, 115, 0.3)' : '',
            paddingBottom: '8px'
          }}
        >
          <h3 
            className={theme} 
            style={{
              color: theme === 'dark' ? '#ffd166' : '',
              margin: '0',
              fontWeight: '600',
              fontSize: '1.25rem',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <span style={{ 
              fontSize: '1.4em', 
              opacity: '0.9', 
              color: theme === 'dark' ? '#ffd166' : '',
              textShadow: theme === 'dark' ? '0 0 8px rgba(255, 209, 102, 0.5)' : ''
            }}>
              𑁋
            </span>
            Baybayin Script
          </h3>
        </div>
        
        <div 
          className={`baybayin-display ${theme}`} 
          style={{
            backgroundColor: theme === 'dark' ? '#171d31' : '',
            border: theme === 'dark' ? '1px solid rgba(75, 86, 115, 0.4)' : '',
            boxShadow: theme === 'dark' ? '0 4px 20px rgba(0, 0, 0, 0.3)' : '',
            textAlign: 'center',
            padding: '24px',
            borderRadius: '8px',
            margin: '12px 0 24px'
          }}
        >
          <div 
            className={`baybayin-text ${theme}`}
            style={{
              color: theme === 'dark' ? '#ffd166' : '',
              fontSize: '3.5rem',
              fontFamily: '"Baybayin Modern", sans-serif',
              margin: '0 0 12px 0',
              textShadow: theme === 'dark' ? '0 2px 8px rgba(255, 209, 102, 0.4)' : ''
            }}
          >
            {wordInfo.baybayin_form}
          </div>
          <div 
            className={`romanized-text ${theme}`}
            style={{
              color: theme === 'dark' ? '#a0c3e2' : '',
              fontSize: '1.2rem'
            }}
          >
            {wordInfo.romanized_form || wordInfo.lemma}
          </div>
        </div>
        
        <div 
          className={`baybayin-info ${theme}`} 
          style={{
            backgroundColor: theme === 'dark' ? 'rgba(23, 29, 49, 0.8)' : '',
            border: theme === 'dark' ? '1px solid rgba(75, 86, 115, 0.4)' : '',
            borderRadius: '8px',
            padding: '16px',
            position: 'relative',
            zIndex: 1
          }}
        >
          <h4 
            className={theme}
            style={{
              color: theme === 'dark' ? '#ffd166' : '',
              marginTop: '0',
              marginBottom: '12px',
              fontWeight: '600'
            }}
          >About Baybayin</h4>
          <p 
            className={theme}
            style={{
              color: theme === 'dark' ? '#a0a8c0' : '',
              fontSize: '0.95rem',
              lineHeight: '1.6',
              marginBottom: '16px'
            }}
          >
            Baybayin is a pre-Spanish Philippine writing system. It is an abugida, 
            or alphasyllabary, which was used in the Philippines prior to Spanish colonization.
            Each character represents a consonant with an inherent a vowel sound, while other 
            vowel sounds are indicated by marks above or below the character.
          </p>
          <div 
            className={`baybayin-character-guide ${theme}`} 
            style={{
              backgroundColor: theme === 'dark' ? 'rgba(33, 39, 59, 0.8)' : '',
              border: theme === 'dark' ? '1px solid rgba(75, 86, 115, 0.4)' : '',
              borderRadius: '8px',
              padding: '16px',
              marginTop: '16px'
            }}
          >
            <h4 
              className={theme}
              style={{
                color: theme === 'dark' ? '#ffd166' : '',
                marginTop: '0',
                marginBottom: '12px',
                fontWeight: '600'
              }}
            >
              Common Characters
            </h4>
            <div 
              className={`character-row ${theme}`}
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '8px',
                justifyContent: 'center'
              }}
            >
              {['ᜀ', 'ᜁ', 'ᜂ', 'ᜃ', 'ᜄ', 'ᜅ', 'ᜆ', 'ᜇ', 'ᜈ'].map(char => (
                <div 
                  className={`baybayin-character ${theme}`} 
                  key={char}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '4px'
                  }}
                >
                  <span 
                    className={`character ${theme}`} 
                    style={{
                      backgroundColor: theme === 'dark' ? 'rgba(33, 39, 59, 0.9)' : '',
                      color: theme === 'dark' ? '#ffd166' : '',
                      border: theme === 'dark' ? '1px solid rgba(255, 209, 102, 0.2)' : '',
                      width: '2.5rem',
                      height: '2.5rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: '4px',
                      fontSize: '1.5rem',
                      fontFamily: '"Baybayin Modern", sans-serif'
                    }}
                  >
                    {char}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderIdioms = () => {
    if (!wordInfo.idioms || !Array.isArray(wordInfo.idioms) || wordInfo.idioms.length === 0) {
      return (
        <div className={`empty-state ${theme}`}>
          <p className={theme}>No idiomatic expressions available for this word.</p>
        </div>
      );
    }
    
    return (
      <div className={`idioms-section ${theme}`}>
        <div className={`section-title ${theme}`}>
          <h3 className={theme}>Idioms & Expressions</h3>
          <span className={`definition-count ${theme}`}>{wordInfo.idioms.length}</span>
        </div>
        
        <div className={`idioms-list ${theme}`}>
          {wordInfo.idioms.map((idiom, index) => {
            if (typeof idiom === 'string') {
              return (
                <div 
                  key={index} 
                  className={`idiom-card ${theme}`}
                >
                  <p className={`idiom-phrase ${theme}`}>{idiom}</p>
                </div>
              );
            }

  return (
              <div 
                key={index} 
                className={`idiom-card ${theme}`}
              >
                {idiom.phrase && <p className={`idiom-phrase ${theme}`}>{idiom.phrase}</p>}
                {idiom.text && (!idiom.phrase || idiom.text !== idiom.phrase) && (
                  <p className={`idiom-phrase ${theme}`}>{idiom.text}</p>
                )}
                {idiom.meaning && (
                  <p className={`idiom-meaning ${theme}`}>
                    <strong className={theme}>Meaning:</strong> {idiom.meaning}
                  </p>
                )}
                {idiom.example && (
                  <p className={`idiom-example ${theme}`}>{idiom.example}</p>
                )}
                {idiom.source && <span className={`idiom-source ${theme}`}>Source: {idiom.source}</span>}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className={`word-details-container ${theme}`}>
      <div className={`word-details ${theme}`}>
        {/* Word header section */}
        <div className={`word-header ${theme}`}>
          <div className="word-header-main">
            <div className="word-title-pronunciation">
              <h2 className={theme}>
                {wordInfo.lemma}
                {wordInfo.baybayin_form && (
                  <span className={`word-baybayin ${theme}`}>{wordInfo.baybayin_form}</span>
                )}
              </h2>
              
              <div className={`word-meta ${theme}`}>
        {wordInfo.language_code && (
                  <span className={`language ${theme}`}>{wordInfo.language_code.toUpperCase()}</span>
                )}
                {wordInfo.preferred_spelling && wordInfo.preferred_spelling !== wordInfo.lemma && (
                  <span className={`preferred-spelling ${theme}`}>Preferred: {wordInfo.preferred_spelling}</span>
                )}
                {(wordInfo as any).is_root_word && (
                  <span className={`root-word-badge ${theme}`}>Root Word</span>
                )}
                {wordInfo.tags && wordInfo.tags.map((tag, idx) => (
                  <span key={idx} className={`tag ${theme}`}>{tag}</span>
                ))}
              </div>
            </div>
      </div>
      
      {wordInfo.pronunciation && (
            <div className={`pronunciation-section ${theme}`}>
              <div className={`pronunciation-content ${theme}`}>
                {wordInfo.pronunciation.text && (
                  <span className={`pronunciation-text ${theme}`}>{wordInfo.pronunciation.text}</span>
                )}
          {wordInfo.pronunciation.ipa && (
                  <span className={`ipa ${theme}`}>{wordInfo.pronunciation.ipa}</span>
          )}
          {wordInfo.pronunciation.audio_url && (
            <button 
                    className={`play-audio ${isAudioPlaying ? 'playing' : ''} ${theme}`}
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
        
        {/* Word details tabs */}
        <div className={`word-content-tabs ${theme}`}>
          <div 
            className={`tab-item ${activeTab === 'definitions' ? 'active' : ''} ${theme}`}
            onClick={() => setActiveTab('definitions')}
          >
            <span className={`tab-icon ${theme}`}>📖</span> Definitions
            {wordInfo.definitions?.length > 0 && (
              <span className={`count ${theme}`}>{wordInfo.definitions.length}</span>
            )}
          </div>
          {wordInfo.etymologies && wordInfo.etymologies.length > 0 && (
            <div 
              className={`tab-item ${activeTab === 'etymology' ? 'active' : ''} ${theme}`}
              onClick={() => setActiveTab('etymology')}
            >
              <span className={`tab-icon ${theme}`}>🔍</span> Etymology
              <span className={`count ${theme}`}>{wordInfo.etymologies.length}</span>
            </div>
          )}
          {wordInfo.relations && Object.keys(wordInfo.relations).length > 0 && (
            <div 
              className={`tab-item ${activeTab === 'relations' ? 'active' : ''} ${theme}`}
              onClick={() => setActiveTab('relations')}
            >
              <span className={`tab-icon ${theme}`}>🔄</span> Relations
              <span className={`count ${theme}`}>{Object.keys(wordInfo.relations).length}</span>
            </div>
          )}
          {wordInfo.idioms && wordInfo.idioms.length > 0 && (
            <div 
              className={`tab-item ${activeTab === 'idioms' ? 'active' : ''} ${theme}`}
              onClick={() => setActiveTab('idioms')}
            >
              <span className={`tab-icon ${theme}`}>💬</span> Idioms
              <span className={`count ${theme}`}>{wordInfo.idioms.length}</span>
        </div>
      )}
          {wordInfo.baybayin_form && (
            <div 
              className={`tab-item ${activeTab === 'baybayin' ? 'active' : ''} ${theme}`}
              onClick={() => setActiveTab('baybayin')}
            >
              <span className={`tab-icon ${theme}`}>𑁋</span> Baybayin
            </div>
          )}
        </div>
        
        <div className={`word-content ${theme}`}>
          {activeTab === 'definitions' && renderDefinitions(wordInfo)}
          {activeTab === 'etymology' && renderEtymology(wordInfo)}
          {activeTab === 'relations' && renderRelatedWords()}
          {activeTab === 'baybayin' && renderBaybayin()}
          {activeTab === 'idioms' && renderIdioms()}
        </div>
      </div>
    </div>
  );
});

export default WordDetails;