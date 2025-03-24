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
    console.log("Rendering related words for wordInfo:", wordInfo);
    
    // Try to use relations from wordInfo first
    if (!wordInfo.relations || Object.keys(wordInfo.relations).every(key => 
      !wordInfo.relations![key] || 
      (Array.isArray(wordInfo.relations![key]) && !(wordInfo.relations![key] as any).length)
    )) {
      console.log("No relations found in wordInfo.relations or all relation arrays are empty");
      
      // If we have network data, try to extract relations from there
      if (networkData && networkData.nodes && networkData.nodes.length > 0) {
        console.log("Attempting to extract relations from network data");
        
        // Create a map of lemmas (excluding the current word)
        const relatedLemmas = networkData.nodes
          .filter(node => node.word && node.word !== wordInfo.lemma)
          .map(node => node.word);
          
        console.log("Found related lemmas in network data:", relatedLemmas);
        
        if (relatedLemmas.length > 0) {
          // Group these into "related" category for display
          const extractedRelations: Record<string, string[]> = {
            'related': relatedLemmas
          };
          
          // Create relation groups similar to what we do with wordInfo.relations
          const relationGroups = [
            {
              type: 'related',
              label: 'Related Words',
              description: 'Words connected to this word in the language network',
              words: relatedLemmas,
              icon: '🔗'
            }
          ];
          
          // Create UI similar to the regular relations display
          return (
            <div className={`relations-section ${theme}`}>
              <div className={`section-title ${theme}`}>
                <h3 className={theme}>Word Relations</h3>
                <span className={`relation-count ${theme}`}>{relatedLemmas.length}</span>
              </div>
              
              <div className={`relation-groups ${theme}`}>
                {relationGroups.map((group, idx) => (
                  <div 
                    className={`relation-group ${group.type} ${expandedGroups[group.type] ? 'expanded' : ''} ${theme}`}
                    key={group.type}
                    onClick={() => toggleGroup(group.type)}
                  >
                    <div 
                      className={`relation-group-header ${expandedGroups[group.type] ? 'expanded' : ''} ${theme}`}
                    >
                      <h4 className={theme}>{group.label}</h4>
                      <span className={`relation-count ${theme}`}>{group.words.length}</span>
                      <span className={`expand-icon ${theme}`}>{expandedGroups[group.type] ? '▼' : '▶'}</span>
                    </div>
                    
                    {expandedGroups[group.type] && (
                      <div className={`relation-group-content ${theme}`}>
                        <p className={`relation-description ${theme}`}>{group.description}</p>
                        <div className={`relation-tags ${theme}`}>
                          {group.words.map((word: string, index: number) => (
                            <span
                              key={index}
                              className={`relation-tag ${theme}`}
                              onClick={(e) => {
                                e.stopPropagation();
                                onWordClick(word);
                              }}
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
        }
      }
      
      // If we don't have relations or network data, show empty state
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
    
    // Continue with existing code to process wordInfo.relations
    // Extract all possible relation types
    const possibleRelationTypes = wordInfo.relations ? Object.keys(wordInfo.relations) : [];
    console.log('Possible relation types:', possibleRelationTypes);
    
    // Process each relation type to gather related words
    const allRelationData: Record<string, string[]> = {};
    possibleRelationTypes.forEach(type => {
      if (wordInfo.relations && wordInfo.relations[type] && Array.isArray(wordInfo.relations[type])) {
        allRelationData[type] = wordInfo.relations[type].map((item: RelatedWord) => item.word);
        console.log(`Processed relation type '${type}':`, allRelationData[type]);
      } else if (
        type === 'root' && 
        wordInfo.relations && 
        wordInfo.relations[type]
      ) {
        // Store the reference to avoid TypeScript errors
        const rootRelation = wordInfo.relations[type];
        
        // Special case for 'root' which may not be an array - checking safely
        if (typeof rootRelation === 'object' && rootRelation !== null && 'word' in rootRelation) {
          console.log(`Found root word:`, (rootRelation as {word: string}).word);
          // Add root word to the appropriate category
          if (!allRelationData['root']) {
            allRelationData['root'] = [];
          }
          allRelationData['root'].push(rootRelation.word);
        }
      }
    });
    
    // Special handling for single root word
    const rootWord = wordInfo.relations.root?.word;
    
    // Extract synonyms words so we can exclude them from related words
    const synonymWords = new Set(allRelationData['synonyms'] || []);
    
    // For the related words category, exclude any words already in synonyms
    if (allRelationData['related']) {
      allRelationData['related'] = allRelationData['related'].filter(word => !synonymWords.has(word));
    }
    
    const hasRelations = Object.values(allRelationData).some(arr => arr.length > 0) || !!rootWord;
    console.log("Has relations:", hasRelations, "Root word:", rootWord);
    console.log("All relation data:", allRelationData);
    
    if (!hasRelations) {
      console.log("No related words found in processed data");
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
        type: 'synonyms', 
        label: 'Synonyms', 
        words: allRelationData['synonyms'] || [], 
        icon: '🔄',
        description: 'Words with the same or similar meaning'
      },
      { 
        type: 'antonyms', 
        label: 'Antonyms', 
        words: allRelationData['antonyms'] || [], 
        icon: '⚔️',
        description: 'Words with opposite meanings'
      },
      { 
        type: 'kaugnay', 
        label: 'Kaugnay', 
        words: allRelationData['kaugnay'] || [], 
        icon: '🔗',
        description: 'Filipino-specific related words'
      },
      { 
        type: 'derived', 
        label: 'Derivatives', 
        words: [...(allRelationData['derived'] || []), ...(allRelationData['derived_from'] || [])], 
        icon: '🌱',
        description: 'Words derived from this word'
      },
      { 
        type: 'root_of', 
        label: 'Root Words', 
        words: allRelationData['root_of'] || [], 
        icon: '🌳',
        description: 'Words for which this word serves as a root'
      },
      { 
        type: 'component_of', 
        label: 'Component Words', 
        words: allRelationData['component_of'] || [], 
        icon: '🧩',
        description: 'Words that use this word as a component'
      },
      { 
        type: 'cognate', 
        label: 'Cognates', 
        words: allRelationData['cognate'] || [], 
        icon: '🌐',
        description: 'Words sharing the same linguistic origin'
      },
      { 
        type: 'etymology', 
        label: 'Etymology Words', 
        words: allRelationData['etymology'] || [], 
        icon: '📜',
        description: 'Words related through etymology'
      },
      {
        type: 'variants', 
        label: 'Variants', 
        words: allRelationData['variants'] || [], 
        icon: '🔀',
        description: 'Alternative forms of this word'
      },
      { 
        type: 'associated', 
        label: 'Associated Words', 
        words: allRelationData['associated'] || [], 
        icon: '🤝',
        description: 'Words associated with this term'
      },
      {
        type: 'related',
        label: 'Related Words',
        words: allRelationData['related'] || [], 
        icon: '📎',
        description: 'Words semantically related to this word'
      },
      { 
        type: 'derivative', 
        label: 'Derivative Words', 
        words: allRelationData['derivative'] || [], 
        icon: '🔰',
        description: 'Words that are derivatives of this word'
      },
      { 
        type: 'other', 
        label: 'Other Relations', 
        words: allRelationData['other'] || [], 
        icon: '📋',
        description: 'Other words with some relationship to this term'
      },
    ].filter(group => group.words.length > 0);
    
    return (
      <div className={`relations-container ${theme}`}>
        {rootWord && (
          <div className={`root-word-section ${theme}`}>
            <h4 className={theme}>Root Word</h4>
            <div className="root-word-tag-container">
              <span 
                className={`relation-tag root-tag ${theme}`}
                onClick={() => onWordClick(rootWord)}
              >
                {rootWord}
              </span>
              <p className={`root-word-description ${theme}`}>
                This word derives from the root word shown above
              </p>
            </div>
          </div>
        )}
        
        <div className={`relation-groups ${theme}`}>
          {relationGroups.map((group, idx) => (
            <div 
              className={`relation-group ${group.type} ${expandedGroups[group.type] ? 'expanded' : ''} ${theme}`}
              key={group.type}
              onClick={() => toggleGroup(group.type)}
            >
              <div 
                className={`relation-group-header ${expandedGroups[group.type] ? 'expanded' : ''} ${theme}`}
              >
                <h4 className={theme}>{group.label}</h4>
                <span className={`relation-count ${theme}`}>{group.words.length}</span>
                <span className={`expand-icon ${theme}`}>{expandedGroups[group.type] ? '▼' : '▶'}</span>
              </div>
              
              {expandedGroups[group.type] && (
                <div className={`relation-group-content ${theme}`}>
                  <p className={`relation-description ${theme}`}>{group.description}</p>
                  <div className={`relation-tags ${theme}`}>
                    {group.words.map((word: string, index: number) => (
                      <span 
                        key={index}
                        className={`relation-tag ${theme}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          onWordClick(word);
                        }}
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
      <div className={`baybayin-section ${theme}`}>
        <div className={`section-title ${theme}`}>
          <h3 className={theme}>Baybayin Script</h3>
        </div>
        
        <div className={`baybayin-display ${theme}`} style={theme === 'dark' ? { backgroundColor: '#171d31', borderColor: 'rgba(75, 86, 115, 0.4)' } : {}}>
          <div className={`baybayin-text ${theme}`}>{wordInfo.baybayin_form}</div>
          <div className={`romanized-text ${theme}`}>{wordInfo.romanized_form || wordInfo.lemma}</div>
        </div>
        
        <div className={`baybayin-info ${theme}`}>
          <h4 className={theme}>About Baybayin</h4>
          <p className={theme}>
            Baybayin is a pre-Spanish Philippine writing system. It is an abugida, 
            or alphasyllabary, which was used in the Philippines prior to Spanish colonization.
            Each character represents a consonant with an inherent a vowel sound, while other 
            vowel sounds are indicated by marks above or below the character.
          </p>
          <div className={`baybayin-character-guide ${theme}`} style={theme === 'dark' ? { backgroundColor: 'rgba(33, 39, 59, 0.7)', border: '1px solid rgba(75, 86, 115, 0.4)', borderRadius: '8px', padding: '16px' } : {}}>
            <h4 className={theme}>Common Characters</h4>
            <div className={`character-row ${theme}`}>
              {['ᜀ', 'ᜁ', 'ᜂ', 'ᜃ', 'ᜄ', 'ᜅ', 'ᜆ', 'ᜇ', 'ᜈ'].map(char => (
                <div className={`baybayin-character ${theme}`} key={char}>
                  <span className={`character ${theme}`} style={theme === 'dark' ? { backgroundColor: 'rgba(33, 39, 59, 0.7)', color: '#ffd166', border: '1px solid rgba(255, 209, 102, 0.2)' } : {}}>{char}</span>
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