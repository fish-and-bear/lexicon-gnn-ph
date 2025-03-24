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
      <div className={`definitions-container ${theme}`}>
        <div className={`section-header ${theme}`}>
          <h3 className={`section-title ${theme}`}>
            <span className={`section-icon ${theme}`}>
              <span className="material-icons">menu_book</span>
            </span>
            Definitions
            <span className={`definition-count ${theme}`}>{word.definitions.length}</span>
          </h3>
          {word.pronunciation && (
            <button 
              className={`audio-button ${theme}`}
              onClick={playAudio}
              aria-label="Play pronunciation"
            >
              <span className="material-icons">volume_up</span>
            </button>
          )}
        </div>
        
        <div className={`section-content ${theme}`}>
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
                      className={`definition-card ${isExpanded ? 'expanded' : ''} ${theme === 'dark' ? 'dark' : ''}`}
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
                            <div className="source-tags">
                              {definition.sources.map((source: string, idx: number) => (
                                <span key={idx} className="source-tag">{source}</span>
                              ))}
                            </div>
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
        
        <div className={`metadata-section ${theme}`}>
          <button 
            className={`metadata-toggle ${showMetadata ? 'expanded' : ''} ${theme}`}
            onClick={toggleMetadata}
          >
            <span className="material-icons">info</span>
            <span className={`toggle-text ${theme}`}>
              {showMetadata ? 'Hide Metadata' : 'Show Metadata'}
            </span>
            <span className="material-icons">
              {showMetadata ? 'expand_less' : 'expand_more'}
            </span>
          </button>
          
          {showMetadata && (
            <div className={`metadata-content ${theme}`}>
              <div className={`metadata-item ${theme}`}>
                <span className={`metadata-label ${theme}`}>ID:</span>
                <span className={`metadata-value ${theme}`}>{word.id}</span>
              </div>
              {word.updated_at && (
                <div className={`metadata-item ${theme}`}>
                  <span className={`metadata-label ${theme}`}>Updated:</span>
                  <span className={`metadata-value ${theme}`}>{formatDate(word.updated_at)}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderEtymology = (word: WordInfo) => {
    // Check if the word has etymologies
    if (!word.etymologies || word.etymologies.length === 0) {
      return (
        <div className={`empty-state ${theme}`}>
          <p className={theme}>No etymology information available for this word.</p>
        </div>
      );
    }
    
    // Use the first etymology in the array
    const etymology = word.etymologies[0];
    
    return (
      <div className={`etymology-container ${theme}`}>
        <div className={`section-header ${theme}`}>
          <h3 className={`section-title ${theme}`}>
            <span className={`section-icon ${theme}`}>
              <span className="material-icons">history_edu</span>
            </span>
            Etymology
          </h3>
        </div>
        
        <div className={`section-content ${theme}`}>
          <div className={`etymology-content ${theme}`}>
            <div className={`etymology-text ${theme}`}>
              <p className={`etymology-paragraph ${theme}`}>{etymology.text}</p>
            </div>
            
            {etymology.components && etymology.components.length > 0 && (
              <div className={`etymology-components ${theme}`}>
                <h4 className={`components-title ${theme}`}>Components:</h4>
                <div className={`component-tags ${theme}`}>
                  {etymology.components.map((component: string, index: number) => (
                    <span 
                      key={index}
                      className={`component-tag ${theme}`}
                      onClick={() => onWordClick(component)}
                    >
                      {component}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {etymology.languages && etymology.languages.length > 0 && (
              <div className={`etymology-languages ${theme}`}>
                <h4 className={`languages-title ${theme}`}>Languages:</h4>
                <div className={`language-tags ${theme}`}>
                  {etymology.languages.map((language: string, index: number) => (
                    <span key={index} className={`language-tag ${theme}`}>
                      {language}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {etymology.sources && etymology.sources.length > 0 && (
              <div className={`etymology-sources ${theme}`}>
                <h4 className={`sources-title ${theme}`}>Sources:</h4>
                <div className={`source-tags ${theme}`}>
                  {etymology.sources.map((source: string, index: number) => (
                    <span key={index} className={`source-tag ${theme}`}>
                      {source}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderRelatedWords = () => {
    console.log('Rendering related words from:', wordInfo);
    console.log('Network data available:', networkData);
    
    // Start with checking relation data from wordInfo
    let hasRelations = false;
    
    // Count total number of related words for the badge
    let totalRelationCount = 0;
    
    // Create a map of related lemmas if we have network data
    let relatedLemmasFromNetwork: Record<string, string[]> = {};
    
    if (networkData && (!wordInfo.relations || Object.keys(wordInfo.relations).length === 0)) {
      console.log('Using network data to find related words');
      // Extract related words from networkData if we don't have relations in wordInfo
      if (networkData.nodes && networkData.nodes.length > 0) {
        // Filter out the current word to find related words
        const relatedNodes = networkData.nodes.filter(node => node.word !== wordInfo.lemma);
        console.log('Found related nodes in network:', relatedNodes);
        
        // Map the related words
        const relatedLemmas = relatedNodes.map(node => node.word);
        console.log('Found related lemmas in network:', relatedLemmas);
        
        if (relatedLemmas.length > 0) {
          // Categorize them as "related" for now
          relatedLemmasFromNetwork['related'] = relatedLemmas;
          hasRelations = true;
          totalRelationCount += relatedLemmas.length;
        }
      }
    }
    
    // Continue with existing code to process wordInfo.relations
    const allRelationData: Record<string, string[]> = {...relatedLemmasFromNetwork};
    
    // Extract all possible relation types
    const possibleRelationTypes = wordInfo.relations ? Object.keys(wordInfo.relations) : [];
    console.log('Possible relation types:', possibleRelationTypes);
    
    // Process each relation type to gather related words
    possibleRelationTypes.forEach(type => {
      if (wordInfo.relations && wordInfo.relations[type] && Array.isArray(wordInfo.relations[type])) {
        allRelationData[type] = wordInfo.relations[type].map((item: RelatedWord) => item.word);
        console.log(`Processed relation type '${type}':`, allRelationData[type]);
        
        // Add to total count
        totalRelationCount += allRelationData[type].length;
        hasRelations = true;
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
          totalRelationCount += 1;
          hasRelations = true;
        }
      }
    });
    
    // Create set of synonym words to filter out from the related words category
    const synonymWords = new Set(allRelationData['synonyms'] || []);
    
    // Create the relation groups with their display information
    const relationGroups = [
      { 
        type: 'synonyms', 
        label: 'Synonyms', 
        icon: 'swap_horiz', 
        description: 'Words with similar meanings',
        words: allRelationData['synonyms'] || []
      },
      { 
        type: 'antonyms', 
        label: 'Antonyms', 
        icon: 'sync_problem', 
        description: 'Words with opposite meanings',
        words: allRelationData['antonyms'] || []
      },
      { 
        type: 'hypernyms', 
        label: 'Hypernyms', 
        icon: 'arrow_upward', 
        description: 'More general terms that include this word',
        words: allRelationData['hypernyms'] || []
      },
      { 
        type: 'hyponyms', 
        label: 'Hyponyms', 
        icon: 'arrow_downward', 
        description: 'More specific instances of this word',
        words: allRelationData['hyponyms'] || []
      },
      { 
        type: 'meronyms', 
        label: 'Meronyms', 
        icon: 'layers', 
        description: 'Parts of what this word represents',
        words: allRelationData['meronyms'] || []
      },
      { 
        type: 'holonyms', 
        label: 'Holonyms', 
        icon: 'category', 
        description: 'Wholes that this word is a part of',
        words: allRelationData['holonyms'] || []
      },
      {
        type: 'root',
        label: 'Root Word',
        icon: 'account_tree',
        description: 'The etymological source of this word',
        words: allRelationData['root'] || []
      },
      {
        type: 'related',
        label: 'Related Words',
        icon: 'link',
        description: 'Words semantically related to this word',
        words: allRelationData['related'] || []
      }
    ].filter(group => group.words.length > 0);

    console.log('Final relation groups:', relationGroups);
    console.log('Total relation count:', totalRelationCount);
    
    return (
      <div className={`relations-container ${theme}`}>
        <div className="section-header">
          <h3 className={`section-title ${theme}`}>
            <span className={`section-icon ${theme}`}>
              <span className="material-icons">link</span>
            </span>
            Related Words
            <span className={`relation-count ${theme}`}>{totalRelationCount}</span>
          </h3>
        </div>
        
        <div className={`section-content ${theme}`}>
          {hasRelations ? (
            <div className={`relation-groups ${theme}`}>
              {/* Display root word first if it exists */}
              {relationGroups.find(group => group.type === 'root') && (
                <div className={`relation-group ${theme}`}>
                  <div 
                    className={`relation-group-header ${expandedGroups['root'] ? 'expanded' : ''} ${theme}`}
                    onClick={() => toggleGroup('root')}
                  >
                    <span className={`relation-group-icon ${theme}`}>
                      <span className="material-icons">account_tree</span>
                    </span>
                    <span className={`relation-group-label ${theme}`}>Root Word</span>
                    <span className={`relation-group-count ${theme}`}>{allRelationData['root']?.length || 0}</span>
                    <span className={`expand-icon ${theme}`}>
                      <span className="material-icons">
                        {expandedGroups['root'] ? 'expand_less' : 'expand_more'}
                      </span>
                    </span>
                  </div>
                  {expandedGroups['root'] && (
                    <div className={`relation-group-content ${theme}`}>
                      <p className={`relation-description ${theme}`}>The etymological source of this word</p>
                      <div className={`related-words ${theme}`}>
                        {allRelationData['root']?.map((word, index) => (
                          <span 
                            key={`root-${index}`}
                            className={`related-word ${theme}`}
                            onClick={() => onWordClick(word)}
                          >
                            {word}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Display other relation groups */}
              {relationGroups
                .filter(group => group.type !== 'root')
                .map(group => (
                <div key={group.type} className={`relation-group ${theme}`}>
                  <div 
                    className={`relation-group-header ${expandedGroups[group.type] ? 'expanded' : ''} ${theme}`}
                    onClick={() => toggleGroup(group.type)}
                  >
                    <span className={`relation-group-icon ${theme}`}>
                      <span className="material-icons">{group.icon}</span>
                    </span>
                    <span className={`relation-group-label ${theme}`}>{group.label}</span>
                    <span className={`relation-group-count ${theme}`}>{group.words.length}</span>
                    <span className={`expand-icon ${theme}`}>
                      <span className="material-icons">
                        {expandedGroups[group.type] ? 'expand_less' : 'expand_more'}
                      </span>
                    </span>
                  </div>
                  {expandedGroups[group.type] && (
                    <div className={`relation-group-content ${theme}`}>
                      <p className={`relation-description ${theme}`}>{group.description}</p>
                      <div className={`related-words ${theme}`}>
                        {group.words.map((word, index) => (
                          <span 
                            key={`${group.type}-${index}`}
                            className={`related-word ${theme}`}
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
          ) : (
            <div className={`empty-state ${theme}`}>
              <p className={theme}>No related words available</p>
              {!networkData && (
                <button 
                  className={`fetch-network-btn ${theme}`}
                  onClick={() => fetchWordNetworkData(wordInfo.lemma)}
                >
                  Fetch word network data
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderBaybayin = () => {
    // Check if word has baybayin data
    if (!wordInfo || !wordInfo.has_baybayin || !wordInfo.baybayin_form) {
      return (
        <div className={`empty-state ${theme}`}>
          <p className={theme}>No Baybayin transliteration available for this word.</p>
        </div>
      );
    }

    return (
      <div className={`baybayin-container ${theme}`}>
        <div className={`section-header ${theme}`}>
          <h3 className={`section-title ${theme}`}>
            <span className={`section-icon ${theme}`}>
              <span className="material-icons">translate</span>
            </span>
            Baybayin Translation
          </h3>
        </div>
        
        <div className={`section-content ${theme}`}>
          <div className={`baybayin-display ${theme}`}>
            <div className={`baybayin-text ${theme}`}>
              {wordInfo.baybayin_form}
            </div>
            <div className={`latin-text ${theme}`}>
              {wordInfo.lemma}
            </div>
          </div>
          
          <div className={`baybayin-info ${theme}`}>
            <p className={theme}>Baybayin is an ancient Philippine syllabary script.</p>
          </div>
        </div>
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
                <div 
                  key={index} 
                  className={`idiom-card ${theme === 'dark' ? 'dark' : ''}`}
                >
                  <p className="idiom-phrase">{idiom}</p>
                </div>
              );
            }
            
            return (
              <div 
                key={index} 
                className={`idiom-card ${theme === 'dark' ? 'dark' : ''}`}
              >
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
    <div className={`word-details-container ${theme}`}>
      <div className={`word-details ${theme}`}>
        <div className={`word-header ${theme}`}>
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
        
        <div className={`word-content-tabs ${theme}`}>
          <div 
            className={`tab-item ${activeTab === 'definitions' ? 'active' : ''} ${theme}`}
            onClick={() => setActiveTab('definitions')}
          >
            <span className="tab-icon">📖</span> Definitions
            {wordInfo.definitions?.length > 0 && (
              <span className="count">{wordInfo.definitions.length}</span>
            )}
          </div>
          {wordInfo.etymologies && wordInfo.etymologies.length > 0 && (
            <div 
              className={`tab-item ${activeTab === 'etymology' ? 'active' : ''} ${theme}`}
              onClick={() => setActiveTab('etymology')}
            >
              <span className="tab-icon">🔍</span> Etymology
              <span className="count">{wordInfo.etymologies.length}</span>
            </div>
          )}
          {wordInfo.relations && Object.keys(wordInfo.relations).length > 0 && (
            <div 
              className={`tab-item ${activeTab === 'relations' ? 'active' : ''} ${theme}`}
              onClick={() => setActiveTab('relations')}
            >
              <span className="tab-icon">🔄</span> Relations
              <span className="count">{Object.keys(wordInfo.relations).length}</span>
            </div>
          )}
          {wordInfo.idioms && wordInfo.idioms.length > 0 && (
            <div 
              className={`tab-item ${activeTab === 'idioms' ? 'active' : ''} ${theme}`}
              onClick={() => setActiveTab('idioms')}
            >
              <span className="tab-icon">💬</span> Idioms
              <span className="count">{wordInfo.idioms.length}</span>
            </div>
          )}
          {wordInfo.baybayin_form && (
            <div 
              className={`tab-item ${activeTab === 'baybayin' ? 'active' : ''} ${theme}`}
              onClick={() => setActiveTab('baybayin')}
            >
              <span className="tab-icon">𑁋</span> Baybayin
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