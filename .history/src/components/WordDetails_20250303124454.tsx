import React, { useCallback, useState, useEffect, useRef, useContext } from 'react';
import { WordInfo, WordNetwork, RelatedWord, NetworkNode } from '../types';
import { fetchWordNetwork } from '../api/wordApi';
import './WordDetails.css';
import './Tabs.css';
import { ThemeContext } from '../contexts/ThemeContext';

interface WordDetailsProps {
  wordInfo: WordInfo;
  onWordClick: (word: string) => void;
}

const WordDetails = React.memo(({ wordInfo, onWordClick }: WordDetailsProps) => {
  const { theme } = useContext(ThemeContext);
  const [expandedSections, setExpandedSections] = useState<string[]>([]);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [expandedDefinitions, setExpandedDefinitions] = useState<Record<string, boolean>>({});
  const [showFullEtymology, setShowFullEtymology] = useState(false);
  const [showMetadata, setShowMetadata] = useState(false);
  const [networkData, setNetworkData] = useState<WordNetwork | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // Active tab state
  const [activeTab, setActiveTab] = useState('definitions');

  // When the word changes, reset expanded states and tab
  useEffect(() => {
    setExpandedSections([]);
    setExpandedGroups({});
    setExpandedDefinitions({});
    setShowFullEtymology(false);
    setShowMetadata(false);
    setNetworkData(null);
    setActiveTab('definitions');
  }, [wordInfo.id]);

  // Function to fetch word network data
  const fetchWordNetworkData = async (word: string, depth = 2) => {
    setIsLoading(true);
    try {
      const data = await fetchWordNetwork(word, {
        depth: depth,
        include_affixes: true,
        include_etymology: true,
        cluster_threshold: 0.6
      });
      console.log('Fetched network data:', data);
      setNetworkData(data);
    } catch (error) {
      console.error('Error fetching word network:', error);
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
    setShowMetadata(!showMetadata);
  };

  const playAudio = () => {
    // Audio playback logic
    console.log('Playing audio...');
  };

  const formatDate = (dateString?: string | null) => {
    if (!dateString) return 'Unknown';
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
    } catch (e) {
      return dateString;
    }
  };

  // Render definitions section
  const renderDefinitions = (word: WordInfo) => {
    if (!word.definitions || word.definitions.length === 0) {
      return (
        <div className={`empty-state ${theme}`}>
          <p className={theme}>No definitions available for this word.</p>
        </div>
      );
    }

    // Group definitions by part of speech
    const definitionsByPos = word.definitions.reduce((groups, def) => {
      const pos = def.part_of_speech?.name_en || 'Unknown';
      if (!groups[pos]) {
        groups[pos] = [];
      }
      groups[pos].push(def);
      return groups;
    }, {} as Record<string, any[]>);

    return (
      <div className={`definitions-section ${theme}`}>
        <div className={`section-title ${theme}`}>
          <h3 className={theme}>Definitions</h3>
          <span className={`definition-count ${theme}`}>{word.definitions.length}</span>
          {word.audio_url && (
            <button 
              className={`audio-button ${theme}`}
              onClick={playAudio}
            >
              <span className="material-icons">volume_up</span>
            </button>
          )}
        </div>
        
        <div className={`definition-groups ${theme}`}>
          {Object.entries(definitionsByPos).map(([posName, definitions]) => (
            <div key={posName} className={`pos-group ${theme}`}>
              <h4 className={`pos-title ${theme}`}>{posName}</h4>
              {definitions.map((def, idx) => (
                <div 
                  key={def.id || idx} 
                  className={`definition-item ${theme}`}
                >
                  <div 
                    className={`definition-header ${expandedDefinitions[def.id || `def-${idx}`] ? 'expanded' : ''} ${theme}`}
                    onClick={() => toggleDefinitionExpand(def.id || `def-${idx}`)}
                  >
                    <div className={`definition-number ${theme}`}>{idx + 1}.</div>
                    <div className={`definition-text ${theme}`}>{def.definition_text || def.text}</div>
                    <div className={`expand-icon ${theme}`}>
                      <span className="material-icons">
                        {expandedDefinitions[def.id || `def-${idx}`] ? 'expand_less' : 'expand_more'}
                      </span>
                    </div>
                  </div>
                  
                  {expandedDefinitions[def.id || `def-${idx}`] && (
                    <div className={`definition-details ${theme}`}>
                      {def.examples && def.examples.length > 0 && (
                        <div className={`examples ${theme}`}>
                          <h5 className={theme}>Examples:</h5>
                          <ul className={theme}>
                            {def.examples.map((ex, i) => (
                              <li key={i} className={theme}>{ex}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Render etymology section
  const renderEtymology = (word: WordInfo) => {
    if (!word.etymologies || word.etymologies.length === 0) {
      return (
        <div className={`empty-state ${theme}`}>
          <p className={theme}>No etymology information available for this word.</p>
        </div>
      );
    }
    
    return (
      <div className={`etymology-section ${theme}`}>
        <div className={`section-title ${theme}`}>
          <h3 className={theme}>Etymology</h3>
          <span className={`count ${theme}`}>{word.etymologies.length}</span>
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
                    <span 
                      key={idx} 
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

  // Render related words section
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
      },
      { 
        type: 'variants', 
        label: 'Variants', 
        icon: 'autorenew', 
        description: 'Alternative forms of this word',
        words: allRelationData['variants'] || []
      },
      { 
        type: 'kaugnay', 
        label: 'Kaugnay', 
        icon: 'assistant', 
        description: 'Filipino-specific related words',
        words: allRelationData['kaugnay'] || []
      },
      { 
        type: 'derived', 
        label: 'Derivatives', 
        icon: 'bolt', 
        description: 'Words derived from this word',
        words: allRelationData['derived'] || []
      }
    ].filter(group => group.words.length > 0);

    console.log('Final relation groups:', relationGroups);
    console.log('Total relation count:', totalRelationCount);
    
    return (
      <div className={`relations-section ${theme}`}>
        <div className={`section-title ${theme}`}>
          <h3 className={theme}>Related Words</h3>
          <span className={`relation-count ${theme}`}>{totalRelationCount}</span>
        </div>
        
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
    );
  };

  // Render baybayin section
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
      <div className={`baybayin-section ${theme}`}>
        <div className={`section-title ${theme}`}>
          <h3 className={theme}>Baybayin Script</h3>
        </div>
        
        <div className={`baybayin-display ${theme}`}>
          <div className={`baybayin-text ${theme}`}>
            {wordInfo.baybayin_form}
          </div>
          <div className={`romanized-text ${theme}`}>
            {wordInfo.romanized_form || wordInfo.lemma}
          </div>
        </div>
        
        <div className={`baybayin-info ${theme}`}>
          <p className={theme}>Baybayin is an ancient Philippine syllabary script.</p>
        </div>
      </div>
    );
  };

  // Render idioms section
  const renderIdioms = () => {
    const idioms = wordInfo.idioms || [];
    
    if (!idioms.length) {
      return (
        <div className={`empty-state ${theme}`}>
          <p className={theme}>No idioms available for this word.</p>
        </div>
      );
    }
    
    return (
      <div className={`idioms-section ${theme}`}>
        <div className={`section-title ${theme}`}>
          <h3 className={theme}>Idioms & Phrases</h3>
          <span className={`idiom-count ${theme}`}>{idioms.length}</span>
        </div>
        
        <div className={`idioms-list ${theme}`}>
          {idioms.map((idiom, index) => (
            <div key={index} className={`idiom-item ${theme}`}>
              <div className={`idiom-phrase ${theme}`}>{idiom.phrase || idiom.text}</div>
              <div className={`idiom-meaning ${theme}`}>{idiom.meaning}</div>
              {idiom.example && (
                <div className={`idiom-example ${theme}`}>
                  <em className={theme}>Example:</em> {idiom.example}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className={`word-details-container ${theme}`}>
      <div className={`word-header ${theme}`}>
        <h2 className={`word-lemma ${theme}`}>{wordInfo.lemma}</h2>
        {wordInfo.pronunciation && (
          <span className={`pronunciation ${theme}`}>/{wordInfo.pronunciation}/</span>
        )}
      </div>
      
      <div className={`word-tabs ${theme}`}>
        <div 
          className={`tab-item ${activeTab === 'definitions' ? 'active' : ''} ${theme}`}
          onClick={() => setActiveTab('definitions')}
        >
          <span className={`tab-icon ${theme}`}>📖</span> Definitions
          <span className={`tab-count ${theme}`}>{wordInfo.definitions?.length || 0}</span>
        </div>
        
        {wordInfo.etymologies && wordInfo.etymologies.length > 0 && (
          <div 
            className={`tab-item ${activeTab === 'etymology' ? 'active' : ''} ${theme}`}
            onClick={() => setActiveTab('etymology')}
          >
            <span className={`tab-icon ${theme}`}>🔍</span> Etymology
          </div>
        )}
        
        <div 
          className={`tab-item ${activeTab === 'relations' ? 'active' : ''} ${theme}`}
          onClick={() => setActiveTab('relations')}
        >
          <span className={`tab-icon ${theme}`}>🔄</span> Relations
          {wordInfo.relations && Object.keys(wordInfo.relations).length > 0 && (
            <span className={`tab-count ${theme}`}>
              {Object.values(wordInfo.relations).flat().length || 0}
            </span>
          )}
        </div>
        
        {wordInfo.idioms && wordInfo.idioms.length > 0 && (
          <div 
            className={`tab-item ${activeTab === 'idioms' ? 'active' : ''} ${theme}`}
            onClick={() => setActiveTab('idioms')}
          >
            <span className={`tab-icon ${theme}`}>💬</span> Idioms
            <span className={`tab-count ${theme}`}>{wordInfo.idioms.length}</span>
          </div>
        )}
        
        {wordInfo.has_baybayin && wordInfo.baybayin_form && (
          <div 
            className={`tab-item ${activeTab === 'baybayin' ? 'active' : ''} ${theme}`}
            onClick={() => setActiveTab('baybayin')}
          >
            <span className={`tab-icon ${theme}`}>𑁋</span> Baybayin
          </div>
        )}
      </div>
      
      <div className={`tab-content ${theme}`}>
        {activeTab === 'definitions' && renderDefinitions(wordInfo)}
        {activeTab === 'etymology' && renderEtymology(wordInfo)}
        {activeTab === 'relations' && renderRelatedWords()}
        {activeTab === 'baybayin' && renderBaybayin()}
        {activeTab === 'idioms' && renderIdioms()}
      </div>
    </div>
  );
});

export default WordDetails;