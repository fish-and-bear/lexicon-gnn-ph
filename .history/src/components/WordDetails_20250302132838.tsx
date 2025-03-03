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

    // Group etymologies by type/category
    const categorizedEtymologies = {
      inheritance: word.etymologies.filter(etym => 
        etym.text?.toLowerCase().includes('inherited') || 
        etym.text?.toLowerCase().includes('proto')),
      influence: word.etymologies.filter(etym => 
        etym.text?.toLowerCase().includes('influenced by') || 
        etym.text?.toLowerCase().includes('borrowed')),
      comparison: word.etymologies.filter(etym => 
        etym.text?.toLowerCase().includes('compare') || 
        etym.text?.toLowerCase().includes('cognate')),
      other: word.etymologies.filter(etym => {
        const lowerText = etym.text?.toLowerCase() || '';
        return !lowerText.includes('inherited') && 
               !lowerText.includes('proto') && 
               !lowerText.includes('influenced by') && 
               !lowerText.includes('borrowed') &&
               !lowerText.includes('compare') &&
               !lowerText.includes('cognate');
      })
    };

    // Extract all unique components for the component cloud
    const allComponents: string[] = [];
    const componentFrequency: Record<string, number> = {};
    
    word.etymologies.forEach(etymology => {
      if (etymology.components) {
        etymology.components.forEach(component => {
          if (!allComponents.includes(component)) {
            allComponents.push(component);
            componentFrequency[component] = 1;
          } else {
            componentFrequency[component]++;
          }
        });
      }
    });

    // Sort components by frequency
    allComponents.sort((a, b) => componentFrequency[b] - componentFrequency[a]);

    // Helper functions for component organization
    const parseComponent = (component: string): { language: string | null, word: string } => {
      // Check if component starts with "English"
      if (component.startsWith("English")) {
        return {
          language: "English",
          word: component.replace("English", "").trim()
        };
      }
      
      // Check if component follows [Language] [component] pattern
      const match = component.match(/^\[([^\]]+)\]\s+(.+)$/);
      if (match) {
        return {
          language: match[1],
          word: match[2]
        };
      }
      
      // Check if component has "inherited from" text
      if (component.includes("inherited from")) {
        const parts = component.split("inherited from");
        return {
          language: parts[1].trim(),
          word: parts[0].trim()
        };
      }
      
      // Default case - no language detected
      return {
        language: null,
        word: component
      };
    };

    const getLanguageClass = (language: string): string => {
      const languageMap: Record<string, string> = {
        'Latin': 'language-latin',
        'Greek': 'language-greek',
        'Germanic': 'language-germanic',
        'English': 'language-germanic', // English is classified as Germanic
        'Romance': 'language-romance',
        'Slavic': 'language-slavic',
        'Sanskrit': 'language-sanskrit',
        'Proto': 'language-proto'
      };
      
      // Check for Proto- prefix
      if (language.startsWith('Proto-')) {
        return 'language-proto';
      }
      
      // Return mapped class or default
      return languageMap[language] || 'language-other';
    };

    const getLanguageIcon = (language: string): string => {
      const languageIcons: Record<string, string> = {
        'Latin': '🏛️',
        'Greek': '🏺',
        'Germanic': '⚔️',
        'English': '🏴',
        'Romance': '🌹',
        'Slavic': '🌲',
        'Sanskrit': '☸️',
        'Proto': '🌱'
      };
      
      // Check for Proto- prefix
      if (language.startsWith('Proto-')) {
        return '🌱';
      }
      
      // Return mapped icon or default
      return languageIcons[language] || '🔤';
    };

    // Consolidated component rendering function
    const renderComponentTag = (component: string, frequency: number = 1, className: string = '') => {
      const { language, word } = parseComponent(component);
      
      return (
        <span 
          key={component} 
          className={`component-tag-enhanced ${className}`}
          onClick={() => onWordClick(component)}
        >
          {language && <strong>{language}</strong>}
          {word}
          {frequency > 1 && (
            <span className="tag-frequency">{frequency}</span>
          )}
        </span>
      );
    };

    // Consolidated function to render organized components
    const renderOrganizedComponents = (components: string[]) => {
      if (!components || components.length === 0) {
        return null;
      }
      
      // Count component frequencies
      const componentFrequencies: Record<string, number> = {};
      components.forEach(component => {
        componentFrequencies[component] = (componentFrequencies[component] || 0) + 1;
      });
      
      // Get unique components sorted by frequency
      const uniqueComponents = [...new Set(components)]
        .sort((a, b) => componentFrequencies[b] - componentFrequencies[a]);
      
      // Group components by language
      const componentsByLanguage: Record<string, string[]> = {};
      
      uniqueComponents.forEach(component => {
        const { language, word } = parseComponent(component);
        if (language) {
          if (!componentsByLanguage[language]) {
            componentsByLanguage[language] = [];
          }
          componentsByLanguage[language].push(component);
        }
      });
      
      // Categorize components
      const protoComponents = uniqueComponents.filter(c => 
        c.toLowerCase().includes('proto') || parseComponent(c).language?.includes('Proto')
      );
      
      const inheritedComponents = uniqueComponents.filter(c => 
        c.toLowerCase().includes('inherited') || 
        (parseComponent(c).language && !c.toLowerCase().includes('proto'))
      );
      
      const variantComponents = uniqueComponents.filter(c => 
        c.toLowerCase().includes('variant') || c.toLowerCase().includes('alternate')
      );
      
      const otherComponents = uniqueComponents.filter(c => 
        !protoComponents.includes(c) && 
        !inheritedComponents.includes(c) && 
        !variantComponents.includes(c)
      );
      
      return (
        <div className="component-organization">
          <h4>Component Organization</h4>
          
          {/* Language-based component cards */}
          <div className="component-cards">
            {Object.entries(componentsByLanguage).map(([language, langComponents]) => (
              <div key={language} className={`component-card ${getLanguageClass(language)}`}>
                <div className="component-card-header">
                  <span className="component-card-icon">{getLanguageIcon(language)}</span>
                  <span className="component-card-title">{language}</span>
                  <span className="component-card-count">{langComponents.length}</span>
                </div>
                <div className="component-card-content">
                  {langComponents.map(component => 
                    renderComponentTag(component, componentFrequencies[component])
                  )}
                </div>
              </div>
            ))}
          </div>
          
          {/* Traditional component cards */}
          <div className="component-cards">
            {protoComponents.length > 0 && (
              <div className="component-card proto">
                <div className="component-card-header">
                  <span className="component-card-icon">🌱</span>
                  <span className="component-card-title">Proto Components</span>
                  <span className="component-card-count">{protoComponents.length}</span>
                </div>
                <div className="component-card-content">
                  {protoComponents.map(component => 
                    renderComponentTag(component, componentFrequencies[component], 'proto')
                  )}
                </div>
              </div>
            )}
            
            {inheritedComponents.length > 0 && (
              <div className="component-card inherited">
                <div className="component-card-header">
                  <span className="component-card-icon">🔄</span>
                  <span className="component-card-title">Inherited Components</span>
                  <span className="component-card-count">{inheritedComponents.length}</span>
                </div>
                <div className="component-card-content">
                  {inheritedComponents.map(component => 
                    renderComponentTag(component, componentFrequencies[component], 'inherited')
                  )}
                </div>
              </div>
            )}
            
            {variantComponents.length > 0 && (
              <div className="component-card variant">
                <div className="component-card-header">
                  <span className="component-card-icon">🔀</span>
                  <span className="component-card-title">Variant Components</span>
                  <span className="component-card-count">{variantComponents.length}</span>
                </div>
                <div className="component-card-content">
                  {variantComponents.map(component => 
                    renderComponentTag(component, componentFrequencies[component], 'variant')
                  )}
                </div>
              </div>
            )}
            
            {otherComponents.length > 0 && (
              <div className="component-card">
                <div className="component-card-header">
                  <span className="component-card-icon">🧩</span>
                  <span className="component-card-title">Other Components</span>
                  <span className="component-card-count">{otherComponents.length}</span>
                </div>
                <div className="component-card-content">
                  {otherComponents.map(component => 
                    renderComponentTag(component, componentFrequencies[component])
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    };

    return (
      <div className="etymology-section">
        {allComponents.length > 0 && renderOrganizedComponents(allComponents)}
        
        <div className="component-cloud">
          <h4>Component Words</h4>
          <div className="component-cloud-tags">
            {allComponents.map((component, idx) => {
              const parsed = parseComponent(component);
              return (
                <span 
                  key={idx} 
                  className={`component-cloud-tag size-${Math.min(3, componentFrequency[component])}`}
                  onClick={() => onWordClick(component)}
                >
                  {parsed.language ? (
                    <>
                      <strong>{parsed.language}</strong> {parsed.word}
                    </>
                  ) : (
                    parsed.word
                  )}
                  <span className="component-frequency">{componentFrequency[component]}</span>
                </span>
              );
            })}
          </div>
        </div>

        <div className="etymology-categories">
          {renderOrganizedComponents(categorizedEtymologies.inheritance)}
          {renderOrganizedComponents(categorizedEtymologies.influence)}
          {renderOrganizedComponents(categorizedEtymologies.comparison)}
          {renderOrganizedComponents(categorizedEtymologies.other)}
        </div>
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

    type RelationType = 'synonyms' | 'antonyms' | 'derived' | 'variants' | 'related';
    
    // Define relation groups with their properties
    const relationGroups = [
      { 
        type: 'synonyms' as RelationType, 
        label: 'Synonyms', 
        words: synonyms, 
        icon: '🔄',
        description: 'Words with the same or similar meaning'
      },
      { 
        type: 'antonyms' as RelationType, 
        label: 'Antonyms', 
        words: antonyms, 
        icon: '⚔️',
        description: 'Words with opposite meanings'
      },
      { 
        type: 'derived' as RelationType, 
        label: 'Derivatives', 
        words: derivatives, 
        icon: '🌱',
        description: 'Words derived from this word'
      },
      { 
        type: 'variants' as RelationType, 
        label: 'Variants', 
        words: variants, 
        icon: '🔀',
        description: 'Alternative forms of this word'
      },
      { 
        type: 'related' as RelationType, 
        label: 'Related Words', 
        words: related, 
        icon: '🔗',
        description: 'Words semantically related to this word'
      },
    ].filter(group => group.words.length > 0);

    const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>(
      relationGroups.reduce((acc, group) => ({
        ...acc,
        [group.type]: true
      }), {} as Record<string, boolean>)
    );

    const toggleGroup = (type: string) => {
      setExpandedGroups(prev => ({
        ...prev,
        [type]: !prev[type]
      }));
    };

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
      <div className="relations-container">
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