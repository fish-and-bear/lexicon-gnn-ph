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

    // Categorize components by type
    const protoComponents = allComponents.filter(comp => comp.toLowerCase().includes('proto'));
    const inheritedComponents = allComponents.filter(comp => 
      comp.toLowerCase().includes('inherited') && !comp.toLowerCase().includes('proto'));
    const variantComponents = allComponents.filter(comp => 
      comp.toLowerCase().includes('earlier') || 
      comp.toLowerCase().includes('variant') || 
      comp.toLowerCase().includes('form'));
    const otherComponents = allComponents.filter(comp => 
      !comp.toLowerCase().includes('proto') && 
      !comp.toLowerCase().includes('inherited') && 
      !comp.toLowerCase().includes('earlier') && 
      !comp.toLowerCase().includes('variant') && 
      !comp.toLowerCase().includes('form'));

    type CategoryTypes = 'inheritance' | 'influence' | 'comparison' | 'other';
    
    const [expandedCategories, setExpandedCategories] = useState<Record<CategoryTypes, boolean>>({
      inheritance: true,
      influence: true,
      comparison: true,
      other: true
    });

    const toggleCategory = (category: CategoryTypes) => {
      setExpandedCategories(prev => ({
        ...prev,
        [category]: !prev[category]
      }));
    };

    // Render a single etymology item
    const renderEtymologyItem = (etymology: any, index: number) => (
      <div key={index} className="etymology-item">
        <p className="etymology-text">{etymology.text}</p>
        
        {etymology.languages && etymology.languages.length > 0 && (
          <div className="etymology-languages">
            {etymology.languages.map((lang: string, idx: number) => (
              <span key={idx} className="language-tag">{lang}</span>
            ))}
          </div>
        )}
        
        {etymology.components && etymology.components.length > 0 && (
          <div className="etymology-components-list">
            <span className="components-label">Components:</span>
            <div className="component-tags">
              {etymology.components.map((component: string, idx: number) => (
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
              {etymology.sources.map((source: string, idx: number) => (
                <span key={idx} className="source-tag">{source}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    );

    // Render a category of etymologies with collapsible header
    const renderEtymologyCategory = (
      category: CategoryTypes, 
      title: string, 
      items: any[], 
      icon: string
    ) => {
      if (items.length === 0) return null;
      
      return (
        <div className={`etymology-category ${category}`}>
          <div 
            className={`category-header ${expandedCategories[category] ? 'expanded' : ''}`}
            onClick={() => toggleCategory(category)}
          >
            <span className="category-icon">{icon}</span>
            <h4>{title}</h4>
            <span className="category-count">{items.length}</span>
            <span className="expand-icon">{expandedCategories[category] ? '▼' : '▶'}</span>
          </div>
          
          {expandedCategories[category] && (
            <div className="category-content">
              {items.map((etymology, index) => renderEtymologyItem(etymology, index))}
            </div>
          )}
        </div>
      );
    };

    // New organized component display
    const renderOrganizedComponents = () => {
      // Determine if we have enough data for a timeline
      const hasTimelineData = protoComponents.length > 0 && 
        (inheritedComponents.length > 0 || variantComponents.length > 0);

      return (
        <div className="component-organization">
          <h4>Etymology Component Organization</h4>
          
          {/* Language Evolution Tree */}
          {hasTimelineData && (
            <div className="language-evolution-tree">
              {protoComponents.length > 0 && (
                <div className="evolution-level proto">
                  {protoComponents.slice(0, 3).map((component, idx) => (
                    <div className="evolution-branch" key={idx}>
                      <div className="evolution-node">
                        <div 
                          className="evolution-node-content"
                          onClick={() => onWordClick(component)}
                        >
                          <span className="evolution-node-title">{component}</span>
                          <span className="evolution-node-subtitle">Proto-Language Form</span>
                        </div>
                      </div>
                      <div className="evolution-connector connector-vertical"></div>
                    </div>
                  ))}
                </div>
              )}
              
              {inheritedComponents.length > 0 && (
                <div className="evolution-level inherited">
                  {inheritedComponents.slice(0, 3).map((component, idx) => (
                    <div className="evolution-branch" key={idx}>
                      <div className="evolution-node">
                        <div 
                          className="evolution-node-content"
                          onClick={() => onWordClick(component)}
                        >
                          <span className="evolution-node-title">{component}</span>
                          <span className="evolution-node-subtitle">Inheritance Path</span>
                        </div>
                      </div>
                      <div className="evolution-connector connector-vertical"></div>
                    </div>
                  ))}
                </div>
              )}
              
              {variantComponents.length > 0 && (
                <div className="evolution-level variant">
                  {variantComponents.slice(0, 3).map((component, idx) => (
                    <div className="evolution-branch" key={idx}>
                      <div className="evolution-node">
                        <div 
                          className="evolution-node-content"
                          onClick={() => onWordClick(component)}
                        >
                          <span className="evolution-node-title">{component}</span>
                          <span className="evolution-node-subtitle">Earlier Form</span>
                        </div>
                      </div>
                      <div className="evolution-connector connector-vertical"></div>
                    </div>
                  ))}
                </div>
              )}
              
              <div className="evolution-level current">
                <div className="evolution-branch">
                  <div className="evolution-node">
                    <div className="evolution-node-content">
                      <span className="evolution-node-title">{word.lemma}</span>
                      <span className="evolution-node-subtitle">Current Word</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Component Cards */}
          <div className="component-cards">
            {protoComponents.length > 0 && (
              <div className="component-card proto">
                <div className="component-card-header">
                  <span className="component-card-icon">🌳</span>
                  <h5 className="component-card-title">Proto-Language Forms</h5>
                  <span className="component-card-count">{protoComponents.length}</span>
                </div>
                <div className="component-card-content">
                  {protoComponents.map((component, idx) => (
                    <span 
                      key={idx} 
                      className="component-tag-enhanced proto"
                      onClick={() => onWordClick(component)}
                    >
                      {component}
                      {componentFrequency[component] > 1 && (
                        <span className="tag-frequency">{componentFrequency[component]}</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {inheritedComponents.length > 0 && (
              <div className="component-card inherited">
                <div className="component-card-header">
                  <span className="component-card-icon">⬇️</span>
                  <h5 className="component-card-title">Inheritance Paths</h5>
                  <span className="component-card-count">{inheritedComponents.length}</span>
                </div>
                <div className="component-card-content">
                  {inheritedComponents.map((component, idx) => (
                    <span 
                      key={idx} 
                      className="component-tag-enhanced inherited"
                      onClick={() => onWordClick(component)}
                    >
                      {component}
                      {componentFrequency[component] > 1 && (
                        <span className="tag-frequency">{componentFrequency[component]}</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {variantComponents.length > 0 && (
              <div className="component-card variant">
                <div className="component-card-header">
                  <span className="component-card-icon">🔄</span>
                  <h5 className="component-card-title">Variants & Earlier Forms</h5>
                  <span className="component-card-count">{variantComponents.length}</span>
                </div>
                <div className="component-card-content">
                  {variantComponents.map((component, idx) => (
                    <span 
                      key={idx} 
                      className="component-tag-enhanced variant"
                      onClick={() => onWordClick(component)}
                    >
                      {component}
                      {componentFrequency[component] > 1 && (
                        <span className="tag-frequency">{componentFrequency[component]}</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {otherComponents.length > 0 && (
              <div className="component-card">
                <div className="component-card-header">
                  <span className="component-card-icon">🧩</span>
                  <h5 className="component-card-title">Other Components</h5>
                  <span className="component-card-count">{otherComponents.length}</span>
                </div>
                <div className="component-card-content">
                  {otherComponents.map((component, idx) => (
                    <span 
                      key={idx} 
                      className="component-tag-enhanced"
                      onClick={() => onWordClick(component)}
                    >
                      {component}
                      {componentFrequency[component] > 1 && (
                        <span className="tag-frequency">{componentFrequency[component]}</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    };

    return (
      <div className="etymology-section">
        {allComponents.length > 0 && renderOrganizedComponents()}
        
        <div className="component-cloud">
          <h4>Component Words</h4>
          <div className="component-cloud-tags">
            {allComponents.map((component, idx) => (
              <span 
                key={idx} 
                className={`component-cloud-tag size-${Math.min(3, componentFrequency[component])}`}
                onClick={() => onWordClick(component)}
              >
                {component}
                <span className="component-frequency">{componentFrequency[component]}</span>
              </span>
            ))}
          </div>
        </div>

        <div className="etymology-categories">
          {renderEtymologyCategory(
            'inheritance', 
            'Inheritance & Proto-Forms', 
            categorizedEtymologies.inheritance,
            '🌳'
          )}
          
          {renderEtymologyCategory(
            'influence', 
            'Influences & Borrowings', 
            categorizedEtymologies.influence,
            '🔄'
          )}
          
          {renderEtymologyCategory(
            'comparison', 
            'Comparisons & Cognates', 
            categorizedEtymologies.comparison,
            '🔍'
          )}
          
          {renderEtymologyCategory(
            'other', 
            'Other Etymology Information', 
            categorizedEtymologies.other,
            'ℹ️'
          )}
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