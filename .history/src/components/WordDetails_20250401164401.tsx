import React, { useCallback, useState, useEffect, useRef } from 'react';
import { WordInfo, WordNetwork, RelatedWord } from '../types';
import { fetchWordNetwork } from '../api/wordApi';
import './WordDetails.css';
import './Tabs.css';

interface WordDetailsProps {
  wordInfo: WordInfo;
  etymologyTree: any;
  isLoadingEtymology: boolean;
  etymologyError: string | null;
  showMetadata: boolean;
  toggleMetadata: () => void;
  onWordLinkClick: (word: string) => void;
}

const WordDetails: React.FC<WordDetailsProps> = React.memo(({ 
  wordInfo, 
  etymologyTree,
  isLoadingEtymology, 
  etymologyError,
  showMetadata,
  toggleMetadata,
  onWordLinkClick
}) => {
  // Simple debug log for development
  console.log("WordDetails component rendering with wordInfo:", wordInfo);
  
  const [activeTab, setActiveTab] = useState<'definitions' | 'relations' | 'etymology' | 'baybayin' | 'idioms' | 'affixations' | 'credits'>('definitions');
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const [expandedDefinitions, setExpandedDefinitions] = useState<Record<string, boolean>>({});
  const [networkData, setNetworkData] = useState<WordNetwork | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});

  useEffect(() => {
    // Reset states when word changes
    setActiveTab('definitions');
    setExpandedDefinitions({});
    setNetworkData(null);
    setIsAudioPlaying(false);
    
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
  }, [wordInfo]);

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

  const renderMetadata = (word: WordInfo) => {
    if (!showMetadata) return null;

    return (
      <div className="metadata-section active">
        <h4>Metadata</h4>
        <div className="metadata-grid">
          <div className="metadata-item"><strong>ID:</strong> {word.id}</div>
          <div className="metadata-item"><strong>Language:</strong> {word.language_code}</div>
          <div className="metadata-item"><strong>Hash:</strong> <span title={word.data_hash}>{word.data_hash?.substring(0, 8)}...</span></div>
          <div className="metadata-item"><strong>Created:</strong> {formatDate(word.created_at)}</div>
          <div className="metadata-item"><strong>Updated:</strong> {formatDate(word.updated_at)}</div>
          <div className="metadata-item"><strong>Verified:</strong> {word.is_verified ? 'Yes' : 'No'}</div>
          
          <div className="metadata-item"><strong>Root Word ID:</strong> {word.root_word_id ?? 'N/A'}</div>
          <div className="metadata-item"><strong>Data Quality:</strong> {word.data_quality_score?.toFixed(2) ?? 'N/A'}</div>
          <div className="metadata-item"><strong>Complexity:</strong> {word.complexity_score?.toFixed(2) ?? 'N/A'}</div>
          <div className="metadata-item"><strong>Frequency:</strong> {word.usage_frequency?.toFixed(2) ?? 'N/A'}</div>
          <div className="metadata-item"><strong>View Count:</strong> {word.view_count ?? 'N/A'}</div>
          <div className="metadata-item"><strong>Last Viewed:</strong> {formatDate(word.last_viewed_at)}</div>
          
          <div className="metadata-item"><strong>Is Root:</strong> {word.is_root ? 'Yes' : 'No'}</div>
          <div className="metadata-item"><strong>Is Proper Noun:</strong> {word.is_proper_noun ? 'Yes' : 'No'}</div>
          <div className="metadata-item"><strong>Is Abbreviation:</strong> {word.is_abbreviation ? 'Yes' : 'No'}</div>
          <div className="metadata-item"><strong>Is Initialism:</strong> {word.is_initialism ? 'Yes' : 'No'}</div>

          {word.verification_notes && (
            <div className="metadata-item metadata-item-full"><strong>Verification Notes:</strong> {word.verification_notes}</div>
          )}
          
          {word.badlit_form && <div className="metadata-item"><strong>Badlit:</strong> {word.badlit_form}</div>}
        </div>
        {word.data_completeness && (
          <div className="data-completeness">
            <h5>Data Completeness</h5>
            <ul>
              {Object.entries(word.data_completeness).map(([key, value]) => (
                <li key={key} className={value ? 'complete' : 'incomplete'}>
                  {key.replace(/_/g, ' ')}: {value ? '✓' : '✗'}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  const renderRelations = (word: WordInfo) => {
    if (!word.relations || Object.keys(word.relations).length === 0) {
      return <div className="empty-state"><p>No relationship information available.</p></div>;
    }

    const availableRelationTypes = Object.keys(word.relations).filter(type => 
      type !== 'root' && word.relations[type] && (!Array.isArray(word.relations[type]) || (word.relations[type] as RelatedWord[]).length > 0)
    );

    if (availableRelationTypes.length === 0 && !word.relations.root) {
       return <div className="empty-state"><p>No relationship information available.</p></div>;
    }

    const preferredOrder = [
      'synonyms', 'antonyms', 'variants', 'related', 'kaugnay', 
      'hypernym', 'hyponym', 'meronym', 'holonym',
      'derived', 'derived_from', 'root_of', 'component_of', 'cognate',
      'see_also', 'compare_with',
      'main', 'derivative', 'etymology', 'associated', 'other'
    ];
    
    const sortedRelationTypes = availableRelationTypes.sort((a, b) => {
      const indexA = preferredOrder.indexOf(a);
      const indexB = preferredOrder.indexOf(b);
      if (indexA === -1 && indexB === -1) return a.localeCompare(b);
      if (indexA === -1) return 1;
      if (indexB === -1) return -1;
      return indexA - indexB;
    });

    return (
      <div className="relations-section">
        {word.relations.root && (
          <div className="relation-group">
            <h4 className="relation-group-title relation-type-root">Root</h4>
            <div className="relation-list">
              <span className="relation-item" onClick={() => onWordLinkClick(word.relations.root!.word)}>
                {word.relations.root!.word}
              </span>
            </div>
          </div>
        )}
        
        {sortedRelationTypes.map((type) => {
          const relations = word.relations[type];
          if (!Array.isArray(relations) || relations.length === 0) return null; 
          
          const isExpanded = expandedGroups[type] !== undefined ? expandedGroups[type] : true;

          return (
            <div key={type} className={`relation-group ${isExpanded ? 'expanded' : ''}`}>
              <h4 className={`relation-group-title relation-type-${type}`} onClick={() => toggleGroup(type)}>
                {type.replace(/_/g, ' ')}
                <span className="relation-count">{relations.length}</span>
                <span className="expand-indicator">{isExpanded ? '▲' : '▼'}</span>
              </h4>
              {isExpanded && (
                <div className="relation-list">
                  {(relations as RelatedWord[]).map((relatedWord, index) => (
                    relatedWord && relatedWord.word ? (
                      <span 
                        key={`${type}-${index}-${relatedWord.word}`} 
                        className="relation-item" 
                        onClick={() => onWordLinkClick(relatedWord.word)}
                      >
                        {relatedWord.word}
                      </span>
                    ) : null
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const renderDefinitions = (word: WordInfo) => {
    if (!word?.definitions || word.definitions.length === 0) {
      return (
        <div className="empty-state">
          <p>No definitions available for this word.</p>
        </div>
      );
    }

    // Group definitions by part of speech
    const definitionsByPos: Record<string, any[]> = {};
    
    word.definitions.forEach(definition => {
      if (!definition) return; // Skip if definition is undefined
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
                  if (!definition) return null; // Skip if definition is undefined
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
                        
                        {definition.sources && Array.isArray(definition.sources) && definition.sources.length > 0 && (
                          <div className="definition-sources">
                            <span className="sources-label">Sources:</span>
                            <div className="source-tags">
                              {definition.sources.map((source: string, idx: number) => (
                                <span key={idx} className="source-tag">{source}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {definition.examples && Array.isArray(definition.examples) && definition.examples.length > 0 && (
                          <div className="definition-examples">
                            <span className="examples-label">Examples:</span>
                            <ul>
                              {definition.examples.map((example: string, idx: number) => (
                                <li key={idx}>{example}</li>
                              ))}
                            </ul>
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
            <p>{etymology.text}</p>
            {etymology.components && etymology.components.length > 0 && (
              <div className="etymology-components">
                <strong>Components:</strong>
                <div className="component-tags">
                  {etymology.components.map((component, idx) => (
                    <span key={idx} className="component-tag" onClick={() => onWordLinkClick(component)}>
                      {component}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {etymology.languages && etymology.languages.length > 0 && (
              <div className="etymology-languages">
                <strong>Languages:</strong> {etymology.languages.join(', ')}
              </div>
            )}
            {etymology.sources && etymology.sources.length > 0 && (
              <div className="etymology-sources">
                <strong>Sources:</strong> {etymology.sources.join(', ')}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderBaybayin = (word: WordInfo) => {
    if (!word.has_baybayin || !word.baybayin_form) {
      return <div className="empty-state"><p>No Baybayin form available.</p></div>;
    }
    return (
      <div className="baybayin-section">
        <h3>Baybayin</h3>
        <p className="baybayin-text">{word.baybayin_form}</p>
        {word.romanized_form && <p>Romanized: {word.romanized_form}</p>}
      </div>
    );
  };

  const renderIdioms = (word: WordInfo) => {
    if (!word.idioms || word.idioms.length === 0) {
      return <div className="empty-state"><p>No idioms available.</p></div>;
    }
    return (
      <div className="idioms-section">
        <h3>Idioms</h3>
        <ul>
          {word.idioms.map((idiom, index) => (
            <li key={index}>
              <strong>{idiom.text || idiom.phrase}:</strong> {idiom.meaning}
              {idiom.example && <em> (e.g., "{idiom.example}")</em>}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderAffixations = (word: WordInfo) => {
    if (!word.affixations || word.affixations.length === 0) {
      return <div className="empty-state"><p>No affixation data available.</p></div>;
    }
    const asRoot = word.affixations.filter(aff => aff.root_word?.lemma === word.lemma);
    const asAffixed = word.affixations.filter(aff => aff.affixed_word?.lemma === word.lemma);

    return (
      <div className="affixations-section">
        {asRoot.length > 0 && (
          <div className="relation-group">
            <h4 className="relation-group-title">Derived Words (as Root)</h4>
            <div className="relation-list">
              {asRoot.map((aff, index) => (
                aff.affixed_word ? (
                  <span key={`asRoot-${index}`} className="relation-item">
                    {aff.affixed_word.lemma} ({aff.affix_type})
                  </span>
                ) : null
              ))}
            </div>
          </div>
        )}
         {word.derived_words && word.derived_words.length > 0 && (
           <div className="relation-group">
              <h4 className="relation-group-title">Derived Words (Direct List)</h4>
              <div className="relation-list">
                {word.derived_words.map((derived, index) => (
                  <span key={`derived-${index}`} className="relation-item">
                    {derived.lemma}
                  </span>
                ))}
              </div>
           </div>
         )}
        {asAffixed.length > 0 && (
          <div className="relation-group">
            <h4 className="relation-group-title">Root Word (as Affixed)</h4>
            <div className="relation-list">
              {asAffixed.map((aff, index) => (
                aff.root_word ? (
                  <span key={`asAffixed-${index}`} className="relation-item">
                    {aff.root_word.lemma} ({aff.affix_type})
                  </span>
                ) : null
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderCredits = (word: WordInfo) => {
    if (!word.credits || word.credits.length === 0) {
      return <div className="empty-state"><p>No credits available.</p></div>;
    }
    return (
      <div className="credits-section">
        <h3>Credits</h3>
        <ul>
          {word.credits.map((credit, index) => (
            <li key={index}>{credit.credit}</li>
          ))}
        </ul>
      </div>
    );
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

  return (
    <div className="word-details-container">
      <div className="word-details">
        <div className="word-header">
          <div className="word-title-row">
            <h2 className="word-lemma">{wordInfo.lemma}</h2>
            {wordInfo.pronunciation?.ipa && (
              <span className="ipa-pronunciation">/ {wordInfo.pronunciation.ipa} /</span>
            )}
            {wordInfo.pronunciation?.audio_url && (
              <button 
                className={`audio-button ${isAudioPlaying ? 'playing' : ''}`} 
                onClick={playAudio} 
                aria-label="Play pronunciation"
              >
                🔊
              </button>
            )}
          </div>
          <div className="word-basic-info">
            {wordInfo.preferred_spelling && wordInfo.preferred_spelling !== wordInfo.lemma && (
              <span className="info-item preferred-spelling">Preferred: {wordInfo.preferred_spelling}</span>
            )}
            <span className="info-item language">Lang: {wordInfo.language_code}</span>
             <button className="metadata-toggle-button" onClick={toggleMetadata}>
              {showMetadata ? 'Hide' : 'Show'} Metadata
            </button>
          </div>
        </div>
         {renderMetadata(wordInfo)}

        <div className="tabs">
          <button className={`tab-button ${activeTab === 'definitions' ? 'active' : ''}`} onClick={() => setActiveTab('definitions')}>Definitions</button>
          <button className={`tab-button ${activeTab === 'relations' ? 'active' : ''}`} onClick={() => setActiveTab('relations')}>Relations</button>
          <button className={`tab-button ${activeTab === 'etymology' ? 'active' : ''}`} onClick={() => setActiveTab('etymology')}>Etymology</button>
          {wordInfo.has_baybayin && (
            <button className={`tab-button ${activeTab === 'baybayin' ? 'active' : ''}`} onClick={() => setActiveTab('baybayin')}>Baybayin</button>
          )}
          {wordInfo.idioms && wordInfo.idioms.length > 0 && (
            <button className={`tab-button ${activeTab === 'idioms' ? 'active' : ''}`} onClick={() => setActiveTab('idioms')}>Idioms</button>
          )}
           {wordInfo.affixations && wordInfo.affixations.length > 0 && (
             <button className={`tab-button ${activeTab === 'affixations' ? 'active' : ''}`} onClick={() => setActiveTab('affixations')}>Affixations</button>
           )}
           {wordInfo.credits && wordInfo.credits.length > 0 && (
             <button className={`tab-button ${activeTab === 'credits' ? 'active' : ''}`} onClick={() => setActiveTab('credits')}>Credits</button>
           )}
        </div>

        <div className="tab-content">
          {activeTab === 'definitions' && renderDefinitions(wordInfo)}
          {activeTab === 'relations' && renderRelations(wordInfo)}
          {activeTab === 'etymology' && renderEtymology(wordInfo)}
          {activeTab === 'baybayin' && renderBaybayin(wordInfo)}
          {activeTab === 'idioms' && renderIdioms(wordInfo)}
           {activeTab === 'affixations' && renderAffixations(wordInfo)}
           {activeTab === 'credits' && renderCredits(wordInfo)}
        </div>
      </div>
    </div>
  );
});

export default WordDetails;