import React, { useCallback, useState, useEffect, useRef } from 'react';
import { WordInfo, WordNetwork, RelatedWord } from '../types';
import { fetchWordNetwork } from '../api/wordApi';
import './WordDetails.css';
import './Tabs.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faStop, faVolumeUp } from '@fortawesome/free-solid-svg-icons';

interface WordDetailsProps {
  wordInfo: WordInfo;
  etymologyTree: any;
  isLoadingEtymology: boolean;
  etymologyError: string | null;
  showMetadata: boolean;
  toggleMetadata: () => void;
  onWordLinkClick: (word: string) => void;
  onEtymologyNodeClick: (node: any) => void;
}

// Helper function to format relation type names
function formatRelationType(type: string): string {
  return type
    .replace(/_/g, ' ') // Replace underscores with spaces
    .replace(/\b\w/g, char => char.toUpperCase()); // Capitalize first letter of each word
}

const WordDetails: React.FC<WordDetailsProps> = React.memo(({ 
  wordInfo, 
  etymologyTree,
  isLoadingEtymology, 
  etymologyError,
  showMetadata,
  toggleMetadata,
  onWordLinkClick,
  onEtymologyNodeClick
}) => {
  // Simple debug log for development
  console.log("WordDetails component rendering with wordInfo:", wordInfo);
  
  const [activeTab, setActiveTab] = useState<'definitions' | 'relations' | 'etymology' | 'baybayin' | 'credits'>('definitions');
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
    
    setExpandedGroups({}); // Initialize as empty, groups added dynamically in renderRelations
    
    // Create audio element if pronunciation available
    const audioPronunciation = wordInfo?.pronunciations?.find(p => p.type === 'audio' && p.value);
    if (audioPronunciation) {
      const audio = new Audio(audioPronunciation.value);
      audio.addEventListener('ended', () => setIsAudioPlaying(false));
      setAudioElement(audio);
      return () => {
        audio.pause();
        audio.removeEventListener('ended', () => setIsAudioPlaying(false));
      };
    }
  }, [wordInfo]); // Dependency only on wordInfo

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
          <div className="metadata-item"><strong>Hash:</strong> <span title={word.data_hash ?? undefined}>{word.data_hash?.substring(0, 8)}...</span></div>
          <div className="metadata-item"><strong>Created:</strong> {formatDate(word.created_at)}</div>
          <div className="metadata-item"><strong>Updated:</strong> {formatDate(word.updated_at)}</div>
          <div className="metadata-item"><strong>Root Word ID:</strong> {word.root_word_id ?? 'N/A'}</div>
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
    const hasOutgoing = word.outgoing_relations && word.outgoing_relations.length > 0;
    const hasIncoming = word.incoming_relations && word.incoming_relations.length > 0;

    if (!hasOutgoing && !hasIncoming) {
      return <div className="empty-state"><p>No relationship information available.</p></div>;
    }

    // Combine and group relations
    const allRelations = [
        ...(word.outgoing_relations || []).map(r => ({ ...r, direction: 'out' as const })),
        ...(word.incoming_relations || []).map(r => ({ ...r, direction: 'in' as const }))
    ]; 

    const grouped: { [key: string]: typeof allRelations } = {};
    allRelations.forEach(rel => {
        // Group by relation_type, ensuring it exists
        const type = rel.relation_type || 'unknown'; 
        if (!grouped[type]) {
            grouped[type] = [];
        }
        grouped[type].push(rel);
    });

    // Sort relation types (example: alphabetical)
    const sortedRelationTypes = Object.keys(grouped).sort((a, b) => a.localeCompare(b));

    if (sortedRelationTypes.length === 0) {
         // This case should ideally not be reached if hasOutgoing/hasIncoming is true, but good safety check
         return <div className="empty-state"><p>No relationship information available.</p></div>;
    }

    return (
      <div className="relations-section">
        {sortedRelationTypes.map((type) => {
          const relationsOfType = grouped[type];
          // Double check length just in case
          if (!relationsOfType || relationsOfType.length === 0) return null; 
          
          // Check expansion state using component state
          const isExpanded = expandedGroups[type] !== undefined ? expandedGroups[type] : true; // Default to expanded

          return (
            <div key={type} className={`relation-group ${isExpanded ? 'expanded' : ''}`}>
              <h4 
                className={`relation-group-title relation-type-${type.toLowerCase().replace(/_/g, '-')}`}
                onClick={() => toggleGroup(type)} // Use the state toggle function
              >
                {formatRelationType(type)} {/* Use helper for display name */} 
                <span className="relation-count">({relationsOfType.length})</span>
                <span className="expand-indicator">{isExpanded ? '▲' : '▼'}</span>
              </h4>
              {isExpanded && (
                <div className="relation-list">
                  {relationsOfType.map((relatedItem, index) => {
                    // Determine the actual related word object based on direction
                    const relatedWord : RelatedWord | undefined = relatedItem.direction === 'out' ? relatedItem.target_word : relatedItem.source_word;
                    
                    // Check if relatedWord and its lemma exist
                    return relatedWord && relatedWord.lemma ? (
                      <span 
                        key={`${type}-${index}-${relatedWord.id}`}
                        className="relation-item" 
                        onClick={() => onWordLinkClick(relatedWord.lemma)} // Use lemma for linking
                      >
                        {relatedWord.lemma} {/* Display lemma */} 
                        {relatedItem.direction === 'in' ? <span className="direction-indicator">(from)</span> : ''}
                      </span>
                    ) : (
                        // Log or display if word data is missing unexpectedly
                        <span key={`${type}-${index}-missing`} className="relation-item missing-data">
                          (Missing Word Data: ID {relatedItem.direction === 'out' ? relatedItem.target_word?.id : relatedItem.source_word?.id})
                        </span>
                    );
                  })}
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
            {wordInfo.pronunciations?.find(p => p.type === 'ipa') && (
              <span className="ipa-pronunciation">/ {wordInfo.pronunciations.find(p => p.type === 'ipa')?.value} /</span>
            )}
            {wordInfo.pronunciations?.find(p => p.type === 'audio' && p.value) && (
              <button 
                className={`audio-button ${isAudioPlaying ? 'playing' : ''}`} 
                onClick={playAudio} 
                aria-label="Play pronunciation"
              >
                <FontAwesomeIcon icon={isAudioPlaying ? faStop : faVolumeUp} />
              </button>
            )}
          </div>
          <div className="word-basic-info">
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
           {wordInfo.credits && wordInfo.credits.length > 0 && (
             <button className={`tab-button ${activeTab === 'credits' ? 'active' : ''}`} onClick={() => setActiveTab('credits')}>Credits</button>
           )}
        </div>

        <div className="tab-content">
          {activeTab === 'definitions' && renderDefinitions(wordInfo)}
          {activeTab === 'relations' && renderRelations(wordInfo)}
          {activeTab === 'etymology' && renderEtymology(wordInfo)}
          {activeTab === 'baybayin' && renderBaybayin(wordInfo)}
           {activeTab === 'credits' && renderCredits(wordInfo)}
        </div>
      </div>
    </div>
  );
});

export default WordDetails;