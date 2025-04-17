import { WordInfo, WordNetwork } from '../types';
import { fetchWordNetwork } from '../api/wordApi';

/**
 * Checks if a word has missing or minimal relations
 * @param wordInfo The word information to check
 * @returns true if relations are missing or minimal
 */
export const hasMissingOrMinimalRelations = (wordInfo: WordInfo): boolean => {
  if (!wordInfo) return true;
  
  const incomingCount = wordInfo.incoming_relations?.length || 0;
  const outgoingCount = wordInfo.outgoing_relations?.length || 0;
  const totalRelations = incomingCount + outgoingCount;
  
  return totalRelations < 3;
};

/**
 * Fetches semantic network data for a word and enhances the WordInfo object
 * This is used as a fallback when relations are missing or insufficient
 * 
 * @param word The word to fetch semantic network data for
 * @param wordInfo Current WordInfo object to enhance
 * @param depth Network depth (default: 2)
 * @param breadth Network breadth (default: 10)
 * @returns A promise that resolves to an enhanced WordInfo with semantic_network data
 */
export const fetchAndAddSemanticNetwork = async (
  word: string,
  wordInfo: WordInfo | undefined,
  depth: number = 2,
  breadth: number = 10
): Promise<WordInfo> => {
  if (!word || !wordInfo) {
    console.warn('Missing word or wordInfo for fetchAndAddSemanticNetwork');
    return wordInfo || {};
  }
  
  try {
    console.log(`Fetching semantic network data for "${word}" as relations fallback`);
    
    // Fetch network data
    const networkData = await fetchWordNetwork(word, {
      depth,
      breadth,
      include_affixes: true,
      include_etymology: true,
      cluster_threshold: 0.3
    });
    
    // Check if we got valid network data
    if (!networkData || !networkData.nodes || !networkData.edges || networkData.nodes.length === 0) {
      console.warn('No valid semantic network data received for fallback');
      return wordInfo;
    }
    
    console.log(`Received semantic network with ${networkData.nodes.length} nodes and ${networkData.links.length} links`);
    
    // Create enhanced copy of wordInfo with semantic network
    return {
      ...wordInfo,
      semantic_network: {
        nodes: networkData.nodes,
        links: networkData.links
      }
    };
  } catch (error) {
    console.error('Error fetching semantic network as fallback:', error);
    return wordInfo; // Return original on error
  }
};

/**
 * Creates relations from semantic network data when regular relations are missing
 * @param wordInfo WordInfo with semantic_network data
 * @returns Array of relations created from the semantic network
 */
export const extractRelationsFromNetwork = (wordInfo: WordInfo): any[] => {
  if (!wordInfo || !wordInfo.semantic_network) {
    return [];
  }
  
  const { nodes, links } = wordInfo.semantic_network;
  if (!nodes || !links || nodes.length === 0 || links.length === 0) {
    return [];
  }
  
  const mainWord = wordInfo.lemma;
  const relations: any[] = [];
  
  // Find the main node ID
  const mainNodeId = nodes.find(n => 
    n.label === mainWord || 
    n.word === mainWord || 
    n.type === 'main'
  )?.id;
  
  if (!mainNodeId) {
    console.warn('Could not find main node in semantic network');
    return [];
  }
  
  // Process each link to create relation objects
  links.forEach(link => {
    // Determine if this is an incoming or outgoing relation
    const isOutgoing = typeof link.source === 'object' 
      ? link.source.id === mainNodeId 
      : link.source === mainNodeId;
      
    const isIncoming = typeof link.target === 'object'
      ? link.target.id === mainNodeId
      : link.target === mainNodeId;
      
    // Skip if neither incoming nor outgoing
    if (!isOutgoing && !isIncoming) {
      return;
    }
    
    // Find the connected node
    const connectedNodeId = isOutgoing 
      ? (typeof link.target === 'object' ? link.target.id : link.target)
      : (typeof link.source === 'object' ? link.source.id : link.source);
      
    const connectedNode = nodes.find(n => n.id === connectedNodeId);
    
    if (!connectedNode) return;
    
    // Skip self-references
    if (connectedNode.label === mainWord || connectedNode.word === mainWord) {
      return;
    }
    
    // Create a relation object
    relations.push({
      id: `semantic-${connectedNode.id}`,
      relation_type: link.type || 'related',
      degree: link.distance || link.degree || 1,
      direction: isOutgoing ? 'outgoing' : 'incoming',
      wordObj: {
        id: connectedNode.id,
        lemma: connectedNode.label || connectedNode.word || String(connectedNode.id),
        has_baybayin: connectedNode.has_baybayin || false,
        baybayin_form: connectedNode.baybayin_form || null
      }
    });
  });
  
  console.log(`Created ${relations.length} fallback relations from semantic network`);
  return relations;
}; 