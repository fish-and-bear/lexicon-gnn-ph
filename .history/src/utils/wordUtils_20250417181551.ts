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
  wordInfo: WordInfo,
  depth: number = 2,
  breadth: number = 10
): Promise<WordInfo | undefined> => {
  try {
    console.log(`Fetching semantic network data for "${word}" as relations fallback`);
    
    // Fetch network data using ApiWordNetwork type from fetchWordNetwork
    const networkData = await fetchWordNetwork(word, {
      depth,
      breadth,
      include_affixes: true,
      include_etymology: true,
      cluster_threshold: 0.3
    });
    
    // Check if we got valid network data (nodes and links are required by ApiWordNetwork)
    if (!networkData || !networkData.nodes || !networkData.links || networkData.nodes.length === 0) {
      console.warn('No valid semantic network data received for fallback');
      // If wordInfo is undefined, we can't enhance it, return undefined
      if (!wordInfo) return undefined; 
      // Otherwise, return the original wordInfo without enhancement
      return wordInfo; 
    }
    
    console.log(`Received semantic network with ${networkData.nodes.length} nodes and ${networkData.links.length} links`);
    
    // If wordInfo is undefined, we can't enhance it. This shouldn't happen based on usage, but handle defensively.
    if (!wordInfo) {
        console.warn('fetchAndAddSemanticNetwork called with undefined wordInfo, returning undefined');
        return undefined;
    }

    // Create enhanced copy of wordInfo with semantic network, matching WordNetwork type
    // Ensure nodes and links are arrays, provide default empty arrays if somehow missing
    const semanticNetworkData: WordNetwork = {
        nodes: networkData.nodes || [],
        links: networkData.links || [],
        metadata: networkData.metadata // Pass metadata along
    };

    return {
      ...wordInfo,
      semantic_network: semanticNetworkData
    };
  } catch (error) {
    console.error('Error fetching semantic network as fallback:', error);
    // Return original wordInfo if it exists, otherwise return undefined to match the promise type
    return wordInfo || undefined; 
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