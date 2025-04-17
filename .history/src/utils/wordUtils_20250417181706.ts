import { WordInfo, WordNetwork } from '../types';
import { fetchWordNetwork, ApiWordNetwork } from '../api/wordApi';

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
  wordInfo: WordInfo | undefined,
  word: string,
  depth: number = 2,
  breadth: number = 10
): Promise<WordInfo | undefined> => {
  try {
    if (!wordInfo) {
      console.warn(`Cannot fetch semantic network for "${word}": wordInfo is undefined.`);
      return undefined;
    }

    console.log(`Fetching semantic network data for "${word}"`);

    // Use params compatible with fetchWordNetwork signature
    const params = {
      // fetchWordNetwork likely takes word as first arg, options as second
      // Let's assume it needs these options:
      depth: depth,
      // breadth: breadth, // Might not be supported by fetchWordNetwork
      // include_affixes: true, // Example option
      // include_etymology: true // Example option
    };

    // Fetch semantic network data - use fetchWordNetwork and type as ApiWordNetwork
    const networkData: ApiWordNetwork = await fetchWordNetwork(word, params); // Use correct function and params

    // Check if we got valid network data (nodes and links are required by ApiWordNetwork)
    if (!networkData || !Array.isArray(networkData.nodes) || !Array.isArray(networkData.links)) {
      console.warn(`Invalid network data received for "${word}". Cannot add semantic network.`);
      return wordInfo; // Return original wordInfo
    }

    // Map ApiWordNetwork to WordNetwork (links -> edges)
    const semanticNetworkData: WordNetwork = {
        nodes: networkData.nodes, // Assuming node structure is compatible enough
        edges: networkData.links.map(link => ({ // Map links to edges
            source: typeof link.source === 'object' ? link.source.id : link.source, // Adjust if source/target are objects
            target: typeof link.target === 'object' ? link.target.id : link.target,
            type: link.type || 'related' // Provide default type if missing
            // Add other properties if needed and available in link
        })),
        metadata: networkData.metadata // Pass metadata along
    };

    // Enhance wordInfo with the fetched semantic network data
    const enhancedWordInfo: WordInfo = {
      ...wordInfo,
      semantic_network: semanticNetworkData,
    };

    return enhancedWordInfo;

  } catch (error) {
    console.error(`Error fetching or adding semantic network for "${word}":`, error);
    // Return the original wordInfo (or undefined if it was initially undefined)
    // Changed return to match Promise<WordInfo | undefined>
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