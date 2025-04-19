import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetwork } from "../types";
import { useTheme } from "../contexts/ThemeContext";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import axios from 'axios';
import NetworkControls from './NetworkControls';

interface WordGraphProps {
  wordNetwork: WordNetwork | null;
  mainWord: string | null;
  onNodeClick: (word: string) => void;
  onNodeSelect: (word: string) => void;
  onNetworkChange: (depth: number, breadth: number) => void;
  initialDepth: number;
  initialBreadth: number;
}

interface CustomNode extends d3.SimulationNodeDatum {
  id: string;
  word: string;
  label?: string;
  group: string;
  connections?: number;
  pinned?: boolean;
  originalId?: number;
  language?: string;
  definitions?: string[];
  path?: Array<{ type: string; word: string }>;
  has_baybayin?: boolean;
  baybayin_form?: string | null;
}

interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
  relationship: string;
  source: string | CustomNode;
  target: string | CustomNode;
}

// Helper to get luminance and decide text color
const getTextColorForBackground = (hexColor: string): string => {
  try {
    const color = d3.color(hexColor);
    if (!color) return '#111'; // Default dark text
    const rgb = color.rgb();
    // Calculate luminance using the standard formula
    const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
    return luminance > 0.5 ? '#111111' : '#f8f8f8'; // Dark text on light bg, light text on dark bg
  } catch (e) {
    console.error("Error parsing color for text:", hexColor, e);
    return '#111'; // Fallback
  }
};

const WordGraph: React.FC<WordGraphProps> = ({
  wordNetwork,
  mainWord,
  onNodeClick,
  onNodeSelect,
  onNetworkChange,
  initialDepth,
  initialBreadth,
}) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(mainWord);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [isLoading, setIsLoading] = useState(false); // Re-add isLoading state
  const [error, setError] = useState<string | null>(null);
  const [isValidNetwork, setIsValidNetwork] = useState(true);
  const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);
  const [filteredRelationships, setFilteredRelationships] = useState<string[]>([]);
  const [forceUpdate, setForceUpdate] = useState<number>(0); // Force remount counter
  const [showDisconnectedNodes, setShowDisconnectedNodes] = useState<boolean>(false);

  const isDraggingRef = useRef(false);
  const isTransitioningRef = useRef(false);
  const prevMainWordRef = useRef<string | null>(null);

  // State for tooltip delay
  const [tooltipTimeoutId, setTooltipTimeoutId] = useState<NodeJS.Timeout | null>(null);

  // Create a key that changes whenever filtered relationships change
  // This will force the graph to completely rebuild
  const filterUpdateKey = useMemo(() => {
    return filteredRelationships.join(',');
  }, [filteredRelationships]);

  useEffect(() => {
    if (!wordNetwork || !wordNetwork.nodes || !Array.isArray(wordNetwork.nodes) || 
        !wordNetwork.edges || !Array.isArray(wordNetwork.edges)) {
      console.error("Invalid wordNetwork structure:", wordNetwork);
      setIsValidNetwork(false);
    } else {
      setIsValidNetwork(true);
    }
  }, [wordNetwork]);

  useEffect(() => {
    setSelectedNodeId(mainWord);
  }, [mainWord]);

  const mapRelationshipToGroup = useCallback((relationship?: string): string => {
    if (!relationship) return 'associated';
    const relLower = relationship.toLowerCase();
    let group: string;
    
    // Add extensive debug logging
    console.log(`[MAP_DEBUG] Mapping relationship: "${relationship}"`);
    
    switch (relLower) {
      case 'main': group = 'main'; break;
      case 'synonym': group = 'synonym'; break;
      case 'antonym': 
      case 'kasalungat': group = 'antonym'; break;
      case 'related': 
      case 'kaugnay':
      case 'kahulugan':
      case 'similar':
      case 'associated': group = 'related'; break;
      case 'variant':
      case 'spelling_variant':
      case 'regional_variant':
      case 'atapat':
      case 'inatapat':
      case 'itapat': group = 'variant'; break;
      // Treat both root and derived types as the same category
      case 'derived': 
      case 'derived_from':
      case 'sahod':
      case 'root':
      case 'root_of':
      case 'isahod': group = 'root'; break;
      case 'affix':
      case 'derivative': group = 'derived'; break;
      case 'hypernym':
      case 'hyponym': group = 'taxonomic'; break;
      case 'meronym':
      case 'holonym':
      case 'part_whole':
      case 'component':
      case 'component_of': group = 'part_whole'; break;
      case 'etymology': group = 'etymology'; break;
      case 'cognate': group = 'cognate'; break;
      case 'see_also':
      case 'compare_with':
      case 'usage': group = 'usage'; break;
      default: group = 'related'; // Default to 'related' instead of 'associated'
    }
    
    // Debug log to track relationship mappings
    console.log(`[MAP_DEBUG] Mapped "${relationship}" to group "${group}"`);
    return group;
  }, []);

  const getNodeColor = useCallback((group: string): string => {
    // Colors organized by semantic relationship categories
    switch (group.toLowerCase()) {
      // Core
      case "main": return "#0e4a86"; // Deep blue - standout color for main word
      
      // Origin group - Reds and oranges
      case "root": return "#e63946"; // Bright red
      case "etymology": return "#d00000"; // Dark red
      case "cognate": return "#ff5c39"; // Light orange
      
      // Meaning group - Blues
      case "synonym": return "#457b9d"; // Medium blue
      case "related": return "#48cae4"; // Light blue
      case "antonym": return "#023e8a"; // Dark blue
      
      // Form group - Purples
      case "variant": return "#7d4fc3"; // Medium purple
      
      // Hierarchy group - Greens
      case "taxonomic": return "#2a9d8f"; // Teal
      case "part_whole": return "#40916c"; // Forest green
      case "component_of": return "#40916c"; // Forest green (same as part_whole)
      
      // Derivation group - Teals
      case "derived": return "#2a9d8f"; // Teal for derived relations
      
      // Info group - Yellows
      case "usage": return "#fcbf49"; // Gold
      
      // Fallback
      default: return "#adb5bd"; // Neutral gray
    }
  }, []);

  const baseLinks = useMemo<{ source: string; target: string; relationship: string }[]>(() => {
    if (!wordNetwork?.nodes || !wordNetwork.edges) return [];
    
    console.log("[BASE] Processing wordNetwork edges:", wordNetwork.edges.length);
    
    const links = wordNetwork.edges
      .map(edge => {
        const sourceNode = wordNetwork.nodes.find(n => n.id === edge.source);
        const targetNode = wordNetwork.nodes.find(n => n.id === edge.target);
        
        if (!sourceNode || !targetNode) {
          console.warn(`Could not find nodes for edge: ${edge.source} -> ${edge.target}`);
          return null;
        }
        
        return {
          source: sourceNode.label,
          target: targetNode.label,
          relationship: edge.type
        };
      })
      .filter((link): link is { source: string; target: string; relationship: string; } => link !== null);
    
    console.log("[BASE] Processed links:", links.length);
    
    if (links.length > 0) {
      console.log("[BASE] Sample links:");
      links.slice(0, 5).forEach(link => {
        console.log(`  ${link.source} -[${link.relationship}]-> ${link.target}`);
      });
    }
    
    return links;
  }, [wordNetwork]);

  const baseNodes = useMemo<CustomNode[]>(() => {
    // Ensure wordNetwork and mainWord exist before proceeding
    if (!wordNetwork?.nodes || !mainWord) {
        return []; // Return empty array if prerequisites are missing
    }

    console.log("[BASE] Processing wordNetwork nodes:", wordNetwork.nodes.length);
    console.log("[BASE] Main word:", mainWord);

    const mappedNodes = wordNetwork.nodes.map(node => {
      let calculatedGroup = 'associated';
      if (node.label === mainWord) {
        calculatedGroup = 'main';
      } else {
        const connectingLink = baseLinks.find(link =>
          (link.source === mainWord && link.target === node.label) ||
          (link.source === node.label && link.target === mainWord)
        );
        calculatedGroup = mapRelationshipToGroup(connectingLink?.relationship);
      }
      
      // Count connections for potential sizing later
       const connections = baseLinks.filter(l => l.source === node.label || l.target === node.label).length;

      return {
        id: node.label,
        word: node.label,
        label: node.label,
        group: calculatedGroup,
        connections: connections, // Store connection count
        originalId: node.id,
        language: node.language,
        definitions: node.definitions,
        path: node.path,
        has_baybayin: node.has_baybayin,
        baybayin_form: node.baybayin_form,
        index: undefined, x: undefined, y: undefined, vx: undefined, vy: undefined, fx: undefined, fy: undefined
      };
    });

    // Filter out duplicate nodes based on id (label), keeping the first occurrence
    const uniqueNodes: CustomNode[] = []; // Explicitly type the array
    const seenIds = new Set<string>();
    for (const node of mappedNodes) {
        if (!seenIds.has(node.id)) {
            uniqueNodes.push(node);
            seenIds.add(node.id);
        }
    }
    console.log("[BASE] Final unique nodes:", uniqueNodes.length);
    
    if (uniqueNodes.length > 0) {
      console.log("[BASE] Sample nodes:");
      uniqueNodes.slice(0, 5).forEach(node => {
        console.log(`  ${node.id} (${node.group}) - connections: ${node.connections}`);
      });
    }
    
    return uniqueNodes; // Now guaranteed to return CustomNode[]
  }, [wordNetwork, mainWord, baseLinks, mapRelationshipToGroup]);

  // Create nodes and links directly based on filter state
  const { filteredNodes, filteredLinks } = useMemo(() => {
    if (!mainWord || baseNodes.length === 0) {
      return { filteredNodes: [], filteredLinks: [] };
    }
    
    console.log("[FILTER] Applying depth/breadth and relationship filters");
    console.log("[FILTER] Main word:", mainWord);
    console.log("[FILTER] Base nodes:", baseNodes.length);
    console.log("[FILTER] Base links:", baseLinks.length);
    console.log("[FILTER] Depth limit:", depth);
    console.log("[FILTER] Breadth limit:", breadth);
    console.log("[FILTER] Show disconnected nodes:", showDisconnectedNodes);
    
    // First, verify if the main word exists in our node set
    const mainWordNode = baseNodes.find(n => n.id === mainWord);
    if (!mainWordNode) {
      console.error(`[FILTER] ERROR: Main word "${mainWord}" not found in baseNodes!`);
      // Return just the single mainWord node if we can construct it
      return {
        filteredNodes: [{ 
          id: mainWord, 
          word: mainWord, 
          label: mainWord, 
          group: 'main',
          index: undefined, x: undefined, y: undefined, vx: undefined, vy: undefined, fx: undefined, fy: undefined 
        }],
        filteredLinks: []
      };
    }
    
    // Step 1: First collect ALL connected nodes based on depth/breadth limits (ignoring relationship filters)
    const nodeMap = new Map(baseNodes.map(n => [n.id, n]));
    const connectedNodeIds = new Set<string>([mainWord]); // Start with main word
    const queue: [string, number][] = [[mainWord, 0]]; // [nodeId, depth]
    const visited = new Set<string>();

    // Log some debug information
    console.log("[FILTER] Starting BFS with mainWord:", mainWord);
    console.log("[FILTER] Node map size:", nodeMap.size);
    
    // Do BFS traversal to find all nodes within depth/breadth
    while (queue.length > 0) {
      const [currentWordId, currentDepth] = queue.shift()!;

      console.log(`[FILTER] Processing node: ${currentWordId} at depth ${currentDepth}`);

      if (currentDepth >= depth) {
        console.log(`[FILTER] Skipping - reached max depth ${depth}`);
        continue;
      }
      
      if (visited.has(currentWordId)) {
        console.log(`[FILTER] Skipping - already visited`);
        continue;
      }
      
      visited.add(currentWordId);

      // Find all links connected to this node
      const relatedLinks = baseLinks.filter(link => {
        // Ensure we handle source/target correctly
        const sourceId = typeof link.source === 'string' ? link.source : 
          (link.source as any)?.id || link.source;
        const targetId = typeof link.target === 'string' ? link.target : 
          (link.target as any)?.id || link.target;
        return sourceId === currentWordId || targetId === currentWordId;
      });
      
      console.log(`[FILTER] Found ${relatedLinks.length} related links for ${currentWordId}`);

      // Get all connected nodes
      const relatedWordIds = relatedLinks.map(link => {
        // Ensure we handle source/target correctly
        const sourceId = typeof link.source === 'string' ? link.source : 
          (link.source as any)?.id || link.source;
        const targetId = typeof link.target === 'string' ? link.target : 
          (link.target as any)?.id || link.target;
        return sourceId === currentWordId ? targetId : sourceId;
      }).filter(id => !visited.has(id));

      console.log(`[FILTER] Found ${relatedWordIds.length} unvisited connected nodes`);
      if (relatedWordIds.length > 0) {
        console.log(`[FILTER] Connected nodes: ${relatedWordIds.join(', ')}`);
      }

      // Sort nodes by relationship type for consistent breadth application
      const sortedWords = [...relatedWordIds].sort((aId, bId) => {
         const aNode = nodeMap.get(aId);
         const bNode = nodeMap.get(bId);
        
        if (!aNode) {
          console.warn(`[FILTER] Node missing from map: ${aId}`);
          return 1; // Push missing nodes to the end
        }
        if (!bNode) {
          console.warn(`[FILTER] Node missing from map: ${bId}`);
          return -1; // Push missing nodes to the end
        }
        
        const aGroup = aNode.group.toLowerCase();
        const bGroup = bNode.group.toLowerCase();
        
        const groupOrder = [
            'main', 'root', 'root_of', 'synonym', 'antonym', 'derived',
            'variant', 'related', 'kaugnay', 'component_of', 'cognate',
            'etymology', 'derivative', 'associated', 'other'
        ];
        
        return groupOrder.indexOf(aGroup) - groupOrder.indexOf(bGroup);
      });

      // Apply breadth limit and add to traversal queue
      const wordsToAdd = sortedWords.slice(0, breadth);
      console.log(`[FILTER] Adding ${wordsToAdd.length}/${sortedWords.length} nodes (breadth limit: ${breadth})`);
      
      wordsToAdd.forEach(wordId => {
         if (nodeMap.has(wordId)) {
          console.log(`[FILTER] Adding to connected set and queue: ${wordId}`);
             connectedNodeIds.add(wordId);
             queue.push([wordId, currentDepth + 1]);
        } else {
          console.warn(`[FILTER] Node not in map, skipping: ${wordId}`);
        }
      });
      
      console.log(`[FILTER] Updated queue size: ${queue.length}`);
      console.log(`[FILTER] Connected nodes so far: ${connectedNodeIds.size}`);
    }

    // Step 2: Create lists of depth-limited nodes and links
    const depthLimitedNodes = baseNodes.filter(node => connectedNodeIds.has(node.id));
    
    // Find links where both source and target are in our depth-limited node set
    const depthLimitedLinks = baseLinks.filter(link => {
      // Ensure we handle both string and object source/target correctly
      const sourceId = typeof link.source === 'string' ? link.source : 
        (link.source as any)?.id || link.source;
      const targetId = typeof link.target === 'string' ? link.target : 
        (link.target as any)?.id || link.target;
      return connectedNodeIds.has(sourceId) && connectedNodeIds.has(targetId);
    });
    
    console.log(`[FILTER] After depth/breadth: ${depthLimitedNodes.length}/${baseNodes.length} nodes and ${depthLimitedLinks.length}/${baseLinks.length} links`);
    
    // Step 3: Apply relationship filtering if any filters are active
    if (filteredRelationships.length === 0) {
      // No relationship filters, so return all depth-limited nodes and links
      console.log(`[FILTER] No relationship filters active - showing all nodes and links`);
      return { 
        filteredNodes: depthLimitedNodes, 
        filteredLinks: depthLimitedLinks 
      };
    }
    
    // Generate group counts for debugging
    const groupCounts = depthLimitedNodes.reduce((counts, node) => {
      const group = node.group.toLowerCase();
      counts[group] = (counts[group] || 0) + 1;
      return counts;
    }, {} as Record<string, number>);
    
    console.log(`[FILTER] Group counts in depth-limited nodes:`, groupCounts);
    console.log(`[FILTER] Currently filtered relationships:`, filteredRelationships);
    
    // Filter nodes by relationship type
    const relationshipFilteredNodes = depthLimitedNodes.filter(node => {
      // Always include the main word node
      if (node.id === mainWord) {
        console.log(`[FILTER] Keeping main word node: ${node.id}`);
        return true;
      }
      
      // Check if this node's group is in the filtered list
      const nodeGroup = node.group.toLowerCase();
      const isGroupFiltered = filteredRelationships.includes(nodeGroup);
      
      if (isGroupFiltered) {
        console.log(`[FILTER] Removing node ${node.id} with filtered group: ${nodeGroup}`);
      } else {
        console.log(`[FILTER] Keeping node ${node.id} with group: ${nodeGroup}`);
      }
      
      // Keep nodes whose group is NOT in the filtered list
      return !isGroupFiltered;
    });
    
    // Only include links where both source and target remain in the filtered node set
    const relationshipFilteredNodeIds = new Set(relationshipFilteredNodes.map(n => n.id));
    const relationshipFilteredLinks = depthLimitedLinks.filter(link => {
      // Ensure both nodes are still included
      const sourceId = typeof link.source === 'string' ? link.source : 
        (link.source as any)?.id || link.source;
      const targetId = typeof link.target === 'string' ? link.target : 
        (link.target as any)?.id || link.target;
      
      const sourceIncluded = relationshipFilteredNodeIds.has(sourceId);
      const targetIncluded = relationshipFilteredNodeIds.has(targetId);
      
      // If either endpoint is filtered out, don't include the link
      if (!sourceIncluded || !targetIncluded) {
        console.log(`[FILTER] Link ${sourceId} -> ${targetId} excluded due to filtered endpoints`);
        return false;
      }
      
      // Check if the link's relationship type is being filtered
      const linkType = link.relationship.toLowerCase();
      
      // Map the link relationship to a group for consistency with node filtering
      const linkGroup = mapRelationshipToGroup(linkType);
      
      // Check if this group is filtered
      const isLinkTypeFiltered = filteredRelationships.includes(linkGroup.toLowerCase());
      
      // Keep links whose relationship is NOT in the filtered list
      if (isLinkTypeFiltered) {
        console.log(`[FILTER] Link ${sourceId} -> ${targetId} excluded due to filtered relationship: ${linkType} (group: ${linkGroup})`);
      }
      return !isLinkTypeFiltered;
    });
    
    // CORRECTED FUNCTIONALITY:
    // Find nodes that would be connected if no filters were applied,
    // but become disconnected after relationship filtering
    const connectedAfterFiltering = new Set<string>([mainWord]); // Always include main word
    
    // Find all nodes that remain connected to the main word through some path
    const bfsQueue: string[] = [mainWord];
    const bfsVisited = new Set<string>([mainWord]);
    
    while (bfsQueue.length > 0) {
      const currentId = bfsQueue.shift()!;
      
      // Find all outgoing links from this node
      relationshipFilteredLinks.forEach(link => {
        const sourceId = typeof link.source === 'string' ? link.source : 
          (link.source as any)?.id || link.source;
        const targetId = typeof link.target === 'string' ? link.target : 
          (link.target as any)?.id || link.target;
        
        // If current node is source, target becomes connected
        if (sourceId === currentId && !bfsVisited.has(targetId)) {
          connectedAfterFiltering.add(targetId);
          bfsVisited.add(targetId);
          bfsQueue.push(targetId);
        }
        
        // If current node is target, source becomes connected
        if (targetId === currentId && !bfsVisited.has(sourceId)) {
          connectedAfterFiltering.add(sourceId);
          bfsVisited.add(sourceId);
          bfsQueue.push(sourceId);
        }
      });
    }
    
    // Find potentially disconnected nodes - these are nodes that passed relationship filtering
    // but have no remaining connections to the main word
    const disconnectedAfterFilteringNodes = relationshipFilteredNodes.filter(
      node => !connectedAfterFiltering.has(node.id) && node.id !== mainWord
    );
    
    console.log(`[FILTER] Found ${disconnectedAfterFilteringNodes.length} nodes that became disconnected due to filtering`);
    
    // Final node set depends on showDisconnectedNodes setting
    const finalNodes = showDisconnectedNodes ? 
      [...relationshipFilteredNodes] : // Keep all nodes that passed relationship filtering
      relationshipFilteredNodes.filter(node => connectedAfterFiltering.has(node.id)); // Only keep connected nodes
    
    // Log the final counts after all filtering
    console.log(`[FILTER] After relationship filtering: ${finalNodes.length}/${depthLimitedNodes.length} nodes and ${relationshipFilteredLinks.length}/${depthLimitedLinks.length} links`);
    console.log(`[FILTER] Connected nodes: ${connectedAfterFiltering.size}, Disconnected nodes: ${disconnectedAfterFilteringNodes.length}`);
    
    // Return the complete filtered data
    return { 
      filteredNodes: finalNodes, 
      filteredLinks: relationshipFilteredLinks 
    };
  }, [baseNodes, baseLinks, mainWord, depth, breadth, filteredRelationships, showDisconnectedNodes, mapRelationshipToGroup]);

  // Now create nodeMap after filteredNodes is defined
  const nodeMap = useMemo(() => {
      return new Map(filteredNodes.map(n => [n.id, n]));
  }, [filteredNodes]);

  // Add transitions for smooth animations
  useEffect(() => {
    // Force a complete rebuild when filters change
    setForceUpdate(prev => prev + 1);
  }, [filteredRelationships]);

  // Completely redesigned toggle filter handler - with clear state updates
  const handleToggleRelationshipFilter = useCallback((relationship: string) => {
    const relationshipLower = relationship.toLowerCase();
    
    console.log(`[FILTER] Toggling filter for '${relationshipLower}'`);
    
    setFilteredRelationships(prevFilters => {
      const isCurrentlyFiltered = prevFilters.includes(relationshipLower);
      let newFilters;
      
      if (isCurrentlyFiltered) {
        // Remove this relationship from filters
        console.log(`[FILTER] Removing '${relationshipLower}' from filters`);
        newFilters = prevFilters.filter(f => f !== relationshipLower);
      } else {
        // Add this relationship to filters
        console.log(`[FILTER] Adding '${relationshipLower}' to filters`);
        newFilters = [...prevFilters, relationshipLower];
      }
      
      console.log(`[FILTER] New filters:`, newFilters);
      return newFilters;
    });
    
    // Force complete graph rebuild
    setForceUpdate(prev => prev + 1);
  }, []);

  const getNodeRadius = useCallback((node: CustomNode) => {
    // Simplified, consistent sizing
    if (node.id === mainWord) return 20;
    if (node.group === 'root') return 16;
    return 13;
  }, [mainWord]);

  const setupZoom = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 8]) // Increased max zoom slightly
      .interpolate(d3.interpolateZoom)
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        if (!isDraggingRef.current) g.attr("transform", event.transform.toString());
      })
      .filter(event => !isDraggingRef.current && !isTransitioningRef.current && !event.ctrlKey && !event.button);
    svg.call(zoom);
    const initialTransform = d3.zoomIdentity.translate(width / 2, height / 2);
    svg.call(zoom.transform, initialTransform);
    return zoom;
  }, []);

  const ticked = useCallback(() => {
      if (!svgRef.current) return;
      const svg = d3.select(svgRef.current);
      const nodeSelection = svg.selectAll<SVGGElement, CustomNode>(".node");
      const linkSelection = svg.selectAll<SVGLineElement, CustomLink>(".link");
      const labelSelection = svg.selectAll<SVGTextElement, CustomNode>(".node-label"); // Select external labels

      // Update node group positions
      nodeSelection.attr("transform", d => `translate(${d.x ?? 0}, ${d.y ?? 0})`);

      // Update link line coordinates
      linkSelection
          .attr("x1", d => (typeof d.source === 'object' ? d.source.x ?? 0 : 0))
          .attr("y1", d => (typeof d.source === 'object' ? d.source.y ?? 0 : 0))
          .attr("x2", d => (typeof d.target === 'object' ? d.target.x ?? 0 : 0))
          .attr("y2", d => (typeof d.target === 'object' ? d.target.y ?? 0 : 0));

      // Update external text label positions (e.g., slightly below node)
      labelSelection
          .attr("x", d => d.x ?? 0)
          .attr("y", d => (d.y ?? 0) + getNodeRadius(d) + 12); // Adjust offset as needed

  }, [getNodeRadius]); // Added getNodeRadius dependency

  const setupSimulation = useCallback((nodes: CustomNode[], links: CustomLink[], width: number, height: number) => {
      simulationRef.current = d3.forceSimulation<CustomNode>(nodes)
        .alphaDecay(0.025) // Slightly slower decay for potentially better label settling
        .velocityDecay(0.4)
        .force("link", d3.forceLink<CustomNode, CustomLink>()
          .id(d => d.id)
          .links(links)
          .distance(110) // Moderate consistent distance
          .strength(0.4))
        .force("charge", d3.forceManyBody<CustomNode>().strength(-300).distanceMax(350)) // Slightly stronger charge for spacing
        // Increase collision radius significantly to account for text labels
        .force("collide", d3.forceCollide<CustomNode>().radius(d => getNodeRadius(d) + 25).strength(1.0))
        // Set simulation center to 0,0
        .force("center", d3.forceCenter(0, 0))
        .on("tick", ticked);

        return simulationRef.current;
  }, [getNodeRadius, ticked]);

  const createDragBehavior = useCallback((simulation: d3.Simulation<CustomNode, CustomLink>) => {
    return d3.drag<SVGGElement, CustomNode>()
      .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
          isDraggingRef.current = true;
          d3.select(event.sourceEvent.target.closest(".node")).classed("dragging", true);
          d3.selectAll(".link").filter((l: any) => l.source.id === d.id || l.target.id === d.id)
             .classed("connected-link", true)
             .raise();
        })
        .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          if (!d.pinned) { d.fx = null; d.fy = null; }
          isDraggingRef.current = false;
          d3.select(event.sourceEvent.target.closest(".node")).classed("dragging", false);
          d3.selectAll(".link.connected-link").classed("connected-link", false);
        });
  }, []);

  const createLinks = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>, linksData: CustomLink[]) => {
      // Draw links first (behind nodes and labels)
      const linkGroup = g.append("g")
      .attr("class", "links")
      .selectAll("line")
          .data(linksData, (d: any) => `${(typeof d.source === 'object' ? d.source.id : d.source)}_${(typeof d.target === 'object' ? d.target.id : d.target)}`)
          .join(
              enter => enter.append("line")
      .attr("class", "link")
                  .attr("stroke", theme === "dark" ? "#666" : "#ccc") // Consistent neutral color
                  .attr("stroke-opacity", 0) // Start transparent
                  .attr("stroke-width", 1.5) // Consistent width
                  .attr("stroke-linecap", "round")
                  .attr("x1", d => (typeof d.source === 'object' ? d.source.x ?? 0 : 0))
                  .attr("y1", d => (typeof d.source === 'object' ? d.source.y ?? 0 : 0))
                  .attr("x2", d => (typeof d.target === 'object' ? d.target.x ?? 0 : 0))
                  .attr("y2", d => (typeof d.target === 'object' ? d.target.y ?? 0 : 0))
                  // Add title element for link tooltip
                  .call(enter => enter.append("title").text((d: CustomLink) => d.relationship))
                  .call(enter => enter.transition().duration(300).attr("stroke-opacity", 0.6)), // Default opacity slightly higher
              update => update
                  // Ensure updates reset to default style before transitions
                  .attr("stroke", theme === "dark" ? "#666" : "#ccc")
                  .attr("stroke-width", 1.5)
                  .call(update => update.transition().duration(300)
                        .attr("stroke-opacity", 0.6)), // Transition opacity on update if needed
              exit => exit
                  .call(exit => exit.transition().duration(300).attr("stroke-opacity", 0))
                  .remove()
          );
      return linkGroup;
  }, [theme]);

  const createNodes = useCallback((
      g: d3.Selection<SVGGElement, unknown, null, undefined>,
      nodesData: CustomNode[],
      simulation: d3.Simulation<CustomNode, CustomLink> | null
      ) => {
    const drag = simulation ? createDragBehavior(simulation) : null;
    
    // Node groups (circles only)
    const nodeGroups = g.append("g")
      .attr("class", "nodes")
      .selectAll("g.node") // More specific selector
        .data(nodesData, (d: any) => (d as CustomNode).id)
      .join(
          enter => {
              const nodeGroup = enter.append("g")
                  .attr("class", d => `node node-group-${d.group} ${d.id === mainWord ? "main-node" : ""}`)
                  .attr("transform", d => `translate(${d.x ?? 0}, ${d.y ?? 0})`)
                  .style("opacity", 0); // Start transparent

              nodeGroup.append("circle")
      .attr("r", d => getNodeRadius(d))
      .attr("fill", d => getNodeColor(d.group))
                  // Subtle outline using darker shade of fill
                  .attr("stroke", d => d3.color(getNodeColor(d.group))?.darker(0.8).formatHex() ?? "#888")
                  .attr("stroke-width", 1.5);

              // NO internal text or title here

              nodeGroup.call(enter => enter.transition().duration(300).style("opacity", 1));
              return nodeGroup;
          },
          update => update,
          exit => exit
              .call(exit => exit.transition().duration(300).style("opacity", 0))
              .remove()
      );

    // External Labels (drawn after nodes/links)
    const labelGroup = g.append("g")
        .attr("class", "labels")
        .selectAll("text.node-label") // More specific selector
        .data(nodesData, (d: any) => (d as CustomNode).id)
        .join(
            enter => {
                const textElement = enter.append("text")
                    .attr("class", "node-label")
      .attr("text-anchor", "middle")
                    .attr("font-size", "10px") // Slightly larger base size for external text
                    .attr("font-weight", d => d.id === mainWord ? "600" : "400")
      .text(d => d.word)
                    .attr("x", d => d.x ?? 0) // Initial position
                    .attr("y", d => (d.y ?? 0) + getNodeRadius(d) + 12)
                    .style("opacity", 0) // Start transparent
                    .style("pointer-events", "none") // Prevent blocking node interactions
                    .style("user-select", "none");

                // Halo for contrast against background
                textElement.clone(true)
                    .lower()
                    .attr("fill", "none")
                    .attr("stroke", theme === "dark" ? "rgba(0,0,0,0.8)" : "rgba(255,255,255,0.8)")
                    .attr("stroke-width", 3)
                    .attr("stroke-linejoin", "round");

                // Set main text fill color based on theme
                textElement.attr("fill", theme === "dark" ? "#eee" : "#222");

                textElement.call(enter => enter.transition().duration(300).style("opacity", 1));
                return textElement;
            },
            update => update, // Could update text content if needed
            exit => exit
                .call(exit => exit.transition().duration(300).style("opacity", 0))
                .remove()
        );

      if (drag) nodeGroups.call(drag as any);
    return nodeGroups; // Return the node groups for interaction setup
  }, [createDragBehavior, getNodeRadius, getNodeColor, theme, mainWord]);

  const setupNodeInteractions = useCallback((
      nodeSelection: d3.Selection<d3.BaseType, CustomNode, SVGGElement, unknown>
  ) => {
      // Add double-click handler for navigation
      nodeSelection.on("dblclick", (event, d) => {
          event.preventDefault();
          event.stopPropagation();
          
          // Navigation: Make this node the new main word
          console.log(`Double-clicked word node: ${d.word}`);
          
          if (onNodeClick) {
            // Always pass the word text directly to the click handler for navigation
            console.log("Double-click - Making this the main word:", d.word);
            onNodeClick(d.word);
          }
      });
      
      // Add single-click handler for highlighting
      nodeSelection
        .on("click", (event, d) => {
          event.stopPropagation();
          if (isDraggingRef.current) return;
          
          console.log(`Single-clicked word node: ${d.word} - Highlighting related nodes`);

          // *** BEGIN ADDITION: Reset all styles before applying new ones ***
          d3.selectAll<SVGGElement, CustomNode>(".node")
            .classed("selected connected", false) // Remove classes
            .transition("reset_node_opacity").duration(150) // Use a named transition
            .style("opacity", 1) // Reset opacity
            .select("circle")
              .attr("stroke-width", 1.5) // Reset stroke width
              .attr("stroke", n => d3.color(getNodeColor(n.group))?.darker(0.8).formatHex() ?? "#888"); // Reset stroke color

          d3.selectAll<SVGLineElement, CustomLink>(".link")
            .classed("highlighted", false)
            .transition("reset_link_opacity").duration(150)
            .attr("stroke", theme === "dark" ? "#666" : "#ccc") // Reset stroke color
            .attr("stroke-opacity", 0.6) // Reset opacity
            .attr("stroke-width", 1.5); // Reset stroke width
          // *** END ADDITION ***
          
          const connectedIds = new Set<string>([d.id]);
          const connectedLinkElements: SVGLineElement[] = [];
           d3.selectAll<SVGLineElement, CustomLink>(".link").filter(l => {
               const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
               const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
               if (sourceId === d.id) { connectedIds.add(targetId); return true; }
               if (targetId === d.id) { connectedIds.add(sourceId); return true; }
               return false;
           }).each(function() { connectedLinkElements.push(this); });

          setSelectedNodeId(d.id);
          
          // Call onNodeSelect to update the word information panel
          if (onNodeSelect) {
            console.log("Updating details panel for:", d.word);
            onNodeSelect(d.word);
          }
          
          // Strong dimming of non-connected elements
          d3.selectAll<SVGGElement, CustomNode>(".node")
              .classed("selected connected", false)
              .filter(n => !connectedIds.has(n.id)) // Filter non-connected
              .transition("dim_node").duration(250)
              .style("opacity", 0.1);
          d3.selectAll<SVGLineElement, CustomLink>(".link")
              .classed("highlighted", false)
              // Filter non-connected links. Need to access link source/target IDs.
              .filter(l => {
                  const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
                  const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
                  return !(connectedIds.has(sourceId) && connectedIds.has(targetId));
              })
              .transition("dim_link").duration(250)
              .attr("stroke", theme === "dark" ? "#444" : "#ddd") // Very faint stroke color
              .attr("stroke-opacity", 0.05)
              .attr("stroke-width", 1.0);

          // Highlight selected node and connected nodes
          const targetNodeElement = d3.select(event.currentTarget as Element);
          targetNodeElement.classed("selected", true)
              .transition("highlight_node").duration(250)
              .style("opacity", 1)
              .select("circle") // Ensure border highlight is applied
                  .attr("stroke-width", 2.5)
                  .attr("stroke", d3.color(getNodeColor(d.group))?.brighter(0.8).formatHex() ?? (theme === "dark" ? "#eee" : "#333"));

          d3.selectAll<SVGGElement, CustomNode>(".node")
              .filter(n => connectedIds.has(n.id) && n.id !== d.id) // Connected but not the clicked one
              .classed("connected", true)
              .transition("highlight_node").duration(250)
              .style("opacity", 1)
              .select("circle") // Reset border if it was dimmed
                   .attr("stroke", n => d3.color(getNodeColor(n.group))?.darker(0.8).formatHex() ?? "#888")
                   .attr("stroke-width", 1.5);

           // Highlight connected links
           d3.selectAll<SVGLineElement, CustomLink>(connectedLinkElements)
            .classed("highlighted", true)
            .raise()
            .transition("highlight_link").duration(250)
            .attr("stroke", (l: CustomLink) => { // Color link based on neighbour
                const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
                const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
                const neighbourNode = sourceId === d.id ? nodeMap.get(targetId) : nodeMap.get(sourceId);
                return neighbourNode ? getNodeColor(neighbourNode.group) : (theme === "dark" ? "#aaa" : "#666");
            })
            .attr("stroke-opacity", 0.9)
            .attr("stroke-width", 2.5);
        })
        .on("mouseover", (event, d) => {
            if (isDraggingRef.current) return;
            if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);

            // Keep only the tooltip logic
            const timeoutId = setTimeout(() => setHoveredNode({ ...d }), 200);
            setTooltipTimeoutId(timeoutId);
        })
        .on("mouseout", (event, d_unknown) => {
            if (isDraggingRef.current) return;
            if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);
            setHoveredNode(null);

            // No style changes needed here
        });
  }, [selectedNodeId, onNodeClick, getNodeRadius, getNodeColor, theme, nodeMap, /* tooltipTimeoutId, */ onNodeSelect]); // REMOVED tooltipTimeoutId dependency

  // Define handleResetZoom before centerOnMainWord
  const handleResetZoom = useCallback(() => {
    if (zoomRef.current && svgRef.current) {
       const svg = d3.select(svgRef.current);
       const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
       const width = containerRect ? containerRect.width : 800;
       const height = containerRect ? containerRect.height : 600;
       const resetTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
       svg.transition().duration(600).ease(d3.easeCubicInOut)
         .call(zoomRef.current.transform, resetTransform);
     }
  }, []);

  const centerOnMainWord = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, nodesToSearch: CustomNode[]) => {
    if (!zoomRef.current || isDraggingRef.current || !mainWord) return;
    const mainNodeData = nodesToSearch.find(n => n.id === mainWord);
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    if (mainNodeData && mainNodeData.x !== undefined && mainNodeData.y !== undefined) {
      const currentTransform = d3.zoomTransform(svg.node()!);
      const targetScale = Math.max(0.5, Math.min(2, currentTransform.k));
      const targetX = width / 2 - mainNodeData.x * targetScale;
      const targetY = height / 2 - mainNodeData.y * targetScale;
      const newTransform = d3.zoomIdentity.translate(targetX, targetY).scale(targetScale);
      svg.transition().duration(750).ease(d3.easeCubicInOut)
         .call(zoomRef.current.transform, newTransform);
    } else {
      // Direct reset instead of calling handleResetZoom
      const resetTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
      svg.transition().duration(600).ease(d3.easeCubicInOut)
        .call(zoomRef.current.transform, resetTransform);
    }
  }, [mainWord]);

  const setupSvgDimensions = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);
    return { width, height };
  }, []);

  useEffect(() => {
    // --- START OF TRY BLOCK ---
    try {
    if (!svgRef.current || !wordNetwork || !mainWord || baseNodes.length === 0) {
      if (svgRef.current) d3.select(svgRef.current).selectAll("*").remove();
      if (simulationRef.current) simulationRef.current.stop();
      setError(null);
      setIsLoading(false);
      return;
    }

      // Declare svg variable ONCE at the top of the effect
      const svg = d3.select(svgRef.current);

    setIsLoading(true);
    setError(null);
    console.log("[GRAPH] Building graph with filtered data");
    console.log(`[GRAPH] Using ${filteredNodes.length} nodes and ${filteredLinks.length} links`);

    if (simulationRef.current) simulationRef.current.stop();
      
    // Clear previous content
    svg.selectAll("*").remove();

    // Set up dimensions and append main group
    const { width, height } = setupSvgDimensions(svg);
    const g = svg.append("g")
      .attr("class", "graph-content")
      .attr("data-filter-key", filteredRelationships.join(','));
    
    // Configure zoom behavior
    const zoom = setupZoom(svg, g);
    zoomRef.current = zoom;

    // Initialize and configure the simulation
    const currentSim = setupSimulation(filteredNodes, filteredLinks, width, height);

    // Create the graph elements
    createLinks(g, filteredLinks);
    const nodeElements = createNodes(g, filteredNodes, currentSim);
    setupNodeInteractions(nodeElements);

    // Pin main node to center if needed
    if (currentSim) {
      const mainNodeData = filteredNodes.find(n => n.id === mainWord);
      if (mainNodeData) {
        mainNodeData.fx = 0;
        mainNodeData.fy = 0;
      }
      currentSim.alpha(1).restart();
    }

    // Center the view on the main word after a brief delay
    setTimeout(() => centerOnMainWord(svg, filteredNodes), 800);

    // Create legend for relationship types
    // [Legend creation code...]

    setIsLoading(false);
      
    // Set up cleanup function
    return () => {
      if (currentSim) currentSim.stop();
      clearTimeout(centerTimeout);
     
      // Remove all event handlers to prevent memory leaks
      if (svgRef.current) {
        const svg = d3.select(svgRef.current);
        svg.selectAll(".node").on("click", null).on("dblclick", null).on("mouseover", null).on("mouseout", null);
        svg.selectAll(".legend-item rect").on("click", null).on("mouseover", null).on("mouseout", null);
      }
    };
  } catch (err) {
    console.error("[D3_EFFECT_ERROR] Error during D3 graph rendering:", err);
    setError(err instanceof Error ? err.message : "An unknown error occurred during graph rendering.");
    setIsValidNetwork(false);
    setIsLoading(false);
    
    // Ensure cleanup runs even on error
    if (simulationRef.current) {
      simulationRef.current.stop();
    }
    if (svgRef.current) {
      d3.select(svgRef.current).selectAll("*").remove();
      if (zoomRef.current) {
        d3.select(svgRef.current).on(".zoom", null);
      }
    }
  }
}, [
  wordNetwork,
  mainWord,
  depth,
  breadth,
  theme, 
  mapRelationshipToGroup,
  getNodeColor, 
  getNodeRadius,
  setupZoom,
  ticked,
  setupSimulation,
  createDragBehavior,
  createLinks, 
  createNodes, 
  setupNodeInteractions, 
  centerOnMainWord,
  setupSvgDimensions,
  filteredNodes,
  filteredLinks,
  filteredRelationships,
  filterUpdateKey,
  forceUpdate,
  handleToggleRelationshipFilter,
  showDisconnectedNodes
]);

  useEffect(() => {
    if (prevMainWordRef.current && prevMainWordRef.current !== mainWord && svgRef.current) {
        const recenterTimeout = setTimeout(() => {
            if(svgRef.current) centerOnMainWord(d3.select(svgRef.current), filteredNodes);
        }, 800);
         return () => clearTimeout(recenterTimeout);
    }
    prevMainWordRef.current = mainWord;
  }, [mainWord, centerOnMainWord, filteredNodes]);

  const handleDepthChange = (newDepth: number) => {
    setDepth(newDepth);
  };

  const handleBreadthChange = (newBreadth: number) => {
    setBreadth(newBreadth); 
  };

  const handleZoom = useCallback((scale: number) => {
      if (zoomRef.current && svgRef.current) {
         d3.select(svgRef.current).transition().duration(300).ease(d3.easeCubicInOut)
           .call(zoomRef.current.scaleBy, scale);
       }
  }, []);

  const renderTooltip = useCallback(() => {
    // Tooltip visibility is now handled by hoveredNode state directly (set via timeout)
    if (!hoveredNode?.id || !hoveredNode?.x || !hoveredNode?.y || !svgRef.current) return null;

    const svgNode = svgRef.current;
    const transform = d3.zoomTransform(svgNode);

    const [screenX, screenY] = transform.apply([hoveredNode.x, hoveredNode.y]);

    const offsetX = (screenX > window.innerWidth / 2) ? -20 - 250 : 20;
    const offsetY = (screenY > window.innerHeight / 2) ? -20 - 80 : 20;
      
      return (
        <div
          className="node-tooltip"
          style={{
            position: "absolute",
            left: `${screenX + offsetX}px`,
            top: `${screenY + offsetY}px`,
            background: theme === "dark" ? "rgba(30, 30, 30, 0.9)" : "rgba(250, 250, 250, 0.9)",
            border: `1.5px solid ${getNodeColor(hoveredNode.group)}`, borderRadius: "8px",
            padding: "10px 14px", maxWidth: "280px", zIndex: 1000, pointerEvents: "none",
            fontFamily: "system-ui, -apple-system, sans-serif",
            transition: "left 0.1s ease-out, top 0.1s ease-out, opacity 0.1s ease-out",
            boxShadow: theme === "dark" ? "0 4px 15px rgba(0,0,0,0.4)" : "0 4px 15px rgba(0,0,0,0.15)",
            opacity: 1,
          }}
        >
           <h4 style={{ margin: 0, marginBottom: '6px', color: getNodeColor(hoveredNode.group), fontSize: '15px' }}>{hoveredNode.id}</h4>
           <div style={{ display: "flex", alignItems: "center", gap: "6px", paddingBottom: "4px" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: getNodeColor(hoveredNode.group), flexShrink: 0 }}></span>
              <span style={{ fontSize: "13px", color: theme === 'dark' ? '#ccc' : '#555', fontWeight: "500" }}>
                  {hoveredNode.group.charAt(0).toUpperCase() + hoveredNode.group.slice(1).replace(/_/g, ' ')}
              </span>
          </div>
           {hoveredNode.definitions && hoveredNode.definitions.length > 0 && (
                <p style={{ fontSize: '12px', color: theme === 'dark' ? '#bbb' : '#666', margin: '6px 0 0 0', fontStyle: 'italic' }}>
                    {hoveredNode.definitions[0].length > 100 ? hoveredNode.definitions[0].substring(0, 97) + '...' : hoveredNode.definitions[0]}
            </p>
          )}
           <div style={{ fontSize: "11px", marginTop: "8px", color: theme === "dark" ? "#8b949e" : "#777777" }}>
               Click to focus | Double-click to pin/unpin
          </div>
        </div>
      );
  }, [hoveredNode, theme, getNodeColor]);

// Function to get the relationship type label with meaningful groups
const getRelationshipTypeLabel = (type: string): { group: string, label: string } => {
  // First, normalize specific subtypes to their general type
  const normalizedType = (() => {
    // Origin/root group normalization
    if (['root_of', 'isahod', 'derived_from', 'sahod'].includes(type.toLowerCase())) return 'root';
    
    // Meaning group normalization
    if (['kaugnay', 'kahulugan', 'similar'].includes(type.toLowerCase())) return 'related';
    if (['kasalungat'].includes(type.toLowerCase())) return 'antonym';
    
    // Form group normalization
    if (['regional_variant', 'atapat', 'inatapat', 'itapat'].includes(type.toLowerCase())) return 'variant';
    
    // Structure group normalization
    if (['component', 'component_of'].includes(type.toLowerCase())) return 'part_whole';
    
    // Derivation group normalization 
    if (['affix', 'derivative'].includes(type.toLowerCase())) return 'derived';
    
    return type;
  })();

  // Now use the normalized type for labels
  switch (normalizedType.toLowerCase()) {
    // Core
    case 'main': return { group: 'Core', label: 'Main Word' };
    
    // Origin/Root group
    case 'root': return { group: 'Origin', label: 'Root/Origin' };
    case 'derived': return { group: 'Origin', label: 'Derived' };
    case 'etymology': return { group: 'Origin', label: 'Etymology' };
    case 'cognate': return { group: 'Origin', label: 'Cognate' };
    
    // Meaning group
    case 'synonym': return { group: 'Meaning', label: 'Synonym' };
    case 'antonym': return { group: 'Meaning', label: 'Antonym' };
    case 'related': return { group: 'Meaning', label: 'Related' };
    
    // Form group
    case 'variant': return { group: 'Form', label: 'Variant' };
    
    // Structure group
    case 'taxonomic': return { group: 'Structure', label: 'Taxonomic' };
    case 'part_whole': return { group: 'Structure', label: 'Components/Parts' };
    
    // Info group
    case 'usage': return { group: 'Info', label: 'Usage Note' };
    
    // Fallback
    default: return { group: 'Other', label: type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) };
  }
};

  if (!isValidNetwork) {
    return (
      <div className="graph-container">
        <div className="error-overlay">
          <p className="error-message">Invalid network data structure. Please try again.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="graph-container">
      <div className="graph-svg-container">
        {isLoading && ( // Use isLoading here
          <div className="loading-overlay"><div className="spinner"></div><p>Loading...</p></div>
        )}
        {error && (
          <div className="error-overlay">
            <p className="error-message">{error}</p>
          </div>
        )}
        {(!wordNetwork || !mainWord || filteredNodes.length === 0 && !error) && (
          <div className="empty-graph-message">Enter a word to see its network.</div>
        )}
        <svg 
          ref={svgRef} 
          className={`graph-svg ${isLoading ? 'loading' : 'loaded'}`}
          key={`graph-${mainWord}-${depth}-${breadth}-${filteredRelationships.join('.')}-${forceUpdate}-${filterUpdateKey}`}
        >
        </svg>
      </div>
      <div className="controls-container">
        <div className="zoom-controls">
          <button onClick={() => handleZoom(1.3)} className="zoom-button" title="Zoom In">+</button>
          <button onClick={() => handleZoom(1 / 1.3)} className="zoom-button" title="Zoom Out">-</button>
          <button onClick={handleResetZoom} className="zoom-button" title="Reset View">Reset</button>
        </div>
        <NetworkControls 
          depth={depth}
          breadth={breadth}
          onDepthChange={handleDepthChange}
          onBreadthChange={handleBreadthChange}
          onChangeCommitted={(_d, _b) => onNetworkChange(depth, breadth)} // Use state variables directly
          className="network-controls"
        />
      </div>
      {renderTooltip()}
    </div>
  );
};

export default React.memo(WordGraph);
