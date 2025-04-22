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
// Import color utility functions
import { 
  mapRelationshipToGroup, 
  getNodeColor, 
  getTextColorForBackground,
  getRelationshipTypeLabel
} from '../utils/colorUtils';

// *** ADD MUI Drawer/Button/Icon imports ***
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import Box from '@mui/material/Box';
// Import an icon, e.g., TuneIcon or SettingsIcon
import TuneIcon from '@mui/icons-material/Tune'; // Or any other suitable icon
// *** ADD MUI List imports for mobile legend ***
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Checkbox from '@mui/material/Checkbox';
import ListSubheader from '@mui/material/ListSubheader';
import Divider from '@mui/material/Divider';

interface WordGraphProps {
  wordNetwork: WordNetwork | null;
  mainWord: string | null;
  onNodeClick: (word: string) => void;
  onNodeSelect: (word: string) => void;
  onNetworkChange: (depth: number, breadth: number) => void;
  initialDepth: number;
  initialBreadth: number;
  isMobile: boolean; // *** Add isMobile prop ***
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
  relationshipToMain?: string; // Add this property for tooltip relationship info
}

interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
  relationship: string;
  source: string | CustomNode;
  target: string | CustomNode;
}

const WordGraph: React.FC<WordGraphProps> = ({
  wordNetwork,
  mainWord,
  onNodeClick,
  onNodeSelect,
  onNetworkChange,
  initialDepth,
  initialBreadth,
  isMobile // *** Destructure isMobile ***
}) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);

  // Log the received mainWord prop
  useEffect(() => {
    console.log(`[WordGraph] Received mainWord prop: '${mainWord}'`);
  }, [mainWord]);

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

  // *** ADD Drawer state ***
  const [controlsOpen, setControlsOpen] = useState(false);

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

  // Add a new click timeout ref
  const clickTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastClickTimeRef = useRef<number>(0);
  const lastClickedNodeRef = useRef<string | null>(null);

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
      
      // Log node details being processed
      // console.log(`[BASE] Processing node: ID=${node.id}, Label='${node.label}', Word='${node.word}'`); 

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
  }, [wordNetwork, mainWord, baseLinks]); // Removed mapRelationshipToGroup dependency as it's now external

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
  }, [baseNodes, baseLinks, mainWord, depth, breadth, filteredRelationships, showDisconnectedNodes]);

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
      .scaleExtent([0.1, 4]) // Increased max zoom slightly
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
    let dragStartTime = 0;
    
    return d3.drag<SVGGElement, CustomNode>()
      .filter(event => {
          // Only initiate drag on primary mouse button
          return !event.ctrlKey && event.button === 0;
      })
      .on("start", (event, d) => {
          // Record when drag started (to distinguish from clicks)
          dragStartTime = Date.now();
          
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
          isDraggingRef.current = true;
          
          // Mark as dragging
          d3.select(event.sourceEvent.target.closest(".node"))
            .classed("dragging", true)
            .select("circle")
            .attr("stroke-dasharray", "3,2");
      })
      .on("drag", (event, d) => { 
          d.fx = event.x; 
          d.fy = event.y; 
      })
      .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          if (!d.pinned) { d.fx = null; d.fy = null; }
          
          // Reset visual state
          d3.select(event.sourceEvent.target.closest(".node"))
            .classed("dragging", false)
            .select("circle")
            .attr("stroke-dasharray", null);
          
          // Calculate how long the drag lasted
          const dragDuration = Date.now() - dragStartTime;
          
          // If drag was very short, don't interfere with click events
          if (dragDuration < 150) {
          isDraggingRef.current = false;
          } else {
            // For longer drags, delay clearing the flag
            setTimeout(() => {
                isDraggingRef.current = false;
            }, 150);
          }
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

  // Optimize hover effect and enhance visual feedback
  const setupNodeInteractions = useCallback((
      nodeSelection: d3.Selection<d3.BaseType, CustomNode, SVGGElement, unknown>
  ) => {
      // Clear all event handlers first
      nodeSelection.on(".click", null)
                  .on(".dblclick", null)
                  .on("mousedown", null)
                  .on("mouseup", null)
                  .on("mouseenter", null)
                  .on("mouseleave", null)
                  .on("mouseover", null)
                  .on("mouseout", null);

      // Enhanced hover effect to show relationship
      nodeSelection.on("mouseenter", function(event, d) {
          if (isDraggingRef.current) return;
          
          // Emphasize this node
          d3.select(this)
            .raise() // Bring to front
            .select("circle")
              .attr("stroke-width", 3)
              .attr("stroke-opacity", 1)
              .attr("filter", "brightness(1.1)");
            
          // Find and highlight connections
          const connectedIds = new Set<string>([d.id]);
          
          // Temporarily highlight connections to this node
          const connectedLinks = d3.selectAll<SVGLineElement, CustomLink>(".link").filter((l: CustomLink) => {
               const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
               const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
              
              // Add connected nodes to set
              if (sourceId === d.id) connectedIds.add(targetId);
              if (targetId === d.id) connectedIds.add(sourceId);
              
              return sourceId === d.id || targetId === d.id;
          });
          
          // Raise and highlight links
          connectedLinks
            .raise()
            .attr("stroke-opacity", 0.85)
            .attr("stroke-width", 2.5)
            .attr("stroke", function(l: CustomLink) {
                  const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
                  const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
                
                // Color based on connected node
                const connectedId = sourceId === d.id ? targetId : sourceId;
                const connectedNode = nodeMap.get(connectedId);
                
                // Use the relationship type to determine color
                return connectedNode ? getNodeColor(connectedNode.group) : (theme === "dark" ? "#aaa" : "#666");
            });
          
          // Highlight connected nodes
          d3.selectAll<SVGGElement, CustomNode>(".node")
            .filter(n => connectedIds.has(n.id) && n.id !== d.id)
            .raise() // Bring to front
              .style("opacity", 1)
            .select("circle")
              .attr("stroke-width", 2)
              .attr("stroke-opacity", 0.9);
              
          // Also highlight connected node labels
          d3.selectAll<SVGTextElement, CustomNode>(".node-label")
            .filter(n => connectedIds.has(n.id))
            .style("opacity", 1)
            .style("font-weight", "bold");
      });
      
      nodeSelection.on("mouseleave", function(event, d) {
          if (isDraggingRef.current) return;
          
          // Reset appearance on mouseout
          d3.select(this).select("circle")
            .attr("stroke-width", d.id === mainWord ? 2.5 : 1.5)
            .attr("stroke-opacity", 0.7)
            .attr("filter", d.id === mainWord ? "brightness(1.15)" : "none")
            .attr("stroke", d3.color(getNodeColor(d.group))?.darker(0.8).formatHex() ?? "#888");
          
          // Reset connected links
          d3.selectAll<SVGLineElement, CustomLink>(".link")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", 1.5)
            .attr("stroke", theme === "dark" ? "#666" : "#ccc");
            
          // Reset connected nodes 
          d3.selectAll<SVGGElement, CustomNode>(".node")
            .style("opacity", n => n.id === mainWord ? 1 : 0.8)
            .select("circle")
              .attr("stroke-width", n => n.id === mainWord ? 2.5 : 1.5)
              .attr("stroke-opacity", 0.7);
              
          // Reset node labels
          d3.selectAll<SVGTextElement, CustomNode>(".node-label")
            .style("opacity", 0.9)
            .style("font-weight", n => n.id === mainWord ? "bold" : "normal");
      });
      
      // Double-click to navigate with improved reliability
      nodeSelection.on("dblclick", function(event, d) {
          event.preventDefault();
          event.stopPropagation();
          
          if (isDraggingRef.current) return;
          
          // Clear any potential timeouts
          if (clickTimeoutRef.current) {
              clearTimeout(clickTimeoutRef.current);
              clickTimeoutRef.current = null;
          }
          
          // Visual effect prior to navigation
          const circleElement = d3.select(this).select("circle");
          const originalFill = circleElement.attr("fill");
          
          // Pulse effect to indicate navigation
          circleElement
            .attr("fill-opacity", 0.7)
            .attr("r", function() { return parseFloat(d3.select(this).attr("r")) * 1.2; });
            
          setTimeout(() => {
              // Reset visual state (though navigation will likely reload the component)
              circleElement
                .attr("fill-opacity", 1)
                .attr("fill", originalFill)
                .attr("r", getNodeRadius(d));
                
              console.log(`Double-click on node: ${d.word} - Navigating`);
              
              // Navigate after visual feedback
              if (onNodeClick) {
                  onNodeClick(d.word);
              }
          }, 120);
      });
      
      // Enhanced tooltip with relationship info
      nodeSelection.on("mouseover", (event, d) => {
            if (isDraggingRef.current) return;
            if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);

          // Check if this is directly connected to main word
          let relationshipToMain = "";
          if (d.id !== mainWord) {
              const link = baseLinks.find(l => 
                  (l.source === mainWord && l.target === d.id) || 
                  (l.source === d.id && l.target === mainWord)
              );
              if (link) {
                  relationshipToMain = link.relationship;
              }
          }
          
          // Pass relationship to tooltip
          setHoveredNode({ 
              ...d, 
              relationshipToMain 
          });
      });
      
      nodeSelection.on("mouseout", (event, d) => {
            if (isDraggingRef.current) return;
            if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);
            setHoveredNode(null);
      });
  }, [mainWord, onNodeClick, getNodeColor, theme, nodeMap, baseLinks, getNodeRadius]);

  // Update initial node layout to ensure main word is centered and visible
  useEffect(() => {
    if (simulationRef.current && mainWord) {
      // Find the main word node data
      const mainNodeData = filteredNodes.find(n => n.id === mainWord);
      if (mainNodeData) {
        // Pin main word node at the center
        mainNodeData.fx = 0;
        mainNodeData.fy = 0;
        // Reset simulation to apply new positions
        simulationRef.current.alpha(0.3).restart();
      }
    }
  }, [filteredNodes, mainWord]);

  // Improved tooltip with relationship info
  const renderTooltip = useCallback(() => {
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
          background: theme === "dark" ? "rgba(30, 30, 30, 0.95)" : "rgba(250, 250, 250, 0.95)",
          border: `1.5px solid ${getNodeColor(hoveredNode.group)}`, 
          borderRadius: "8px",
          padding: "10px 14px", 
          maxWidth: "280px", 
          zIndex: 1000, 
          pointerEvents: "none",
          fontFamily: "system-ui, -apple-system, sans-serif",
          transition: "opacity 0.15s ease-out, transform 0.15s ease-out",
          boxShadow: theme === "dark" ? "0 4px 15px rgba(0,0,0,0.4)" : "0 4px 15px rgba(0,0,0,0.15)",
          opacity: 1,
          transform: "translateY(0)",
          animation: "fadeInTooltip 0.2s ease-out",
        }}
      >
         <h4 style={{ margin: 0, marginBottom: '6px', color: getNodeColor(hoveredNode.group), fontSize: '15px' }}>{hoveredNode.id}</h4>
         
         {/* Relationship to main word */}
         {hoveredNode.id !== mainWord && (hoveredNode as any).relationshipToMain && (
           <div style={{ 
             display: "flex", 
             alignItems: "center", 
             gap: "6px", 
             paddingBottom: "4px",
             background: theme === "dark" ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.03)",
             padding: "5px 8px",
             borderRadius: "4px",
             marginBottom: "5px"
           }}>
             <span style={{ 
               fontSize: "11px", 
               color: theme === "dark" ? "#aaa" : "#666", 
               fontWeight: "500",
               whiteSpace: "nowrap"
             }}>
               {mainWord} 
               <span style={{ margin: "0 4px", opacity: 0.7 }}>→</span> 
               <span style={{ 
                 fontStyle: "italic", 
                 color: theme === "dark" ? "#ddd" : "#333",
                 fontWeight: "600" 
               }}>
                 {(hoveredNode as any).relationshipToMain}
               </span>
             </span>
           </div>
         )}
         
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
         <div style={{ 
           fontSize: "11px", 
           marginTop: "8px", 
           color: theme === "dark" ? "#8b949e" : "#777777",
           display: "flex",
           justifyContent: "center",
           gap: "12px",
           borderTop: theme === "dark" ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.06)",
           paddingTop: "6px"
         }}>
           <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
             <span style={{ 
               fontSize: "10px", 
               background: theme === "dark" ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.06)", 
               borderRadius: "3px", 
               padding: "1px 4px"
             }}>Double-click</span>
             <span>Navigate</span>
           </div>
        </div>
      </div>
    );
  }, [hoveredNode, theme, getNodeColor, mainWord]);

  // Improved visual styling
  useEffect(() => {
    const styleId = "graph-interaction-styles";
    if (!document.getElementById(styleId)) {
      const styleElement = document.createElement("style");
      styleElement.id = styleId;
      styleElement.textContent = `
        .node {
          cursor: pointer;
          opacity: 0.8;
          transition: opacity 0.2s ease-out;
        }
        .node[data-id="${mainWord}"] {
          opacity: 1;
        }
        .node[data-id="${mainWord}"] circle {
          stroke-width: 2.5px !important;
          filter: brightness(1.15);
        }
        .node-label {
          pointer-events: none;
          transition: opacity 0.2s ease-out, font-weight 0.2s ease-out;
        }
        .node[data-id="${mainWord}"] + .node-label {
          font-weight: bold;
          opacity: 1;
        }
        .link {
          stroke-opacity: 0.6;
          transition: stroke-opacity 0.2s ease-out, stroke-width 0.2s ease-out;
        }
        .node.dragging {
          cursor: grabbing;
        }
        
        /* Graph tooltip keyframe animation */
        @keyframes fadeInTooltip {
          from { opacity: 0; transform: translateY(5px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        /* Add a subtle highlight effect on node hover */
        .node:hover circle {
          filter: brightness(1.1);
        }
      `;
      document.head.appendChild(styleElement);
    }
    
    return () => {
      const existingStyle = document.getElementById(styleId);
      if (existingStyle) {
        existingStyle.remove();
      }
    };
  }, [mainWord]);

  // Performance optimization: Debounce window resize events
  useEffect(() => {
    let resizeTimer: number | null = null;
    
    const handleResize = () => {
      if (resizeTimer) window.clearTimeout(resizeTimer);
      
      resizeTimer = window.setTimeout(() => {
        if (svgRef.current) {
          // Resize the SVG and adjust positions
          const containerRect = svgRef.current.parentElement?.getBoundingClientRect();
          if (containerRect) {
            const { width, height } = containerRect;
            d3.select(svgRef.current)
              .attr("width", width)
              .attr("height", height);
            
            // Force layout update
            if (simulationRef.current) {
              simulationRef.current.alpha(0.3).restart();
            }
          }
        }
      }, 200);
    };
    
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (resizeTimer) window.clearTimeout(resizeTimer);
    };
  }, []);

  // Ensure the main word node is always highlighted
  useEffect(() => {
    if (svgRef.current && mainWord) {
      // Update node styling for main word
      d3.select(svgRef.current)
        .selectAll(".node")
        .attr("data-id", d => (d as any).id)
        .classed("main-word", d => (d as any).id === mainWord);
        
      // Update label styling
      d3.select(svgRef.current)
        .selectAll(".node-label")
        .attr("data-id", d => (d as any).id)
        .style("font-weight", d => (d as any).id === mainWord ? "bold" : "normal")
        .style("opacity", d => (d as any).id === mainWord ? 1 : 0.9);
    }
  }, [mainWord, filteredNodes]);

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
      
      // Use the declared svg variable
      svg.selectAll("*").remove();

    const { width, height } = setupSvgDimensions(svg);
    const g = svg.append("g")
      .attr("class", "graph-content")
      .attr("data-filter-key", filteredRelationships.join(','));
      const zoom = setupZoom(svg, g);
      zoomRef.current = zoom;

    const currentSim = setupSimulation(filteredNodes, filteredLinks, width, height);

    createLinks(g, filteredLinks);
    const nodeElements = createNodes(g, filteredNodes, currentSim);
    setupNodeInteractions(nodeElements);

    if (currentSim) {
      const mainNodeData = filteredNodes.find(n => n.id === mainWord);
      if (mainNodeData) {
          mainNodeData.fx = 0;
          mainNodeData.fy = 0;
      }
      currentSim.alpha(1).restart();
    }

      // Use the declared svg variable
    setTimeout(() => centerOnMainWord(svg, filteredNodes), 800);

    // Conditionally render the legend only on desktop
    if (!isMobile) {
      // Create a more elegant single-column legend with auto-sizing
      const legendContainer = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${Math.max(width - legendWidth - 20, 10)}, 20)`); // Responsive positioning

      // Define refined legend properties
      const legendPadding = 12; // Restored original padding
      const legendItemHeight = 22; 
      const dotRadius = 5; 
      const textPadding = 16; // Restored original padding 
      const categorySpacing = 10; 
      const maxLabelWidth = 110; // Restored original width

      // Organize relation types by category - ensure each type appears in only one category
      const categories = [
        { name: "Core", types: ["main"] },
        { name: "Origin", types: ["root", "root_of", "etymology"] },
        { name: "Derived", types: ["derived", "derived_from", "derivative"] },
        { name: "Meaning", types: ["synonym", "antonym", "related", "similar"] },
        { name: "Cultural", types: ["kaugnay", "kahulugan", "kasalungat"] },
        { name: "Form", types: ["variant", "spelling_variant", "regional_variant", "abbreviation", "form_of"] },
        { name: "Translation", types: ["itapat", "atapat", "inatapat"] },
        { name: "Structure", types: ["hypernym", "hyponym", "meronym", "holonym", "taxonomic"] },
        { name: "Part-Whole", types: ["part_whole", "component", "component_of"] },
        { name: "Other", types: ["affix", "usage", "associated", "other"] }
    ];

    // Pre-measure text to determine legend width
    const tempText = svg.append("text")
      .attr("font-size", "11px") // Slightly larger text
      .attr("font-weight", "500")
      .style("opacity", 0);

    let maxTextWidth = 0;
    let maxCategoryWidth = 0;
    
    // Also measure toggle text to ensure it fits
    tempText.text("Show disconnected nodes");
    const toggleTextWidth = tempText.node()?.getBBox().width || 0;

    // Measure category headers
    categories.forEach(category => {
      tempText.text(category.name);
      const categoryWidth = tempText.node()?.getBBox().width || 0;
      maxCategoryWidth = Math.max(maxCategoryWidth, categoryWidth);
      
      // Measure each label
      category.types.forEach(type => {
        tempText.text(getRelationshipTypeLabel(type).label);
        const textWidth = tempText.node()?.getBBox().width || 0;
        maxTextWidth = Math.max(maxTextWidth, Math.min(textWidth, maxLabelWidth));
      });
    });

    tempText.remove();

    // Calculate legend dimensions based on text measurements
    // Be more conservative with width calculations
    const toggleRequiredWidth = toggleTextWidth + 42 + (legendPadding * 2); // Further reduced spacing
    const legendWidth = Math.max(
      maxCategoryWidth + 5, // Minimal extra space
      maxTextWidth + textPadding + 5, // Minimal extra space
      toggleRequiredWidth
    ) + (legendPadding * 2);

    // Find total rows for legend layout
    let totalRows = 0;
    categories.forEach(cat => {
      // Each category needs 1 row for header + rows for items
      totalRows += 1 + cat.types.length;
    });

    // Calculate legend height with more spacing
    const legendHeight = (totalRows * legendItemHeight) + 
                        ((categories.length - 1) * categorySpacing) + 
                        (legendPadding * 2) + 
                        50 + // Add extra padding for the title and instructions
                        40; // Add extra space for the checkbox option

    // Add refined legend background rectangle
      legendContainer.append("rect")
        .attr("width", legendWidth)
      .attr("height", legendHeight)
      .attr("rx", 10)
      .attr("ry", 10)
      .attr("fill", theme === "dark" ? "rgba(28, 30, 38, 0.85)" : "rgba(255, 255, 255, 0.92)")
      .attr("stroke", theme === "dark" ? "rgba(255, 255, 255, 0.12)" : "rgba(0, 0, 0, 0.06)")
      .attr("stroke-width", 1);

    // Add elegant title with original font size
      legendContainer.append("text")
        .attr("x", legendWidth / 2)
      .attr("y", legendPadding + 7)
        .attr("text-anchor", "middle")
      .attr("font-weight", "600")
      .attr("font-size", "12px") // Restored original size
      .attr("fill", theme === "dark" ? "#eee" : "#333")
        .text("Relationship Types");
      
    // Add subtitle with instructions
    legendContainer.append("text")
      .attr("x", legendWidth / 2)
      .attr("y", legendPadding + 22)
      .attr("text-anchor", "middle")
      .attr("font-weight", "400")
      .attr("font-size", "9px")
      .attr("fill", theme === "dark" ? "#aaa" : "#666")
      .text("Click to filter by type");
    
    // Add subtle divider line after title
    legendContainer.append("line")
      .attr("x1", legendPadding)
      .attr("y1", legendPadding + 30)
      .attr("x2", legendWidth - legendPadding)
      .attr("y2", legendPadding + 30)
      .attr("stroke", theme === "dark" ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.06)")
      .attr("stroke-width", 1);

    // Track current y position for legend items
    let yPos = legendPadding + 38; // More space after title and instructions

    // Render each category
    categories.forEach((category, categoryIndex) => {
      // Add category header with refined styling
      yPos += legendItemHeight;
      
      // Add category name with original font size
      const categoryTextElement = legendContainer.append("text")
        .attr("x", legendPadding)
        .attr("y", yPos)
        .attr("font-weight", "600")
        .attr("font-size", "11px") // Restored original size
        .attr("fill", theme === "dark" ? "#ccc" : "#555")
        .text(category.name);
      
      const categoryTextBBox = categoryTextElement.node()?.getBBox();
      if (categoryTextBBox) {
        // Add subtle background for category headers
        legendContainer.append("rect")
          .attr("x", legendPadding - 4)
          .attr("y", yPos - categoryTextBBox.height + 2)
          .attr("width", categoryTextBBox.width + 8)
          .attr("height", categoryTextBBox.height + 4)
          .attr("rx", 3)
          .attr("fill", theme === "dark" ? "rgba(255, 255, 255, 0.07)" : "rgba(0, 0, 0, 0.04)")
          .lower(); // Move behind text
      }
      
      // Add category items
      category.types.forEach(type => {
        // Calculate y position for each item
        yPos += legendItemHeight;
        
        // Check if this relationship type is filtered out
        const isFiltered = filteredRelationships.includes(type.toLowerCase());
        
        // Create legend entry group with hover interaction
        const entry = legendContainer.append("g")
          .attr("transform", `translate(${legendPadding}, ${yPos})`)
            .attr("class", "legend-item")
          .attr("data-type", type);
          
        // Create a hover/click target rectangle
        entry.append("rect")
          .attr("width", legendWidth - (legendPadding * 2))
          .attr("height", legendItemHeight)
          .attr("x", -5)
          .attr("y", -12)
          .attr("rx", 4)
          .attr("fill", isFiltered ? (theme === "dark" ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.03)") : "transparent")
          .attr("cursor", "pointer")
          .on("mouseover", function(this: SVGRectElement) {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("fill", theme === "dark" ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.05)");
            d3.select(this.parentNode as SVGGElement).select("circle")
              .transition()
              .duration(200)
              .attr("r", dotRadius * 1.3);
            d3.select(this.parentNode as SVGGElement).select("text")
              .transition()
              .duration(200)
              .attr("font-weight", "600");
          })
          .on("mouseout", function(this: SVGRectElement) {
            const parentGroup = d3.select(this.parentNode as SVGGElement);
            const relType = parentGroup.attr("data-type");
            const isCurrentlyFiltered = filteredRelationships.includes(relType.toLowerCase());
            
            d3.select(this)
              .transition()
              .duration(200)
              .attr("fill", isCurrentlyFiltered ? (theme === "dark" ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.03)") : "transparent");
            parentGroup.select("circle")
              .transition()
              .duration(200)
              .attr("r", dotRadius);
            parentGroup.select("text")
              .transition()
              .duration(200)
              .attr("font-weight", "500");
          })
          .on("click", function(this: SVGRectElement) {
            const parentGroup = d3.select(this.parentNode as SVGGElement);
            const relType = parentGroup.attr("data-type");
            
            // Check if this relationship type is currently filtered
            const isFiltered = filteredRelationships.includes(relType.toLowerCase());
            
            // Toggle filtering for this relationship type
            console.log(`[FILTER DEBUG] Legend click: ${relType} - currently ${isFiltered ? 'filtered' : 'visible'}`);
            handleToggleRelationshipFilter(relType);
            
            // Update legend item visual state
            d3.select(this)
              .transition()
              .duration(300)
              .attr("fill", isFiltered ? "transparent" : (theme === "dark" ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.03)"));
            
            parentGroup.select("text")
              .transition()
              .duration(300)
              .style("text-decoration", isFiltered ? "none" : "line-through")
              .style("opacity", isFiltered ? 1 : 0.7);
            
            parentGroup.select("circle")
              .transition()
              .duration(300)
              .style("opacity", isFiltered ? 1 : 0.5);
          });
        
        // Add color dot with enhanced styling
        entry.append("circle")
          .attr("cx", 5)
            .attr("cy", 0)
          .attr("r", dotRadius)
          .attr("fill", getNodeColor(type))
          .attr("stroke", theme === "dark" ? "rgba(255, 255, 255, 0.15)" : "rgba(0, 0, 0, 0.08)")
          .attr("stroke-width", 0.5)
          .style("opacity", isFiltered ? 0.5 : 1);
        
        // Get label text with possible truncation
        const labelText = getRelationshipTypeLabel(type).label;
        const availableWidth = legendWidth - textPadding - (legendPadding * 2) - 10; // Account for dot width and padding
        const textElement = entry.append("text")
          .attr("x", textPadding)
            .attr("y", 0)
          .attr("dy", ".25em")
          .attr("font-size", "11px") // Restored original size
          .attr("font-weight", "500")
            .attr("fill", theme === "dark" ? "#ddd" : "#333")
          .style("text-decoration", isFiltered ? "line-through" : "none")
          .style("opacity", isFiltered ? 0.7 : 1);
        
        // Check if text needs truncation without actually truncating yet
        textElement.text(labelText);
        const textWidth = textElement.node()?.getBBox().width || 0;
        
        // If text overflows, truncate it
        if (textWidth > availableWidth) {
          // Calculate how many characters we can fit
          const approxCharsPerWidth = labelText.length / textWidth;
          const visibleChars = Math.floor(approxCharsPerWidth * availableWidth) - 3;
          textElement.text(labelText.substring(0, visibleChars) + "...");
          
          // Add title for tooltip on hover
          entry.append("title").text(labelText);
        }
      });
      
      // Add spacing after each category (except the last one)
      if (categoryIndex < categories.length - 1) {
        yPos += categorySpacing;
      }
    });

    // Add checkbox for disconnected nodes
    // Calculate position below the legend
    const checkboxY = yPos + legendItemHeight + 10;
    
    // Add a subtle divider line before the checkbox
    legendContainer.append("line")
      .attr("x1", legendPadding)
      .attr("y1", checkboxY - 15)
      .attr("x2", legendWidth - legendPadding)
      .attr("y2", checkboxY - 15)
      .attr("stroke", theme === "dark" ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.06)")
      .attr("stroke-width", 1);
    
    // Add a container for the toggle switch
    const toggleContainer = legendContainer.append("g")
      .attr("transform", `translate(${legendPadding}, ${checkboxY})`)
      .attr("class", "disconnected-nodes-option")
      .style("cursor", "pointer");
    
    // Skip adding a background rectangle for the toggle area entirely
    
    // Create a simpler, more elegant toggle switch
    // 1. Create pill background with cleaner styling
    const toggleTrack: d3.Selection<SVGRectElement, unknown, null, undefined> = toggleContainer.append("rect")
      .attr("width", 30) // Smaller toggle
      .attr("height", 14) // Smaller toggle
      .attr("rx", 7)
      .attr("x", 0)
      .attr("y", -5)
      .attr("fill", showDisconnectedNodes ? 
        (theme === "dark" ? "#4873c4" : "#3873e8") : 
        (theme === "dark" ? "#505868" : "#d9dee6"))
      .attr("stroke", "none")
      .attr("filter", theme === "dark" ? 
        "drop-shadow(0px 1px 1px rgba(0, 0, 0, 0.2))" : 
        "drop-shadow(0px 1px 1px rgba(0, 0, 0, 0.1))");
    
    // 2. Skip complicated gradients that might look bad
    
    // 3. Add toggle handle with cleaner styling
    const toggleHandle: d3.Selection<SVGCircleElement, unknown, null, undefined> = toggleContainer.append("circle")
      .attr("r", 5) // Smaller handle
      .attr("cx", showDisconnectedNodes ? 21 : 9) // Adjusted positions
      .attr("cy", 2)
      .attr("fill", theme === "dark" ? "#ffffff" : "#ffffff")
      .attr("stroke", theme === "dark" ? "rgba(0, 0, 0, 0.05)" : "rgba(0, 0, 0, 0.1)")
      .attr("stroke-width", 0.5)
      .attr("filter", "drop-shadow(0px 1px 1px rgba(0, 0, 0, 0.15))");
    
    // 4. Skip inner shadow on handle for simpler appearance
    
    // 5. Add simple but clear label with fixed width
    toggleContainer.append("text")
      .attr("x", 38) // Reverted to original spacing
      .attr("y", 2)
      .attr("dominant-baseline", "middle")
      .attr("font-size", "11px") // Restored original size
      .attr("font-weight", "500")
      .attr("fill", theme === "dark" ? "#ddd" : "#444")
      .text("Show disconnected nodes");
    
    // Add click handler for the toggle with simplified transitions
    toggleContainer.on("click", function() {
      const newValue = !showDisconnectedNodes;
      setShowDisconnectedNodes(newValue);
      
      // Update toggle track color with simple transition
      toggleTrack
        .transition().duration(250)
        .attr("fill", newValue ? 
          (theme === "dark" ? "#4873c4" : "#3873e8") : 
          (theme === "dark" ? "#505868" : "#d9dee6"));
      
      // Move handle with simple transition
      toggleHandle
        .transition()
        .duration(250)
        .attr("cx", newValue ? 21 : 9); // Match the positions from above
      });
    } // End of !isMobile conditional block for legend 

    setIsLoading(false);
      
    // Tooltip depends on state now, so keep it outside useEffect cleanup?
    const centerTimeout = setTimeout(() => {
         if (svgRef.current) centerOnMainWord(svg, filteredNodes);
     }, 800);

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
    // --- END OF TRY BLOCK, START OF CATCH BLOCK ---
    } catch (err) {
      console.error("[D3_EFFECT_ERROR] Error during D3 graph rendering:", err);
      setError(err instanceof Error ? err.message : "An unknown error occurred during graph rendering.");
      setIsValidNetwork(false); // Mark network as invalid on error
      setIsLoading(false); // Ensure loading state is reset
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
    // --- END OF CATCH BLOCK ---
  }, [
     wordNetwork,
     mainWord,
     depth,
     breadth,
    theme, 
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
   baseLinks,
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

  // *** NEW FUNCTION: Render Legend Content as React Elements ***
  const renderLegendContent = useCallback(() => {
    // Use the same category structure as desktop legend to ensure consistency
    const categories = [
      { name: "Core", types: ["main"] },
      { name: "Origin", types: ["root", "root_of", "etymology"] },
      { name: "Derived", types: ["derived", "derived_from", "derivative"] },
      { name: "Meaning", types: ["synonym", "antonym", "related", "similar"] },
      { name: "Cultural", types: ["kaugnay", "kahulugan", "kasalungat"] },
      { name: "Form", types: ["variant", "spelling_variant", "regional_variant", "abbreviation", "form_of"] },
      { name: "Translation", types: ["itapat", "atapat", "inatapat"] },
      { name: "Structure", types: ["hypernym", "hyponym", "meronym", "holonym", "taxonomic"] },
      { name: "Part-Whole", types: ["part_whole", "component", "component_of"] },
      { name: "Other", types: ["affix", "usage", "associated", "other"] }
    ];

    return (
      <List 
        dense 
        sx={{ 
          bgcolor: 'background.paper', 
          borderRadius: 1, 
          mt: 1, 
          maxHeight: 'calc(100vh - 250px)', 
          overflow: 'auto', 
          width: '100%'
        }}
      >
        <ListSubheader sx={{ bgcolor: 'transparent', lineHeight: '24px', py: 1, fontWeight: 'medium' }}>
          Relationship Types (Click to filter)
        </ListSubheader>
        
        {/* Add types to map parameters */}
        {categories.map((category: { name: string; types: string[] }, categoryIndex: number) => (
          <React.Fragment key={category.name}>
            {categoryIndex > 0 && <Divider component="li" variant="middle" />}
            <ListSubheader sx={{ bgcolor: 'transparent', fontWeight: 'bold', lineHeight: '36px' }}>{category.name}</ListSubheader>
            {/* Add type to map parameter */}
            {category.types.map((type: string) => {
              const isFiltered = filteredRelationships.includes(type.toLowerCase());
              const labelId = `legend-checkbox-label-${type}`;

              return (
                <ListItem
                  key={type}
                  button
                  onClick={() => handleToggleRelationshipFilter(type)}
                  sx={{
                    opacity: isFiltered ? 0.6 : 1,
                    textDecoration: isFiltered ? 'line-through' : 'none',
                    py: 0.5 // Adjust vertical padding
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 'auto', mr: 1.5 }}>
                    <Box
                      component="span"
                      sx={{
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        bgcolor: getNodeColor(type),
                        display: 'inline-block',
                        border: theme === 'dark' ? '1px solid rgba(255,255,255,0.3)' : '1px solid rgba(0,0,0,0.2)'
                      }}
                    />
                  </ListItemIcon>
                  <ListItemText id={labelId} primary={getRelationshipTypeLabel(type).label} sx={{ m: 0 }} />
                </ListItem>
              );
            })}
          </React.Fragment>
        ))}
        <Divider component="li" />
        {/* Disconnected Nodes Toggle */}
        <ListItem
          button
          onClick={() => setShowDisconnectedNodes(prev => !prev)}
          sx={{ py: 0.5 }}
        >
          <ListItemIcon sx={{ minWidth: 'auto', mr: 1.5 }}>
             <Checkbox
                edge="start"
                checked={showDisconnectedNodes}
                tabIndex={-1}
                disableRipple
                size="small"
                inputProps={{ 'aria-labelledby': 'legend-checkbox-label-disconnected' }}
              />
          </ListItemIcon>
          <ListItemText id="legend-checkbox-label-disconnected" primary="Show disconnected nodes" sx={{ m: 0 }}/>
        </ListItem>
      </List>
    );
  }, [filteredRelationships, handleToggleRelationshipFilter, showDisconnectedNodes, getNodeColor, getRelationshipTypeLabel, theme]);

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
      
      {/* CONTROLS AREA */} 
      <div className="controls-container">
        {/* Zoom Controls - Always visible */} 
        <div className="zoom-controls">
          <button onClick={() => handleZoom(1.3)} className="zoom-button" title="Zoom In">+</button>
          <button onClick={() => handleZoom(1 / 1.3)} className="zoom-button" title="Zoom Out">-</button>
          <button onClick={handleResetZoom} className="zoom-button reset-zoom-button" title="Reset View">Reset</button> 
        </div>

        {/* Conditional Network Controls */} 
        {!isMobile ? (
          // Desktop: Static controls
          <NetworkControls 
            depth={depth}
            breadth={breadth}
            onDepthChange={handleDepthChange}
            onBreadthChange={handleBreadthChange}
            onChangeCommitted={(_d, _b) => onNetworkChange(depth, breadth)}
            className="network-controls"
          />
        ) : (
          // Mobile: Button to open Drawer
          <IconButton 
            onClick={() => setControlsOpen(true)}
            className="network-controls-trigger" 
            aria-label="Open network controls"
            title="Network Controls"
            sx={{ 
                position: 'absolute',
                bottom: 8, // Adjusted spacing
                right: 8, // Adjusted spacing
                // Theme-aware styling
                bgcolor: theme === 'dark' ? 'rgba(40, 48, 68, 0.8)' : 'rgba(255, 255, 255, 0.85)', 
                backdropFilter: 'blur(3px)', 
                border: theme === 'dark' ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0,0,0,0.08)',
                color: theme === 'dark' ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.6)', 
                zIndex: 1500, // Ensure it's on top
                '&:hover': { 
                  color: theme === 'dark' ? 'rgba(255, 255, 255, 0.95)' : 'rgba(0, 0, 0, 0.9)',
                  bgcolor: theme === 'dark' ? 'rgba(50, 60, 80, 0.9)' : 'rgba(245, 245, 245, 0.95)' 
                }
            }}
          >
            <TuneIcon />
          </IconButton>
        )}
      </div> {/* Closing tag for controls-container */} 
      
      {/* Mobile Controls Drawer */} 
      {isMobile && (
        <Drawer
          anchor="bottom"
          open={controlsOpen}
          onClose={() => setControlsOpen(false)}
          PaperProps={{
              sx: {
                  borderTopLeftRadius: 16,
                  borderTopRightRadius: 16,
              }
          }}
        >
          <Box sx={{ p: 2, pt: 1, display: 'flex', flexDirection: 'column', maxHeight: '100%' }}> {/* Allow Box to control height */}
             {/* Grab handle */}
            <Box sx={{
                width: 40,
                height: 6,
                bgcolor: 'grey.300',
                borderRadius: 3,
                mx: 'auto',
                mb: 1,
             }} />
             {/* Title */}
            <Typography variant="h6" sx={{ textAlign: 'center', mb: 1 }}>Network Controls</Typography>
             {/* Controls */} 
            <NetworkControls 
                depth={depth}
                breadth={breadth}
                onDepthChange={handleDepthChange}
                onBreadthChange={handleBreadthChange}
                onChangeCommitted={(_d, _b) => onNetworkChange(depth, breadth)}
            />
             {/* *** RENDER LEGEND CONTENT IN DRAWER *** */}
            {renderLegendContent()}
          </Box>
        </Drawer>
      )}

      {/* Tooltip */} 
      {renderTooltip()}
    </div> // Closing tag for graph-container
  );
};

export default React.memo(WordGraph);
