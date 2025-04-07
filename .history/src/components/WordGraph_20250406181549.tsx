import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetworkResponse as WordNetwork, NetworkNode, NetworkEdge } from "../types";
import { useTheme } from "../contexts/ThemeContext";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import axios from 'axios';

interface WordGraphProps {
  wordNetwork: WordNetwork | null;
  mainWord: string | null;
  onNodeClick: (word: string) => void;
  onNetworkChange: (depth: number) => void;
  initialDepth: number;
  initialBreadth?: number;
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

interface CustomLink {
  source: string | CustomNode;
  target: string | CustomNode;
  relationship: string;
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
  onNetworkChange,
  initialDepth,
  initialBreadth = 15,
}) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(mainWord);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [depth, setDepth] = useState<number>(initialDepth || 2);
  const [breadth, setBreadth] = useState<number>(initialBreadth || 15);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isValidNetwork, setIsValidNetwork] = useState(true);
  const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);

  const isDraggingRef = useRef(false);
  const isTransitioningRef = useRef(false);
  const lastClickTimeRef = useRef(0);
  const prevMainWordRef = useRef<string | null>(null);

  // State for tooltip delay
  const [tooltipTimeoutId, setTooltipTimeoutId] = useState<NodeJS.Timeout | null>(null);

  // Use useState for the base data that comes directly from the API
  const [rawNodes, setRawNodes] = useState<NetworkNode[]>([]);
  const [rawLinks, setRawLinks] = useState<NetworkEdge[]>([]);

  useEffect(() => {
    if (!wordNetwork || !wordNetwork.nodes || !Array.isArray(wordNetwork.nodes) || 
        !wordNetwork.edges || !Array.isArray(wordNetwork.edges)) {
      console.error("Invalid wordNetwork structure:", wordNetwork);
      setIsValidNetwork(false);
    } else {
      setIsValidNetwork(true);
      setRawNodes(wordNetwork.nodes);
      setRawLinks(wordNetwork.edges);
    }
  }, [wordNetwork]);

  useEffect(() => {
    setSelectedNodeId(mainWord);
  }, [mainWord]);

  const mapRelationshipToGroup = useCallback((relationship?: string): string => {
    if (!relationship) return 'associated';
    const relLower = relationship.toLowerCase();
    switch (relLower) {
      case 'main': return 'main';
      case 'synonym': return 'synonym';
      case 'antonym': return 'antonym';
      case 'related': return 'related';
      case 'kaugnay': return 'related';
      case 'variant':
      case 'spelling_variant':
      case 'regional_variant': return 'variant';
        case 'derived': 
      case 'derived_from':
      case 'root_of': return 'derived';
      case 'root': return 'root';
      case 'hypernym':
      case 'hyponym': return 'taxonomic';
      case 'meronym':
      case 'holonym': return 'part_whole';
      case 'etymology': return 'etymology';
      case 'component_of': return 'component_of';
      case 'cognate': return 'cognate';
      case 'see_also':
      case 'compare_with': return 'usage';
      default: return 'associated';
    }
  }, []);

  const getNodeColor = useCallback((group: string): string => {
    // Get base color based on node group
    let baseColor: string;
    switch (group) {
      case "main": baseColor = "#1d3557"; break;
      case "root": baseColor = "#e63946"; break;
      case "derived": baseColor = "#2a9d8f"; break;
      case "synonym": baseColor = "#457b9d"; break;
      case "antonym": baseColor = "#f77f00"; break;
      case "variant": baseColor = "#f4a261"; break;
      case "related": baseColor = "#fcbf49"; break;
      case "taxonomic": baseColor = "#8338ec"; break;
      case "part_whole": baseColor = "#3a86ff"; break;
      case "usage": baseColor = "#0ead69"; break;
      case "etymology": baseColor = "#3d5a80"; break;
      case "component_of": baseColor = "#ffb01f"; break;
      case "cognate": baseColor = "#9381ff"; break;
      case "associated": baseColor = "#adb5bd"; break;
      default: baseColor = "#6c757d"; break;
    }
    
    // Apply color intensity adjustment
    const intensity = 100; // Use 100 as default if undefined
    if (intensity !== 100) {
      try {
        const color = d3.color(baseColor);
        if (color) {
          if (intensity > 100) {
            // Increase brightness for values > 100%
            const brightenFactor = (intensity - 100) / 50;
            return color.brighter(brightenFactor).formatHex();
          } else {
            // Decrease brightness for values < 100%
            const darkenFactor = (100 - intensity) / 100;
            return color.darker(darkenFactor).formatHex();
          }
        }
      } catch (e) {
        console.error("Error adjusting color intensity:", e);
      }
    }
    
    return baseColor;
  }, []);

  const memoizedLinks = useMemo<{ source: string; target: string; relationship: string }[]>(() => {
    if (!rawNodes || !rawLinks || rawNodes.length === 0 || rawLinks.length === 0) {
      console.log("Cannot create memoizedLinks - missing rawNodes or rawLinks data");
      return [];
    }

    console.log("Creating memoizedLinks from:", { rawNodesLength: rawNodes.length, rawLinksLength: rawLinks.length });
    
    return rawLinks
      .map((edge: NetworkEdge) => {
        const sourceNode = rawNodes.find((n: NetworkNode) => n.id === edge.source);
        const targetNode = rawNodes.find((n: NetworkNode) => n.id === edge.target);

        if (!sourceNode || !targetNode) {
          console.warn(`Could not find nodes for edge: ${edge.source} -> ${edge.target}`);
          return null;
        }
        
        if (!sourceNode.lemma || !targetNode.lemma) {
          console.warn(`Missing lemma in nodes for edge: ${edge.source} -> ${edge.target}`);
          // Use ID as fallback if lemma is missing
          return {
            source: sourceNode.lemma || `id_${sourceNode.id}`,
            target: targetNode.lemma || `id_${targetNode.id}`,
            relationship: edge.type
          };
        }

        return {
          source: sourceNode.lemma,
          target: targetNode.lemma,
          relationship: edge.type
        };
      })
      .filter((link): link is { source: string; target: string; relationship: string; } => link !== null);
  }, [rawNodes, rawLinks]);

  const memoizedNodes = useMemo<CustomNode[]>(() => {
    if (!rawNodes || !mainWord || rawNodes.length === 0) {
      console.log("Cannot create memoizedNodes - missing rawNodes or mainWord", { 
        hasRawNodes: !!rawNodes, 
        rawNodesLength: rawNodes?.length || 0,
        mainWord 
      });
      return [];
    }

    console.log("Creating memoizedNodes for:", mainWord);
    
    const mappedNodes = rawNodes.map((node: NetworkNode): CustomNode => {
      if (!node.lemma) {
        console.warn(`Node missing lemma, using id_${node.id} as fallback`, node);
      }
      
      const nodeId = node.lemma || `id_${node.id}`;
      let calculatedGroup = 'associated';
      
      if (nodeId === mainWord) {
        calculatedGroup = 'main';
      } else {
        const connectingLink = memoizedLinks.find(link =>
          (link.source === mainWord && link.target === nodeId) ||
          (link.source === nodeId && link.target === mainWord)
        );
        calculatedGroup = mapRelationshipToGroup(connectingLink?.relationship);
      }
      
      const connections = memoizedLinks.filter(l => l.source === nodeId || l.target === nodeId).length;

      return {
        id: nodeId,
        word: nodeId,
        label: node.lemma || `id_${node.id}`,
        group: calculatedGroup,
        connections: connections,
        originalId: node.id,
        language: node.language_code ?? undefined,
        index: undefined, x: undefined, y: undefined, vx: undefined, vy: undefined, fx: undefined, fy: undefined
      };
    });

    const uniqueNodes: CustomNode[] = [];
    const seenIds = new Set<string>();
    for (const node of mappedNodes) {
        if (node.id && !seenIds.has(node.id)) {
            uniqueNodes.push(node);
            seenIds.add(node.id);
        }
    }
    
    console.log(`Created ${uniqueNodes.length} unique nodes from ${mappedNodes.length} raw nodes`);
    return uniqueNodes;
  }, [rawNodes, mainWord, memoizedLinks, mapRelationshipToGroup]);

  const filteredNodes = useMemo<CustomNode[]>(() => {
    if (!mainWord || memoizedNodes.length === 0) {
      console.log("Cannot create filteredNodes - missing mainWord or memoizedNodes is empty", {
        mainWord,
        memoizedNodesLength: memoizedNodes.length
      });
      return [];
    }
    
    // Check if mainWord exists in nodes
    const mainNode = memoizedNodes.find(n => n.id === mainWord);
    if (!mainNode) {
      console.warn(`Main word "${mainWord}" not found in nodes, cannot build network`);
      return memoizedNodes.slice(0, Math.min(memoizedNodes.length, 20)); // Return some nodes as fallback
    }
    
    // Map for quick node lookups by ID
    const nodeMap = new Map(memoizedNodes.map(n => [n.id, n]));
    // Set to track which nodes we'll include in the final result
    const nodesToInclude = new Set<string>([mainWord]);
    // Queue for BFS traversal: [nodeId, depth]
    const queue: [string, number][] = [[mainWord, 0]];
    // Track visited nodes to avoid cycles
    const visited = new Set<string>();
    
    console.log(`Filtering network with depth=${depth}, breadth=${breadth}, nodes=${memoizedNodes.length}, edges=${memoizedLinks.length}`);
    
    // Process using breadth-first search
    while (queue.length > 0) {
      const [currentId, currentDepth] = queue.shift()!;
      
      // Skip if we've reached max depth
      if (currentDepth >= depth) continue;
      
      // Mark as visited
      visited.add(currentId);
      
      // Find all links connected to this node
      const connectedLinks = memoizedLinks.filter(link => 
        link.source === currentId || link.target === currentId
      );
      
      console.log(`Node ${currentId} at depth ${currentDepth} has ${connectedLinks.length} connections`);
      
      // If we have more links than our breadth parameter allows, sort by priority
      let linksToProcess = connectedLinks;
      if (connectedLinks.length > breadth) {
        // Sort links by priority: synonyms and antonyms first, then other types
        connectedLinks.sort((a, b) => {
          // Helper function to get priority score for relationship types
          const getPriority = (type: string): number => {
            type = (type || "").toLowerCase();
            if (type.includes('synonym')) return 1;
            if (type.includes('antonym')) return 2;
            if (type.includes('root')) return 3;
            if (type.includes('derived')) return 4;
            if (type.includes('hypernym') || type.includes('hyponym')) return 5;
            return 10; // Other types
          };
          
          return getPriority(a.relationship) - getPriority(b.relationship);
        });
        
        // Limit to breadth parameter
        linksToProcess = connectedLinks.slice(0, breadth);
        console.log(`Limited from ${connectedLinks.length} to ${linksToProcess.length} connections due to breadth limit`);
      }
      
      // Process each selected link
      for (const link of linksToProcess) {
        // Get the ID of the connected node
        const connectedId = link.source === currentId ? link.target : link.source;
        
        // Skip if already visited
        if (visited.has(connectedId)) continue;
        
        // Add to our result set and enqueue for further processing
        if (nodeMap.has(connectedId)) {
          nodesToInclude.add(connectedId);
          // Only enqueue if we haven't reached max depth
          if (currentDepth < depth - 1) {
            queue.push([connectedId, currentDepth + 1]);
          }
        }
      }
    }
    
    const result = memoizedNodes.filter(node => nodesToInclude.has(node.id));
    console.log(`Filtered nodes: ${result.length} of ${memoizedNodes.length} included in final result`);
    
    // If we somehow ended up with no nodes (which shouldn't happen since at least the main node should be included),
    // return at least the main node
    if (result.length === 0 && mainNode) {
      console.warn("Filtered to 0 nodes, returning main node as fallback");
      return [mainNode];
    }
    
    return result;
  }, [memoizedNodes, memoizedLinks, mainWord, depth, breadth]);

  // Memoize nodeMap for use in multiple callbacks
  const nodeMap = useMemo(() => {
      return new Map(filteredNodes.map(n => [n.id, n]));
  }, [filteredNodes]);

  // Update useEffect for the filteredLinks to use memoizedLinks
  const filteredLinks = useMemo<CustomLink[]>(() => {
    if (!rawLinks) return [];
    
    // We need to filter links to only include those between filteredNodes
    const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
    
    return memoizedLinks
      .filter(link => filteredNodeIds.has(link.source) && filteredNodeIds.has(link.target))
      .map(link => ({
        source: link.source,
        target: link.target,
        relationship: link.relationship
      }));
  }, [memoizedLinks, filteredNodes, rawLinks]);

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
        // Set simulation center to center of the screen
        .force("center", d3.forceCenter(width / 2, height / 2))
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
      nodeSelection
        .on("click", (event, d) => {
          event.stopPropagation();
          if (isDraggingRef.current) return;
          
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

           if (onNodeClick) onNodeClick(d.id);
        })
        .on("mouseover", (event, d) => {
            if (isDraggingRef.current) return;
            const nodeElement = d3.select(event.currentTarget as Element);
            if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);

            // Hover Effect: Highlight border only
            nodeElement.select("circle")
               .transition("hover_border").duration(100)
               .attr("stroke-width", 2.5)
               .attr("stroke", d3.color(getNodeColor(d.group))?.brighter(0.8).formatHex() ?? (theme === "dark" ? "#eee" : "#333"));
            nodeElement.raise();
            // NO dimming of others

            const timeoutId = setTimeout(() => { setHoveredNode({ ...d }); }, 200);
            setTooltipTimeoutId(timeoutId);
        })
        .on("mouseout", (event, d_unknown) => {
            if (isDraggingRef.current) return;
            if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);
            setHoveredNode(null);

            const nodeElement = d3.select(event.currentTarget as Element);
            const d = d_unknown as CustomNode;

            // Revert hover effect for the circle
             const circleSelection = nodeElement.select<SVGCircleElement>("circle").data([d]);

             // Transition width first
             circleSelection.transition("hover_border_width").duration(150)
                  .attr("stroke-width", (n: CustomNode) => n.id === selectedNodeId ? 2.5 : (n.pinned ? 3 : 1.5));

            // Then transition stroke color, always reverting to default dark unless pinned/selected
             circleSelection.transition("hover_border_color").duration(150)
                 .attr("stroke", (n: CustomNode) => {
                     const baseColor = getNodeColor(n.group);
                     if (n.id === selectedNodeId) { // Keep selected highlight color
                         return d3.color(baseColor)?.brighter(0.8).formatHex() ?? baseColor ?? (theme === "dark" ? "#eee" : "#333");
                     }
                     if (n.pinned) { // Keep pinned color
                         return baseColor;
                     }
                     // Default dark color
                     return d3.color(baseColor)?.darker(0.8).formatHex() ?? baseColor ?? "#888";
                 });

             // NO restoration of other opacities needed
        })
        .on("dblclick", (event, d_unknown) => {
             event.preventDefault();
             const d = d_unknown as CustomNode; // Cast early
             d.pinned = !d.pinned;
             d.fx = d.pinned ? d.x : null;
             d.fy = d.pinned ? d.y : null;

             // Select the circle and re-bind the typed data *before* the transition
             const circleSelection = d3.select(event.currentTarget as Element)
                                     .select<SVGCircleElement>('circle')
                                     .data([d]); // Re-bind typed data

             // Now apply transitions using the correctly typed selection
             circleSelection.transition().duration(150)
                 .attr("stroke-width", (n: CustomNode) => n.id === selectedNodeId ? 2.5 : (n.pinned ? 3 : 1.5))
                 .attr("stroke-dasharray", (n: CustomNode) => n.pinned ? "5,3" : "none")
                 .attr("stroke", (n: CustomNode) => {
                     const baseColor = getNodeColor(n.group);
                     let finalColor: string;
                     if (n.pinned) {
                         finalColor = baseColor;
                     } else {
                         if (n.id === selectedNodeId) {
                             finalColor = d3.color(baseColor)?.brighter(0.8).formatHex() ?? baseColor ?? (theme === "dark" ? "#eee" : "#333");
                         } else {
                             finalColor = d3.color(baseColor)?.darker(0.8).formatHex() ?? baseColor ?? "#888";
                         }
                     }
                     return finalColor;
                 });
        });
  }, [selectedNodeId, onNodeClick, getNodeRadius, getNodeColor, theme, nodeMap]);

  // Define handleResetZoom before centerOnMainWord
  const handleResetZoom = useCallback(() => {
    if (zoomRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current);
      const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
      
      // Default dimensions if container can't be measured
      const width = containerRect ? containerRect.width : 800;
      const height = containerRect ? containerRect.height : 600;
      
      // Create a transform that centers the view with scale 1
      const resetTransform = d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(1);
      
      // Apply with animation
      svg.transition()
        .duration(600)
        .ease(d3.easeCubicOut)
        .call(zoomRef.current.transform, resetTransform)
        .on("end", () => {
          console.log("View reset");
        });
    }
  }, []);

  const centerOnMainWord = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, nodesToSearch: CustomNode[]) => {
    if (!zoomRef.current || isDraggingRef.current || !mainWord) return;
    
    // Find the main node data
    const mainNodeData = nodesToSearch.find(n => n.id === mainWord);
    
    // Get the container dimensions
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    if (!containerRect) return;
    
    const width = containerRect.width;
    const height = containerRect.height;
    
    // Check if we found the main node and it has position data
    if (mainNodeData && mainNodeData.x !== undefined && mainNodeData.y !== undefined) {
      // Create a transform that centers the main node 
      const currentTransform = d3.zoomTransform(svg.node()!);
      
      // Use consistent scale - not too close, not too far
      const targetScale = 1.0;
      
      // Calculate position to center the main node
      const targetX = width / 2 - mainNodeData.x * targetScale;
      const targetY = height / 2 - mainNodeData.y * targetScale;
      
      // Create new transform and animate to it
      const newTransform = d3.zoomIdentity
        .translate(targetX, targetY)
        .scale(targetScale);
      
      // Use transition for smooth animation
      isTransitioningRef.current = true;
      svg.transition()
        .duration(750)
        .ease(d3.easeCubicOut)
        .call(zoomRef.current.transform, newTransform)
        .on("end", () => {
          isTransitioningRef.current = false;
          console.log("Centered on main word:", mainWord);
        });
    } else {
      // If we can't find the main node, just reset to center
      console.log("Couldn't find main node position for centering, using default position");
      handleResetZoom();
    }
  }, [mainWord, handleResetZoom]);

  const setupSvgDimensions = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);
    return { width, height };
  }, []);

  // Main effect for rendering the graph
  useEffect(() => {
    if (!svgRef.current || !wordNetwork || !mainWord) {
      console.log("Cannot render graph:", {
        hasSvgRef: !!svgRef.current,
        hasWordNetwork: !!wordNetwork,
        hasMainWord: !!mainWord,
        filteredNodesLength: filteredNodes.length
      });
      return;
    }
    
    console.log(`Rendering graph with ${filteredNodes.length} nodes and ${filteredLinks.length} links. Depth: ${depth}, Breadth: ${breadth}`);

    // Clear any existing elements and stop any running simulations
    const svgElement = d3.select(svgRef.current);
    svgElement.selectAll("*").remove();
    if (simulationRef.current) simulationRef.current.stop();

    // Get container dimensions
    const containerRect = svgElement.node()?.parentElement?.getBoundingClientRect() || { width: 800, height: 600 };
    const width = containerRect.width;
    const height = containerRect.height;

    // Create the zoom behavior
    zoomRef.current = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 8])
      .on("zoom", (event) => {
        if (!isDraggingRef.current && !isTransitioningRef.current) {
          graph.attr("transform", event.transform.toString());
        }
      });
      
    svgElement.call(zoomRef.current);

    // Create the main container for all graph elements
    const graph = svgElement.append("g")
      .attr("class", "everything");

    // Create the simulation
    simulationRef.current = d3.forceSimulation<CustomNode>(filteredNodes)
      .alpha(0.9) // Higher alpha for better initial positioning
      .alphaDecay(0.03) // Slower decay for smoother animation
      .velocityDecay(0.4)
      .force("link", d3.forceLink<CustomNode, CustomLink>(filteredLinks)
        .id((d: CustomNode) => d.id)
        .distance(80) // Slightly larger distance
        .strength(0.5)) // Slightly stronger link force
      .force("charge", d3.forceManyBody<CustomNode>()
        .strength(-350) // Stronger repulsion
        .distanceMax(400)) // Larger distance max
      .force("collide", d3.forceCollide<CustomNode>()
        .radius(d => getNodeRadius(d) + 25)
        .strength(0.8)) // Stronger collision detection
      .force("center", d3.forceCenter(width / 2, height / 2))
      .on("tick", tick);

    // Create links
    const links = graph.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(filteredLinks)
      .enter()
      .append("line")
      .attr("stroke-width", 1.5)
      .attr("stroke", theme === "dark" ? "#666" : "#999")
      .attr("stroke-opacity", 0.6);

    // Create nodes as groups
    const nodes = graph.append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(filteredNodes)
      .enter()
      .append("g")
      .attr("class", d => `node ${d.id === mainWord ? "main-node" : ""}`)
      .call(d3.drag<SVGGElement, CustomNode>()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded) as any)
      .on("click", (event, d) => {
        if (!isDraggingRef.current && d.id !== mainWord) {
          onNodeClick(d.id);
        }
      });

    // Add circles to node groups
    nodes.append("circle")
      .attr("r", d => getNodeRadius(d))
      .attr("fill", d => getNodeColor(d.group))
      .attr("stroke", theme === "dark" ? "#444" : "#fff")
      .attr("stroke-width", 1.5);

    // Add text labels
    nodes.append("text")
      .text(d => d.word)
      .attr("font-size", d => d.id === mainWord ? "14px" : "12px")
      .attr("text-anchor", "middle")
      .attr("dy", d => getNodeRadius(d) + 15)
      .attr("fill", theme === "dark" ? "#eee" : "#333")
      .attr("pointer-events", "none"); // Prevent text from interfering with clicks

    // Drag functions
    function dragStarted(event: any, d: CustomNode) {
      if (!event.active) simulationRef.current?.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
      isDraggingRef.current = true;
    }

    function dragged(event: any, d: CustomNode) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragEnded(event: any, d: CustomNode) {
      if (!event.active) simulationRef.current?.alphaTarget(0);
      d.fx = null;
      d.fy = null;
      isDraggingRef.current = false;
    }

    // Tick function
    function tick() {
      links
        .attr("x1", d => (typeof d.source === 'object' ? d.source.x ?? 0 : 0))
        .attr("y1", d => (typeof d.source === 'object' ? d.source.y ?? 0 : 0))
        .attr("x2", d => (typeof d.target === 'object' ? d.target.x ?? 0 : 0))
        .attr("y2", d => (typeof d.target === 'object' ? d.target.y ?? 0 : 0));

      nodes.attr("transform", d => `translate(${d.x ?? 0}, ${d.y ?? 0})`);
    }

    // Center the graph on the main word after it has stabilized
    setTimeout(() => {
      centerOnMainWord(svgElement, filteredNodes);
    }, 300);

    // Clean up function
    return () => {
      if (simulationRef.current) simulationRef.current.stop();
    };
  }, [filteredNodes, filteredLinks, getNodeColor, getNodeRadius, mainWord, theme, onNodeClick, centerOnMainWord, depth, breadth]);

  useEffect(() => {
    if (prevMainWordRef.current && prevMainWordRef.current !== mainWord && svgRef.current) {
        const recenterTimeout = setTimeout(() => {
            if(svgRef.current) centerOnMainWord(d3.select(svgRef.current), filteredNodes);
        }, 800);
         return () => clearTimeout(recenterTimeout);
    }
    prevMainWordRef.current = mainWord;
  }, [mainWord, centerOnMainWord, filteredNodes]);

  const handleZoom = useCallback((scale: number) => {
    if (zoomRef.current && svgRef.current) {
      // Apply zoom transformation with animation
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .ease(d3.easeCubicOut)
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

  // Update local state when props change
  useEffect(() => {
    if (initialDepth !== depth) {
      console.log(`Updating depth from ${depth} to ${initialDepth}`);
      setDepth(initialDepth);
    }
  }, [initialDepth, depth]);

  useEffect(() => {
    if (initialBreadth !== breadth) {
      console.log(`Updating breadth from ${breadth} to ${initialBreadth}`);
      setBreadth(initialBreadth);
    }
  }, [initialBreadth, breadth]);

  // Update local state when wordNetwork prop changes
  useEffect(() => {
    if (wordNetwork) {
      const nodes = Array.isArray(wordNetwork.nodes) ? wordNetwork.nodes : [];
      const edges = Array.isArray(wordNetwork.edges) ? wordNetwork.edges : [];
      
      console.log(`WordGraph received data: ${nodes.length} nodes, ${edges.length} edges`);
      console.log('Network nodes:', nodes);
      console.log('Network edges:', edges);
      
      setRawNodes(nodes);
      setRawLinks(edges);
    } else {
      setRawNodes([]);
      setRawLinks([]);
    }
  }, [wordNetwork]);

  // Add a safety check for empty nodes
  const renderFallbackGraph = useCallback(() => {
    if (!svgRef.current) return;
    
    console.log("Rendering fallback graph since normal graph failed");
    setError("Unable to generate proper graph visualization. Showing simplified view.");
    
    // Clear existing content
    const svgElement = d3.select(svgRef.current);
    svgElement.selectAll("*").remove();
    
    // Get container dimensions
    const containerRect = svgElement.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    
    // Create a simple graph with just the main word
    const g = svgElement.append("g")
      .attr("class", "fallback-graph")
      .attr("transform", `translate(${width/2}, ${height/2})`);
    
    // Add a circle for the main word
    g.append("circle")
      .attr("r", 30)
      .attr("fill", getNodeColor("main"))
      .attr("stroke", theme === "dark" ? "#444" : "#fff")
      .attr("stroke-width", 2);
    
    // Add text for the main word
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", ".3em")
      .attr("fill", getTextColorForBackground(getNodeColor("main")))
      .attr("font-weight", "bold")
      .text(mainWord || "Word");
    
    // Add a message
    svgElement.append("text")
      .attr("x", width/2)
      .attr("y", height/2 + 60)
      .attr("text-anchor", "middle")
      .attr("fill", theme === "dark" ? "#aaa" : "#666")
      .text("Limited data available for this word");
  }, [mainWord, theme, getNodeColor]);
  
  useEffect(() => {
    // Check if we have data but no graph is rendered
    if (mainWord && wordNetwork && wordNetwork.nodes && wordNetwork.nodes.length > 0 && 
        filteredNodes.length === 0 && svgRef.current) {
      console.warn("Data available but filtered nodes is empty - rendering fallback");
      renderFallbackGraph();
    }
  }, [mainWord, wordNetwork, filteredNodes, renderFallbackGraph]);

  if (!isValidNetwork) {
    return (
      <div className="graph-container">
        <div className="error-overlay">
          <p className="error-message">Invalid network data structure. Please try again.</p>
          <p className="error-details">
            {wordNetwork ? 
              `Received ${wordNetwork.nodes?.length || 0} nodes and ${wordNetwork.edges?.length || 0} edges, but data structure is invalid.` : 
              'No data received from API.'
            }
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="graph-container">
      <div className="graph-svg-container">
        {isLoading && (
          <div className="loading-overlay"><div className="spinner"></div><p>Loading...</p></div>
        )}
        {error && (
          <div className="error-overlay">
            <p className="error-message">{error}</p>
          </div>
        )}
         {(!wordNetwork || !mainWord || filteredNodes.length === 0 && !isLoading && !error) && (
           <div className="empty-graph-message">Enter a word to see its network.</div>
        )}
        <svg ref={svgRef} className={`graph-svg ${isLoading ? 'loading' : 'loaded'}`}>
        </svg>
      </div>
      <div className="controls-container">
        <div className="zoom-controls">
           <button onClick={() => handleZoom(1.3)} className="zoom-button" title="Zoom In">+</button>
           <button onClick={() => handleZoom(1 / 1.3)} className="zoom-button" title="Zoom Out">-</button>
           <button onClick={handleResetZoom} className="zoom-button" title="Reset View">Reset</button>
        </div>
      </div>
      {renderTooltip()}
    </div>
  );
};

export default WordGraph;
