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
import NetworkControls from "./NetworkControls";
import axios from 'axios';

interface WordGraphProps {
  wordNetwork: WordNetwork | null;
  mainWord: string | null;
  onNodeClick: (word: string) => void;
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
    switch (group) {
      case "main": return "#1d3557";
      case "root": return "#e63946";
      case "derived": return "#2a9d8f";
      case "synonym": return "#457b9d";
      case "antonym": return "#f77f00";
      case "variant": return "#f4a261";
      case "related": return "#fcbf49";
      case "taxonomic": return "#8338ec";
      case "part_whole": return "#3a86ff";
      case "usage": return "#0ead69";
      case "etymology": return "#3d5a80";
      case "component_of": return "#ffb01f";
      case "cognate": return "#9381ff";
      case "associated": return "#adb5bd";
      default: return "#6c757d";
    }
  }, []);

  const baseLinks = useMemo<{ source: string; target: string; relationship: string }[]>(() => {
    if (!wordNetwork?.nodes || !wordNetwork.edges) return [];
    
    return wordNetwork.edges
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
  }, [wordNetwork]);

  const baseNodes = useMemo<CustomNode[]>(() => {
    // Ensure wordNetwork and mainWord exist before proceeding
    if (!wordNetwork?.nodes || !mainWord) {
        return []; // Return empty array if prerequisites are missing
    }

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
    return uniqueNodes; // Now guaranteed to return CustomNode[]
  }, [wordNetwork, mainWord, baseLinks, mapRelationshipToGroup]);

  const filteredNodes = useMemo<CustomNode[]>(() => {
    if (!mainWord || baseNodes.length === 0) {
      return [];
    }
    
    const nodeMap = new Map(baseNodes.map(n => [n.id, n]));
    const connectedNodeIds = new Set<string>([mainWord]);
    const queue: [string, number][] = [[mainWord, 0]];
    const visited = new Set<string>();

    while (queue.length > 0) {
      const [currentWordId, currentDepth] = queue.shift()!;

      if (currentDepth >= depth || visited.has(currentWordId)) continue;
      visited.add(currentWordId);

      const relatedLinks = baseLinks.filter(link =>
        link.source === currentWordId || link.target === currentWordId
      );

      const relatedWordIds = relatedLinks.map(link => {
        return link.source === currentWordId ? link.target : link.source;
      }).filter(id => !visited.has(id));

      const sortedWords = relatedWordIds.sort((aId, bId) => {
         const aNode = nodeMap.get(aId);
         const bNode = nodeMap.get(bId);
         const aGroup = aNode ? aNode.group : 'associated';
         const bGroup = bNode ? bNode.group : 'associated';
        const groupOrder = [
            'main', 'root', 'root_of', 'synonym', 'antonym', 'derived',
            'variant', 'related', 'kaugnay', 'component_of', 'cognate',
            'etymology', 'derivative', 'associated', 'other'
        ];
        return groupOrder.indexOf(aGroup) - groupOrder.indexOf(bGroup);
      });

      sortedWords.slice(0, breadth).forEach(wordId => {
         if (nodeMap.has(wordId)) {
             connectedNodeIds.add(wordId);
             queue.push([wordId, currentDepth + 1]);
         }
      });
    }

    return baseNodes.filter((node) => connectedNodeIds.has(node.id));
  }, [baseNodes, baseLinks, mainWord, depth, breadth]);

  // Memoize nodeMap for use in multiple callbacks
  const nodeMap = useMemo(() => {
      return new Map(filteredNodes.map(n => [n.id, n]));
  }, [filteredNodes]);

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
        handleResetZoom();
    }
  }, [mainWord]); // Removed handleResetZoom from dependencies

  const setupSvgDimensions = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);
    return { width, height };
  }, []);

  useEffect(() => {
    if (!svgRef.current || !wordNetwork || !mainWord || baseNodes.length === 0) {
      if (svgRef.current) d3.select(svgRef.current).selectAll("*").remove();
      if (simulationRef.current) simulationRef.current.stop();
      setError(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    if (simulationRef.current) simulationRef.current.stop();
      
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

    const { width, height } = setupSvgDimensions(svg);
    const g = svg.append("g").attr("class", "graph-content");
      const zoom = setupZoom(svg, g);
      zoomRef.current = zoom;

    console.log("[Graph Effect] Base links count:", baseLinks.length);
    const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
    console.log("[Graph Effect] Filtered node IDs:", Array.from(filteredNodeIds));

    console.log("[Graph Effect] Base links sample before filter:", JSON.stringify(baseLinks.slice(0, 10).map(l => ({ s: l.source, t: l.target, r: l.relationship }))));
    console.log("[Graph Effect] Filtered node IDs Set content:", JSON.stringify(Array.from(filteredNodeIds)));

    const currentFilteredLinks = baseLinks.filter(link => {
      const sourceId = typeof link.source === 'object' && link.source !== null ? (link.source as CustomNode).id : link.source as string;
      const targetId = typeof link.target === 'object' && link.target !== null ? (link.target as CustomNode).id : link.target as string;
      return filteredNodeIds.has(sourceId) && filteredNodeIds.has(targetId);
    }) as CustomLink[];
    console.log("[Graph Effect] Filtered links count AFTER filtering:", currentFilteredLinks.length);
    if(currentFilteredLinks.length > 0 && currentFilteredLinks.length < 10) {
         console.log("[Graph Effect] Filtered links sample AFTER filtering:", JSON.stringify(currentFilteredLinks.map(l => ({s: l.source, t: l.target}))));
      }

    if(currentFilteredLinks.length === 0 && filteredNodes.length > 1) {
         console.warn("Graph has nodes but no links connect them within the current depth/breadth.");
    }

    const currentSim = setupSimulation(filteredNodes, currentFilteredLinks, width, height);

    createLinks(g, currentFilteredLinks);
    const nodeElements = createNodes(g, filteredNodes, currentSim);
    setupNodeInteractions(nodeElements);

    if (currentSim) {
      currentSim.nodes(filteredNodes);
      const linkForce = currentSim.force<d3.ForceLink<CustomNode, CustomLink>>("link");
      if (linkForce) {
        linkForce.links(currentFilteredLinks);
      }
      // Pin the main node to simulation center (0,0)
      const mainNodeData = filteredNodes.find(n => n.id === mainWord);
      if (mainNodeData) {
          mainNodeData.fx = 0;
          mainNodeData.fy = 0;
      }
      currentSim.alpha(1).restart();
    }

    setTimeout(() => centerOnMainWord(svg, filteredNodes), 800);

      const legendContainer = svg.append("g")
        .attr("class", "legend")
      .attr("transform", `translate(${width - 160}, 25)`);
      
    const allRelationGroups = [
        { type: 'main', label: 'Main Word' },
        { type: 'root', label: 'Root' },
        { type: 'derived', label: 'Derived/Root Of' },
        { type: 'synonym', label: 'Synonym' },
        { type: 'antonym', label: 'Antonym' },
        { type: 'variant', label: 'Variant' },
        { type: 'related', label: 'Related/Kaugnay' },
        { type: 'taxonomic', label: 'Taxonomic' },
        { type: 'part_whole', label: 'Part/Whole' },
        { type: 'usage', label: 'Usage Note' },
        { type: 'etymology', label: 'Etymology' },
        { type: 'component_of', label: 'Component Of' },
        { type: 'cognate', label: 'Cognate' },
        { type: 'associated', label: 'Associated' }
    ];
    const legendItemHeight = 20;
    const legendWidth = 150;
    const legendPadding = 15;
    const legendX = width - legendWidth - legendPadding;

      legendContainer.append("rect")
        .attr("width", legendWidth)
        .attr("height", allRelationGroups.length * legendItemHeight + legendPadding * 2 + 10)
        .attr("rx", 6).attr("ry", 6)
        .attr("fill", theme === "dark" ? "rgba(30, 30, 30, 0.85)" : "rgba(255, 255, 255, 0.85)")
        .attr("stroke", theme === "dark" ? "rgba(255, 255, 255, 0.15)" : "rgba(0, 0, 0, 0.1)");

      legendContainer.append("text")
        .attr("x", legendWidth / 2)
        .attr("y", legendPadding + 2)
        .attr("text-anchor", "middle")
        .attr("font-weight", "600").attr("font-size", "11px")
        .attr("fill", theme === "dark" ? "#eee" : "#222")
        .text("Relationship Types");
      
     allRelationGroups.forEach((item, i) => {
        const legendEntry = legendContainer.append("g")
            .attr("class", "legend-item")
            .attr("transform", `translate(${legendPadding}, ${i * legendItemHeight + legendPadding + 25})`);

        legendEntry.append("circle")
            .attr("class", "legend-color")
            .attr("cx", 0)
            .attr("cy", 0)
            .attr("r", 6)
          .attr("fill", getNodeColor(item.type));
        
        legendEntry.append("text")
            .attr("class", "legend-label")
            .attr("x", 15)
            .attr("y", 0)
            .attr("dy", ".35em")
            .attr("font-size", "10px")
            .attr("fill", theme === "dark" ? "#ddd" : "#333")
          .text(item.label);

        // Add tooltip to legend item group
        legendEntry.append("title").text(`Relationship Type: ${item.label}`);
      });

      setIsLoading(false);
      
    // Tooltip depends on state now, so keep it outside useEffect cleanup?
    const centerTimeout = setTimeout(() => {
         if (svgRef.current) centerOnMainWord(svg, filteredNodes);
     }, 800);

      return () => {
      if (currentSim) currentSim.stop();
       clearTimeout(centerTimeout);
    };
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
     baseLinks
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

  const handleDepthChange = useCallback((newDepth: number) => {
    setDepth(newDepth);
    onNetworkChange(newDepth, breadth);
  }, [breadth, onNetworkChange]);

  const handleBreadthChange = useCallback((newBreadth: number) => {
    setBreadth(newBreadth);
    onNetworkChange(depth, newBreadth);
  }, [depth, onNetworkChange]);

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
        <NetworkControls 
          depth={depth}
          breadth={breadth}
          onDepthChange={handleDepthChange}
          onBreadthChange={handleBreadthChange}
        />
      </div>
      {renderTooltip()}
    </div>
  );
};

export default React.memo(WordGraph);
