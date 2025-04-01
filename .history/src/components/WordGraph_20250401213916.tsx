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
  // State to hold current dimensions, triggers re-render on resize
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

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
    if (!wordNetwork?.nodes || !mainWord) return [];

    return wordNetwork.nodes.map(node => {
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
    if (node.id === mainWord) return 22;
    if (node.group === 'root') return 18;
    const connFactor = node.connections ? Math.min(node.connections, 5) : 0; // Cap connection influence
    return 12 + connFactor; // Base size + connection bonus
  }, [mainWord]);

  const setupZoom = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, g: d3.Selection<SVGGElement, unknown, null, undefined>, width: number, height: number) => {
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

      nodeSelection.attr("transform", d => `translate(${d.x ?? 0}, ${d.y ?? 0})`);

      linkSelection
          .attr("x1", d => (typeof d.source === 'object' ? d.source.x ?? 0 : 0))
          .attr("y1", d => (typeof d.source === 'object' ? d.source.y ?? 0 : 0))
          .attr("x2", d => (typeof d.target === 'object' ? d.target.x ?? 0 : 0))
          .attr("y2", d => (typeof d.target === 'object' ? d.target.y ?? 0 : 0));
  }, []);

  const setupSimulation = useCallback((nodes: CustomNode[], links: CustomLink[], width: number, height: number) => {
      simulationRef.current = d3.forceSimulation<CustomNode>(nodes)
        .alphaDecay(0.022)
        .velocityDecay(0.4)
        .force("link", d3.forceLink<CustomNode, CustomLink>()
          .id(d => d.id)
          .links(links)
          .distance(110)
          .strength(0.4))
        .force("charge", d3.forceManyBody<CustomNode>().strength(-300).distanceMax(300))
        .force("collide", d3.forceCollide<CustomNode>().radius(d => getNodeRadius(d) + 6).strength(1.0))
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
    return g.append("g")
      .attr("class", "links")
      .selectAll("line")
        .data(linksData)
      .join("line")
      .attr("class", "link")
        .attr("stroke", theme === "dark" ? "#555" : "#bbb")
        .attr("stroke-opacity", 0.5)
      .attr("stroke-width", 1.5);
  }, [theme]);

  const createNodes = useCallback((
      g: d3.Selection<SVGGElement, unknown, null, undefined>,
      nodesData: CustomNode[],
      simulation: d3.Simulation<CustomNode, CustomLink> | null
      ) => {
    const drag = simulation ? createDragBehavior(simulation) : null;
    const nodeGroups = g.append("g")
      .attr("class", "nodes")
      .selectAll("g")
        .data(nodesData)
      .join("g")
        .attr("class", d => `node node-group-${d.group} ${d.id === mainWord ? "main-node" : ""}`);
    
      if (drag) nodeGroups.call(drag as any);
    
    nodeGroups.append("circle")
      .attr("r", d => getNodeRadius(d))
      .attr("fill", d => getNodeColor(d.group))
        .attr("stroke", d => d3.color(getNodeColor(d.group))?.darker(0.6).formatHex() ?? (theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)")) // Outline based on fill color
        .attr("stroke-width", 1);
    
    nodeGroups.append("text")
        .attr("dy", ".35em")
      .attr("text-anchor", "middle")
        .attr("font-size", d => d.id === mainWord ? "11px" : "9px")
        .attr("font-weight", d => d.id === mainWord ? "600" : "400")
      .text(d => d.word)
          .clone(true)
          .lower()
          .attr("fill", "none")
          .attr("stroke", theme === "dark" ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.7)")
          .attr("stroke-width", 3.5)
          .attr("stroke-linejoin", "round");

       nodeGroups.select("text:not([stroke])")
          .attr("fill", d => getTextColorForBackground(getNodeColor(d.group)))
      .style("pointer-events", "none");
      
      nodeGroups.append("title").text(d => `${d.word}
Group: ${d.group}
Connections: ${d.connections ?? 0}`);
    return nodeGroups;
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
               if (sourceId === d.id) {
                 connectedIds.add(targetId);
                 return true;
               }
               if (targetId === d.id) {
                 connectedIds.add(sourceId);
                 return true;
               }
               return false;
           }).each(function() { connectedLinkElements.push(this); });

          setSelectedNodeId(d.id);
          
          d3.selectAll(".node").classed("selected connected", false)
            .transition().duration(200).style("opacity", 0.1);
          d3.selectAll<SVGLineElement, CustomLink>(".link")
              .classed("highlighted", false)
              .transition().duration(200)
              .attr("stroke", theme === "dark" ? "#555" : "#bbb")
              .attr("stroke-opacity", 0.05)
              .attr("stroke-width", 1.5);

          const targetNodeElement = d3.select(event.currentTarget as Element);
          targetNodeElement.classed("selected", true)
            .transition().duration(200)
            .style("opacity", 1);

           d3.selectAll<SVGLineElement, CustomLink>(connectedLinkElements)
            .classed("highlighted", true)
            .raise()
            .transition().duration(200)
            .attr("stroke", (l: CustomLink) => {
                const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
                const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
                const targetNode = sourceId === d.id ? nodeMap.get(targetId) : nodeMap.get(sourceId);
                return targetNode ? getNodeColor(targetNode.group) : (theme === "dark" ? "#aaa" : "#666");
            })
            .attr("stroke-opacity", 0.9)
            .attr("stroke-width", 2.5);

           if (onNodeClick) onNodeClick(d.id);
        })
        .on("mouseover", (event, d) => {
            if (isDraggingRef.current) return;
            const nodeElement = d3.select(event.currentTarget as Element);
            // Clear any existing tooltip timeout
            if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);

            // Hover Effect: Highlight border and raise
            nodeElement.select("circle")
               .transition().duration(100)
               .attr("stroke-width", 2.5)
               .attr("stroke", theme === "dark" ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.8)");
            nodeElement.raise();
            // Dim others slightly on hover
            d3.selectAll<SVGGElement, CustomNode>(".node:not(:hover)").transition().duration(100).style("opacity", 0.6);
            d3.selectAll<SVGLineElement, CustomLink>(".link").transition().duration(100).attr("stroke-opacity", 0.2);

            // Set timeout to show tooltip
            const timeoutId = setTimeout(() => {
            setHoveredNode({ ...d });
            }, 200); // 200ms delay
            setTooltipTimeoutId(timeoutId);
        })
        .on("mouseout", (event, d_unknown) => {
            if (isDraggingRef.current) return;
            // Clear tooltip timeout and hide tooltip immediately
            if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);
            setHoveredNode(null);

            const nodeElement = d3.select(event.currentTarget as Element);
            const d = d_unknown as CustomNode; // Cast the datum early

            // Revert hover effect for the circle
             nodeElement.select<SVGCircleElement>("circle")
                .data([d]) // Re-bind the correctly typed datum
                .transition().duration(150)
                .attr("stroke-width", (n: CustomNode) => n.id === selectedNodeId ? 2.5 : (n.pinned ? 3 : 1)) // Keep selected/pinned width
                .attr("stroke", (n: CustomNode) => n.pinned ? getNodeColor(n.group) : (d3.color(getNodeColor(n.group))?.darker(0.6).formatHex() ?? (theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"))); // Keep pinned color, otherwise default dark shade

             // Restore opacity for all nodes and links
             d3.selectAll<SVGGElement, CustomNode>(".node").transition().duration(150).style("opacity", 1);
             d3.selectAll<SVGLineElement, CustomLink>(".link").transition().duration(150).attr("stroke-opacity", 0.5);

             // REMOVED: Complex logic to re-apply selected styles on mouseout

        })
        .on("dblclick", (event, d) => {
             event.preventDefault();
             d.pinned = !d.pinned;
             d.fx = d.pinned ? d.x : null;
             d.fy = d.pinned ? d.y : null;
             d3.select(event.currentTarget as Element).select('circle')
                 .transition().duration(150)
                 .attr("stroke-width", d.pinned ? 3 : (d.id === selectedNodeId ? 2.5 : 1))
                 .attr("stroke-dasharray", d.pinned ? "4,3" : "none")
                 .attr("stroke", d.pinned ? getNodeColor(d.group) : (theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"));
        });
  }, [selectedNodeId, onNodeClick, getNodeRadius, getNodeColor, theme, nodeMap]);

  // Define handleResetZoom before centerOnMainWord
  const handleResetZoom = useCallback(() => {
    if (zoomRef.current && svgRef.current) {
       const svg = d3.select(svgRef.current);
       // Use state dimensions
       const { width, height } = dimensions;
       const resetTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
       svg.transition().duration(600).ease(d3.easeCubicInOut)
         .call(zoomRef.current.transform, resetTransform);
     }
  }, [dimensions]);

  const centerOnMainWord = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, nodesToSearch: CustomNode[]) => {
    if (!zoomRef.current || isDraggingRef.current || !mainWord) return;
    const mainNodeData = nodesToSearch.find(n => n.id === mainWord);
    // Use state dimensions
    const { width, height } = dimensions;
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
  }, [mainWord, dimensions, handleResetZoom]);

  const setupSvgDimensions = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, container: HTMLElement) => {
    const rect = container.getBoundingClientRect();
    const newWidth = rect.width > 0 ? rect.width : 800; // Fallback width
    const newHeight = rect.height > 0 ? rect.height : 600; // Fallback height
    svg.attr("width", newWidth).attr("height", newHeight).attr("viewBox", `0 0 ${newWidth} ${newHeight}`);
    return { width: newWidth, height: newHeight };
  }, []);

  // --- MAIN EFFECT HOOK (with ResizeObserver) ---
  useEffect(() => {
    const svgElement = svgRef.current;
    const containerElement = svgElement?.parentElement; // Observe the container

    if (!svgElement || !containerElement || !wordNetwork || !mainWord || baseNodes.length === 0) {
      if (svgElement) d3.select(svgElement).selectAll("*").remove();
      if (simulationRef.current) simulationRef.current.stop();
      setError(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    if (simulationRef.current) simulationRef.current.stop();

    const svg = d3.select(svgElement);
    svg.selectAll("*").remove(); // Clear previous render

    // Initial setup based on container size
    let currentWidth = dimensions.width;
    let currentHeight = dimensions.height;
    const initialDims = setupSvgDimensions(svg, containerElement);
    currentWidth = initialDims.width;
    currentHeight = initialDims.height;
    // Update state immediately for first render if needed
    if (currentWidth !== dimensions.width || currentHeight !== dimensions.height) {
        setDimensions({ width: currentWidth, height: currentHeight });
    }

    const g = svg.append("g").attr("class", "graph-content");
    // Pass current dimensions to setupZoom
    const zoom = setupZoom(svg, g, currentWidth, currentHeight);
    zoomRef.current = zoom;

    // --- Link Filtering (remains the same) ---
    console.log("[Graph Effect] Base links count:", baseLinks.length);
    const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
    console.log("[Graph Effect] Filtered node IDs:", Array.from(filteredNodeIds));
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

    // --- Simulation & Drawing (Pass dimensions) ---
    const currentSim = setupSimulation(filteredNodes, currentFilteredLinks, currentWidth, currentHeight);
    createLinks(g, currentFilteredLinks);
    const nodeElements = createNodes(g, filteredNodes, currentSim);
    setupNodeInteractions(nodeElements);

    if (currentSim) {
        // ... update simulation nodes/links ...
       currentSim.nodes(filteredNodes);
      const linkForce = currentSim.force<d3.ForceLink<CustomNode, CustomLink>>("link");
      if (linkForce) {
        linkForce.links(currentFilteredLinks);
      }

       // Pin the main node (relative to center force)
       const mainNodeData = filteredNodes.find(n => n.id === mainWord);
       if (mainNodeData) {
           mainNodeData.fx = currentWidth / 2;
           mainNodeData.fy = currentHeight / 2;
       }
       currentSim.alpha(1).restart();
    }

    // --- Legend (Position adjusted slightly, better with CSS) ---
    const legendWidth = 150;
    const legendPadding = 15;
    const legendX = currentWidth - legendWidth - legendPadding; // Position from right edge
    const legendY = legendPadding; // Position from top edge

    const legendContainer = svg.append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${legendX}, ${legendY})`);
      
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
            .attr("transform", `translate(${legendPadding}, ${i * legendItemHeight + legendPadding + 25})`);

        legendEntry.append("circle")
            .attr("cx", 0)
            .attr("cy", 0)
            .attr("r", 6)
          .attr("fill", getNodeColor(item.type));
        
        legendEntry.append("text")
            .attr("x", 15)
            .attr("y", 0)
            .attr("dy", ".35em")
            .attr("font-size", "10px")
            .attr("fill", theme === "dark" ? "#ddd" : "#333")
          .text(item.label);
      });

    setIsLoading(false);

    // Initial centering (might need adjustment after resize)
    const initialCenterTimeout = setTimeout(() => {
        if (svgRef.current) centerOnMainWord(svg, filteredNodes);
    }, 800);

    // --- ResizeObserver Setup ---
    let animationFrameId: number | null = null;
    const resizeObserver = new ResizeObserver(entries => {
        // Use requestAnimationFrame to debounce resize updates
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
        animationFrameId = requestAnimationFrame(() => {
            if (!entries || entries.length === 0) return;
            const entry = entries[0];
            const { width: newWidth, height: newHeight } = entry.contentRect;

            if (newWidth > 0 && newHeight > 0) {
                 console.log(`[Resize] New dimensions: ${newWidth}x${newHeight}`);
                 // Update state to trigger re-render and pass new dimensions
                 setDimensions({ width: newWidth, height: newHeight });

                 // --- Option 1: Just update viewBox and let zoom handle it ---
                 // svg.attr("width", newWidth).attr("height", newHeight).attr("viewBox", `0 0 ${newWidth} ${newHeight}`);
                 // --- Option 2: More involved - update forces, recenter zoom? ---
                 // This might be needed if aspect ratio changes drastically
                 // svg.attr("width", newWidth).attr("height", newHeight).attr("viewBox", `0 0 ${newWidth} ${newHeight}`);
                 // const centerForce = currentSim?.force<d3.ForceCenter<CustomNode>>("center");
                 // if (centerForce) centerForce.x(newWidth / 2).y(newHeight / 2);
                 // // Potentially adjust zoom/pan to keep focus?
                 // // handleResetZoom(); // Or a smarter recenter
                 // currentSim?.alpha(0.3).restart(); // Give simulation a kick
            }
        });
    });

    resizeObserver.observe(containerElement);

    // Cleanup function
    return () => {
      if (currentSim) currentSim.stop();
      clearTimeout(initialCenterTimeout);
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
    };
  }, [
     wordNetwork, mainWord, depth, breadth, theme,
     mapRelationshipToGroup, getNodeColor, getNodeRadius, setupZoom, ticked,
     setupSimulation, createDragBehavior, createLinks, createNodes, setupNodeInteractions,
     centerOnMainWord, setupSvgDimensions, filteredNodes, baseLinks,
     dimensions
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

  const handleDepthChange = useCallback((event: Event, newValue: number | number[]) => {
    const newDepth = Array.isArray(newValue) ? newValue[0] : newValue;
    setDepth(newDepth);
  }, []);

  const handleBreadthChange = useCallback((event: Event, newValue: number | number[]) => {
    const newBreadth = Array.isArray(newValue) ? newValue[0] : newValue;
    setBreadth(newBreadth);
  }, []);

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
        <div className="graph-controls">
          <div className="slider-container">
             <Typography variant="caption" sx={{ mr: 1 }}>Depth: {depth}</Typography>
             <Slider value={depth} onChange={handleDepthChange} onChangeCommitted={() => onNetworkChange(depth, breadth)} aria-labelledby="depth-slider" step={1} marks min={1} max={5} size="small" sx={{ width: 100 }}/>
          </div>
          <div className="slider-container">
             <Typography variant="caption" sx={{ mr: 1 }}>Breadth: {breadth}</Typography>
             <Slider value={breadth} onChange={handleBreadthChange} onChangeCommitted={() => onNetworkChange(depth, breadth)} aria-labelledby="breadth-slider" step={1} marks min={5} max={20} size="small" sx={{ width: 100 }}/>
          </div>
        </div>
      </div>
      {renderTooltip()}
    </div>
  );
};

export default React.memo(WordGraph);
