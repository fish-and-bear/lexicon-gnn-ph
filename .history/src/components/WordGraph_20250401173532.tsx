import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetwork, NetworkNode } from "../types";
import { useTheme } from "../contexts/ThemeContext";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import axios from 'axios';

interface WordGraphProps {
  wordNetwork: WordNetwork;
  mainWord: string;
  onNodeClick: (word: string) => void;
  onNetworkChange: (depth: number, breadth: number) => void;
  initialDepth: number;
  initialBreadth: number;
}

interface CustomNode extends d3.SimulationNodeDatum {
  id: string;
  word: string;
  group: string;
  connections?: number;
  pinned?: boolean;
  info?: NetworkNode;
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
  onNetworkChange,
  initialDepth,
  initialBreadth,
}) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isValidNetwork, setIsValidNetwork] = useState(true);

  // Check if wordNetwork has the expected structure
  useEffect(() => {
    if (!wordNetwork || !wordNetwork.nodes || !Array.isArray(wordNetwork.nodes) || 
        !wordNetwork.edges || !Array.isArray(wordNetwork.edges)) {
      console.error("Invalid wordNetwork structure:", wordNetwork);
      setIsValidNetwork(false);
    } else {
      setIsValidNetwork(true);
    }
  }, [wordNetwork]);

  // Update selectedNodeId when mainWord changes
  useEffect(() => {
    setSelectedNodeId(mainWord);
  }, [mainWord]);

  // Update the getNodeRelation function to reflect current main word
  const getNodeRelation = useCallback((word: string, node: NetworkNode): string => {
    if (word === mainWord) return "main";
    
    // Direct node type check
    if (node.type === 'root') return "root";
    if (node.type === 'root_of') return "root_of";
    if (node.type === 'cognate') return "cognate";
    if (node.type === 'component_of') return "component_of";
    if (node.type === 'kaugnay') return "related"; // Map kaugnay to related
    
    // Check path to determine relationship
    if (node.path && node.path.length > 0) {
      const lastPathItem = node.path[node.path.length - 1];
      switch (lastPathItem.type.toLowerCase()) {
        case 'synonym': return "synonym";
        case 'antonym': return "antonym";
        case 'derived': 
        case 'derived_from': return "derived";
        case 'variant': return "variant";
        case 'related': return "related";
        case 'kaugnay': return "related"; // Map kaugnay to related
        case 'etymology': return "etymology";
        case 'root_of': return "root_of";
        case 'component_of': return "component_of";
        case 'cognate': return "cognate";
        case 'associated': return "associated";
        default: return "other";
      }
    }
    
    return "associated";
  }, [mainWord]);

  const getNodeColor = useCallback((group: string): string => {
    switch (group) {
      case "main":
        return "#1d3557"; // Deep blue - brand primary
      case "root":
        return "#e63946"; // Crimson red - brand accent
      case "root_of": 
        return "#2a9d8f"; // Teal - brand secondary
      case "synonym":
        return "#457b9d"; // Medium blue
      case "antonym":
        return "#e63946"; // Crimson red
      case "derived":
        return "#2a9d8f"; // Teal
      case "variant":
        return "#f4a261"; // Orange
      case "related":
        return "#ffd166"; // Golden yellow - use the brighter kaugnay color for related
      case "component_of":
        return "#ffb01f"; // Amber
      case "cognate":
        return "#9381ff"; // Light purple
      case "etymology":
        return "#3d5a80"; // Navy blue
      case "associated":
        return "#48cae4"; // Light blue
      case "derivative":
        return "#14b8a6"; // Medium teal
      default:
        return "#94a3b8"; // Light slate gray
    }
  }, []);

  const nodes: CustomNode[] = useMemo(() => {
    if (!wordNetwork || !wordNetwork.nodes || !Array.isArray(wordNetwork.nodes)) {
      return [];
    }
    
    return wordNetwork.nodes.map(node => ({
      id: node.word,
      word: node.word,
      group: getNodeRelation(node.word, node),
      info: node,
      connections: 0,
      pinned: false
    }));
  }, [wordNetwork, getNodeRelation]);

  const links: CustomLink[] = useMemo(() => {
    if (!wordNetwork || !wordNetwork.edges || !Array.isArray(wordNetwork.edges)) {
      return [];
    }
    
    return wordNetwork.edges
      .map(edge => {
        const sourceNode = wordNetwork.nodes.find(n => n.id === edge.source);
        const targetNode = wordNetwork.nodes.find(n => n.id === edge.target);
        
        if (!sourceNode || !targetNode) {
          return null;
        }
        
        return {
          source: sourceNode.label,
          target: targetNode.label,
          relationship: edge.type
        };
      })
      .filter(link => link !== null) as CustomLink[];
  }, [wordNetwork]);

  const filteredNodes = useMemo(() => {
    if (!mainWord || nodes.length === 0) {
      return nodes;
    }
    
    const connectedNodes = new Set<string>([mainWord]);
    const queue: [string, number][] = [[mainWord, 0]];
    const visited = new Set<string>();

    while (queue.length > 0) {
      const [currentWord, currentDepth] = queue.shift()!;
      if (currentDepth >= depth) break;
      visited.add(currentWord);

      const relatedWords = links
        .filter(link => 
          (typeof link.source === 'string' ? link.source : link.source.id) === currentWord ||
          (typeof link.target === 'string' ? link.target : link.target.id) === currentWord
        )
        .map(link => {
          const otherWord = (typeof link.source === 'string' ? link.source : link.source.id) === currentWord
            ? (typeof link.target === 'string' ? link.target : link.target.id)
            : (typeof link.source === 'string' ? link.source : link.source.id);
          return otherWord;
        })
        .filter(word => !visited.has(word));

      const sortedWords = relatedWords.sort((a, b) => {
        const aNode = nodes.find(node => node.id === a);
        const bNode = nodes.find(node => node.id === b);
        const aGroup = aNode ? aNode.group : 'other';
        const bGroup = bNode ? bNode.group : 'other';
        const groupOrder = [
          'main', 
          'root', 
          'root_of',
          'synonym', 
          'antonym', 
          'derived', 
          'variant', 
          'related',
          'kaugnay', 
          'component_of',
          'cognate',
          'etymology', 
          'derivative', 
          'associated', 
          'other'
        ];
        return groupOrder.indexOf(aGroup) - groupOrder.indexOf(bGroup);
      });

      sortedWords.slice(0, breadth).forEach(word => {
        connectedNodes.add(word);
        queue.push([word, currentDepth + 1]);
      });
    }

    return nodes.filter((node) => connectedNodes.has(node.id));
  }, [nodes, links, mainWord, depth, breadth]);

  // Store simulation directly rather than in a ref
  const [simulationObject, setSimulationObject] = useState<d3.Simulation<CustomNode, undefined> | null>(null);

  // Add a ref to track drag state
  const isDraggingRef = useRef(false);
  // Add a transition ref for tracking whether we're in the middle of a smooth transition
  const isTransitioningRef = useRef(false);
  // Add lastClickTimeRef to prevent double-clicks being processed as single clicks
  const lastClickTimeRef = useRef(0);
  // Add previous main word ref to track transitions
  const prevMainWordRef = useRef<string | null>(null);

  // Improve the setupSvgDimensions function to ensure proper sizing
  const setupSvgDimensions = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    
    // Set explicit dimensions on the SVG
    svg.attr("width", width)
       .attr("height", height)
       .attr("viewBox", `0 0 ${width} ${height}`);
    
    return { width, height };
  }, []);

  // Update setupZoom for even smoother interactions
  const setupZoom = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    
    // Create zoom behavior with improved settings
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 3]) // More reasonable scale limits
      // Smoother zoom with interpolation
      .interpolate(d3.interpolateZoom)
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        // Don't apply zoom during drag operations (prevents conflicts)
        if (!isDraggingRef.current) {
          g.attr("transform", event.transform.toString());
        }
      })
      .filter(event => {
        // Only handle zoom events if we're not dragging a node
        // and not in middle of a transition
        if (isDraggingRef.current || isTransitioningRef.current) return false;
        
        // Allow mouse wheel, double-click, and touch events for zooming
        return !event.ctrlKey && !event.button;
      });

    // Apply zoom to SVG with easing functions
    svg.call(zoom);
    
    // Calculate center transform and apply it smoothly
    const initialTransform = d3.zoomIdentity.translate(width / 2, height / 2);
    svg.transition()
      .duration(150) // Even faster initial centering for immediate feedback
      .ease(d3.easeExpOut) // Use exponential ease out for more natural feel
      .call(zoom.transform, initialTransform);
    
    return zoom;
  }, []);

  // Add the getNodeRadius function before setupSimulation
  const getNodeRadius = useCallback((node: CustomNode) => {
    // Return different radius based on node type
    if (node.id === mainWord) return 25; // Main word is larger
    if (node.group === 'root') return 20; // Root words are medium
    if (node.connections && node.connections > 3) return 18; // Highly connected
    return 14; // Default size
  }, [mainWord]);

  // Move setupSimulation function before its usage
  const setupSimulation = useCallback((nodes: CustomNode[], links: CustomLink[]) => {
    try {
      // Create a new simulation with optimized parameters
      const simulation = d3.forceSimulation<CustomNode>()
        .nodes(nodes)
        .alphaDecay(0.015)    // Slower decay for more gradual settling
        .velocityDecay(0.55)  // Higher damping for smoother motion
        .force("link", d3.forceLink<CustomNode, CustomLink>()
          .id(d => d.id)
          .links(links)
          .distance(link => {
            // Adjust link distance based on relationship type
            switch (link.relationship) {
              case "synonym": return 80;
              case "antonym": return 120;
              case "derived": return 100;
              default: return 90;
            }
          })
          .strength(0.6))     // Moderate link strength
        .force("charge", d3.forceManyBody<CustomNode>()
          .strength(-120)     // Repulsive force
          .distanceMax(300)   // Limit long-range effects
          .theta(0.9))        // Accuracy/performance tradeoff
        .force("collide", d3.forceCollide<CustomNode>()
          .radius(d => getNodeRadius(d) + 2)
          .strength(0.7)
          .iterations(2))
        .force("center", d3.forceCenter(0, 0))
        .force("x", d3.forceX(0).strength(0.05)) // Gentle force toward center
        .force("y", d3.forceY(0).strength(0.05));

      // Update node positions on each tick
      simulation.on("tick", () => {
        if (!svgRef.current) return;

        // Use safely typed selections
        const nodeSelection = d3.select(svgRef.current).selectAll<SVGGElement, CustomNode>(".node");
        const linkSelection = d3.select(svgRef.current).selectAll<SVGLineElement, CustomLink>(".link");
        
        // Apply transforms safely
        nodeSelection.attr("transform", d => {
          const x = d.x || 0;
          const y = d.y || 0;
          return `translate(${x}, ${y})`;
        });

        linkSelection
          .attr("x1", d => {
            return (typeof d.source === 'object' && d.source !== null) ? d.source.x || 0 : 0;
          })
          .attr("y1", d => {
            return (typeof d.source === 'object' && d.source !== null) ? d.source.y || 0 : 0;
          })
          .attr("x2", d => {
            return (typeof d.target === 'object' && d.target !== null) ? d.target.x || 0 : 0;
          })
          .attr("y2", d => {
            return (typeof d.target === 'object' && d.target !== null) ? d.target.y || 0 : 0;
          });
      });

      // Store the simulation in state for later access
      setSimulationObject(simulation);
      return simulation;
    } catch (error) {
      console.error("Error setting up simulation:", error);
      setError("Failed to set up graph simulation. Please try again.");
      return null;
    }
  }, [getNodeRadius]);

  /**
   * Creates the drag behavior for nodes
   */
  const createDragBehavior = useCallback((simulation: d3.Simulation<CustomNode, CustomLink>) => {
    if (!simulation) return d3.drag<SVGGElement, CustomNode>();
    
    // Return a properly configured drag behavior
    return d3.drag<SVGGElement, CustomNode>()
      .on("start", (event, d) => {
        try {
          // Stop any ongoing simulation when drag starts
          if (!event.active) simulation.alphaTarget(0.3).restart();
          
          // Set the fixed position to current position
          d.fx = d.x;
          d.fy = d.y;
          
          // Safely get the node element
          const targetElement = event.sourceEvent.target;
          if (!targetElement) return;
          
          const nodeElement = targetElement.closest ? 
            d3.select(targetElement.closest(".node")) : null;
            
          if (nodeElement && !nodeElement.empty()) {
            nodeElement.classed("dragging", true);
            
            const circleSelection = nodeElement.select("circle");
            if (!circleSelection.empty()) {
              circleSelection
                .transition()
                .duration(150)
                .attr("r", getNodeRadius(d) * 1.1);
            }
          }
            
          // Highlight connected links safely
          d3.selectAll(".link").each(function(linkData: any) {
            if (!linkData) return;
            
            try {
              const link = d3.select(this);
              if (link.empty()) return;
              
              const source = typeof linkData.source === 'object' ? 
                (linkData.source ? linkData.source.id : null) : linkData.source;
              const target = typeof linkData.target === 'object' ? 
                (linkData.target ? linkData.target.id : null) : linkData.target;
                
              if (source === d.id || target === d.id) {
                link.classed("connected-link", true)
                  .transition()
                  .duration(150)
                  .attr("stroke-opacity", 0.8)
                  .attr("stroke-width", 2);
              }
            } catch (err) {
              console.error("Error highlighting link:", err);
            }
          });
            
          isDraggingRef.current = true;
        } catch (error) {
          console.error("Error in drag start:", error);
        }
      })
      .on("drag", (event, d) => {
        try {
          // Update the fixed position
          d.fx = event.x;
          d.fy = event.y;
          // No need to manually update DOM elements - the simulation's tick will do that
        } catch (error) {
          console.error("Error in drag move:", error);
        }
      })
      .on("end", (event, d) => {
        try {
          // Gradually return to simulation if not active
          if (!event.active) simulation.alphaTarget(0);
          
          // Keep the node fixed in its final position if pinned
          if (!d.pinned) {
            d.fx = null;
            d.fy = null;
          }
          
          // Safely get the node element
          const targetElement = event.sourceEvent.target;
          if (!targetElement) return;
          
          const nodeElement = targetElement.closest ? 
            d3.select(targetElement.closest(".node")) : null;
            
          if (nodeElement && !nodeElement.empty()) {
            nodeElement.classed("dragging", false);
            
            const circleSelection = nodeElement.select("circle");
            if (!circleSelection.empty()) {
              circleSelection
                .transition()
                .duration(300)
                .attr("r", (d: any) => getNodeRadius(d));
            }
          }
            
          // Reset connected links safely
          d3.selectAll(".connected-link")
            .classed("connected-link", false)
            .transition()
            .duration(300)
            .attr("stroke-opacity", 0.3)
            .attr("stroke-width", 1.5);
            
          isDraggingRef.current = false;
        } catch (error) {
          console.error("Error in drag end:", error);
          isDraggingRef.current = false;
        }
      });
  }, [getNodeRadius]);

  /**
   * Creates the links between nodes
   */
  const createLinks = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    return g.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("class", "link")
      .attr("stroke", d => {
        switch (d.relationship) {
          case "synonym": return theme === "dark" ? "#4CAF50" : "#2E7D32";
          case "antonym": return theme === "dark" ? "#F44336" : "#C62828";
          case "derived": return theme === "dark" ? "#2196F3" : "#1565C0";
          case "related": return theme === "dark" ? "#9C27B0" : "#6A1B9A";
          default: return theme === "dark" ? "#78909C" : "#546E7A";
        }
      })
      .attr("stroke-opacity", 0.3)
      .attr("stroke-width", 1.5);
  }, [links, theme]);

  /**
   * Creates the nodes of the graph
   */
  const createNodes = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>, simulation: d3.Simulation<CustomNode, CustomLink> | null) => {
    // Create drag behavior
    const drag = simulation ? createDragBehavior(simulation) : null;
    
    // Create the node groups
    const nodeGroups = g.append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(filteredNodes)
      .join("g")
      .attr("class", d => `node ${d.id === mainWord ? "main-node" : ""} ${d.group}`);
    
    // Apply drag behavior if available
    if (drag) {
      nodeGroups.call(drag as any);
    }
    
    // Add circles for nodes
    nodeGroups.append("circle")
      .attr("r", d => getNodeRadius(d))
      .attr("fill", d => getNodeColor(d.group))
      .attr("stroke", theme === "dark" ? "#DDD" : "#333")
      .attr("stroke-width", d => d.pinned ? 2 : 1)
      .attr("stroke-dasharray", d => d.pinned ? "3,2" : "none");
    
    // Add text labels
    nodeGroups.append("text")
      .attr("dy", ".3em")
      .attr("text-anchor", "middle")
      .text(d => d.word)
      .attr("font-size", d => `${Math.min(16, 10 + getNodeRadius(d) / 2)}px`)
      .attr("font-weight", d => d.id === mainWord ? "bold" : "normal")
      .attr("fill", theme === "dark" ? "#FFF" : "#000")
      .style("pointer-events", "none");
      
    // Add interaction enhancers 
    nodeGroups.append("title")
      .text(d => `${d.word}\nGroup: ${d.group}`);
      
    return nodeGroups;
  }, [filteredNodes, createDragBehavior, getNodeRadius, getNodeColor, theme, mainWord]);

  // Simplified node interactions
  const setupNodeInteractions = useCallback((
    nodeSelection: d3.Selection<d3.BaseType, CustomNode, SVGGElement, unknown>,
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>
  ) => {
    try {
      nodeSelection
        .on("click", (event, d) => {
          // Prevent event propagation
          event.stopPropagation();
          
          // Don't register clicks during dragging
          if (isDraggingRef.current) return;
          
          // Handle node selection
          setSelectedNodeId(d.id);
          
          // Highlight this node and its connections
          d3.selectAll(".node").classed("selected", false);
          d3.selectAll(".link").classed("highlighted", false);
          
          const currentTarget = event.currentTarget;
          if (!currentTarget) return;
          
          d3.select(currentTarget as Element)
            .classed("selected", true)
            .transition()
            .duration(200)
            .select("circle")
            .attr("r", getNodeRadius(d) * 1.2);
          
          // Highlight connected links and nodes safely
          d3.selectAll(".link")
            .each(function(linkData: any) {
              if (!linkData) return;
              
              try {
                const link = d3.select(this);
                if (link.empty()) return;
                
                const source = typeof linkData.source === 'object' ? 
                  (linkData.source ? linkData.source.id : null) : linkData.source;
                const target = typeof linkData.target === 'object' ? 
                  (linkData.target ? linkData.target.id : null) : linkData.target;
                
                if (source === d.id || target === d.id) {
                  link.classed("highlighted", true)
                    .transition()
                    .duration(200)
                    .attr("stroke-opacity", 0.8)
                    .attr("stroke-width", 2);
                    
                  // Find and highlight connected nodes
                  const connectedId = source === d.id ? target : source;
                  if (connectedId) {
                    d3.selectAll(`.node`)
                      .filter(function(n: any) {
                        return n && n.id === connectedId;
                      })
                      .classed("connected", true);
                  }
                }
              } catch (err) {
                console.error("Error highlighting connected elements:", err);
              }
            });
          
          // Show word details
          if (onNodeClick) {
            onNodeClick(d.id);
          }
        })
        .on("mouseover", (event, d) => {
          try {
            // Skip hover effects during dragging
            if (isDraggingRef.current) return;
            
            const currentTarget = event.currentTarget;
            if (!currentTarget) return;
            
            d3.select(currentTarget as Element)
              .transition()
              .duration(150)
              .select("circle")
              .attr("r", getNodeRadius(d) * 1.1);

            setHoveredNode({ ...d });
          } catch (error) {
            console.error("Error in node mouseover:", error);
          }
        })
        .on("mouseout", (event, d) => {
          try {
            // Skip hover effects during dragging
            if (isDraggingRef.current) return;
            
            // Only reset radius if not the selected node
            if (d.id !== selectedNodeId) {
              const currentTarget = event.currentTarget;
              if (!currentTarget) return;
              
              d3.select(currentTarget as Element)
                .transition()
                .duration(200)
                .select("circle")
                .attr("r", getNodeRadius(d));
            }

            setHoveredNode(null);
          } catch (error) {
            console.error("Error in node mouseout:", error);
          }
        })
        .on("dblclick", (event, d) => {
          try {
            // Handle double click to set as main word
            if (onNodeClick) {
              onNodeClick(d.id);
            }
          } catch (error) {
            console.error("Error in node double-click:", error);
          }
        });
      
      return nodeSelection;
    } catch (error) {
      console.error("Error setting up node interactions:", error);
      return nodeSelection;
    }
  }, [isDraggingRef, selectedNodeId, onNodeClick, getNodeRadius]);

  // Create a better function to center on the main word
  const centerOnMainWord = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    if (!zoomRef.current) return;
    
    // Only center if not currently dragging
    if (isDraggingRef.current) return;
    
    const mainNode = filteredNodes.find(n => n.id === mainWord);
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    
    if (mainNode && mainNode.x !== undefined && mainNode.y !== undefined) {
      // Calculate distance from center
      const dist = Math.sqrt(mainNode.x * mainNode.x + mainNode.y * mainNode.y);
      if (dist > 20) { // Only recenter if it's drifted significantly
        // Flag that we're doing a transition
        isTransitioningRef.current = true;
        
        // Create transform that centers the main node
        const newTransform = d3.zoomIdentity
          .translate(width/2 - mainNode.x, height/2 - mainNode.y)
          .scale(1);
          
        // Apply transform with a smooth transition
        svg.transition()
          .duration(600) // Slightly faster for better responsiveness
          .ease(d3.easeCubicInOut) // Smoother easing
          .call(zoomRef.current.transform, newTransform)
          .on("end", () => {
            // Reset transition flag when done
            isTransitioningRef.current = false;
          });
      }
    }
  }, [mainWord, filteredNodes]);

  // Update the updateGraph function to incorporate more seamless transitions
  const updateGraph = useCallback(() => {
    if (!svgRef.current) return;

    setIsLoading(true);
    setError(null);

    try {
      // Clear the previous content and stop any running simulation
      if (simulationObject) {
        simulationObject.stop();
      }
      
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      // Set up dimensions
      const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
      const width = containerRect ? containerRect.width : 800;
      const height = containerRect ? containerRect.height : 600;
      
      // Ensure SVG has proper dimensions
      svg.attr("width", width)
         .attr("height", height);
      
      // Create a root group that will contain all elements
      const g = svg.append("g")
        .attr("class", "graph-content")
        .attr("transform", `translate(${width/2}, ${height/2})`);
      
      // Set up zoom behavior
      const zoom = setupZoom(svg, g);
      zoomRef.current = zoom;

      // Check if we have valid data to render
      if (filteredNodes.length === 0) {
        setIsLoading(false);
        setError("No nodes to display. Please try a different word or adjust the depth/breadth settings.");
        return;
      }

      // Set up the simulation with our filtered nodes and links
      const sim = setupSimulation(filteredNodes, links);

      // Create links first (so they appear behind nodes)
      const link = createLinks(g);
      
      // Create nodes with drag behavior
      const node = createNodes(g, sim);

      // Set up node interactions if we have a valid simulation
      if (sim) {
        setupNodeInteractions(node, svg);

        // Update the simulation with our data
        sim.nodes(filteredNodes);
        const linkForce = sim.force<d3.ForceLink<CustomNode, CustomLink>>("link");
        if (linkForce) {
          // Filter links to ensure they connect to nodes that exist in our filtered set
          const validLinks = links.filter(link => {
            try {
              const sourceId = typeof link.source === 'object' ? 
                (link.source ? link.source.id : '') : link.source;
              const targetId = typeof link.target === 'object' ? 
                (link.target ? link.target.id : '') : link.target;
                
              return sourceId && targetId && 
                filteredNodes.some(node => node.id === sourceId) && 
                filteredNodes.some(node => node.id === targetId);
            } catch (error) {
              console.error("Error filtering link:", error);
              return false;
            }
          });
          
          linkForce.links(validLinks);
        }

        // Restart the simulation to begin layout
        sim.alpha(1).restart();

        // Center on main word after a brief delay
        setTimeout(() => {
          centerOnMainWord(svg);
        }, 500);
      }
      
      // Add a simple legend for relationship types
      const legendContainer = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - 120}, 20)`);
      
      const relationTypes = [
        { type: 'main', label: 'Main Word' },
        { type: 'root', label: 'Root Word' },
        { type: 'synonym', label: 'Synonym' },
        { type: 'antonym', label: 'Antonym' },
        { type: 'derived', label: 'Derivative' },
        { type: 'related', label: 'Related' }
      ];
      
      // Add background for legend
      legendContainer.append("rect")
        .attr("width", 110)
        .attr("height", relationTypes.length * 20 + 30)
        .attr("rx", 5)
        .attr("ry", 5)
        .attr("fill", theme === "dark" ? "rgba(0, 0, 0, 0.7)" : "rgba(255, 255, 255, 0.9)")
        .attr("stroke", theme === "dark" ? "rgba(255, 255, 255, 0.2)" : "rgba(0, 0, 0, 0.1)");
      
      // Add title
      legendContainer.append("text")
        .attr("x", 55)
        .attr("y", 15)
        .attr("text-anchor", "middle")
        .attr("font-weight", "bold")
        .attr("font-size", "10px")
        .attr("fill", theme === "dark" ? "#fff" : "#333")
        .text("Relationship Types");
      
      // Add legend items
      relationTypes.forEach((item, i) => {
        const g = legendContainer.append("g")
          .attr("transform", `translate(10, ${i * 20 + 30})`);
        
        // Color dot
        g.append("circle")
          .attr("r", 5)
          .attr("fill", getNodeColor(item.type));
        
        // Label
        g.append("text")
          .attr("x", 10)
          .attr("y", 3)
          .attr("font-size", "9px")
          .attr("fill", theme === "dark" ? "#fff" : "#333")
          .text(item.label);
      });

      setIsLoading(false);
      
      // Return cleanup function
      return () => {
        if (sim) sim.stop();
      };
    } catch (err) {
      console.error("Error updating graph:", err);
      setError("An error occurred while updating the graph. Please try again.");
      setIsLoading(false);
    }
  }, [
    filteredNodes, 
    links, 
    theme, 
    getNodeColor, 
    createLinks, 
    createNodes, 
    setupNodeInteractions, 
    setupSimulation, 
    setupZoom, 
    centerOnMainWord,
    simulationObject
  ]);

  // Helper function to add a subtle grid background
  const addGridBackground = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>, width: number, height: number) => {
    const gridSize = 30;
    const lineColor = theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)';
    
    // Create a group for the grid
    const grid = g.append("g").attr("class", "grid-background");
    
    // Add vertical grid lines
    for (let x = 0; x <= width; x += gridSize) {
      grid.append("line")
        .attr("x1", x)
        .attr("y1", 0)
        .attr("x2", x)
        .attr("y2", height)
        .attr("stroke", lineColor)
        .attr("stroke-width", 1);
    }
    
    // Add horizontal grid lines
    for (let y = 0; y <= height; y += gridSize) {
      grid.append("line")
        .attr("x1", 0)
        .attr("y1", y)
        .attr("x2", width)
        .attr("y2", y)
        .attr("stroke", lineColor)
        .attr("stroke-width", 1);
    }
  }, [theme]);

  const handleZoom = useCallback((scale: number) => {
    if (zoomRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition()
        .duration(300) // Slightly shorter for better responsiveness
        .ease(d3.easeCubicInOut) // Smoother easing function
        .call(zoomRef.current.scaleBy, scale);
    }
  }, []);

  const handleResetZoom = useCallback(() => {
    if (zoomRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current);
      const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
      const width = containerRect ? containerRect.width : 800;
      const height = containerRect ? containerRect.height : 600;
      
      // Reset zoom with a smooth transition to center
      svg.transition()
        .duration(500) // Slightly shorter for better responsiveness
        .ease(d3.easeCubicInOut) // Smoother easing function
        .call(zoomRef.current.transform, d3.zoomIdentity.translate(width/2, height/2));
        
      // After resetting zoom, center on main word
      setTimeout(() => centerOnMainWord(svg), 600);
    }
  }, [centerOnMainWord]);

  const handleDepthChange = useCallback((event: Event, newValue: number | number[]) => {
    const newDepth = Array.isArray(newValue) ? newValue[0] : newValue;
    setDepth(newDepth);
    onNetworkChange(newDepth, breadth);
  }, [onNetworkChange, breadth]);

  const handleBreadthChange = useCallback((event: Event, newValue: number | number[]) => {
    const newBreadth = Array.isArray(newValue) ? newValue[0] : newValue;
    setBreadth(newBreadth);
    onNetworkChange(depth, newBreadth);
  }, [onNetworkChange, depth]);

  const legendItems = useMemo(
    () => [
      { key: "main", label: "Main Word" },
      { key: "derivative", label: "Derivative" },
      { key: "etymology", label: "Etymology" },
      { key: "root", label: "Root" },
      { key: "associated", label: "Associated Word" },
      { key: "other", label: "Other" },
    ],
    []
  );

  useEffect(() => {
    const cleanupFunction = updateGraph();
    window.addEventListener("resize", updateGraph);
    return () => {
      if (typeof cleanupFunction === "function") {
        cleanupFunction();
      }
      window.removeEventListener("resize", updateGraph);
    };
  }, [updateGraph, wordNetwork, mainWord, depth, breadth]);

  const renderTooltip = useCallback(() => {
    try {
      if (!hoveredNode || typeof hoveredNode.x === 'undefined' || typeof hoveredNode.y === 'undefined') {
        return null;
      }
      
      // Get the full node data
      if (!wordNetwork || !wordNetwork.nodes) return null;
      
      const nodeData = wordNetwork.nodes.find(n => n.word === hoveredNode.id);
      if (!nodeData) return null;
      
      // Extract the most relevant definition
      let definition = "No definition available";
      let partOfSpeech = "";
      
      // First check nodeData's definitions
      if (nodeData.definitions && Array.isArray(nodeData.definitions) && nodeData.definitions.length > 0) {
        // Try to get a definition object first
        const defObj = nodeData.definitions.find(d => typeof d === 'object' && d !== null);
        if (defObj) {
          // Extract text from the definition object
          if ((defObj as any).text) {
            definition = (defObj as any).text;
          } else if ((defObj as any).definition_text) {
            definition = (defObj as any).definition_text;
          }
          
          // Try to get part of speech
          if ((defObj as any).pos) {
            partOfSpeech = (defObj as any).pos;
          } else if ((defObj as any).part_of_speech) {
            partOfSpeech = (defObj as any).part_of_speech;
          }
        } else {
          // If no object, try to get a string definition
          const defString = nodeData.definitions.find(d => typeof d === 'string');
          if (defString) definition = defString;
        }
      }
      
      // If still no definition, check the info object
      if (definition === "No definition available" && hoveredNode.info && hoveredNode.info.definitions) {
        if (Array.isArray(hoveredNode.info.definitions) && hoveredNode.info.definitions.length > 0) {
          const infoDefObj = hoveredNode.info.definitions.find((d: any) => typeof d === 'object' && d !== null);
          if (infoDefObj) {
            if ((infoDefObj as any).text) {
              definition = (infoDefObj as any).text;
            } else if ((infoDefObj as any).definition_text) {
              definition = (infoDefObj as any).definition_text;
            }
          } else {
            const infoDefString = hoveredNode.info.definitions.find((d: any) => typeof d === 'string');
            if (infoDefString) definition = infoDefString;
          }
        }
      }
      
      // Get relationship label
      let relationshipLabel = "Associated Word"; // Default
      if (hoveredNode.group) {
        relationshipLabel = hoveredNode.group.charAt(0).toUpperCase() + 
          hoveredNode.group.slice(1).replace(/_/g, ' ');
        if (hoveredNode.group === "main") {
          relationshipLabel = "Main Word";
        }
      }
      
      return (
        <div
          className="node-tooltip"
          style={{
            position: "absolute",
            left: `${hoveredNode.x + 10}px`,
            top: `${hoveredNode.y + 10}px`,
            background: theme === "dark" ? "rgba(13, 17, 23, 0.95)" : "rgba(255, 255, 255, 0.95)",
            border: `2px solid ${getNodeColor(hoveredNode.group)}`,
            borderRadius: "6px",
            padding: "10px 12px",
            maxWidth: "250px",
            zIndex: 1000,
            pointerEvents: "none",
            fontFamily: "system-ui, -apple-system, sans-serif"
          }}
        >
          <div style={{ 
            display: "flex", 
            justifyContent: "space-between", 
            alignItems: "center",
            marginBottom: "6px"
          }}>
            <h4 style={{ 
              margin: 0, 
              color: theme === "dark" ? "#ffffff" : "#333333",
              fontSize: "16px",
              fontWeight: "bold"
            }}>
              {hoveredNode.id}
            </h4>
            {partOfSpeech && (
              <span style={{ 
                fontSize: "12px", 
                color: theme === "dark" ? "#a0a0a0" : "#666666",
                fontStyle: "italic",
                padding: "2px 6px",
                background: theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.05)",
                borderRadius: "4px"
              }}>
                {partOfSpeech}
              </span>
            )}
          </div>
          
          <div style={{
            padding: "6px 0",
            borderTop: `1px solid ${theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
            borderBottom: definition !== "No definition available" ? 
              `1px solid ${theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}` : "none",
            marginBottom: definition !== "No definition available" ? "6px" : 0
          }}>
            <div style={{ 
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}>
              <span style={{ 
                display: "inline-block",
                width: "10px",
                height: "10px",
                borderRadius: "50%",
                background: getNodeColor(hoveredNode.group)
              }}></span>
              <span style={{ 
                fontSize: "13px", 
                color: getNodeColor(hoveredNode.group),
                fontWeight: "500"
              }}>
                {relationshipLabel}
              </span>
            </div>
          </div>
          
          {definition !== "No definition available" && (
            <p style={{ 
              margin: 0, 
              fontSize: "13px", 
              color: theme === "dark" ? "#e0e0e0" : "#333333",
              lineHeight: "1.4",
            }}>
              {definition}
            </p>
          )}
          
          {definition === "No definition available" && (
            <p style={{ 
              margin: 0, 
              fontSize: "13px", 
              color: theme === "dark" ? "#a0a0a0" : "#777777",
              fontStyle: "italic",
              lineHeight: "1.4",
            }}>
              No definition available
            </p>
          )}
          
          <div style={{
            fontSize: "11px",
            marginTop: "6px",
            color: theme === "dark" ? "#8b949e" : "#666666",
            fontStyle: "italic"
          }}>
            Click to make this the main word
          </div>
        </div>
      );
    } catch (error) {
      console.error("Error rendering tooltip:", error);
      return null;
    }
  }, [hoveredNode, wordNetwork, theme, getNodeColor]);

  // Effect to track main word changes and handle smooth transitions
  useEffect(() => {
    if (prevMainWordRef.current && prevMainWordRef.current !== mainWord) {
      // A new main word was selected, ensure graph is centered
      if (svgRef.current) {
        setTimeout(() => {
          try {
            // Fix: Check that svgRef.current is not null before passing to d3.select
            if (svgRef.current) {
              centerOnMainWord(d3.select(svgRef.current));
            }
          } catch (error) {
            console.error("Error centering on main word:", error);
          }
        }, 400); // Slightly faster for better responsiveness
      }
    }
    
    // Update the reference
    prevMainWordRef.current = mainWord;
  }, [mainWord, centerOnMainWord]);

  // Return early if network is invalid
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
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Loading...</p>
          </div>
        )}
        {error && (
          <div className="error-overlay">
            <p className="error-message">{error}</p>
            <button onClick={updateGraph} className="retry-button">Retry</button>
          </div>
        )}
        <svg ref={svgRef} className={`graph-svg ${isLoading ? 'loading' : 'loaded'}`}>
          {/* SVG content will be rendered here by D3 */}
        </svg>
      </div>
      <div className="controls-container">
        <div className="zoom-controls">
          <button onClick={() => handleZoom(1.2)} className="zoom-button">
            +
          </button>
          <button onClick={() => handleZoom(1 / 1.2)} className="zoom-button">
            -
          </button>
          <button onClick={handleResetZoom} className="zoom-button">
            Reset
          </button>
        </div>
        <div className="graph-controls">
          <div className="slider-container">
            <Typography variant="caption">Depth: {depth}</Typography>
            <Slider
              value={depth}
              onChange={handleDepthChange}
              aria-labelledby="depth-slider"
              step={1}
              marks
              min={1}
              max={5}
              size="small"
            />
          </div>
          <div className="slider-container">
            <Typography variant="caption">Breadth: {breadth}</Typography>
            <Slider
              value={breadth}
              onChange={handleBreadthChange}
              aria-labelledby="breadth-slider"
              step={1}
              marks
              min={5}
              max={20}
              size="small"
            />
          </div>
        </div>
      </div>
      {renderTooltip()}
    </div>
  );
};

export default React.memo(WordGraph);
