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
        source: sourceNode.word,
        target: targetNode.word,
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

  // Add utility function to calculate the actual position considering transformations
  const getTransformedPosition = useCallback((x: number, y: number): { x: number, y: number } => {
    if (!svgRef.current) return { x, y };
    
    try {
      // Get the transformation matrix
      const g = d3.select(svgRef.current).select('g.graph-content').node() as SVGGElement;
      if (!g) return { x, y };
      
      // Get current transformation
      const transform = d3.zoomTransform(svgRef.current);
      
      // Apply transformation to coordinates
      return {
        x: transform.applyX(x),
        y: transform.applyY(y)
      };
    } catch (error) {
      console.error("Error calculating transformed position:", error);
      return { x, y };
    }
  }, []);

  // Refined function to get proper node radius 
  const getNodeRadius = useCallback((d: CustomNode) => {
    // Base radius on node type
    let baseRadius = 12;
    
    // Main word is larger
    if (d.id === mainWord) {
      baseRadius = 20;
    } 
    // Important node types are larger
    else if (d.group === 'root' || d.group === 'root_of') {
      baseRadius = 16;
    }
    // Adjust by connections if available
    else if (d.connections) {
      const connectionBonus = Math.min(5, Math.log(d.connections + 1));
      baseRadius += connectionBonus;
    }
    
    return baseRadius;
  }, [mainWord]);

  // Better function to center the graph content
  const centerOnMainWord = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    if (!zoomRef.current || !svgRef.current) return;
    
    try {
      // Get container dimensions
      const svgElement = svgRef.current;
      const containerRect = svgElement.parentElement?.getBoundingClientRect();
      if (!containerRect) return;
      
      const width = containerRect.width;
      const height = containerRect.height;
      
      // Find the main word node
      const mainNode = d3.select(svgElement).select(`.node.main-node`);
      
      if (!mainNode.empty()) {
        // Get the transform attribute to extract current position
        const transform = mainNode.attr("transform");
        if (!transform) return;
        
        // Extract x and y values using a regex
        const translateMatch = /translate\(([^,]+),\s*([^)]+)\)/.exec(transform);
        if (!translateMatch) return;
        
        const nodeX = parseFloat(translateMatch[1]);
        const nodeY = parseFloat(translateMatch[2]);
        
        // Calculate the translation needed to center the main word node
        const centerX = width / 2 - nodeX;
        const centerY = height / 2 - nodeY;
        
        // Apply the transformation smoothly
        svg.transition()
          .duration(500)
          .call(zoomRef.current.transform, 
            d3.zoomIdentity
              .translate(centerX, centerY)
              .scale(1) // Reset scale to 1
          );
      } else {
        // If main node not found, center the graph in general
        svg.transition()
          .duration(500)
          .call(zoomRef.current.transform, 
            d3.zoomIdentity
              .translate(width / 2, height / 2)
              .scale(1)
          );
      }
    } catch (error) {
      console.error("Error centering on main word:", error);
    }
  }, []);

  // Improved zoom setup
  const setupZoom = useCallback((
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    g: d3.Selection<SVGGElement, unknown, null, undefined>
  ) => {
    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.25, 4]) // Allow more zooming range
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
        
        // Force tooltip update on zoom
        if (hoveredNode) {
          setHoveredNode({...hoveredNode});
        }
      });
      
    // Initialize with a slight zoom in for better visibility
    svg.call(zoom)
       .call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(1));
    
    // Double click to zoom in and center on clicked point
    svg.on("dblclick.zoom", (event) => {
      const transform = d3.zoomTransform(svg.node()!);
      const newScale = transform.k * 1.5;
      
      // Get the container dimensions
      const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
      const width = containerRect ? containerRect.width : 800;
      const height = containerRect ? containerRect.height : 600;
      
      // Center on point and zoom in
      const coords = d3.pointer(event);
      const x = coords[0];
      const y = coords[1];
      
      svg.transition()
         .duration(400)
         .call(zoom.transform, d3.zoomIdentity
           .translate(width/2 - newScale * x, height/2 - newScale * y)
           .scale(newScale)
         );
    });

    return zoom;
  }, [hoveredNode]);

  // Update setupSimulation to ensure main word is in center
  const setupSimulation = useCallback((nodes: CustomNode[], links: CustomLink[]) => {
    try {
      if (nodes.length === 0) return null;
      
      // Create a new simulation with optimized parameters
      const simulation = d3.forceSimulation<CustomNode>()
        .nodes(nodes)
        .alphaDecay(0.014)    // Slightly slower decay for more stable layout
        .velocityDecay(0.4)   // Less damping for more natural motion 
        .alpha(0.5)           // Higher initial energy for better settling
        .alphaTarget(0)       // Target zero energy (equilibrium state)
        .force("charge", d3.forceManyBody<CustomNode>()
          .strength(d => d.id === mainWord ? -800 : -300)  // Stronger repulsion for main word
          .distanceMax(500)   // Allow longer range effects for better spacing
          .theta(0.8))        // Good balance of accuracy/performance
        .force("link", d3.forceLink<CustomNode, CustomLink>()
          .id(d => d.id)
          .links(links)
          .distance(link => {
            // Adjust link distance based on relationship type
            const source = typeof link.source === 'object' ? link.source.id : link.source;
            const target = typeof link.target === 'object' ? link.target.id : link.target;
            
            // Keep main word's connections at a consistent distance
            if (source === mainWord || target === mainWord) {
              return 120;
            }
            
            // Other relationships
            switch (link.relationship) {
              case "synonym": return 80;
              case "antonym": return 150;
              case "derived": return 100;
              case "root":
              case "root_of": return 120;
              default: return 100;
            }
          })
          .strength(link => {
            // Customize strength based on relationship
            const source = typeof link.source === 'object' ? link.source.id : link.source;
            const target = typeof link.target === 'object' ? link.target.id : link.target;
            
            // Main word connections have stronger links
            if (source === mainWord || target === mainWord) {
              return 0.7;
            }
            
            return 0.5; // Default strength
          }))
        .force("center", d3.forceCenter(0, 0).strength(0.1)) // Stronger centering force
        .force("collide", d3.forceCollide<CustomNode>()
          .radius(d => getNodeRadius(d) * 1.5) // Increased collision radius for better spacing
          .strength(0.8)                        // Stronger collision prevention
          .iterations(2))                       // Multiple iterations for stability
        .force("x", d3.forceX(0).strength(0.05))  // Stronger force toward center x
        .force("y", d3.forceY(0).strength(0.05)); // Stronger force toward center y

      // Find and initialize the main word node in the center
      const mainNode = nodes.find(n => n.id === mainWord);
      if (mainNode) {
        // Position it at center initially
        mainNode.x = 0;
        mainNode.y = 0;
        mainNode.fx = 0;
        mainNode.fy = 0;
      }

      // Update node and link positions on each tick
      simulation.on("tick", () => {
        if (!svgRef.current) return;

        // Select nodes and links with proper typing
        const nodeSelection = d3.select(svgRef.current).selectAll<SVGGElement, CustomNode>(".node");
        const linkSelection = d3.select(svgRef.current).selectAll<SVGLineElement, CustomLink>(".link");
        
        // Update node positions
        nodeSelection.attr("transform", d => {
          // Handle undefined positions gracefully
          const x = d.x || 0;
          const y = d.y || 0;
          return `translate(${x}, ${y})`;
        });

        // Update link positions
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
  }, [getNodeRadius, mainWord]);

  /**
   * Improved drag behavior for seamless interactions
   */
  const createDragBehavior = useCallback((simulation: d3.Simulation<CustomNode, CustomLink>) => {
    if (!simulation) return d3.drag<SVGGElement, CustomNode>();
    
    // Return a properly configured drag behavior
    return d3.drag<SVGGElement, CustomNode>()
      .on("start", (event, d) => {
        try {
          // Reset any currently running simulation to make it responsive
          if (!event.active) simulation.alphaTarget(0.2).restart();
          
          // Fix the node position at current coordinates
          d.fx = d.x;
          d.fy = d.y;
          
          // Add visual feedback for dragging state
          const currentTarget = event.sourceEvent.currentTarget;
          let nodeElement;
          
          if (currentTarget && currentTarget instanceof Element) {
            nodeElement = d3.select(currentTarget.closest('.node'));
          }
          
          if (nodeElement && !nodeElement.empty()) {
            nodeElement.classed("dragging", true);
            
            const circle = nodeElement.select("circle");
            if (!circle.empty()) {
              circle
                .transition()
                .duration(50) // Quick feedback
                .attr("r", getNodeRadius(d) * 1.2)
                .attr("stroke-width", 2);
            }
          }
          
          // Highlight links that connect to this node
          d3.selectAll(".link").each(function() {
            const linkData = d3.select(this).datum() as any;
            if (!linkData) return;
            
            const source = typeof linkData.source === 'object' ? 
              linkData.source?.id : linkData.source;
            const target = typeof linkData.target === 'object' ? 
              linkData.target?.id : linkData.target;
              
            if (source === d.id || target === d.id) {
              d3.select(this)
                .classed("highlighted", true)
                .transition()
                .duration(50)
                .attr("stroke-opacity", 0.8)
                .attr("stroke-width", 2);
            }
          });
          
          isDraggingRef.current = true;
        } catch (error) {
          console.error("Error starting drag:", error);
        }
      })
      .on("drag", (event, d) => {
        try {
          // Update node position to follow the cursor
          d.fx = event.x;
          d.fy = event.y;
          
          // Update related nodes to maintain relationships
          // This creates a more cohesive feel as connections follow along
          d3.selectAll(".link").each(function() {
            const linkData = d3.select(this).datum() as any;
            if (!linkData) return;
            
            // Skip processing if not related to current node
            const source = typeof linkData.source === 'object' ? 
              linkData.source : null;
            const target = typeof linkData.target === 'object' ? 
              linkData.target : null;
              
            // Apply a slight pull toward the dragged node
            if (source && source.id === d.id && target) {
              // Move target slightly toward source
              if (!target.fx && !target.fy) { // Only pull unfixed nodes
                const pullFactor = 0.03; // Gentle pull
                target.x = target.x! + (d.fx! - target.x!) * pullFactor;
                target.y = target.y! + (d.fy! - target.y!) * pullFactor;
              }
            } else if (target && target.id === d.id && source) {
              // Move source slightly toward target
              if (!source.fx && !source.fy) { // Only pull unfixed nodes
                const pullFactor = 0.03; // Gentle pull
                source.x = source.x! + (d.fx! - source.x!) * pullFactor;
                source.y = source.y! + (d.fy! - source.y!) * pullFactor;
              }
            }
          });
        } catch (error) {
          console.error("Error during drag:", error);
        }
      })
      .on("end", (event, d) => {
        try {
          // Return simulation to normal
          if (!event.active) simulation.alphaTarget(0);
          
          // Release node unless it's pinned
          if (!d.pinned) {
            d.fx = null;
            d.fy = null;
          }
          
          // Reset visual state
          const currentTarget = event.sourceEvent.currentTarget;
          let nodeElement;
          
          if (currentTarget && currentTarget instanceof Element) {
            nodeElement = d3.select(currentTarget.closest('.node'));
          }
          
          if (nodeElement && !nodeElement.empty()) {
            nodeElement.classed("dragging", false);
            
            const circle = nodeElement.select("circle");
            if (!circle.empty()) {
              circle
                .transition()
                .duration(300)
                .attr("r", getNodeRadius(d))
                .attr("stroke-width", d.pinned ? 2 : 1);
            }
          }
          
          // Reset highlighted links
          d3.selectAll(".link.highlighted")
            .classed("highlighted", false)
            .transition()
            .duration(300)
            .attr("stroke-opacity", 0.4)
            .attr("stroke-width", 1.5);
          
          isDraggingRef.current = false;
        } catch (error) {
          console.error("Error ending drag:", error);
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

  // Improved updateGraph for better centering
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
      // Position it explicitly at the center of the SVG
      const g = svg.append("g")
        .attr("class", "graph-content")
        .attr("transform", `translate(${width/2}, ${height/2})`);
      
      // Set up zoom behavior with proper centering
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

        // Important: Make the main word the center
        const mainWordNode = filteredNodes.find(n => n.id === mainWord);
        if (mainWordNode) {
          // Fix the main word at the center initially
          mainWordNode.fx = 0;
          mainWordNode.fy = 0;
          
          // Then release it after initial stabilization
          setTimeout(() => {
            if (mainWordNode) {
              // Only release if not pinned
              if (!mainWordNode.pinned) {
                mainWordNode.fx = null;
                mainWordNode.fy = null;
              }
              // Center the view on the main word
              centerOnMainWord(svg);
            }
          }, 1000);
        } else {
          // If no main word found, just center the graph generally
          setTimeout(() => {
            centerOnMainWord(svg);
          }, 800);
        }
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
    simulationObject,
    mainWord
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

  // Update the renderTooltip function to correctly position the tooltip
  const renderTooltip = useCallback(() => {
    try {
      if (!hoveredNode || typeof hoveredNode.x === 'undefined' || typeof hoveredNode.y === 'undefined') {
        return null;
      }
      
      // Get the full node data
      if (!wordNetwork || !wordNetwork.nodes) return null;
      
      const nodeData = wordNetwork.nodes.find(n => n.word === hoveredNode.id);
      if (!nodeData) return null;
      
      // Calculate correct position to account for zoom and pan
      const { x, y } = getTransformedPosition(hoveredNode.x, hoveredNode.y);
      
      // Determine container dimensions to prevent tooltip from going off-screen
      const containerRect = svgRef.current?.getBoundingClientRect();
      const width = containerRect?.width || 800;
      const height = containerRect?.height || 600;
      
      // Calculate tooltip dimensions (estimate)
      const tooltipWidth = 250;
      const tooltipHeight = 150;
      
      // Adjust position to keep tooltip inside container
      let tooltipX = x + 15;
      let tooltipY = y - 10;
      
      // Prevent tooltip from extending beyond right edge
      if (tooltipX + tooltipWidth > width) {
        tooltipX = x - tooltipWidth - 15;
      }
      
      // Prevent tooltip from extending beyond bottom edge
      if (tooltipY + tooltipHeight > height) {
        tooltipY = y - tooltipHeight - 15;
      }
      
      // Ensure tooltip isn't positioned off-screen
      tooltipX = Math.max(10, tooltipX);
      tooltipY = Math.max(10, tooltipY);
      
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
            left: `${tooltipX}px`,
            top: `${tooltipY}px`,
            background: theme === "dark" ? "rgba(13, 17, 23, 0.95)" : "rgba(255, 255, 255, 0.95)",
            border: `2px solid ${getNodeColor(hoveredNode.group)}`,
            borderRadius: "6px",
            padding: "10px 12px",
            maxWidth: "250px",
            zIndex: 1000,
            pointerEvents: "none",
            fontFamily: "system-ui, -apple-system, sans-serif",
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.15)",
            transition: "opacity 0.15s ease-in-out",
            opacity: 0.95
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
              maxHeight: "150px",
              overflow: "auto"
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
  }, [hoveredNode, wordNetwork, theme, getNodeColor, getTransformedPosition]);

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
