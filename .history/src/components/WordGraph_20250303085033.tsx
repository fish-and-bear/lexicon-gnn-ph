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
  const containerRef = useRef<HTMLDivElement>(null);
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

  const getNodeColor = useCallback((node: CustomNode | string): string => {
    // If node is a CustomNode object, extract the group
    const group = typeof node === 'object' ? node.group : node;
    
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

  // Transform coordinates from SVG space to screen space
  const getTransformedPosition = useCallback((x: number, y: number) => {
    if (!svgRef.current) return { x, y };
    
    try {
      // Get the SVG element and its CTM (Current Transformation Matrix)
      const svg = svgRef.current;
      const pt = svg.createSVGPoint();
      pt.x = x;
      pt.y = y;
      
      // Get the current transform from the zoom behavior
      const transform = d3.zoomTransform(svg);
      
      // Apply the transform to get the actual position
      const transformedX = transform.applyX(x);
      const transformedY = transform.applyY(y);
      
      return { x: transformedX, y: transformedY };
    } catch (error) {
      console.error("Error transforming coordinates:", error);
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
    if (!zoomRef.current || !svgRef.current || !containerRef.current) return;
    
    try {
      // Get container dimensions
      const containerWidth = containerRef.current.clientWidth;
      const containerHeight = containerRef.current.clientHeight;
      
      // Find the main word node using a better selector
      const mainNodeElement = Array.from(svgRef.current.querySelectorAll('.node'))
        .find(node => node.classList.contains('main-node'));
      
      if (mainNodeElement) {
        // Get the transform attribute to extract current position
        const transform = mainNodeElement.getAttribute("transform");
        if (!transform) return;
        
        // Extract x and y values using a regex
        const translateMatch = /translate\(([^,]+),\s*([^)]+)\)/.exec(transform);
        if (!translateMatch) return;
        
        const nodeX = parseFloat(translateMatch[1]);
        const nodeY = parseFloat(translateMatch[2]);
        
        // Get current zoom transform
        const currentTransform = d3.zoomTransform(svgRef.current);
        
        // Apply the transformation to center on main word
        svg.transition()
          .duration(750)
          .call(zoomRef.current.transform, 
            d3.zoomIdentity
              .translate(containerWidth/2 - nodeX, containerHeight/2 - nodeY)
              .scale(currentTransform.k) // Maintain current zoom level
          );
          
        console.log("Centering on main word:", mainWord, "at position:", nodeX, nodeY);
      } else {
        // If main node not found, center the graph in general
        svg.transition()
          .duration(750)
          .call(zoomRef.current.transform, 
            d3.zoomIdentity
              .translate(containerWidth/2, containerHeight/2)
              .scale(1)
          );
          
        console.log("Main node not found, centering graph generally");
      }
    } catch (error) {
      console.error("Error centering on main word:", error);
    }
  }, [mainWord]);

  // Improved zoom setup to initialize with centered position
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
      
    // Get container dimensions
    const containerWidth = containerRef.current?.clientWidth || 800;
    const containerHeight = containerRef.current?.clientHeight || 600;
      
    // Initialize with centered position
    svg.call(zoom)
       .call(zoom.transform, d3.zoomIdentity
         .translate(containerWidth/2, containerHeight/2)
         .scale(1));
    
    // Double click to zoom in and center on clicked point
    svg.on("dblclick.zoom", (event) => {
      const transform = d3.zoomTransform(svg.node()!);
      const newScale = transform.k * 1.5;
      
      // Get the point coordinates in the current view
      const coords = d3.pointer(event);
      const x = coords[0];
      const y = coords[1];
      
      svg.transition()
         .duration(400)
         .call(zoom.transform, d3.zoomIdentity
           .translate(containerWidth/2 - newScale * x, containerHeight/2 - newScale * y)
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
        .alphaDecay(0.01)     // Slower decay for better layout
        .velocityDecay(0.3)   // Less damping for more natural motion 
        .alpha(0.8)           // Higher initial energy for better settling
        .alphaTarget(0)       // Target zero energy (equilibrium state)
        .force("charge", d3.forceManyBody<CustomNode>()
          .strength(d => d.id === mainWord ? -1000 : -400)  // Stronger repulsion for main word
          .distanceMax(600)   // Allow longer range effects for better spacing
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
              return 150; // Increased distance for better visibility
            }
            
            // Other relationships
            switch (link.relationship) {
              case "synonym": return 100;
              case "antonym": return 180;
              case "derived": return 130;
              case "root":
              case "root_of": return 150;
              default: return 120;
            }
          })
          .strength(link => {
            // Customize strength based on relationship
            const source = typeof link.source === 'object' ? link.source.id : link.source;
            const target = typeof link.target === 'object' ? link.target.id : link.target;
            
            // Main word connections have stronger links
            if (source === mainWord || target === mainWord) {
              return 0.5; // Reduced from 0.7 to allow more natural positioning
            }
            
            return 0.3; // Reduced from 0.5 for more natural layout
          }))
        .force("center", d3.forceCenter(0, 0).strength(0.2)) // Stronger centering force
        .force("collide", d3.forceCollide<CustomNode>()
          .radius(d => getNodeRadius(d) * 1.8) // Increased collision radius for better spacing
          .strength(0.7)                        // Collision prevention strength
          .iterations(3))                       // More iterations for stability
        .force("x", d3.forceX(0).strength(0.07))  // Stronger force toward center x
        .force("y", d3.forceY(0).strength(0.07)); // Stronger force toward center y

      // Find and initialize the main word node in the center
      const mainNode = nodes.find(n => n.id === mainWord);
      if (mainNode) {
        // Position it exactly at center initially and fix it
        mainNode.x = 0;
        mainNode.y = 0;
        mainNode.fx = 0;
        mainNode.fy = 0;
        
        // Add a special visual marker for main node
        mainNode.pinned = true;
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
        })
        .attr("data-id", d => d.id); // Add data-id for easier selection
        
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
      
      // Log for debugging
      console.log("Simulation initialized with", nodes.length, "nodes and", links.length, "links");
      console.log("Main word:", mainWord, "node fixed at center");
      
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
            // Use direct parent selection instead of closest
            let parent = currentTarget;
            while (parent && !parent.classList.contains('node') && parent.parentElement) {
              parent = parent.parentElement;
            }
            
            if (parent && parent.classList.contains('node')) {
              nodeElement = d3.select(parent);
            }
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
      .attr("stroke", theme === "dark" ? "#444" : "#ddd")
      .attr("stroke-width", d => d.id === mainWord ? 3 : 1.5);
    
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

  // Improved updateGraph for better centering and functionality
  const updateGraph = useCallback(() => {
    try {
      // Skip if there's no SVG element or no data
      if (!svgRef.current || !wordNetwork?.nodes?.length) {
        return;
      }

      // Clear any previous content
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      // Get container dimensions for responsive sizing
      const parentRect = svgRef.current.parentElement?.getBoundingClientRect() || 
        { width: 800, height: 600 };
      const width = parentRect.width;
      const height = parentRect.height;

      // Set SVG dimensions to match container
      svg
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      // Create root group for graph content with initial centered position
      const g = svg.append("g")
        .attr("class", "graph-content")
        .attr("transform", `translate(${width/2}, ${height/2})`);

      // Setup zoom behavior
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.25, 4])
        .on("zoom", (event) => {
          g.attr("transform", event.transform);
        });

      svg.call(zoom);
      zoomRef.current = zoom;

      // Create simulation with nodes and links
      const simulation = setupSimulation(filteredNodes, links);

      // Create links
      const linkSelection = g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(links)
        .enter()
        .append("line")
        .attr("class", d => `link ${d.relationship}`)
        .attr("stroke", d => {
          switch (d.relationship) {
            case "synonym": return "#4CAF50";
            case "antonym": return "#F44336";
            case "derived": return "#2196F3";
            case "root":
            case "root_of": return "#9C27B0";
            default: return "#9E9E9E";
          }
        })
        .attr("stroke-width", 1.5)
        .attr("stroke-opacity", 0.6);

      // Create nodes with drag behavior attached
      const nodeGroups = g.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(filteredNodes)
        .enter()
        .append("g")
        .attr("class", d => {
          const classes = ["node"];
          if (d.id === mainWord) classes.push("main-word");
          if (d.id === selectedNodeId) classes.push("selected");
          return classes.join(" ");
        })
        .attr("data-id", d => d.id);

      // Create the drag behavior only if we have a simulation
      const drag = simulation ? d3.drag<SVGGElement, CustomNode>()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
          d3.select(event.sourceEvent.currentTarget).classed("dragging", true);
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
          isDraggingRef.current = true;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d3.select(event.sourceEvent.currentTarget).classed("dragging", false);
          // Only unfix non-pinned nodes
          if (!d.pinned) {
            d.fx = null;
            d.fy = null;
          }
          // Set a timeout to reset dragging state to prevent click events from firing immediately after drag
          setTimeout(() => {
            isDraggingRef.current = false;
          }, 100);
        }) : null;

      // Apply drag behavior if available
      if (drag) {
        nodeGroups.call(drag);
      }

      // Add circles for nodes
      nodeGroups.append("circle")
        .attr("r", d => getNodeRadius(d))
        .attr("fill", d => getNodeColor(d.group))
        .attr("stroke", theme === "dark" ? "#444" : "#ddd")
        .attr("stroke-width", d => d.id === mainWord ? 3 : 1.5);

      // Add labels for nodes
      nodeGroups.append("text")
        .attr("dy", 5)
        .attr("text-anchor", "middle")
        .text(d => d.word)
        .attr("font-size", d => `${Math.min(16, 10 + getNodeRadius(d) / 2)}px`)
        .attr("font-weight", d => d.id === mainWord ? "bold" : "normal")
        .attr("fill", theme === "dark" ? "#FFF" : "#000")
        .style("pointer-events", "none");

      // Set up node click interactions
      nodeGroups
        .on("click", (event, d) => {
          if (isDraggingRef.current) return;
          setSelectedNodeId(d.id);
          onNodeClick(d.id);
        })
        .on("mouseover", (event, d) => {
          if (isDraggingRef.current) return;
          setHoveredNode({...d});
        })
        .on("mouseout", () => {
          setHoveredNode(null);
        });

      // Set up simulation tick updates
      if (simulation) {
        simulation.on("tick", () => {
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

          // Update node positions
          nodeGroups
            .attr("transform", d => {
              // Handle undefined positions gracefully
              const x = d.x || 0;
              const y = d.y || 0;
              return `translate(${x}, ${y})`;
            });
        });
      }

      // Pin the main word in the center
      const mainNode = filteredNodes.find(n => n.id === mainWord);
      if (mainNode) {
        mainNode.fx = 0;
        mainNode.fy = 0;
        
        // Release the main node after some stabilization
        setTimeout(() => {
          if (!mainNode.pinned) {
            mainNode.fx = null;
            mainNode.fy = null;
          }
        }, 3000);
      }

      // Set the simulation object in state for external access
      setSimulationObject(simulation);

      // Return a proper cleanup function
      return () => {
        if (simulation) {
          simulation.stop();
        }
      };
    } catch (error) {
      console.error("Error updating graph:", error);
      setError("Failed to update graph visualization");
      return () => {}; // Return an empty function as cleanup
    }
  }, [
    wordNetwork, 
    filteredNodes, 
    links, 
    mainWord, 
    setupSimulation, 
    getNodeRadius,
    getNodeColor,
    theme,
    selectedNodeId,
    onNodeClick
  ]);

  // Fix the useEffect that uses updateGraph to handle the cleanup function correctly
  useEffect(() => {
    // Call updateGraph and store its cleanup function
    const cleanup = updateGraph();
    
    // Add resize event listener
    window.addEventListener("resize", updateGraph);
    
    // Return a combined cleanup function
    return () => {
      // Call the updateGraph cleanup function if it exists
      if (cleanup && typeof cleanup === "function") {
        cleanup();
      }
      // Remove the resize event listener
      window.removeEventListener("resize", updateGraph);
    };
  }, [updateGraph, wordNetwork, mainWord, depth, breadth]);

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

  // Render tooltip for hovered node
  const renderTooltip = () => {
    if (!hoveredNode || !svgRef.current) return null;
    
    // Get the node info
    const nodeInfo = hoveredNode.info;
    if (!nodeInfo) return null;

    // Position the tooltip relative to the hovered node
    const position = hoveredNode;
    if (!position.x || !position.y) return null;

    // Get the SVG element's position and transform
    const svgRect = svgRef.current.getBoundingClientRect();
    const transform = d3.zoomTransform(svgRef.current);
    
    // Calculate the tooltip position, accounting for zoom and pan
    const tooltipX = (position.x * transform.k) + transform.x + svgRect.left;
    const tooltipY = (position.y * transform.k) + transform.y + svgRect.top;
    
    return (
      <div 
        className="tooltip" 
        style={{
          position: 'fixed',
          left: `${tooltipX + 20}px`, 
          top: `${tooltipY - 10}px`,
          zIndex: 100,
          backgroundColor: theme === 'dark' ? '#333' : '#fff',
          color: theme === 'dark' ? '#fff' : '#333',
          border: '1px solid #ccc',
          borderRadius: '4px',
          padding: '8px',
          boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
          pointerEvents: 'none',
          maxWidth: '300px'
        }}
      >
        <div><strong>{hoveredNode.word}</strong></div>
        {nodeInfo.type && <div>Type: {nodeInfo.type}</div>}
        {nodeInfo.definitions && nodeInfo.definitions.length > 0 && (
          <div>{typeof nodeInfo.definitions[0] === 'string' 
            ? nodeInfo.definitions[0] 
            : (nodeInfo.definitions[0] as any)?.text || ''}</div>
        )}
      </div>
    );
  };

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
    <div className="word-graph-container" style={{ width: '100%', height: '100%', position: 'relative' }}>
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-text">Generating word network...</div>
        </div>
      )}
      
      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      
      <svg 
        ref={svgRef} 
        className="word-graph"
        style={{ width: '100%', height: '100%' }}
      />
      
      {renderTooltip()}
      
      <div className="graph-controls">
        <div className="control-group">
          <label>Depth: {depth}</label>
          <input 
            type="range" 
            min="1" 
            max="3" 
            value={depth} 
            onChange={e => setDepth(parseInt(e.target.value))} 
            onMouseUp={() => onNetworkChange(depth, breadth)}
          />
        </div>
        <div className="control-group">
          <label>Breadth: {breadth}</label>
          <input 
            type="range" 
            min="1" 
            max="10" 
            value={breadth} 
            onChange={e => setBreadth(parseInt(e.target.value))} 
            onMouseUp={() => onNetworkChange(depth, breadth)}
          />
        </div>
      </div>
    </div>
  );
};

export default React.memo(WordGraph);
