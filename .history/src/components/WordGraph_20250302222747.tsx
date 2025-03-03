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
  group: string;
  info: NetworkNode;
}

interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
  source: string | CustomNode;
  target: string | CustomNode;
  type: string;
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

  const getNodeRelation = useCallback((word: string, node: NetworkNode): string => {
    if (word === mainWord) return "main";
    
    // Direct node type check
    if (node.type === 'root') return "root";
    if (node.type === 'root_of') return "root_of";
    if (node.type === 'cognate') return "cognate";
    if (node.type === 'component_of') return "component_of";
    if (node.type === 'kaugnay') return "kaugnay";
    
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
        case 'kaugnay': return "kaugnay";
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
        return "#8d99ae"; // Gray-blue
      case "kaugnay":
        return "#ffd166"; // Golden yellow (brighter than before)
      case "component_of":
        return "#ffb01f"; // Amber (brighter than before)
      case "cognate":
        return "#9381ff"; // Light purple (brand compatible)
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
      group: getNodeRelation(node.word, node),
      info: node
    }));
  }, [wordNetwork, getNodeRelation]);

  const links: CustomLink[] = useMemo(() => {
    if (!wordNetwork || !wordNetwork.edges || !Array.isArray(wordNetwork.edges)) {
      return [];
    }
    
    return wordNetwork.edges.map(edge => {
      const sourceNode = wordNetwork.nodes.find(n => n.id === edge.source);
      const targetNode = wordNetwork.nodes.find(n => n.id === edge.target);
      
      if (!sourceNode || !targetNode) {
        return null;
      }
      
      return {
        source: sourceNode.word,
        target: targetNode.word,
        type: edge.type
      };
    }).filter(link => link !== null) as CustomLink[];
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

  const dragBehavior = d3.drag<SVGGElement, CustomNode>()
    .on("start", (event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) => {
      // Make the simulation more reactive during drag
      if (!event.active && simulationObject) {
        simulationObject.alphaTarget(0.3).restart();
      }
      
      // Set fixed position during drag
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
      
      // Visually enlarge the node being dragged for better feedback
      d3.select(event.sourceEvent.currentTarget)
        .select("circle")
        .transition()
        .duration(200)
        .attr("r", event.subject.group === "main" ? 30 : 20);
        
      // Visually emphasize connections from this node
      d3.selectAll(".link")
        .filter((d: any) => {
          const source = typeof d.source === 'object' ? d.source.id : d.source;
          const target = typeof d.target === 'object' ? d.target.id : d.target;
          return source === event.subject.id || target === event.subject.id;
        })
        .transition()
        .duration(200)
        .attr("stroke-width", 2.5)
        .attr("stroke-opacity", 1);
    })
    .on("drag", (event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) => {
      // Update node position during drag
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    })
    .on("end", (event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) => {
      // Cool down simulation when drag ends
      if (!event.active && simulationObject) {
        simulationObject.alphaTarget(0);
      }
      
      // Reset node size
      d3.select(event.sourceEvent.currentTarget)
        .select("circle")
        .transition()
        .duration(300)
        .attr("r", event.subject.group === "main" ? 25 : 15);
        
      // Reset link appearance
      d3.selectAll(".link")
        .transition()
        .duration(300)
        .attr("stroke-width", 1.5)
        .attr("stroke-opacity", 0.6);
      
      // Keep position fixed if shift key is pressed (pin the node)
      if (event.sourceEvent.shiftKey) {
        // If shift key is pressed, keep the node pinned
        d3.select(event.sourceEvent.currentTarget)
          .select("circle")
          .attr("stroke", "#fca311")
          .attr("stroke-width", 3);
      } else {
        // Allow the node to move freely again
        event.subject.fx = null;
        event.subject.fy = null;
      }
    });

  // Set up improved simulation with better physics
  const setupSimulation = useCallback((width: number, height: number) => {
    // Filter links to only include nodes that are in the filtered set
    const validLinks = links.filter(link => 
      filteredNodes.some(node => node.id === (typeof link.source === 'string' ? link.source : link.source.id)) &&
      filteredNodes.some(node => node.id === (typeof link.target === 'string' ? link.target : link.target.id))
    );

    // Create enhanced simulation with more nuanced forces
    const sim = d3
      .forceSimulation<CustomNode>(filteredNodes)
      .force(
        "link",
        d3.forceLink<CustomNode, CustomLink>(validLinks)
          .id((d) => d.id)
          .distance((link) => {
            // Customize link distance based on relationship type
            // This improves the visual organization of the graph
            const source = typeof link.source === 'object' ? link.source : { group: 'unknown' };
            const target = typeof link.target === 'object' ? link.target : { group: 'unknown' };
            
            // Main connections closer together
            if (source.group === 'main' || target.group === 'main') return 90;
            
            // Similar words closer
            if (source.group === 'synonym' || target.group === 'synonym') return 100;
            
            // Different types of words further apart
            return 130;
          })
          .strength((link) => {
            // Make some connections stronger than others
            const source = typeof link.source === 'object' ? link.source : { group: 'unknown' };
            const target = typeof link.target === 'object' ? link.target : { group: 'unknown' };
            
            // Strong connection to main word
            if (source.group === 'main' || target.group === 'main') return 0.7;
            
            // Moderate connection for synonyms
            if (source.group === 'synonym' || target.group === 'synonym') return 0.5;
            
            // Default connection strength
            return 0.3;
          })
      )
      // Charge force pushes nodes apart - use type assertion for CustomNode
      .force("charge", d3.forceManyBody<CustomNode>()
        .strength(d => d.group === 'main' ? -400 : -250)
        .distanceMax(350)
      )
      // Center force pulls nodes toward center of visualization
      .force("center", d3.forceCenter(width / 2, height / 2))
      // Collision detection prevents node overlap
      .force(
        "collide",
        d3.forceCollide<CustomNode>()
          .radius(d => d.group === 'main' ? 30 : 20)
          .strength(0.7)
          .iterations(3)
      )
      // Add x and y forces for more balanced layout
      .force("x", d3.forceX(width / 2).strength(0.05))
      .force("y", d3.forceY(height / 2).strength(0.05))
      // Slower decay for smoother animation
      .alphaDecay(0.028);
      
    // Store the simulation in state
    setSimulationObject(sim);
    
    return sim;
  }, [filteredNodes, links]);

  // Function to create more visually appealing links
  const createLinks = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const validLinks = links.filter(link => 
      filteredNodes.some(node => node.id === (typeof link.source === 'string' ? link.source : link.source.id)) &&
      filteredNodes.some(node => node.id === (typeof link.target === 'string' ? link.target : link.target.id))
    );

    return g
      .selectAll(".link")
      .data(validLinks)
      .join("line")
      .attr("class", "link")
      .attr("stroke", (d: CustomLink) => {
        // Use target node color for link
        const targetNode = typeof d.target === 'object' ? d.target : 
          filteredNodes.find(n => n.id === d.target);
        return targetNode ? getNodeColor(targetNode.group) : "#ccc";
      })
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.6)
      .attr("stroke-linecap", "round")
      // Add hover interaction
      .on("mouseover", function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr("stroke-width", 3)
          .attr("stroke-opacity", 1);
      })
      .on("mouseout", function() {
        d3.select(this)
          .transition()
          .duration(300)
          .attr("stroke-width", 1.5)
          .attr("stroke-opacity", 0.6);
      });
  }, [filteredNodes, links, getNodeColor]);

  // Function to create more visually appealing and interactive nodes
  const createNodes = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const node = g
      .selectAll<SVGGElement, CustomNode>(".node")
      .data(filteredNodes)
      .join("g")
      .attr(
        "class",
        (d: CustomNode) => `node ${d.id === selectedNodeId ? "selected" : ""} ${d.group}`
      )
      .call(dragBehavior as any);

    // Create circle with improved styling
    node.append("circle")
      .attr("r", (d: CustomNode) => (d.group === "main" ? 25 : 15))
      .attr("fill", (d: CustomNode) => getNodeColor(d.group))
      .attr("stroke", "#ffffff")
      .attr("stroke-width", (d: CustomNode) => d.id === selectedNodeId ? 3 : 2)
      .attr("stroke-opacity", 0.8)
      // Add shadow for depth
      .attr("filter", "drop-shadow(0px 2px 3px rgba(0, 0, 0, 0.2))");

    // Improve text visibility and styling
    node.append("text")
      .text((d: CustomNode) => d.id)
      .attr("dy", 30)
      .attr("text-anchor", "middle")
      .attr("fill", theme === "dark" ? "#e0e0e0" : "#333")
      .style("font-size", (d: CustomNode) => d.group === "main" ? "12px" : "10px")
      .style("font-weight", (d: CustomNode) => d.group === "main" ? "bold" : "normal")
      // Add text shadow for better readability
      .style("text-shadow", "0px 0px 3px rgba(255, 255, 255, 0.7)")
      // Prevent text from interfering with node interactions
      .style("pointer-events", "none");

    return node;
  }, [filteredNodes, selectedNodeId, theme, getNodeColor]);

  const setupNodeInteractions = useCallback((node: d3.Selection<SVGGElement, CustomNode, SVGGElement, unknown>, svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    node
      .on("click", (event: MouseEvent, d: CustomNode) => {
        // Prevent event propagation to avoid unwanted behaviors
        event.stopPropagation();
        
        // First, reset styling on all nodes
        d3.selectAll(".node").classed("selected", false);
        d3.selectAll(".node circle").attr("stroke-width", 2);
        
        // Apply selected styling to clicked node
        d3.select(event.currentTarget as Element)
          .classed("selected", true)
          .select("circle")
          .attr("stroke-width", 3)
          .attr("r", d.group === "main" ? 28 : 18);
          
        // Provide visual feedback for connected nodes
        showConnectedNodes(d.id);
        
        setSelectedNodeId(d.id);
        onNodeClick(d.id);
      })
      .on("mouseover", (event: MouseEvent, d: CustomNode) => {
        const [x, y] = d3.pointer(event, svg.node());
        setHoveredNode({ ...d, x, y });
        
        // Enlarge the hovered node
        d3.select(event.currentTarget as Element).select("circle")
          .transition()
          .duration(200)
          .attr("r", d.group === "main" ? 28 : 18);
          
        // Highlight connected links
        highlightConnectedLinks(d.id, true);
      })
      .on("mouseout", (event: MouseEvent, d: CustomNode) => {
        setHoveredNode(null);
        
        // Return to normal size if not selected
        if (d.id !== selectedNodeId) {
          d3.select(event.currentTarget as Element).select("circle")
            .transition()
            .duration(200)
            .attr("r", d.group === "main" ? 25 : 15);
        }
        
        // Reset link highlighting
        highlightConnectedLinks(d.id, false);
      });
    
    // Double click to pin/unpin nodes
    node.on("dblclick", (event: MouseEvent, d: CustomNode) => {
      event.stopPropagation(); // Prevent event bubbling
      
      if (d.fx !== null || d.fy !== null) {
        // Unpin node
        d.fx = null;
        d.fy = null;
        d3.select(event.currentTarget as Element).select("circle")
          .attr("stroke", "#ffffff");
      } else {
        // Pin node
        d.fx = d.x;
        d.fy = d.y;
        d3.select(event.currentTarget as Element).select("circle")
          .attr("stroke", "#fca311"); // Use amber color for pinned nodes
      }
    });
  }, [onNodeClick, selectedNodeId]);
  
  // Helper function to highlight connected links
  const highlightConnectedLinks = useCallback((nodeId: string, isHighlighted: boolean) => {
    d3.selectAll(".link")
      .filter((d: any) => {
        const source = typeof d.source === 'object' ? d.source.id : d.source;
        const target = typeof d.target === 'object' ? d.target.id : d.target;
        return source === nodeId || target === nodeId;
      })
      .transition()
      .duration(isHighlighted ? 200 : 500)
      .attr("stroke-width", isHighlighted ? 2.5 : 1.5)
      .attr("stroke-opacity", isHighlighted ? 1 : 0.6);
  }, []);
  
  // Helper function to emphasize connected nodes
  const showConnectedNodes = useCallback((nodeId: string) => {
    // Find nodes connected to this one
    const connectedNodeIds = links
      .filter(link => {
        const source = typeof link.source === 'object' ? link.source.id : link.source;
        const target = typeof link.target === 'object' ? link.target.id : link.target;
        return source === nodeId || target === nodeId;
      })
      .map(link => {
        const source = typeof link.source === 'object' ? link.source.id : link.source;
        const target = typeof link.target === 'object' ? link.target.id : link.target;
        return source === nodeId ? target : source;
      });
    
    // Apply subtle highlighting to connected nodes
    d3.selectAll(".node")
      .filter((d: any) => connectedNodeIds.includes(d.id))
      .select("circle")
      .attr("stroke-width", 2.5)
      .attr("stroke-opacity", 1);
    
    // Also highlight the connections
    highlightConnectedLinks(nodeId, true);
  }, [links, highlightConnectedLinks]);

  const updateGraph = useCallback(() => {
    if (!svgRef.current) return;

    setIsLoading(true);
    setError(null);

    try {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const { width, height } = setupSvgDimensions(svg);
      const g = svg.append("g");
      
      const zoom = setupZoom(svg, g);
      zoomRef.current = zoom;

      // Initialize the simulation
      const sim = setupSimulation(width, height);

      const link = createLinks(g);
      const node = createNodes(g);

      setupNodeInteractions(node, svg);

      sim.nodes(filteredNodes);
      sim.force<d3.ForceLink<CustomNode, CustomLink>>("link")?.links(links.filter(link => 
        filteredNodes.some(node => node.id === (typeof link.source === 'string' ? link.source : link.source.id)) &&
        filteredNodes.some(node => node.id === (typeof link.target === 'string' ? link.target : link.target.id))
      ));

      sim.on("tick", () => {
        link
          .attr("x1", (d: CustomLink) => (d.source as CustomNode).x!)
          .attr("y1", (d: CustomLink) => (d.source as CustomNode).y!)
          .attr("x2", (d: CustomLink) => (d.target as CustomNode).x!)
          .attr("y2", (d: CustomLink) => (d.target as CustomNode).y!);

        node.attr("transform", (d: CustomNode) => `translate(${d.x!},${d.y!})`);
      });

      // Add legend
      const legendContainer = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - 120}, 20)`);
      
      const relationTypes = [
        { type: 'main', label: 'Main Word' },
        { type: 'root', label: 'Root Word' },
        { type: 'synonym', label: 'Synonym' },
        { type: 'antonym', label: 'Antonym' },
        { type: 'derived', label: 'Derivative' },
        { type: 'related', label: 'Related' },
        { type: 'kaugnay', label: 'Kaugnay' },
        { type: 'component_of', label: 'Component' },
        { type: 'cognate', label: 'Cognate' }
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

      return () => {
        if (sim) sim.stop();
      };
    } catch (err) {
      console.error("Error updating graph:", err);
      setError("An error occurred while updating the graph. Please try again.");
      setIsLoading(false);
    }
  }, [filteredNodes, links, theme, getNodeColor, createLinks, createNodes, setupNodeInteractions, setupSimulation]);

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
      svg.transition().duration(300).call(zoomRef.current.scaleBy, scale);
    }
  }, []);

  const handleResetZoom = useCallback(() => {
    if (zoomRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg
        .transition()
        .duration(300)
        .call(zoomRef.current.transform, d3.zoomIdentity);
    }
  }, []);

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

  const setupSvgDimensions = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    svg.attr("width", width).attr("height", height);
    return { width, height };
  };

  const setupZoom = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        g.attr("transform", event.transform.toString());
      });

    svg.call(zoom);
    return zoom;
  };

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
    if (!hoveredNode || typeof hoveredNode.x === 'undefined' || typeof hoveredNode.y === 'undefined') {
      return null;
    }
    
    return (
      <div
        className="tooltip"
        style={{
          position: "absolute",
          left: `${hoveredNode.x + 10}px`,
          top: `${hoveredNode.y + 10}px`,
        }}
      >
        <h3>{hoveredNode.info.word}</h3>
        {hoveredNode.info.definitions && hoveredNode.info.definitions.map((def, index) => (
          <p key={index}>{def}</p>
        ))}
      </div>
    );
  }, [hoveredNode]);

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
