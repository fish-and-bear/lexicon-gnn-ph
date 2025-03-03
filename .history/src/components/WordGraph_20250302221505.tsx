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

  // Enhanced drag behavior for better interactivity
  const dragBehavior = useCallback(() => d3.drag<SVGGElement, CustomNode>()
    .on("start", (event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) => {
      if (!event.active) {
        // Apply stronger forces when dragging starts
        simulation.current?.alphaTarget(0.3).restart();
      }
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
      
      // Visual feedback on drag start
      d3.select(event.sourceEvent.currentTarget).select("circle")
        .transition().duration(200)
        .attr("r", (d: CustomNode) => (d.group === "main" ? 28 : 18));
    })
    .on("drag", (event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) => {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
      
      // Update connected links during drag for better visual feedback
      d3.selectAll(".link")
        .filter((d: any) => {
          const source = typeof d.source === 'object' ? d.source.id : d.source;
          const target = typeof d.target === 'object' ? d.target.id : d.target;
          return source === event.subject.id || target === event.subject.id;
        })
        .attr("stroke-width", 2)
        .attr("stroke-opacity", 1);
    })
    .on("end", (event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) => {
      if (!event.active) {
        // Gradually cool the simulation back down
        simulation.current?.alphaTarget(0);
      }
      
      // Reset node size after drag ends
      d3.select(event.sourceEvent.currentTarget).select("circle")
        .transition().duration(300)
        .attr("r", (d: CustomNode) => (d.group === "main" ? 25 : 15));
      
      // Reset link appearance
      d3.selectAll(".link")
        .attr("stroke-width", 1)
        .attr("stroke-opacity", 0.6);
      
      // Keep position fixed if shift key is pressed during drag end
      if (!event.sourceEvent.shiftKey) {
        event.subject.fx = null;
        event.subject.fy = null;
      }
    }), []);

  // Store simulation in a ref to access it in event handlers
  const simulation = useRef<d3.Simulation<CustomNode, undefined>>();
  
  // Set up improved simulation with better physics
  const setupSimulation = useCallback((width: number, height: number) => {
    const validLinks = links.filter(link => 
      filteredNodes.some(node => node.id === (typeof link.source === 'string' ? link.source : link.source.id)) &&
      filteredNodes.some(node => node.id === (typeof link.target === 'string' ? link.target : link.target.id))
    );

    const sim = d3
      .forceSimulation<CustomNode>(filteredNodes)
      .force(
        "link",
        d3.forceLink<CustomNode, CustomLink>(validLinks)
          .id((d) => d.id)
          .distance(link => {
            // Dynamic link distance based on relationship type
            const source = typeof link.source === 'object' ? link.source : { group: 'unknown' };
            const target = typeof link.target === 'object' ? link.target : { group: 'unknown' };
            
            // Shorter distances for more related words
            if (source.group === 'main' || target.group === 'main') return 80;
            if (source.group === 'synonym' || target.group === 'synonym') return 90;
            if (source.group === 'related' || target.group === 'related') return 100;
            
            return 120; // Default distance
          })
          .strength(link => {
            // Stronger connections for certain relationships
            const source = typeof link.source === 'object' ? link.source : { group: 'unknown' };
            const target = typeof link.target === 'object' ? link.target : { group: 'unknown' };
            
            if (source.group === 'main' || target.group === 'main') return 0.7;
            if (source.group === 'synonym' || target.group === 'synonym') return 0.5;
            
            return 0.3; // Default strength
          })
      )
      .force("charge", d3.forceManyBody()
        .strength(d => d.group === 'main' ? -450 : -300)
        .distanceMax(350)
      )
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force(
        "collide",
        d3.forceCollide<CustomNode>()
          .radius(d => d.group === 'main' ? 30 : 20)
          .strength(0.7)
          .iterations(3)
      )
      // Add additional forces for more dynamic layout
      .force("x", d3.forceX(width / 2).strength(0.05))
      .force("y", d3.forceY(height / 2).strength(0.05))
      .alphaDecay(0.028); // Slightly slower decay for smoother animation
      
    simulation.current = sim;
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

  // Function to create more interactive nodes
  const createNodes = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const node = g
      .selectAll<SVGGElement, CustomNode>(".node")
      .data(filteredNodes)
      .join("g")
      .attr(
        "class",
        (d: CustomNode) => `node ${d.id === selectedNodeId ? "selected" : ""} ${d.group}`
      )
      .call(dragBehavior() as any);

    // Add visual feedback on node hover
    node.on("mouseover", function(event, d) {
      d3.select(this).select("circle")
        .transition()
        .duration(200)
        .attr("r", d.group === "main" ? 30 : 20)
        .attr("stroke-width", 3);
        
      // Highlight connected links and nodes
      d3.selectAll(".link")
        .filter((link: any) => {
          const source = typeof link.source === 'object' ? link.source.id : link.source;
          const target = typeof link.target === 'object' ? link.target.id : link.target;
          return source === d.id || target === d.id;
        })
        .transition()
        .duration(200)
        .attr("stroke-width", 3)
        .attr("stroke-opacity", 1);
        
      // Highlight connected nodes
      d3.selectAll(".node")
        .filter((node: CustomNode) => {
          return links.some(link => {
            const source = typeof link.source === 'object' ? link.source.id : link.source;
            const target = typeof link.target === 'object' ? link.target.id : link.target;
            return (source === d.id && target === node.id) || 
                   (target === d.id && source === node.id);
          });
        })
        .select("circle")
        .transition()
        .duration(200)
        .attr("stroke-width", 2);
    })
    .on("mouseout", function(event, d) {
      if (d.id !== selectedNodeId) {
        d3.select(this).select("circle")
          .transition()
          .duration(300)
          .attr("r", d.group === "main" ? 25 : 15)
          .attr("stroke-width", 2);
      }
      
      // Reset link appearance
      d3.selectAll(".link")
        .transition()
        .duration(300)
        .attr("stroke-width", 1.5)
        .attr("stroke-opacity", 0.6);
        
      // Reset node appearance
      d3.selectAll(".node")
        .select("circle")
        .transition()
        .duration(300)
        .attr("stroke-width", d.id === selectedNodeId ? 3 : 2);
    });

    // Enhanced circle with pulse animation for main node
    const circles = node.append("circle")
      .attr("r", (d: CustomNode) => (d.group === "main" ? 25 : 15))
      .attr("fill", (d: CustomNode) => getNodeColor(d.group))
      .attr("stroke", "#fff")
      .attr("stroke-width", (d: CustomNode) => d.id === selectedNodeId ? 3 : 2)
      .attr("stroke-opacity", 0.8);
    
    // Add subtle pulse animation to main node
    circles.filter((d: CustomNode) => d.group === "main")
      .append("animate")
      .attr("attributeName", "r")
      .attr("values", "25;27;25")
      .attr("dur", "2s")
      .attr("repeatCount", "indefinite");

    // Make labels more legible
    node
      .append("text")
      .text((d: CustomNode) => d.id)
      .attr("dy", 30)
      .attr("text-anchor", "middle")
      .attr("fill", theme === "dark" ? "#e0e0e0" : "#333")
      .style("font-size", (d: CustomNode) => d.group === "main" ? "12px" : "10px")
      .style("font-weight", (d: CustomNode) => d.group === "main" ? "bold" : "normal")
      .style("pointer-events", "none") // Prevent text from interfering with node interactions
      .style("text-shadow", "0 0 3px rgba(255,255,255,0.7), 0 0 2px rgba(255,255,255,0.5)")
      .style("user-select", "none");

    return node;
  }, [filteredNodes, selectedNodeId, theme, getNodeColor, dragBehavior]);

  const setupNodeInteractions = useCallback((node: d3.Selection<SVGGElement, CustomNode, SVGGElement, unknown>, svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    node
      .on("click", (event: MouseEvent, d: CustomNode) => {
        // Prevent event propagation to avoid unwanted behaviors
        event.stopPropagation();
        
        // Create a selection effect
        d3.selectAll(".node").classed("selected", false);
        d3.select(event.currentTarget).classed("selected", true);
        
        // Visual feedback on selection
        d3.selectAll(".node circle")
          .transition()
          .duration(300)
          .attr("stroke-width", 2);
          
        d3.select(event.currentTarget).select("circle")
          .transition()
          .duration(300)
          .attr("stroke-width", 3);
        
        setSelectedNodeId(d.id);
        onNodeClick(d.id);
      })
      .on("mouseover", (event: MouseEvent, d: CustomNode) => {
        const [x, y] = d3.pointer(event, svg.node());
        setHoveredNode({ ...d, x, y });
      })
      .on("mouseout", () => setHoveredNode(null));
      
    // Handle double-click to "pin" nodes
    node.on("dblclick", (event: MouseEvent, d: CustomNode) => {
      event.stopPropagation();
      if (d.fx && d.fy) {
        // Unpin the node
        d.fx = null;
        d.fy = null;
        d3.select(event.currentTarget).select("circle")
          .transition()
          .duration(300)
          .attr("stroke", "#fff");
      } else {
        // Pin the node
        d.fx = d.x;
        d.fy = d.y;
        d3.select(event.currentTarget).select("circle")
          .transition()
          .duration(300)
          .attr("stroke", "#ffd700"); // Gold color for pinned nodes
      }
    });
  }, [onNodeClick]);

  const updateGraph = useCallback(() => {
    if (!svgRef.current) return;

    setIsLoading(true);
    setError(null);

    try {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const { width, height } = setupSvgDimensions(svg);
      const g = svg.append("g");
      
      // Add a subtle grid background for better orientation
      addGridBackground(g, width, height);
      
      const zoom = setupZoom(svg, g);
      zoomRef.current = zoom;

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

      // Add interactive legend
      addInteractiveLegend(svg, width, height);

      setIsLoading(false);

      return () => {
        sim.stop();
      };
    } catch (err) {
      console.error("Error updating graph:", err);
      setError("An error occurred while updating the graph. Please try again.");
      setIsLoading(false);
    }
  }, [filteredNodes, links, theme, onNodeClick, selectedNodeId, getNodeColor, createLinks, createNodes, setupNodeInteractions, setupSimulation]);

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

  // Helper function to add an interactive legend
  const addInteractiveLegend = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, width: number, height: number) => {
    const relationTypes = [
      { type: 'main', label: 'Main Word' },
      { type: 'root', label: 'Root Word' },
      { type: 'synonym', label: 'Synonym' },
      { type: 'antonym', label: 'Antonym' },
      { type: 'derived', label: 'Derivative' },
      { type: 'variant', label: 'Variant' },
      { type: 'related', label: 'Related' },
      { type: 'kaugnay', label: 'Kaugnay' },
      { type: 'component_of', label: 'Component' },
      { type: 'cognate', label: 'Cognate' }
    ];
    
    const legend = svg.append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${width - 160}, 20)`);
    
    // Add background rect
    legend.append("rect")
      .attr("width", 150)
      .attr("height", relationTypes.length * 25 + 35)
      .attr("rx", 8)
      .attr("ry", 8)
      .attr("fill", theme === 'dark' ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.9)')
      .attr("stroke", theme === 'dark' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)')
      .attr("stroke-width", 1);
    
    // Add title
    legend.append("text")
      .attr("x", 75)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .attr("font-weight", "bold")
      .attr("font-size", "12px")
      .attr("fill", theme === 'dark' ? '#fff' : '#333')
      .text("Word Relationships");
    
    // Add legend items
    relationTypes.forEach((item, i) => {
      const g = legend.append("g")
        .attr("transform", `translate(10, ${i * 25 + 35})`)
        .attr("class", "legend-item")
        .style("cursor", "pointer")
        .on("mouseover", function() {
          // Highlight nodes of this type
          d3.selectAll(`.node.${item.type} circle`)
            .transition()
            .duration(200)
            .attr("r", (d: any) => d.group === "main" ? 30 : 20)
            .attr("stroke-width", 3);
            
          d3.select(this).select("text")
            .transition()
            .duration(200)
            .attr("font-weight", "bold");
        })
        .on("mouseout", function() {
          // Reset node appearance
          d3.selectAll(`.node.${item.type} circle`)
            .transition()
            .duration(300)
            .attr("r", (d: any) => d.group === "main" ? 25 : 15)
            .attr("stroke-width", 2);
            
          d3.select(this).select("text")
            .transition()
            .duration(300)
            .attr("font-weight", "normal");
        });
      
      // Color circle
      g.append("circle")
        .attr("r", 6)
        .attr("fill", getNodeColor(item.type));
      
      // Label
      g.append("text")
        .attr("x", 15)
        .attr("y", 4)
        .attr("font-size", "11px")
        .attr("fill", theme === 'dark' ? '#fff' : '#333')
        .text(item.label);
    });
  }, [theme, getNodeColor]);

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
