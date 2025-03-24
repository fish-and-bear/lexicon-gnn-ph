import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetwork, WordNetworkGraph, NetworkNode } from "../types";
import { useTheme } from "../contexts/ThemeContext";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";

interface WordGraphProps {
  wordNetwork: WordNetworkGraph;
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
  language?: string;
  has_baybayin?: boolean;
  baybayin_form?: string | null;
  definitions?: string[];
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
  source: string | CustomNode;
  target: string | CustomNode;
  relationship: string;
  weight?: number;
}

function isWordNetworkGraph(network: WordNetwork): network is WordNetworkGraph {
  return 'nodes' in network && 'edges' in network;
}

const WordGraphComponent: React.FC<WordGraphProps> = ({
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
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isDraggingRef = useRef(false);
  
  // Normalize data and create nodes
  const nodes: CustomNode[] = useMemo(() => {
    if (!isWordNetworkGraph(wordNetwork) || !Array.isArray(wordNetwork.nodes)) {
      console.error("Invalid nodes data:", wordNetwork);
      return [];
    }
    
    try {
      return wordNetwork.nodes.map(node => ({
        id: String(node.id),
        word: String(node.word),
        group: String(node.type || 'other'),
        language: node.language,
        has_baybayin: node.has_baybayin,
        baybayin_form: node.baybayin_form,
        definitions: node.definitions
      }));
    } catch (error) {
      console.error("Error processing nodes:", error);
      return [];
    }
  }, [wordNetwork]);
  
  // Create links from edges
  const links: CustomLink[] = useMemo(() => {
    if (!isWordNetworkGraph(wordNetwork) || !Array.isArray(wordNetwork.edges)) {
      console.error("Invalid edges data:", wordNetwork);
      return [];
    }
    
    try {
      return wordNetwork.edges.map(edge => ({
        source: String(edge.source),
        target: String(edge.target),
        relationship: String(edge.type || 'other'),
        weight: edge.weight
      }));
    } catch (error) {
      console.error("Error processing edges:", error);
      return [];
    }
  }, [wordNetwork]);
  
  // Filter nodes based on depth and breadth
  const filteredNodes = useMemo(() => {
    if (nodes.length === 0) return [];
    
    try {
      // Find the main node
      const mainNode = nodes.find(node => node.id === mainWord);
      if (!mainNode) {
        console.error("Main word node not found:", mainWord);
        return nodes.slice(0, 20); // Return some nodes to avoid empty graph
      }
      
      const connectedNodes = new Set<string>([mainWord]);
      const queue: [string, number][] = [[mainWord, 0]];
      const visited = new Set<string>([mainWord]);
      
      // BFS to find connected nodes
      while (queue.length > 0) {
        const [currentWord, currentDepth] = queue.shift()!;
        if (currentDepth >= depth) break;
        
        // Find all links connected to the current word
        const relatedLinks = links.filter(link => {
          const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
          const targetId = typeof link.target === 'string' ? link.target : link.target.id;
          return sourceId === currentWord || targetId === currentWord;
        });
        
        // Get the connected words from those links
        const relatedWords = relatedLinks.map(link => {
          const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
          const targetId = typeof link.target === 'string' ? link.target : link.target.id;
          return sourceId === currentWord ? targetId : sourceId;
        }).filter(word => !visited.has(word));
        
        // Take only up to breadth number of related words
        relatedWords.slice(0, breadth).forEach(word => {
          connectedNodes.add(word);
          visited.add(word);
          queue.push([word, currentDepth + 1]);
        });
      }
      
      // Return filtered nodes
      return nodes.filter(node => connectedNodes.has(node.id));
    } catch (error) {
      console.error("Error filtering nodes:", error);
      return nodes.slice(0, 20); // Return some nodes to avoid empty graph
    }
  }, [nodes, links, mainWord, depth, breadth]);
  
  // Basic node radius based on type
  const getNodeRadius = useCallback((node: CustomNode) => {
    if (node.id === mainWord) return 20;
    if (node.group === 'root' || node.group === 'synonym') return 15;
    return 12;
  }, [mainWord]);
  
  // Node color based on group/type
  const getNodeColor = useCallback((group: string) => {
    switch (group) {
      case 'main': return '#ee6c4d';
      case 'root': return '#3d5a80';
      case 'synonym': return '#98c1d9';
      case 'antonym': return '#e63946';
      case 'derived': return '#48cae4';
      case 'variant': return '#457b9d';
      case 'related': return '#a8dadc';
      default: return '#adb5bd';
    }
  }, []);
  
  // Legend items
  const legendItems = useMemo(() => [
    { key: "main", label: "Main Word" },
    { key: "root", label: "Root Word" },
    { key: "synonym", label: "Synonym" },
    { key: "antonym", label: "Antonym" },
    { key: "derived", label: "Derived" },
    { key: "related", label: "Related" },
    { key: "other", label: "Other" }
  ], []);
  
  // Render legend
  const renderLegend = useCallback(() => {
    return (
      <div style={{
        position: 'absolute',
        top: '20px',
        right: '20px',
        backgroundColor: theme === 'dark' ? 'rgba(40, 40, 40, 0.8)' : 'rgba(255, 255, 255, 0.8)',
        border: `1px solid ${theme === 'dark' ? '#555' : '#ddd'}`,
        borderRadius: '8px',
        padding: '12px 15px',
        fontSize: '13px',
        maxWidth: '180px',
        backdropFilter: 'blur(4px)'
      }}>
        <div style={{ 
          fontWeight: 'bold', 
          marginBottom: '10px', 
          fontSize: '14px',
          borderBottom: `1px solid ${theme === 'dark' ? '#555' : '#eee'}`,
          paddingBottom: '6px',
          color: theme === 'dark' ? '#fff' : '#333'
        }}>
          Word Types
        </div>
        {legendItems.map(item => (
          <div key={item.key} style={{
            display: 'flex',
            alignItems: 'center',
            marginBottom: '8px',
            color: theme === 'dark' ? '#eee' : '#444'
          }}>
            <div style={{ 
              width: '14px', 
              height: '14px', 
              marginRight: '10px', 
              backgroundColor: getNodeColor(item.key),
              borderRadius: '50%',
              border: `1px solid ${theme === 'dark' ? '#666' : '#ccc'}`
            }} />
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    );
  }, [legendItems, getNodeColor, theme]);
  
  // Render tooltip with node info
  const renderTooltip = useCallback(() => {
    if (!hoveredNode) return null;
    
    return (
      <div 
        style={{
          position: 'absolute',
          top: hoveredNode.y ? hoveredNode.y + 30 : 0,
          left: hoveredNode.x ? hoveredNode.x : 0,
          background: theme === 'dark' ? 'rgba(40, 40, 40, 0.95)' : 'rgba(255, 255, 255, 0.95)',
          color: theme === 'dark' ? '#eee' : '#333',
          padding: '12px 16px',
          borderRadius: '8px',
          boxShadow: theme === 'dark' ? '0 4px 12px rgba(0, 0, 0, 0.4)' : '0 4px 12px rgba(0, 0, 0, 0.15)',
          border: `1px solid ${theme === 'dark' ? '#555' : '#ddd'}`,
          zIndex: 1000,
          pointerEvents: 'none',
          minWidth: '180px',
          maxWidth: '280px',
          backdropFilter: 'blur(5px)',
          fontSize: '13px'
        }}
      >
        <div style={{ 
          fontWeight: 'bold', 
          marginBottom: '8px', 
          fontSize: '16px',
          color: theme === 'dark' ? '#fff' : '#222',
          borderBottom: `1px solid ${theme === 'dark' ? '#666' : '#eee'}`,
          paddingBottom: '8px'
        }}>
          {hoveredNode.word}
        </div>
        
        <div style={{ 
          display: 'flex',
          alignItems: 'center',
          marginBottom: '8px'
        }}>
          <div style={{ 
            width: '10px', 
            height: '10px', 
            marginRight: '8px',
            backgroundColor: getNodeColor(hoveredNode.group),
            borderRadius: '50%'
          }}></div>
          <span style={{ 
            color: theme === 'dark' ? '#bbb' : '#666',
            fontSize: '12px',
            fontStyle: 'italic'
          }}>
            {hoveredNode.group.charAt(0).toUpperCase() + hoveredNode.group.slice(1)}
          </span>
        </div>
        
        {hoveredNode.definitions && hoveredNode.definitions.length > 0 && (
          <div style={{ 
            marginTop: '10px',
            lineHeight: '1.4',
            color: theme === 'dark' ? '#ddd' : '#555' 
          }}>
            {typeof hoveredNode.definitions[0] === 'string' 
              ? hoveredNode.definitions[0]
              : (hoveredNode.definitions[0] as any)?.text || ''}
          </div>
        )}
      </div>
    );
  }, [hoveredNode, theme, getNodeColor]);
  
  // Main graph update function
  const updateGraph = useCallback(() => {
    if (!svgRef.current || !containerRef.current || nodes.length === 0) {
      console.warn("Cannot update graph: missing elements or no nodes");
      return;
    }

    console.log("Updating graph with", nodes.length, "nodes and", links.length, "links");
    setIsLoading(true);
    setError(null);

    try {
      // Clear previous SVG content
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      // Set up SVG dimensions
      const containerRect = containerRef.current.getBoundingClientRect();
      const width = containerRect.width;
      const height = containerRect.height;

      svg
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-width / 2, -height / 2, width, height]);

      // Create the simulation
      const simulation = d3.forceSimulation<CustomNode>(nodes)
        .force("link", d3.forceLink<CustomNode, CustomLink>(links)
          .id(d => d.id)
          .distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter())
        .force("collide", d3.forceCollide().radius(30));

      // Create the links
      const link = svg.append("g")
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", d => Math.sqrt(d.weight || 1));

      // Create the nodes
      const node = svg.append("g")
        .selectAll("g")
        .data(nodes)
        .join("g")
        .call(d3.drag<any, CustomNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
            isDraggingRef.current = true;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
            setTimeout(() => {
              isDraggingRef.current = false;
            }, 100);
          }));

      // Add circles to nodes
      node.append("circle")
        .attr("r", 20)
        .attr("fill", d => getNodeColor(d.group))
        .attr("stroke", "#fff")
        .attr("stroke-width", d => d.id === mainWord ? 3 : 1);

      // Add labels to nodes
      node.append("text")
        .text(d => d.word)
        .attr("x", 0)
        .attr("y", 30)
        .attr("text-anchor", "middle")
        .attr("fill", theme === "dark" ? "#fff" : "#000");

      // Add hover and click events
      node
        .on("mouseover", (event, d) => {
          setHoveredNode({
            ...d,
            x: event.pageX,
            y: event.pageY
          });
        })
        .on("mouseout", () => setHoveredNode(null))
        .on("click", (event, d) => {
          if (!isDraggingRef.current) {
            setSelectedNodeId(d.id);
            onNodeClick(d.word);
          }
        });

      // Update positions on tick
      simulation.on("tick", () => {
        link
          .attr("x1", d => (d.source as CustomNode).x || 0)
          .attr("y1", d => (d.source as CustomNode).y || 0)
          .attr("x2", d => (d.target as CustomNode).x || 0)
          .attr("y2", d => (d.target as CustomNode).y || 0);

        node
          .attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);
      });

      // Add zoom behavior
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on("zoom", (event) => {
          svg.selectAll("g").attr("transform", event.transform);
        });

      svg.call(zoom);

      // Center the main word
      const mainNode = nodes.find(n => n.id === mainWord);
      if (mainNode) {
        mainNode.fx = 0;
        mainNode.fy = 0;
        simulation.alpha(1).restart();
        setTimeout(() => {
          if (mainNode) {
            mainNode.fx = null;
            mainNode.fy = null;
          }
        }, 2000);
      }

      setIsLoading(false);
    } catch (error) {
      console.error("Error rendering graph:", error);
      setError("Failed to render graph visualization");
      setIsLoading(false);
    }
  }, [nodes, links, mainWord, theme, onNodeClick]);
  
  // Effect to update graph when data or filters change
  useEffect(() => {
    updateGraph();
  }, [updateGraph]);
  
  // Effect to update depth and breadth from props
  useEffect(() => {
    setDepth(initialDepth);
    setBreadth(initialBreadth);
  }, [initialDepth, initialBreadth]);
  
  // Effect to update selected node when main word changes
  useEffect(() => {
    setSelectedNodeId(mainWord);
  }, [mainWord]);
  
  // Reset zoom handler
  const handleResetZoom = useCallback(() => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition()
        .duration(750)
        .call(
          d3.zoom<SVGSVGElement, unknown>().transform as any, 
          d3.zoomIdentity
        );
    }
  }, []);
  
  return (
    <div 
      ref={containerRef} 
      className="word-graph-container"
      style={{ 
        width: '100%', 
        height: '100%',
        position: 'relative',
        backgroundColor: theme === 'dark' ? '#1a1a1a' : '#ffffff'
      }}
    >
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner">Loading graph...</div>
        </div>
      )}
      
      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={updateGraph}>Retry</button>
        </div>
      )}

      <svg
        ref={svgRef}
        style={{
          width: '100%',
          height: '100%',
          cursor: isDraggingRef.current ? 'grabbing' : 'grab'
        }}
      >
        <g className="graph-content" />
      </svg>

      {renderLegend()}
      {renderTooltip()}

      <div className="controls-container">
        <div className="slider-container">
          <label>
            Depth: {depth}
            <Slider
              min={1}
              max={5}
              step={1}
              value={depth}
              onChange={(_, value) => {
                const newDepth = Array.isArray(value) ? value[0] : value;
                setDepth(newDepth);
                onNetworkChange(newDepth, breadth);
              }}
            />
          </label>
        </div>

        <div className="slider-container">
          <label>
            Breadth: {breadth}
            <Slider
              min={1}
              max={20}
              step={1}
              value={breadth}
              onChange={(_, value) => {
                const newBreadth = Array.isArray(value) ? value[0] : value;
                setBreadth(newBreadth);
                onNetworkChange(depth, newBreadth);
              }}
            />
          </label>
        </div>

        <button 
          className="reset-zoom-button"
          onClick={handleResetZoom}
        >
          Reset Zoom
        </button>
      </div>
    </div>
  );
};

export default WordGraphComponent; 