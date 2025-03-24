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
  const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isDraggingRef = useRef(false);
  
  // Add debugging logs
  console.log('WordGraphComponent received props:', {
    wordNetwork,
    mainWord,
    initialDepth,
    initialBreadth
  });
  
  console.log('WordNetwork structure:', {
    hasNodes: wordNetwork?.nodes?.length,
    hasEdges: wordNetwork?.edges?.length,
    nodeCount: wordNetwork?.nodes?.length || 0,
    edgeCount: wordNetwork?.edges?.length || 0
  });
  
  // Create nodes from wordNetwork
  const nodes: CustomNode[] = useMemo(() => {
    if (!wordNetwork?.nodes || !Array.isArray(wordNetwork.nodes)) {
      console.error("Invalid nodes data:", wordNetwork);
      return [];
    }

    return wordNetwork.nodes.map(node => ({
      id: String(node.id),
      word: String(node.word),
      group: String(node.type || 'other'),
      language: node.language,
      has_baybayin: node.has_baybayin,
      baybayin_form: node.baybayin_form,
      definitions: node.definitions || []
    }));
  }, [wordNetwork]);
  
  // Create links from wordNetwork
  const links: CustomLink[] = useMemo(() => {
    if (!wordNetwork?.edges || !Array.isArray(wordNetwork.edges)) {
      console.error("Invalid edges data:", wordNetwork);
      return [];
    }

    return wordNetwork.edges.map(edge => ({
      source: String(edge.source),
      target: String(edge.target),
      relationship: String(edge.type || 'other'),
      weight: edge.weight || 1
    }));
  }, [wordNetwork]);
  
  // Node color based on group/type
  const getNodeColor = useCallback((group: string) => {
    switch (group.toLowerCase()) {
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
  
  // Node radius based on type
  const getNodeRadius = useCallback((node: CustomNode) => {
    if (node.id === mainWord) return 25;
    switch (node.group.toLowerCase()) {
      case 'root': return 20;
      case 'synonym': return 18;
      case 'antonym': return 18;
      case 'derived': return 15;
      default: return 12;
    }
  }, [mainWord]);
  
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

    try {
      setIsLoading(true);
      setError(null);

      // Clear previous content
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      // Get container dimensions
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      const centerX = width / 2;
      const centerY = height / 2;

      // Create container group
      const g = svg.append("g");

      // Add zoom behavior with performance optimization
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on("zoom", (event) => {
          requestAnimationFrame(() => {
            g.attr("transform", event.transform);
          });
        });

      svg.call(zoom);

      // Create the simulation with optimized forces
      const simulation = d3.forceSimulation<CustomNode>(nodes)
        .force("link", d3.forceLink<CustomNode, CustomLink>(links)
          .id(d => d.id)
          .distance(d => {
            // Adjust distance based on node types
            const sourceType = (d.source as CustomNode).type;
            const targetType = (d.target as CustomNode).type;
            if (sourceType === 'main' || targetType === 'main') return 120;
            if (sourceType === 'root' || targetType === 'root') return 100;
            return 80;
          }))
        .force("charge", d3.forceManyBody<CustomNode>()
          .strength(d => d.type === 'main' ? -1000 : -500)
          .distanceMax(300))
        .force("center", d3.forceCenter(centerX, centerY))
        .force("collide", d3.forceCollide<CustomNode>()
          .radius(d => getNodeRadius(d) * 1.5)
          .strength(0.7))
        .velocityDecay(0.6) // Reduce oscillation
        .alphaMin(0.001); // Lower alpha min for faster stabilization

      simulationRef.current = simulation;

      // Create virtual DOM elements for better performance
      const linkGroup = g.append("g").attr("class", "links");
      const nodeGroup = g.append("g").attr("class", "nodes");

      // Create the links with minimal attributes
      const link = linkGroup
        .selectAll<SVGLineElement, CustomLink>("line")
        .data(links)
        .join("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", d => Math.sqrt(d.weight || 1));

      // Create the nodes with optimized rendering
      const node = nodeGroup
        .selectAll<SVGGElement, CustomNode>("g")
        .data(nodes)
        .join("g")
        .attr("class", d => `node ${d.id === mainWord ? "main-node" : ""}`)
        .call(d3.drag<SVGGElement, CustomNode>()
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
          }) as any);

      // Add circles to nodes
      node.append("circle")
        .attr("r", getNodeRadius)
        .attr("fill", d => getNodeColor(d.group))
        .attr("stroke", theme === "dark" ? "#fff" : "#000")
        .attr("stroke-width", d => d.id === mainWord ? 3 : 1.5);

      // Add labels with better positioning
      node.append("text")
        .text(d => d.word)
        .attr("dy", d => getNodeRadius(d) + 15)
        .attr("text-anchor", "middle")
        .attr("fill", theme === "dark" ? "#fff" : "#000")
        .style("font-size", "12px")
        .style("pointer-events", "none");

      // Optimize event handlers
      const handleNodeHover = (event: MouseEvent, d: CustomNode) => {
        setHoveredNode({
          ...d,
          x: event.pageX,
          y: event.pageY
        });
      };

      const handleNodeClick = (event: MouseEvent, d: CustomNode) => {
        if (!isDraggingRef.current) {
          setSelectedNodeId(d.id);
          onNodeClick(d.word);
        }
      };

      node
        .on("mouseover", handleNodeHover)
        .on("mouseout", () => setHoveredNode(null))
        .on("click", handleNodeClick);

      // Optimize tick function for better performance
      let tickCount = 0;
      const maxTicks = 300; // Limit total number of ticks

      simulation.on("tick", () => {
        tickCount++;
        if (tickCount > maxTicks) {
          simulation.stop();
          return;
        }

        // Use requestAnimationFrame for smooth updates
        requestAnimationFrame(() => {
          link
            .attr("x1", d => (d.source as CustomNode).x || 0)
            .attr("y1", d => (d.source as CustomNode).y || 0)
            .attr("x2", d => (d.target as CustomNode).x || 0)
            .attr("y2", d => (d.target as CustomNode).y || 0);

          node.attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);
        });
      });

      // Fix main word position initially
      const mainNode = nodes.find(n => n.id === mainWord);
      if (mainNode) {
        mainNode.fx = centerX;
        mainNode.fy = centerY;
        simulation.alpha(1).restart();
        
        // Release after initial positioning
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
  }, [nodes, links, mainWord, theme, getNodeColor, getNodeRadius, onNodeClick]);
  
  // Update graph when data changes
  useEffect(() => {
    updateGraph();
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [updateGraph]);
  
  // Update depth and breadth from props
  useEffect(() => {
    setDepth(initialDepth);
    setBreadth(initialBreadth);
  }, [initialDepth, initialBreadth]);
  
  // Reset zoom handler
  const handleResetZoom = useCallback(() => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.transition()
      .duration(750)
      .call(
        d3.zoom<SVGSVGElement, unknown>().transform as any,
        d3.zoomIdentity
      );
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
      />

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