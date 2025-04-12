import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
  Dispatch,
  SetStateAction
} from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetwork, NetworkNode } from "../types";
import { useTheme } from "../contexts/ThemeContext";
import NetworkControls from './NetworkControls';
import { alpha } from '@mui/material/styles';

interface WordGraphProps {
  wordNetwork: WordNetwork;
  mainWord: string;
  onNodeClick: (nodeId: string | null) => void;
  onNodeDoubleClick: (word: string) => void;
  onNetworkChange: (depth: number, breadth: number) => Promise<void>;
  selectedNodeId: string | null;
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

function isWordNetwork(network: any): network is WordNetwork {
  return network && Array.isArray(network.nodes) && Array.isArray(network.edges);
}

const WordGraphComponent: React.FC<WordGraphProps> = ({
  wordNetwork,
  mainWord,
  onNodeClick,
  onNodeDoubleClick,
  onNetworkChange,
  selectedNodeId,
  initialDepth,
  initialBreadth,
}) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isDraggingRef = useRef(false);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const gRef = useRef<SVGGElement | null>(null);
  
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

    return wordNetwork.nodes.map((node: NetworkNode) => ({
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

    return wordNetwork.edges.map((edge: {source: number, target: number, type: string}) => ({
      source: String(edge.source),
      target: String(edge.target),
      relationship: String(edge.type || 'other'),
      weight: 1
    }));
  }, [wordNetwork]);
  
  // Node color based on group/type
  const getNodeColor = useCallback((group: string) => {
    // Use theme-aware colors if possible, otherwise fallback
    const colors = theme === 'dark' ? 
      { // Dark theme colors (example)
        main: '#ff8a65', root: '#ef5350', synonym: '#64b5f6',
        antonym: '#f06292', derived: '#4dd0e1', variant: '#7986cb',
        related: '#81c784', other: '#90a4ae'
      } : 
      { // Light theme colors
        main: '#ee6c4d', root: '#3d5a80', synonym: '#98c1d9',
        antonym: '#e63946', derived: '#48cae4', variant: '#457b9d',
        related: '#a8dadc', other: '#adb5bd'
      };
    return colors[group.toLowerCase() as keyof typeof colors] || colors.other;
  }, [theme]);
  
  // Node radius based on type
  const getNodeRadius = useCallback((node: CustomNode) => {
    if (node.word === mainWord) return 15; // Slightly smaller main node
    switch (node.group.toLowerCase()) {
      case 'root': return 12;
      case 'synonym': return 10;
      case 'antonym': return 10;
      case 'derived': return 8;
      default: return 6;
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
    const legendBgColor = theme === 'dark' ? alpha('#2d3748', 0.85) : alpha('#ffffff', 0.85);
    const legendBorderColor = theme === 'dark' ? alpha('#ffffff', 0.1) : alpha('#000000', 0.1);
    const legendTextColor = theme === 'dark' ? '#e2e8f0' : '#4a5568';
    const headerColor = theme === 'dark' ? '#f7fafc' : '#2d3748';

    return (
      <div style={{
        position: 'absolute',
        top: '20px',
        right: '20px',
        backgroundColor: legendBgColor,
        border: `1px solid ${legendBorderColor}`,
        borderRadius: '8px',
        padding: '12px 15px',
        fontSize: '13px',
        maxWidth: '180px',
        backdropFilter: 'blur(4px)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        <div style={{ 
          fontWeight: '600', 
          marginBottom: '10px', 
          fontSize: '14px',
          borderBottom: `1px solid ${legendBorderColor}`,
          paddingBottom: '6px',
          color: headerColor
        }}>
          Word Types
        </div>
        {legendItems.map(item => (
          <div key={item.key} style={{
            display: 'flex',
            alignItems: 'center',
            marginBottom: '8px',
            color: legendTextColor
          }}>
            <div style={{ 
              width: '12px', 
              height: '12px', 
              marginRight: '10px', 
              backgroundColor: getNodeColor(item.key),
              borderRadius: '50%',
              border: `1px solid ${alpha(getNodeColor(item.key), 0.5)}`
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
    
    const tooltipBg = theme === 'dark' ? alpha('#1a202c', 0.95) : alpha('#ffffff', 0.95);
    const tooltipBorder = theme === 'dark' ? alpha('#ffffff', 0.1) : alpha('#000000', 0.1);
    const tooltipText = theme === 'dark' ? '#cbd5e0' : '#4a5568';
    const headerText = theme === 'dark' ? '#f7fafc' : '#1a202c';
    
    return (
      <div 
        style={{
          position: 'absolute',
          // Adjust positioning to be relative to the node
          top: (hoveredNode.y ?? 0) + 15, // Offset below node
          left: (hoveredNode.x ?? 0) + 15, // Offset right of node
          background: tooltipBg,
          color: tooltipText,
          padding: '10px 14px',
          borderRadius: '6px',
          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)',
          border: `1px solid ${tooltipBorder}`,
          zIndex: 1000,
          pointerEvents: 'none',
          minWidth: '160px',
          maxWidth: '260px',
          backdropFilter: 'blur(5px)',
          fontSize: '13px',
          fontFamily: 'sans-serif',
          transition: 'opacity 0.2s ease-in-out', // Smooth fade
          opacity: 1
        }}
      >
        {/* Header */}
        <div style={{ 
          fontWeight: '600', 
          marginBottom: '8px', 
          fontSize: '15px',
          color: headerText,
          borderBottom: `1px solid ${tooltipBorder}`,
          paddingBottom: '6px'
        }}>
          {hoveredNode.word}
        </div>
        
        {/* Word Type */}
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
            borderRadius: '50%',
            border: `1px solid ${alpha(getNodeColor(hoveredNode.group), 0.5)}`
          }}></div>
          <span style={{ 
            textTransform: 'capitalize',
            fontSize: '12px',
            fontWeight: '500',
            opacity: 0.8
          }}>
            {hoveredNode.group}
          </span>
        </div>

        {/* Baybayin */}
        {hoveredNode.has_baybayin && hoveredNode.baybayin_form && (
          <div style={{ marginTop: '8px', marginBottom: '8px' }}>
            <span style={{
              fontFamily: 'Noto Sans Baybayin, sans-serif',
              fontSize: '1.5em', // Larger Baybayin
              padding: '2px 6px',
              borderRadius: '4px',
              backgroundColor: alpha(getNodeColor(hoveredNode.group), 0.15),
              lineHeight: 1
            }}>
              {hoveredNode.baybayin_form}
            </span>
          </div>
        )}

        {/* Definitions Snippet */}
        {hoveredNode.definitions && hoveredNode.definitions.length > 0 && (
          <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: `1px dashed ${tooltipBorder}` }}>
            <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '12px', opacity: 0.9 }}>
              {hoveredNode.definitions.slice(0, 2).map((def, index) => (
                <li key={index} style={{ marginBottom: '4px' }}>{def}</li>
              ))}
              {hoveredNode.definitions.length > 2 && <li style={{ opacity: 0.7 }}>...</li>}
            </ul>
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

      // Create container group with hardware acceleration
      const g = svg.append("g")
        .attr("class", "graph-group")
        .style("transform-origin", "center center")
        .style("will-change", "transform");
      gRef.current = g.node(); // Store reference to the group

      // Add zoom behavior with optimized performance
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .extent([[0, 0], [width, height]])
        .scaleExtent([0.3, 5])
        .on("zoom", (event) => {
          const transform = event.transform;
          requestAnimationFrame(() => {
            g.style("transform", `translate(${transform.x}px,${transform.y}px) scale(${transform.k})`);
          });
        });

      svg.call(zoom);

      // --- DEBUG LOGS START ---
      console.log("Nodes passed to simulation:", JSON.stringify(nodes));
      console.log("Links passed to simulation:", JSON.stringify(links));
      // --- DEBUG LOGS END ---

      // Create the simulation with optimized forces
      const simulation = d3.forceSimulation<CustomNode>(nodes)
        .force("link", d3.forceLink<CustomNode, CustomLink>(links)
          .id(d => d.id)
          .distance(d => {
            const sourceGroup = (d.source as CustomNode).group;
            const targetGroup = (d.target as CustomNode).group;
            if (sourceGroup === 'main' || targetGroup === 'main') return 120;
            if (sourceGroup === 'root' || targetGroup === 'root') return 100;
            return 80;
          }))
        .force("charge", d3.forceManyBody<CustomNode>()
          .strength(d => d.group === 'main' ? -1000 : -500)
          .distanceMax(300)
          .theta(0.8))
        .force("center", d3.forceCenter(centerX, centerY))
        .force("collide", d3.forceCollide<CustomNode>()
          .radius(d => getNodeRadius(d) * 1.5)
          .strength(0.7)
          .iterations(2))
        .velocityDecay(0.6)
        .alphaMin(0.001)
        .alphaDecay(0.02);

      simulationRef.current = simulation;

      // Create virtual DOM elements for better performance
      const defs = svg.append("defs");
      
      // Create gradients for node fills
      const gradients = new Map([
        ['main', ['#ee6c4d', '#e63946']],
        ['root', ['#3d5a80', '#457b9d']],
        ['synonym', ['#98c1d9', '#8ecae6']],
        ['antonym', ['#e63946', '#d62828']],
        ['derived', ['#48cae4', '#00b4d8']],
        ['variant', ['#457b9d', '#2c7da0']],
        ['related', ['#a8dadc', '#90e0ef']],
        ['other', ['#adb5bd', '#6c757d']]
      ]);

      gradients.forEach(([color1, color2], type) => {
        const gradient = defs.append("linearGradient")
          .attr("id", `gradient-${type}`)
          .attr("x1", "0%")
          .attr("y1", "0%")
          .attr("x2", "100%")
          .attr("y2", "100%");

        gradient.append("stop")
          .attr("offset", "0%")
          .attr("stop-color", color1);

        gradient.append("stop")
          .attr("offset", "100%")
          .attr("stop-color", color2);
      });

      // Create the links with minimal attributes
      const linkGroup = g.append("g")
        .attr("class", "links")
        .style("will-change", "transform");

      const link = linkGroup
        .selectAll<SVGLineElement, CustomLink>("line")
        .data(links)
        .join("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", d => Math.sqrt(d.weight || 1));

      // Create the nodes with optimized rendering
      const nodeGroup = g.append("g")
        .attr("class", "nodes")
        .style("will-change", "transform");

      const node = nodeGroup
        .selectAll<SVGGElement, CustomNode>("g")
        .data(nodes)
        .join("g")
        .attr("class", d => `node ${d.word === mainWord ? "main-node" : ""}`)
        .style("will-change", "transform")
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

      // Add circles to nodes with gradients
      node.append("circle")
        .attr("r", getNodeRadius)
        .attr("fill", d => `url(#gradient-${d.group.toLowerCase()})`)
        .attr("stroke", theme === "dark" ? "#fff" : "#000")
        .attr("stroke-width", d => d.word === mainWord ? 3 : 1.5);

      // Add labels with better positioning and rendering
      node.append("text")
        .text(d => d.word)
        .attr("dy", d => getNodeRadius(d) + 15)
        .attr("text-anchor", "middle")
        .attr("fill", theme === "dark" ? "#fff" : "#000")
        .style("font-size", "12px")
        .style("pointer-events", "none")
        .style("paint-order", "stroke")
        .style("stroke", theme === "dark" ? "#000" : "#fff")
        .style("stroke-width", "3px")
        .style("stroke-linecap", "round")
        .style("stroke-linejoin", "round");

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
          const newSelectedId = selectedNodeId === d.id ? null : d.id;
          onNodeClick(newSelectedId);
        }
      };

      node
        .on("mouseover", handleNodeHover)
        .on("mouseout", () => setHoveredNode(null))
        .on("click", handleNodeClick);

      // Optimize tick function for better performance
      let tickCount = 0;
      const maxTicks = 300;
      let lastTickTime = performance.now();
      const minTickInterval = 1000 / 60; // Cap at 60fps

      simulation.on("tick", () => {
        tickCount++;
        if (tickCount > maxTicks) {
          simulation.stop();
          return;
        }

        const now = performance.now();
        if (now - lastTickTime < minTickInterval) return;
        lastTickTime = now;

        // Use requestAnimationFrame for smooth updates
        requestAnimationFrame(() => {
          link
            .attr("x1", d => (d.source as CustomNode).x || 0)
            .attr("y1", d => (d.source as CustomNode).y || 0)
            .attr("x2", d => (d.target as CustomNode).x || 0)
            .attr("y2", d => (d.target as CustomNode).y || 0);

          node.style("transform", d => `translate(${d.x || 0}px,${d.y || 0}px)`);
        });
      });

      // Fix main word position initially
      const mainNode = nodes.find(n => n.word === mainWord);
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
      // --- DEBUG LOGS START ---
      console.error("Error setting up graph simulation:", error);
      // --- DEBUG LOGS END ---
      console.error("Error rendering graph:", error);
      setError("Failed to render graph visualization");
      setIsLoading(false);
    }
  }, [nodes, links, mainWord, theme, getNodeColor, getNodeRadius, onNodeClick, selectedNodeId]);
  
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
  
  // Handle depth/breadth changes
  const handleDepthChange = (newDepth: number) => {
    if (newDepth !== depth) {
      setDepth(newDepth);
      setIsLoading(true);
      setError(null);
      onNetworkChange(newDepth, breadth)
        .catch(err => {
          console.error("Network change failed:", err);
          setError("Failed to update network depth.");
          setDepth(depth); // Revert on error
        })
        .finally(() => setIsLoading(false));
    }
  };

  const handleBreadthChange = (newBreadth: number) => {
    if (newBreadth !== breadth) {
      setBreadth(newBreadth);
      setIsLoading(true);
      setError(null);
      onNetworkChange(depth, newBreadth)
        .catch(err => {
          console.error("Network change failed:", err);
          setError("Failed to update network breadth.");
          setBreadth(breadth); // Revert on error
        })
        .finally(() => setIsLoading(false));
    }
  };
  
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
  
  // Render empty state
  const renderEmptyState = useCallback(() => {
    return (
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        textAlign: 'center',
        color: theme === 'dark' ? '#fff' : '#333',
        maxWidth: '80%'
      }}>
        <h3 style={{ marginBottom: '1rem' }}>No graph data available</h3>
        <p style={{ marginBottom: '1rem' }}>
          Try adjusting the depth and breadth settings or searching for a different word.
        </p>
        <div className="controls" style={{ marginTop: '2rem' }}>
          <NetworkControls
            depth={depth}
            breadth={breadth}
            onDepthChange={handleDepthChange}
            onBreadthChange={handleBreadthChange}
            onChangeCommitted={(d, b) => onNetworkChange(d, b)}
          />
        </div>
        {wordNetwork && (
          <pre style={{
            marginTop: '2rem',
            textAlign: 'left',
            background: theme === 'dark' ? '#2a2a2a' : '#f5f5f5',
            padding: '1rem',
            borderRadius: '4px',
            overflow: 'auto',
            maxHeight: '200px',
            fontSize: '12px'
          }}>
            {JSON.stringify(wordNetwork, null, 2)}
          </pre>
        )}
      </div>
    );
  }, [depth, breadth, theme, wordNetwork, onNetworkChange, handleDepthChange, handleBreadthChange]);
  
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

      {(!nodes.length || !links.length) ? renderEmptyState() : (
        <>
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
            <NetworkControls
              depth={depth}
              breadth={breadth}
              onDepthChange={handleDepthChange}
              onBreadthChange={handleBreadthChange}
              onChangeCommitted={(d, b) => onNetworkChange(d, b)}
            />
            <button 
              className="reset-zoom-button"
              onClick={handleResetZoom}
            >
              Reset Zoom
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default WordGraphComponent; 