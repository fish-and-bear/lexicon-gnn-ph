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
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isDraggingRef = useRef(false);
  
  // Normalize data and create nodes
  const nodes: CustomNode[] = useMemo(() => {
    if (!wordNetwork?.nodes || !Array.isArray(wordNetwork.nodes)) {
      console.error("Invalid nodes data:", wordNetwork);
      return [];
    }
    
    try {
      return wordNetwork.nodes.map(node => ({
        id: String(node.id),
        word: node.word,
        group: node.type || "other",
        info: node
      }));
    } catch (error) {
      console.error("Error processing nodes:", error);
      return [];
    }
  }, [wordNetwork]);
  
  // Create links from edges
  const links: CustomLink[] = useMemo(() => {
    if (!wordNetwork?.edges || !Array.isArray(wordNetwork.edges)) {
      console.error("Invalid edges data:", wordNetwork);
      return [];
    }
    
    try {
      return wordNetwork.edges.map(edge => ({
        source: String(edge.source),
        target: String(edge.target),
        relationship: edge.type || "other"
      }));
    } catch (error) {
      console.error("Error processing links:", error);
      return [];
    }
  }, [wordNetwork]);
  
  // Filter nodes based on depth and breadth
  const filteredNodes = useMemo(() => {
    if (nodes.length === 0) return [];
    
    try {
      const mainNode = nodes.find(node => node.id === mainWord);
      if (!mainNode) return [];
      
      const connectedNodes = new Set<string>([mainWord]);
      const queue: [string, number][] = [[mainWord, 0]];
      const visited = new Set<string>([mainWord]);
      
      while (queue.length > 0) {
        const [currentWord, currentDepth] = queue.shift()!;
        if (currentDepth >= depth) break;
        
        const relatedLinks = links.filter(link => 
          (typeof link.source === 'string' ? link.source : link.source.id) === currentWord ||
          (typeof link.target === 'string' ? link.target : link.target.id) === currentWord
        );
        
        const relatedWords = relatedLinks.map(link => {
          const isSource = (typeof link.source === 'string' ? link.source : link.source.id) === currentWord;
          return isSource 
            ? (typeof link.target === 'string' ? link.target : link.target.id)
            : (typeof link.source === 'string' ? link.source : link.source.id);
        }).filter(word => !visited.has(word));
        
        // Take only up to breadth number of related words
        relatedWords.slice(0, breadth).forEach(word => {
          connectedNodes.add(word);
          visited.add(word);
          queue.push([word, currentDepth + 1]);
        });
      }
      
      return nodes.filter(node => connectedNodes.has(node.id));
    } catch (error) {
      console.error("Error filtering nodes:", error);
      return [];
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
  
  // Render tooltip with node info
  const renderTooltip = useCallback(() => {
    if (!hoveredNode) return null;
    
    return (
      <div 
        style={{
          position: 'absolute',
          top: hoveredNode.y ? hoveredNode.y + 30 : 0,
          left: hoveredNode.x ? hoveredNode.x : 0,
          background: theme === 'dark' ? '#333' : '#fff',
          color: theme === 'dark' ? '#fff' : '#333',
          padding: '8px 12px',
          borderRadius: '4px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
          zIndex: 1000,
          pointerEvents: 'none',
          minWidth: '150px',
          maxWidth: '250px'
        }}
      >
        <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{hoveredNode.word}</div>
        <div>Type: {hoveredNode.group}</div>
        {hoveredNode.info?.definitions && hoveredNode.info.definitions.length > 0 && (
          <div style={{ marginTop: '5px' }}>
            {hoveredNode.info.definitions[0]}
          </div>
        )}
      </div>
    );
  }, [hoveredNode, theme]);
  
  // Main graph update function
  const updateGraph = useCallback((): (() => void) | void => {
    try {
      // Validate required refs and data
      if (!svgRef.current || !containerRef.current) return;
      if (filteredNodes.length === 0) {
        setError("No nodes to display. Try increasing depth or breadth.");
        return;
      }
      
      setError(null);
      setIsLoading(true);
      
      // Clear previous SVG content
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();
      
      // Set up SVG dimensions
      const containerRect = containerRef.current.getBoundingClientRect();
      const width = containerRect.width;
      const height = containerRect.height;
      
      svg.attr("width", width)
         .attr("height", height);
      
      // Create root group
      const g = svg.append("g")
         .attr("transform", `translate(${width / 2}, ${height / 2})`);
      
      // Prepare the links that actually exist in our filtered nodes
      const validLinks = links.filter(link => {
        const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
        const targetId = typeof link.target === 'string' ? link.target : link.target.id;
        return filteredNodes.some(n => n.id === sourceId) && 
               filteredNodes.some(n => n.id === targetId);
      });
      
      // Set up basic simulation
      const simulation = d3.forceSimulation<CustomNode>(filteredNodes)
        .force("link", d3.forceLink<CustomNode, CustomLink>()
          .id(d => d.id)
          .links(validLinks)
          .distance(100))
        .force("charge", d3.forceManyBody<CustomNode>().strength(-300))
        .force("center", d3.forceCenter(0, 0))
        .force("collide", d3.forceCollide<CustomNode>().radius(d => getNodeRadius(d) * 1.5));
      
      // Create links
      const linkElements = g.append("g")
        .selectAll("line")
        .data(validLinks)
        .join("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", 1.5);
      
      // Create node groups
      const nodeGroups = g.append("g")
        .selectAll("g")
        .data(filteredNodes)
        .join("g")
        .attr("class", d => `node ${d.id === mainWord ? "main-node" : ""}`)
        .call((selection) => {
          // Create and apply drag behavior
          const drag = d3.drag<SVGGElement, CustomNode>()
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
              if (!d.pinned) {
                d.fx = null;
                d.fy = null;
              }
              setTimeout(() => {
                isDraggingRef.current = false;
              }, 100);
            });
          
          selection.call(drag as any);
        });
      
      // Add circles for nodes
      nodeGroups.append("circle")
        .attr("r", d => getNodeRadius(d))
        .attr("fill", d => getNodeColor(d.group))
        .attr("stroke", "#fff")
        .attr("stroke-width", d => d.id === mainWord ? 3 : 1.5);
      
      // Add text labels below nodes
      nodeGroups.append("text")
        .attr("dy", d => getNodeRadius(d) + 14)
        .attr("text-anchor", "middle")
        .text(d => d.word)
        .attr("font-size", "12px")
        .attr("fill", theme === "dark" ? "#fff" : "#333");
      
      // Node interactions
      nodeGroups
        .on("click", (event, d) => {
          if (isDraggingRef.current) return;
          setSelectedNodeId(d.id);
          onNodeClick(d.id);
        })
        .on("mouseover", (event, d) => {
          setHoveredNode({
            ...d,
            x: event.pageX,
            y: event.pageY
          });
        })
        .on("mouseout", () => {
          setHoveredNode(null);
        });
      
      // Fix the main word in the center initially
      const mainNode = filteredNodes.find(n => n.id === mainWord);
      if (mainNode) {
        mainNode.fx = 0;
        mainNode.fy = 0;
        setTimeout(() => {
          if (mainNode) {
            mainNode.fx = null;
            mainNode.fy = null;
          }
        }, 3000);
      }
      
      // Update positions on simulation tick
      simulation.on("tick", () => {
        linkElements
          .attr("x1", d => {
            const source = typeof d.source === 'object' ? d.source : filteredNodes.find(n => n.id === d.source);
            return source ? source.x || 0 : 0;
          })
          .attr("y1", d => {
            const source = typeof d.source === 'object' ? d.source : filteredNodes.find(n => n.id === d.source);
            return source ? source.y || 0 : 0;
          })
          .attr("x2", d => {
            const target = typeof d.target === 'object' ? d.target : filteredNodes.find(n => n.id === d.target);
            return target ? target.x || 0 : 0;
          })
          .attr("y2", d => {
            const target = typeof d.target === 'object' ? d.target : filteredNodes.find(n => n.id === d.target);
            return target ? target.y || 0 : 0;
          });
        
        nodeGroups
          .attr("transform", d => `translate(${d.x || 0}, ${d.y || 0})`);
      });
      
      // Add zoom behavior
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.2, 3])
        .on("zoom", (event) => {
          g.attr("transform", event.transform);
        });
      
      svg.call(zoom);
      
      // Hide loading indicator
      setIsLoading(false);
      
      // Cleanup function
      return () => {
        simulation.stop();
      };
    } catch (error) {
      console.error("Error updating graph:", error);
      setError("Failed to create graph visualization. Please try again.");
      setIsLoading(false);
      return undefined;
    }
  }, [filteredNodes, links, mainWord, getNodeRadius, getNodeColor, theme, onNodeClick]);
  
  // Effect to update graph when data or filters change
  useEffect(() => {
    const cleanup = updateGraph();
    
    // Handle window resize
    window.addEventListener("resize", updateGraph);
    
    return () => {
      if (typeof cleanup === "function") {
        cleanup();
      }
      window.removeEventListener("resize", updateGraph);
    };
  }, [updateGraph, filteredNodes.length]);
  
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
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      {isLoading && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          background: 'rgba(0,0,0,0.3)',
          zIndex: 100
        }}>
          <div style={{ color: '#fff', padding: '20px', background: 'rgba(0,0,0,0.7)', borderRadius: '8px' }}>
            Loading graph...
          </div>
        </div>
      )}
      
      {error && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: theme === 'dark' ? '#333' : '#fff',
          color: theme === 'dark' ? '#fff' : '#333',
          padding: '15px 20px',
          borderRadius: '8px',
          boxShadow: '0 4px 15px rgba(0,0,0,0.15)',
          zIndex: 100,
          maxWidth: '80%'
        }}>
          <p>{error}</p>
          <button 
            onClick={() => setError(null)}
            style={{
              padding: '6px 12px',
              background: theme === 'dark' ? '#444' : '#eee',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Dismiss
          </button>
        </div>
      )}
      
      <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
      
      {renderTooltip()}
      
      <div className="controls-container" style={{
        position: 'absolute',
        bottom: '20px',
        left: '20px',
        right: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '15px',
        background: theme === 'dark' ? 'rgba(30,30,30,0.9)' : 'rgba(240,240,240,0.9)',
        borderRadius: '8px',
        boxShadow: '0 4px 15px rgba(0,0,0,0.15)',
        zIndex: 10
      }}>
        <div className="graph-controls" style={{ display: 'flex', gap: '20px', flexGrow: 1 }}>
          <div className="slider-container" style={{ display: 'flex', alignItems: 'center', gap: '10px', flexGrow: 1 }}>
            <Typography style={{ minWidth: '80px', color: theme === 'dark' ? '#fff' : '#333' }}>
              Depth: {depth}
            </Typography>
            <Slider
              min={1}
              max={3}
              step={1}
              value={depth}
              onChange={(_, newValue) => {
                const value = Array.isArray(newValue) ? newValue[0] : newValue;
                setDepth(value);
                onNetworkChange(value, breadth);
              }}
              style={{ color: theme === 'dark' ? '#6f8aee' : '#3f51b5' }}
            />
          </div>
          
          <div className="slider-container" style={{ display: 'flex', alignItems: 'center', gap: '10px', flexGrow: 1 }}>
            <Typography style={{ minWidth: '80px', color: theme === 'dark' ? '#fff' : '#333' }}>
              Breadth: {breadth}
            </Typography>
            <Slider
              min={1}
              max={10}
              step={1}
              value={breadth}
              onChange={(_, newValue) => {
                const value = Array.isArray(newValue) ? newValue[0] : newValue;
                setBreadth(value);
                onNetworkChange(depth, value);
              }}
              style={{ color: theme === 'dark' ? '#6f8aee' : '#3f51b5' }}
            />
          </div>
        </div>
        
        <div className="zoom-controls">
          <button
            className="zoom-button"
            onClick={handleResetZoom}
            style={{
              padding: '8px 15px',
              background: theme === 'dark' ? '#444' : '#eee',
              color: theme === 'dark' ? '#fff' : '#333',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            Reset View
          </button>
        </div>
      </div>
    </div>
  );
};

export default WordGraph; 