import React, { useEffect, useRef, useCallback, useState, useMemo } from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetworkResponse } from "../types";
import { useTheme } from "../contexts/ThemeContext";

// Export NodeData type
export interface NodeData extends d3.SimulationNodeDatum {
  id: number;
  lemma: string;
  language_code?: string;
  main?: boolean;
  depth?: number;
  fx?: number | null; // Fixed x position for dragging
  fy?: number | null; // Fixed y position for dragging
  // Add index?, x?, y?, vx?, vy? from SimulationNodeDatum if needed explicitly, 
  // but extending the interface is usually sufficient.
}

interface LinkData extends d3.SimulationLinkDatum<NodeData> {
  id: string;
  // source and target will be NodeData after simulation initializes links
  source: number | NodeData; 
  target: number | NodeData;
  type: string;
}

interface WordNetwork {
  nodes: NodeData[];
  edges: LinkData[];
  stats?: {
    node_count: number;
    edge_count: number;
    depth?: number;
    breadth?: number;
  };
}

interface WordGraphProps {
  wordNetwork: WordNetwork;
  mainWord: string; // Lemma of the main word
  onNodeClick: (nodeData: NodeData) => void;
  onNetworkChange: (newDepth: number) => void;
  initialDepth: number;
  initialBreadth: number;
  theme?: string; // Optional theme prop
}

const WordGraph: React.FC<WordGraphProps> = ({
  wordNetwork,
  mainWord,
  onNodeClick,
  onNetworkChange,
  initialDepth,
  initialBreadth,
  theme
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const simulationRef = useRef<d3.Simulation<NodeData, LinkData>>();
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown>>();
  const { theme: contextTheme } = useTheme();
  const [tooltipData, setTooltipData] = useState<{
    x: number, 
    y: number, 
    text: string,
    group: string,
    id: number | string
  } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // Find the main node ID based on the lemma
  const mainNodeId = useMemo(() => {
    const mainNode = wordNetwork.nodes.find(n => n.lemma === mainWord);
    return mainNode ? mainNode.id : null;
  }, [wordNetwork.nodes, mainWord]);

  // Memoize nodes and links to prevent unnecessary re-renders
  const nodes = useMemo(() => wordNetwork.nodes.map(n => ({ ...n })), [wordNetwork.nodes]);
  const links = useMemo(() => wordNetwork.edges.map(l => ({ ...l })), [wordNetwork.edges]);

  // Debounce resize handler
  const debounce = (func: (...args: any[]) => void, delay: number) => {
    let timeoutId: NodeJS.Timeout;
    return (...args: any[]) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        func(...args);
      }, delay);
    };
  };

  const updateDimensions = useCallback(() => {
    if (containerRef.current) {
      const { width, height } = containerRef.current.getBoundingClientRect();
      setDimensions({ width, height });
      if (simulationRef.current) {
        simulationRef.current.force('center', d3.forceCenter(width / 2, height / 2));
        simulationRef.current.alpha(0.3).restart(); // Reheat simulation on resize
      }
    }
  }, []);

  useEffect(() => {
    const debouncedUpdate = debounce(updateDimensions, 150);
    const resizeObserver = new ResizeObserver(debouncedUpdate);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
      updateDimensions(); // Initial dimensions
    }
    return () => resizeObserver.disconnect();
  }, [updateDimensions]);

  useEffect(() => {
    if (!svgRef.current || !nodes.length || dimensions.width === 0) return;

    const svg = d3.select(svgRef.current);
    const { width, height } = dimensions;

    svg.selectAll('*').remove(); // Clear previous elements

    const g = svg.append('g');

    // --- Setup Simulation --- 
    if (!simulationRef.current) {
        simulationRef.current = d3.forceSimulation<NodeData, LinkData>(nodes)
            .force('link', d3.forceLink<NodeData, LinkData>(links)
                               .id(d => d.id)
                               .distance(link => (link.source as NodeData).id === mainNodeId || (link.target as NodeData).id === mainNodeId ? 100 : 70) // Check ID on NodeData
                               .strength(0.6))
            .force('charge', d3.forceManyBody().strength(-200)) // Increased repulsion
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide<NodeData>().radius(d => (d.id === mainNodeId ? 30 : 20)));
    } else {
        // Update simulation with new data
        simulationRef.current.nodes(nodes);
        const linkForce = simulationRef.current.force('link') as d3.ForceLink<NodeData, LinkData>; 
        linkForce.links(links);
        simulationRef.current.alpha(0.3).restart(); // Reheat simulation
    }

    const simulation = simulationRef.current;

    // --- Draw Links --- 
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter().append('line')
        .attr('class', d => `link ${d.type || 'default'}`) 
        .attr('stroke-width', 1.5);

    // --- Draw Nodes --- 
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter().append('g')
        .attr('class', d => `node ${d.id === mainNodeId ? 'main-node' : ''}`)
        .call(drag(simulation))
        .on('click', (event, d: NodeData) => { // Add explicit type
          event.stopPropagation(); // Prevent zoom trigger on node click
          onNodeClick(d);
        });

    node.append('circle')
        .attr('r', (d: NodeData) => d.id === mainNodeId ? 18 : 12) // Add explicit type
        .attr('fill', (d: NodeData) => { // Add explicit type
            // Example: Color based on language or depth if needed
            if (d.id === mainNodeId) return 'hsl(30, 100%, 50%)'; // Orange for main
            return 'hsl(210, 50%, 70%)'; // Blueish for others
        });

    node.append('text')
        .text((d: NodeData) => d.lemma) // Add explicit type
        .attr('dy', '0.35em');

    // --- Simulation Ticker --- 
    simulation.on('tick', () => {
        link
            .attr('x1', d => (d.source as NodeData).x ?? 0) // Access x/y safely
            .attr('y1', d => (d.source as NodeData).y ?? 0)
            .attr('x2', d => (d.target as NodeData).x ?? 0)
            .attr('y2', d => (d.target as NodeData).y ?? 0);

        node
            .attr('transform', (d: NodeData) => `translate(${d.x ?? 0},${d.y ?? 0})`); // Access x/y safely
    });

    // --- Zoom Functionality --- 
    if (!zoomRef.current) {
        zoomRef.current = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.3, 4]) // Zoom range
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
    }
    svg.call(zoomRef.current);

    // Initial zoom slightly out if many nodes
    if (nodes.length > 50) {
      svg.transition().duration(500).call(
        zoomRef.current.transform, 
        d3.zoomIdentity.translate(width/4, height/4).scale(0.5) 
      );
    }

    // Cleanup function
    return () => {
      simulation.stop();
    };

  }, [nodes, links, dimensions, mainNodeId, onNodeClick]); // Dependencies

  // --- Drag Handler --- 
  const drag = (simulation: d3.Simulation<NodeData, LinkData>) => {
    function dragstarted(event: d3.D3DragEvent<SVGGElement, NodeData, any>, d: NodeData) { // Use NodeData
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x ?? 0; // Use nullish coalescing
      d.fy = d.y ?? 0; // Use nullish coalescing
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, NodeData, any>, d: NodeData) { // Use NodeData
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, NodeData, any>, d: NodeData) { // Use NodeData
      if (!event.active) simulation.alphaTarget(0);
      // Keep node fixed after drag, remove if you want it to resettle
      // d.fx = null; 
      // d.fy = null;
    }

    return d3.drag<SVGGElement, NodeData>() // Specify NodeData type here
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  };

  // Performance optimization: Memoize color function to use CSS variables
  const getNodeColor = useMemo(() => {
    return (type: string): string => {
      // Get computed styles to access CSS variables
      const isLightTheme = contextTheme !== 'dark';
      const colors = {
        main: 'var(--color-root, #e63946)',
        synonym: isLightTheme ? 'var(--color-main, #1d3557)' : 'var(--color-derivative, #457b9d)', 
        antonym: isLightTheme ? 'var(--secondary-color, #e63946)' : 'var(--color-root, #e63946)', 
        hypernym: isLightTheme ? 'var(--color-etymology, #2a9d8f)' : 'var(--color-etymology, #2a9d8f)', 
        hyponym: isLightTheme ? 'var(--color-derivative, #457b9d)' : 'var(--primary-color, #ffd166)', 
        related: isLightTheme ? 'var(--color-associated, #fca311)' : 'var(--accent-color, #e09f3e)', 
        default: isLightTheme ? 'var(--color-default, #6c757d)' : 'var(--color-default, #6c757d)'
      };
      
      return colors[type as keyof typeof colors] || colors.default;
    };
  }, [contextTheme]);

  // Effect to update props to parent when depth changes
  useEffect(() => {
    onNetworkChange(initialDepth);
  }, [initialDepth, onNetworkChange]);

  // Function to get relationship description for tooltip
  const getRelationDescription = (group: string): string => {
    switch(group) {
      case 'main': return 'Main word';
      case 'synonym': return 'Word with similar meaning';
      case 'antonym': return 'Word with opposite meaning';
      case 'hypernym': return 'Broader or more general term';
      case 'hyponym': return 'More specific or specialized term';
      case 'related': return 'Related word or concept';
      default: return 'Connected word';
    }
  };

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', overflow: 'hidden' }} className={`graph-svg-container ${theme || contextTheme}`}>
      <svg ref={svgRef} width={dimensions.width} height={dimensions.height} className="word-graph-svg" />
      {tooltipData && !isDragging && (
        <div 
          className="graph-tooltip" 
          style={{
            position: 'absolute',
            left: `${tooltipData.x + 15}px`,
            top: `${tooltipData.y - 15}px`,
            backgroundColor: contextTheme === 'dark' ? 'rgba(26, 32, 44, 0.92)' : 'rgba(255, 255, 255, 0.92)',
            color: contextTheme === 'dark' ? '#E2E8F0' : '#2D3748',
            padding: '8px 12px',
            borderRadius: '8px',
            boxShadow: '0 4px 10px rgba(0, 0, 0, 0.15)',
            border: `1px solid ${contextTheme === 'dark' ? '#4A5568' : '#E2E8F0'}`,
            zIndex: 1000,
            maxWidth: '250px',
            pointerEvents: 'none'
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{tooltipData.text}</div>
          <div style={{ 
            fontSize: '12px', 
            color: contextTheme === 'dark' ? '#A0AEC0' : '#4A5568'
          }}>
            {getRelationDescription(tooltipData.group)}
          </div>
        </div>
      )}
    </div>
  );
};

export default WordGraph; 