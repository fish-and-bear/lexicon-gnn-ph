import React, { useEffect, useRef, useCallback, useState, useMemo } from "react";
import * as d3 from "d3";
import "./WordGraph.css";
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

  // Node click handler
  const handleNodeClick = useCallback((node: NodeData) => {
    if (onNodeClick && node.lemma) {
      onNodeClick({
        id: node.id,
        lemma: node.lemma,
        language_code: node.language_code || 'tl',
        main: node.main || false
      });
    }
  }, [onNodeClick]);

  // Update the graph based on data changes
  useEffect(() => {
    if (!svgRef.current || !nodes || nodes.length === 0) return;

    // Clear existing SVG content
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const { width, height } = dimensions;

    const g = svg.append('g');

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any)
      .on('dblclick.zoom', null); // Disable double-click zoom

    const centerZoom = () => {
      svg.call((zoom as any).transform, 
        d3.zoomIdentity
          .translate(width / 2, height / 2)
          .scale(0.8)
      );
    };

    // Initialize with a centered view
    centerZoom();

    // Make sure we have the right data structure
    const validLinks = links.filter(
      link => 
        nodes.some(node => node.id === link.source) && 
        nodes.some(node => node.id === link.target)
    );

    // Create the simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(validLinks).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('x', d3.forceX(width / 2).strength(0.05))
      .force('y', d3.forceY(height / 2).strength(0.05))
      .force('collision', d3.forceCollide().radius(50));

    // Add the links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(validLinks)
      .enter()
      .append('line')
      .attr('class', 'link')
      .attr('stroke', d => getRelationColor(d.type))
      .attr('stroke-width', 1.5);

    // Add the nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', d => `node ${d.id === mainNodeId ? 'main-node' : ''}`)
      .call(drag(simulation) as any)
      .on('click', (event, d: NodeData) => {
        event.stopPropagation();
        handleNodeClick(d);
      });

    // Add a circle to each node
    node.append('circle')
      .attr('r', d => d.id === mainNodeId ? 20 : 14)
      .attr('fill', d => getNodeColor(d))
      .attr('stroke', '#fff')
      .attr('stroke-width', d => d.id === mainNodeId ? 3 : 1.5);

    // Add text labels
    node.append('text')
      .attr('dx', 0)
      .attr('dy', d => d.id === mainNodeId ? 30 : 25)
      .attr('text-anchor', 'middle')
      .text(d => d.lemma || 'Unknown')
      .style('font-size', d => d.id === mainNodeId ? '12px' : '10px')
      .style('fill', '#333')
      .style('pointer-events', 'none')
      .each(function() {
        // Wrap text if needed
        const text = d3.select(this);
        const words = text.text().split(/\s+/);
        if (words.length === 1) return; // No need to wrap single words
        
        text.text('');
        let lineNumber = 0;
        let line: string[] = [];
        let tspan = text.append('tspan').attr('x', 0).attr('dy', 0);
        
        words.forEach(word => {
          line.push(word);
          tspan.text(line.join(' '));
          
          if (tspan.node()!.getComputedTextLength() > 80) {
            line.pop();
            tspan.text(line.join(' '));
            line = [word];
            tspan = text.append('tspan')
              .attr('x', 0)
              .attr('dy', ++lineNumber * 12)
              .text(word);
          }
        });
      });

    // Update node and link positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as NodeData).x!)
        .attr('y1', d => (d.source as NodeData).y!)
        .attr('x2', d => (d.target as NodeData).x!)
        .attr('y2', d => (d.target as NodeData).y!);

      node
        .attr('transform', d => `translate(${d.x}, ${d.y})`);
    });
    
    // Stop simulation after a short time to reduce CPU usage
    setTimeout(() => {
      simulation.stop();
    }, 5000);

    return () => {
      simulation.stop();
    };
  }, [nodes, links, dimensions, mainNodeId, handleNodeClick]);

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

  // Color helper functions
  const getRelationColor = (type?: string): string => {
    switch (type) {
      case 'synonym': return '#4caf50'; // Green
      case 'antonym': return '#f44336'; // Red
      case 'hypernym': return '#2196f3'; // Blue
      case 'hyponym': return '#9c27b0'; // Purple
      case 'related': return '#ff9800'; // Orange
      case 'etymology': return '#00bcd4'; // Cyan
      case 'affixation': return '#ffc107'; // Amber
      default: return '#757575'; // Grey
    }
  };

  const getNodeColor = (node: NodeData): string => {
    if (node.id === mainNodeId) return '#e65100'; // Deep orange for main node
    if (node.main) return '#ff9800'; // Orange for important nodes
    
    // Color based on language code
    switch (node.language_code) {
      case 'tl': return '#2196f3'; // Blue for Tagalog
      case 'es': return '#f44336'; // Red for Spanish
      case 'en': return '#4caf50'; // Green for English
      case 'zh': return '#ffc107'; // Yellow for Chinese
      case 'ja': return '#9c27b0'; // Purple for Japanese
      case 'ms': return '#00bcd4'; // Cyan for Malay
      default: return '#78909c'; // Blue grey for others
    }
  };

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