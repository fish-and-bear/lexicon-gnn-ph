import React, { useEffect, useRef, useCallback, useState } from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetworkResponse } from "../types";
import { useTheme } from "../contexts/ThemeContext";

interface WordGraphProps {
  wordNetwork: WordNetworkResponse | null;
  mainWord: string | null;
  onNodeClick: (word: string) => void;
  onNetworkChange: (depth: number) => void;
  initialDepth: number;
  initialBreadth?: number;
}

const WordGraph2: React.FC<WordGraphProps> = ({
  wordNetwork,
  mainWord,
  onNodeClick,
  onNetworkChange,
  initialDepth,
  initialBreadth = 15,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();
  const [tooltipData, setTooltipData] = useState<{
    x: number, 
    y: number, 
    text: string,
    group: string,
    id: number | string
  } | null>(null);

  // Color scheme for different relation types - more vibrant colors
  const getNodeColor = useCallback((type: string): string => {
    const colors = {
      main: '#FF5722', // Bright Orange for main word
      synonym: theme === 'dark' ? '#8BC34A' : '#689F38', // Lime Green for synonyms
      antonym: theme === 'dark' ? '#F44336' : '#D32F2F', // Red for antonyms
      hypernym: theme === 'dark' ? '#29B6F6' : '#0288D1', // Light Blue for hypernyms
      hyponym: theme === 'dark' ? '#BA68C8' : '#8E24AA', // Light Purple for hyponyms
      related: theme === 'dark' ? '#FFA726' : '#FB8C00', // Orange for related words
      default: theme === 'dark' ? '#90A4AE' : '#78909C' // Gray for default/unknown
    };
    
    return colors[type as keyof typeof colors] || colors.default;
  }, [theme]);

  // Effect to update props to parent when depth changes
  useEffect(() => {
    onNetworkChange(initialDepth);
  }, [initialDepth, onNetworkChange]);

  // Effect to handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (svgRef.current && containerRef.current) {
        updateGraph();
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Main graph rendering function
  const updateGraph = useCallback(() => {
    if (!svgRef.current || !containerRef.current) return;
    
    // Clear any existing graph
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    if (!wordNetwork || !mainWord) {
      // Display a message for empty state
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      
      const g = svg.append('g')
        .attr('transform', `translate(${width / 2}, ${height / 2})`)
        .attr('class', 'word-graph');
      
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', theme === 'dark' ? '#aaa' : '#666')
        .attr('font-size', '16px')
        .text("Enter a word to explore connections");
      
      return;
    }

    // Get container dimensions
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    // Create the SVG group for the graph
    const g = svg.append('g')
      .attr('class', 'word-graph');

    // Check if we have valid data to display
    if (!wordNetwork.nodes || wordNetwork.nodes.length === 0) {
      console.warn("No nodes found in wordNetwork data");
      
      const g = svg.append('g')
        .attr('transform', `translate(${width / 2}, ${height / 2})`)
        .attr('class', 'word-graph');
      
      // Center text showing the main word
      g.append('circle')
        .attr('r', 40)
        .attr('fill', getNodeColor('main'))
        .attr('stroke', theme === 'dark' ? '#fff' : '#000')
        .attr('stroke-width', 2);
        
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '16px')
        .attr('font-weight', 'bold')
        .text(mainWord);
      
      // Add message below
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('y', 70)
        .attr('fill', theme === 'dark' ? '#aaa' : '#666')
        .attr('font-size', '14px')
        .text("No connections found");
      
      return;
    }

    // Check edgesArray - handle both edges and links properties for compatibility
    const edgesArray = wordNetwork.edges || [];
    
    // Create arrowhead marker for directed edges with improved style
    svg.append("defs").selectAll("marker")
      .data(["arrow-synonym", "arrow-antonym", "arrow-hypernym", "arrow-hyponym", "arrow-default", "arrow-related"])
      .enter().append("marker")
      .attr("id", d => d)
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 25)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("fill", d => {
        if (d === "arrow-synonym") return theme === 'dark' ? '#8BC34A' : '#689F38';
        if (d === "arrow-antonym") return theme === 'dark' ? '#F44336' : '#D32F2F';
        if (d === "arrow-hypernym") return theme === 'dark' ? '#29B6F6' : '#0288D1';
        if (d === "arrow-hyponym") return theme === 'dark' ? '#BA68C8' : '#8E24AA';
        if (d === "arrow-related") return theme === 'dark' ? '#FFA726' : '#FB8C00';
        return theme === 'dark' ? '#90A4AE' : '#78909C'; // default
      })
      .attr("d", "M0,-5L10,0L0,5");

    // Add animated marker definitions with improved animations
    svg.append("defs").append("style").text(`
      @keyframes dash {
        to {
          stroke-dashoffset: -30;
        }
      }
      
      .animated-link {
        stroke-dasharray: 6, 6;
        animation: dash 10s linear infinite;
      }
      
      @keyframes breathe {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
      }
      
      .main-node circle {
        animation: breathe 3s ease-in-out infinite;
        transform-origin: center center;
        transform-box: fill-box;
      }
    `);

    // Process nodes data with improved position initialization
    const nodes = wordNetwork.nodes.map((node, index) => {
      let nodeType = 'default';
      
      // Make sure lemma is a string
      const nodeLemma = String(node.lemma || '');
      
      // Main word gets special treatment
      if (nodeLemma === mainWord) {
        nodeType = 'main';
      } 
      // Check all edges to determine relationship type
      else if (edgesArray && edgesArray.length > 0) {
        // First, find the ID of the main word node
        const mainNodeId = wordNetwork.nodes.find(n => n.lemma === mainWord)?.id;
        
        if (mainNodeId) {
          // Find any edge connecting this node to the main word
          const relation = edgesArray.find(e => 
            (e.source === mainNodeId && e.target === node.id) || 
            (e.source === node.id && e.target === mainNodeId)
          );
          
          // If we found a direct relationship, use its type
          if (relation) {
            nodeType = relation.type || 'default';
          }
        }
      }
      
      // Improved initial position calculation
      const angle = (index / wordNetwork.nodes.length) * 2 * Math.PI;
      const radiusFactor = nodeLemma === mainWord ? 0 : Math.min(width, height) * 0.3;
      const x = width/2 + Math.cos(angle) * radiusFactor;
      const y = height/2 + Math.sin(angle) * radiusFactor;
      
      return {
        id: node.id,
        label: nodeLemma,
        group: nodeType,
        isMain: nodeLemma === mainWord,
        // Add these properties for d3's simulation
        index: index,
        x: x,
        y: y,
        vx: 0,
        vy: 0,
        fx: nodeLemma === mainWord ? width/2 : undefined,
        fy: nodeLemma === mainWord ? height/2 : undefined
      };
    });

    // Create a map of nodes by ID for faster edge lookups
    const nodesById = new Map();
    nodes.forEach(node => {
      nodesById.set(node.id, node);
    });

    // Process edges data with more reliable ID handling
    const links = edgesArray
      .filter(edge => {
        // Ensure source and target can be parsed as numbers
        const source = typeof edge.source === 'object' ? edge.source.id : 
                     typeof edge.source === 'number' ? edge.source : 
                     parseInt(String(edge.source), 10);
                     
        const target = typeof edge.target === 'object' ? edge.target.id : 
                     typeof edge.target === 'number' ? edge.target : 
                     parseInt(String(edge.target), 10);
                     
        // Only include edges where both source and target nodes exist
        const sourceExists = !isNaN(source) && nodesById.has(source);
        const targetExists = !isNaN(target) && nodesById.has(target);
        
        return sourceExists && targetExists && !isNaN(source) && !isNaN(target);
      })
      .map(edge => {
        // Get the source and target IDs
        const sourceId = typeof edge.source === 'object' ? edge.source.id : 
                        typeof edge.source === 'number' ? edge.source : 
                        parseInt(String(edge.source), 10);
                        
        const targetId = typeof edge.target === 'object' ? edge.target.id : 
                        typeof edge.target === 'number' ? edge.target : 
                        parseInt(String(edge.target), 10);
        
        // Get the actual node objects
        const sourceNode = nodesById.get(sourceId);
        const targetNode = nodesById.get(targetId);
        
        return {
          id: edge.id || `${sourceId}-${targetId}`,
          source: sourceNode,
          target: targetNode,
          type: edge.type || 'default'
        };
      });

    // Create force simulation with optimized forces
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100).strength(0.7))
      .force('charge', d3.forceManyBody().strength(-800))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => d.isMain ? 50 : 25))
      .force('x', d3.forceX(width / 2).strength(0.1))
      .force('y', d3.forceY(height / 2).strength(0.1))
      .alphaDecay(0.05);

    // Create links with cleaner styling
    const link = g.selectAll('.link')
      .data(links)
      .join('line')
      .attr('class', d => `link ${Math.random() > 0.5 ? 'animated-link' : ''}`)
      .attr('stroke', (d: any) => {
        // Link color based on relationship type
        switch (d.type) {
          case 'synonym': return theme === 'dark' ? '#8BC34A' : '#689F38';
          case 'antonym': return theme === 'dark' ? '#F44336' : '#D32F2F';
          case 'hypernym': return theme === 'dark' ? '#29B6F6' : '#0288D1';
          case 'hyponym': return theme === 'dark' ? '#BA68C8' : '#8E24AA';
          case 'related': return theme === 'dark' ? '#FFA726' : '#FB8C00';
          default: return theme === 'dark' ? '#78909C' : '#455A64';
        }
      })
      .attr('stroke-width', 3)
      .attr('stroke-opacity', 0.7)
      .attr('marker-end', (d: any) => {
        return d.type === 'hypernym' || d.type === 'hyponym' ? `url(#arrow-${d.type})` : null;
      });

    // Create node groups with enhanced interactivity
    const node = g.selectAll('.node')
      .data(nodes)
      .join('g')
      .attr('class', d => `node ${(d as any).isMain ? 'main-node' : ''}`)
      .on('click', (event, d: any) => {
        event.preventDefault();
        event.stopPropagation();
        if (d.label && d.label !== mainWord) {
          onNodeClick(d.label);
        }
      })
      .on('mouseover', function(event, d: any) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', (d: any) => (d.isMain ? 38 : 24));
          
        // Show improved tooltip with more information
        setTooltipData({
          x: event.pageX,
          y: event.pageY,
          text: d.label,
          group: d.group,
          id: d.id
        });
      })
      .on('mouseout', function() {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', (d: any) => (d.isMain ? 34 : 20));
          
        // Clear tooltip
        setTooltipData(null);
      });

    // Add drag behavior with improved handling
    node.call(d3.drag<SVGGElement, any>()
      .on('start', function(event, d: any) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', function(event, d: any) {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', function(event, d: any) {
        if (!event.active) simulation.alphaTarget(0);
        // Only keep main word fixed
        if (!d.isMain) {
          d.fx = null;
          d.fy = null;
        }
      }) as any);

    // Add circles to nodes with cleaner, modern styling
    node.append('circle')
      .attr('r', (d: any) => d.isMain ? 34 : 20)
      .attr('fill', (d: any) => getNodeColor(d.group))
      .attr('stroke', theme === 'dark' ? '#fff' : '#222')
      .attr('stroke-width', (d: any) => d.isMain ? 2 : 1.5)
      .style('cursor', 'pointer');

    // Add text labels with improved readability
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('fill', (d: any) => {
        // Calculate text color based on background for better readability
        const nodeType = d.group;
        const isDark = ['antonym', 'hypernym', 'hyponym'].includes(nodeType) || theme === 'dark';
        return isDark ? '#ffffff' : '#000000';
      })
      .attr('font-size', (d: any) => d.isMain ? '14px' : '11px')
      .attr('font-weight', (d: any) => d.isMain ? 'bold' : 'normal')
      .text((d: any) => d.label)
      .attr('pointer-events', 'none')
      .style('user-select', 'none');

    // Tick function to update positions during simulation
    simulation.on('tick', () => {
      // Constrain to view bounds
      nodes.forEach(d => {
        // Add bounds to avoid nodes moving offscreen
        d.x = Math.max(30, Math.min(width - 30, d.x || 0));
        d.y = Math.max(30, Math.min(height - 30, d.y || 0));
      });
      
      // Update link positions
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);
        
      // Update node positions
      node.attr('transform', (d: any) => `translate(${d.x}, ${d.y})`);
    });

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.2, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Initial zoom level - center the graph
    svg.call((zoom as any).transform, d3.zoomIdentity.translate(0, 0).scale(0.9));
    
    // Start simulation with a stronger alpha
    simulation.alpha(1).restart();

    // Add legend with better positioning and style
    const legendData = [
      { label: 'Main Word', color: getNodeColor('main') },
      { label: 'Synonym', color: getNodeColor('synonym') },
      { label: 'Antonym', color: getNodeColor('antonym') },
      { label: 'Hypernym', color: getNodeColor('hypernym') },
      { label: 'Hyponym', color: getNodeColor('hyponym') },
      { label: 'Related', color: getNodeColor('related') }
    ];

    const legendBg = svg.append('rect')
      .attr('x', 10) 
      .attr('y', 10)
      .attr('width', 120)
      .attr('height', legendData.length * 20 + 10)
      .attr('rx', 5)
      .attr('ry', 5)
      .attr('fill', theme === 'dark' ? 'rgba(30,30,30,0.8)' : 'rgba(255,255,255,0.9)')
      .attr('stroke', theme === 'dark' ? '#444' : '#ddd')
      .attr('stroke-width', 1);
      
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(20, 20)`);

    legendData.forEach((item, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      legendItem.append('circle')
        .attr('r', 6)
        .attr('fill', item.color)
        .attr('stroke', theme === 'dark' ? '#fff' : '#222')
        .attr('stroke-width', 1);
      
      legendItem.append('text')
        .attr('x', 12)
        .attr('y', 4)
        .attr('fill', theme === 'dark' ? '#eee' : '#333')
        .attr('font-size', '11px')
        .text(item.label);
    });

    // Clean up when component unmounts
    return () => {
      simulation.stop();
    };
  }, [wordNetwork, mainWord, onNodeClick, theme, getNodeColor]);

  // Run graph update when dependencies change
  useEffect(() => {
    updateGraph();
  }, [updateGraph]);

  // Function to get better relationship description for tooltip
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
    <div ref={containerRef} className="graph-container">
      <svg 
        ref={svgRef} 
        className="word-graph-svg"
        width="100%"
        height="100%"
      />
      {tooltipData && (
        <div 
          className="enhanced-tooltip" 
          style={{
            position: 'absolute',
            left: `${tooltipData.x + 15}px`,
            top: `${tooltipData.y - 15}px`,
            backgroundColor: theme === 'dark' ? 'rgba(30,30,30,0.95)' : 'rgba(255,255,255,0.95)',
            color: theme === 'dark' ? '#fff' : '#333',
            padding: '8px 12px',
            borderRadius: '6px',
            boxShadow: '0 3px 8px rgba(0,0,0,0.2)',
            zIndex: 1000,
            border: `1px solid ${theme === 'dark' ? '#444' : '#ddd'}`,
            maxWidth: '250px',
            pointerEvents: 'none'
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '2px' }}>{tooltipData.text}</div>
          <div style={{ 
            fontSize: '11px', 
            opacity: 0.9,
            color: theme === 'dark' ? '#aaa' : '#666'
          }}>
            {getRelationDescription(tooltipData.group)}
          </div>
        </div>
      )}
    </div>
  );
};

export default WordGraph2;