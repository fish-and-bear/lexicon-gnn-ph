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
  const { theme } = useTheme();
  const [tooltipData, setTooltipData] = useState<{x: number, y: number, text: string} | null>(null);

  // Color scheme for different relation types
  const getNodeColor = useCallback((type: string): string => {
    const colors = {
      main: '#FF7700', // Bright Orange for main word
      synonym: theme === 'dark' ? '#4CAF50' : '#2E7D32', // Green for synonyms
      antonym: theme === 'dark' ? '#F44336' : '#D32F2F', // Red for antonyms
      hypernym: theme === 'dark' ? '#2196F3' : '#1565C0', // Blue for hypernyms
      hyponym: theme === 'dark' ? '#9C27B0' : '#7B1FA2', // Purple for hyponyms
      related: theme === 'dark' ? '#FF9800' : '#F57C00', // Orange for related words
      default: theme === 'dark' ? '#90A4AE' : '#607D8B' // Gray for default/unknown
    };
    
    return colors[type as keyof typeof colors] || colors.default;
  }, [theme]);

  // Effect to update props to parent when depth changes
  useEffect(() => {
    onNetworkChange(initialDepth);
  }, [initialDepth, onNetworkChange]);

  useEffect(() => {
    if (!svgRef.current) return;
    
    // Clear any existing graph
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    if (!wordNetwork || !mainWord) {
      // Display a message for empty state
      const width = svg.node()?.parentElement?.clientWidth || 600;
      const height = svg.node()?.parentElement?.clientHeight || 400;
      
      const g = svg.append('g')
        .attr('transform', `translate(${width / 2}, ${height / 2})`)
        .attr('class', 'word-graph');
      
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', theme === 'dark' ? '#fff' : '#fff')
        .attr('font-size', '16px')
        .text("Enter a word to explore connections");
      
      return;
    }

    // Check if we have valid data to display
    if (!wordNetwork.nodes || wordNetwork.nodes.length === 0) {
      const width = svg.node()?.parentElement?.clientWidth || 600;
      const height = svg.node()?.parentElement?.clientHeight || 400;
      
      const g = svg.append('g')
        .attr('transform', `translate(${width / 2}, ${height / 2})`)
        .attr('class', 'word-graph');
      
      // Center text showing the main word
      g.append('circle')
        .attr('r', 40)
        .attr('fill', getNodeColor('main'))
        .attr('stroke', theme === 'dark' ? '#fff' : '#000')
        .attr('stroke-width', 2)
        .style('filter', 'drop-shadow(0 0 8px rgba(255, 119, 0, 0.5))');
        
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

    // Print debug info
    console.log(`Rendering graph with ${wordNetwork.nodes.length} nodes and ${wordNetwork.edges?.length || 0} edges`);
    
    // Get container dimensions
    const width = svg.node()?.parentElement?.clientWidth || 600;
    const height = svg.node()?.parentElement?.clientHeight || 400;

    // Create the SVG group for the graph
    const g = svg.append('g')
      .attr('class', 'word-graph');

    // Create arrowhead marker for directed edges
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
        if (d === "arrow-synonym") return theme === 'dark' ? '#4CAF50' : '#2E7D32';
        if (d === "arrow-antonym") return theme === 'dark' ? '#F44336' : '#D32F2F';
        if (d === "arrow-hypernym") return theme === 'dark' ? '#2196F3' : '#1565C0';
        if (d === "arrow-hyponym") return theme === 'dark' ? '#9C27B0' : '#7B1FA2';
        if (d === "arrow-related") return theme === 'dark' ? '#FF9800' : '#F57C00';
        return theme === 'dark' ? '#90A4AE' : '#607D8B'; // default
      })
      .attr("d", "M0,-5L10,0L0,5");

    // Add animated marker definitions
    svg.append("defs").append("style").text(`
      @keyframes dash {
        to {
          stroke-dashoffset: -20;
        }
      }
      
      .animated-link {
        stroke-dasharray: 5, 5;
        animation: dash 5s linear infinite;
      }
      
      @keyframes pulse {
        0% {
          filter: drop-shadow(0 0 5px rgba(255, 119, 0, 0.5));
        }
        50% {
          filter: drop-shadow(0 0 10px rgba(255, 119, 0, 0.8));
        }
        100% {
          filter: drop-shadow(0 0 5px rgba(255, 119, 0, 0.5));
        }
      }
      
      .main-node circle {
        animation: pulse 2s infinite;
      }
    `);

    // Process nodes data with improved position initialization
    const nodes = wordNetwork.nodes.map((node, index) => {
      let nodeType = 'default';
      
      // Main word gets special treatment
      if (node.lemma === mainWord) {
        nodeType = 'main';
      } 
      // Check all edges to determine relationship type
      else if (wordNetwork.edges && wordNetwork.edges.length > 0) {
        // First, find the ID of the main word node
        const mainNodeId = wordNetwork.nodes.find(n => n.lemma === mainWord)?.id;
        
        if (mainNodeId) {
          // Find any edge connecting this node to the main word
          const relation = wordNetwork.edges.find(e => 
            (e.source === mainNodeId && e.target === node.id) || 
            (e.source === node.id && e.target === mainNodeId)
          );
          
          // If we found a direct relationship, use its type
          if (relation) {
            nodeType = relation.type || 'default';
          }
        }
      }
      
      // Setup initial positions in a circle around the center
      const angle = (index / wordNetwork.nodes.length) * 2 * Math.PI;
      const radius = node.lemma === mainWord ? 0 : 150; // Larger radius for better spacing
      const x = width/2 + Math.cos(angle) * radius;
      const y = height/2 + Math.sin(angle) * radius;
      
      return {
        id: node.id,
        label: node.lemma,
        group: nodeType,
        isMain: node.lemma === mainWord,
        // Add these properties to satisfy d3's SimulationNodeDatum interface
        index: index,
        x: x, // Main word in center
        y: y, // Main word in center
        vx: 0,
        vy: 0,
        fx: node.lemma === mainWord ? width/2 : undefined, // Fix main word in center
        fy: node.lemma === mainWord ? height/2 : undefined  // Fix main word in center
      };
    });

    // Process edges data with more reliable ID handling
    const links = (wordNetwork.edges || [])
      .filter(edge => {
        // Ensure source and target can be parsed as numbers
        const source = typeof edge.source === 'number' ? edge.source : parseInt(String(edge.source), 10);
        const target = typeof edge.target === 'number' ? edge.target : parseInt(String(edge.target), 10);
        return !isNaN(source) && !isNaN(target);
      })
      .map(edge => {
        // Convert string IDs to numbers if needed
        const source = typeof edge.source === 'number' ? edge.source : parseInt(String(edge.source), 10);
        const target = typeof edge.target === 'number' ? edge.target : parseInt(String(edge.target), 10);
        
        // Find actual node objects
        const sourceNode = nodes.find(n => n.id === source);
        const targetNode = nodes.find(n => n.id === target);
        
        if (!sourceNode || !targetNode) {
          console.warn(`Edge references missing node: ${source} → ${target}`);
          return null;
        }
        
        return {
          id: `${source}-${target}`,
          source: sourceNode,
          target: targetNode,
          type: edge.type || 'default'
        };
      })
      .filter(link => link !== null) as any[];

    console.log(`Processed ${nodes.length} nodes and ${links.length} links`);

    // Create force simulation with stronger forces and better centering
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(120).strength(0.8))
      .force('charge', d3.forceManyBody().strength(-800))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => d.isMain ? 60 : 30))
      .force('x', d3.forceX(width / 2).strength(0.1))
      .force('y', d3.forceY(height / 2).strength(0.1))
      .alphaTarget(0.1)
      .alphaDecay(0.02);

    // Create links first (so they appear behind nodes) with better styling
    const link = g.selectAll('.link')
      .data(links)
      .join('line')
      .attr('class', d => `link ${Math.random() > 0.3 ? 'animated-link' : ''}`) // More animated links
      .attr('stroke', (d: any) => {
        // Link color based on relationship type
        switch (d.type) {
          case 'synonym': return theme === 'dark' ? '#4caf50' : '#2e7d32'; // Green for synonyms
          case 'antonym': return theme === 'dark' ? '#f44336' : '#c62828'; // Red for antonyms
          case 'hypernym': return theme === 'dark' ? '#2196f3' : '#1565c0'; // Blue for hypernyms
          case 'hyponym': return theme === 'dark' ? '#9c27b0' : '#7b1fa2'; // Purple for hyponyms
          case 'related': return theme === 'dark' ? '#ff9800' : '#f57c00'; // Orange for related
          default: return theme === 'dark' ? '#78909c' : '#455a64'; // Gray for default
        }
      })
      .attr('stroke-width', 3)
      .attr('stroke-opacity', 0.8)
      .attr('marker-end', (d: any) => {
        return d.type === 'hypernym' || d.type === 'hyponym' ? `url(#arrow-${d.type})` : null;
      });

    // Create node groups with better event handling
    const node = g.selectAll('.node')
      .data(nodes)
      .join('g')
      .attr('class', d => `node ${(d as any).isMain ? 'main-node' : ''}`)
      .on('click', (event, d: any) => {
        event.preventDefault();
        event.stopPropagation();
        if (d.label && d.label !== mainWord) {
          console.log(`Node clicked: ${d.label}`);
          onNodeClick(d.label);
        }
      })
      .on('mouseover', function(event, d: any) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', (d: any) => (d.isMain ? 38 : 24))
          .attr('stroke-width', 3);
          
        // Show tooltip
        setTooltipData({
          x: event.pageX,
          y: event.pageY,
          text: `${d.label} (${d.group})`
        });
      })
      .on('mouseout', function() {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', (d: any) => (d.isMain ? 34 : 20))
          .attr('stroke-width', (d: any) => (d.isMain ? 2 : 1.5));
          
        // Hide tooltip
        setTooltipData(null);
      });

    // Add drag behavior
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

    // Add circles to nodes with enhanced styling
    node.append('circle')
      .attr('r', (d: any) => d.isMain ? 34 : 20)
      .attr('fill', (d: any) => getNodeColor(d.group))
      .attr('stroke', theme === 'dark' ? '#fff' : '#222')
      .attr('stroke-width', (d: any) => d.isMain ? 2 : 1.5)
      .attr('class', 'node-circle')
      .style('filter', (d: any) => d.isMain ? 
        'drop-shadow(0 0 10px rgba(255, 119, 0, 0.7))' : 
        'drop-shadow(0px 3px 5px rgba(0,0,0,0.3))');

    // Add text labels to nodes with better text styling
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('fill', (d: any) => {
        // Calculate text color based on background for better readability
        const nodeType = d.group;
        const isDark = ['antonym', 'hypernym', 'hyponym'].includes(nodeType) || theme === 'dark';
        return isDark ? '#ffffff' : '#000000';
      })
      .attr('font-size', (d: any) => d.isMain ? '16px' : '12px')
      .attr('font-weight', (d: any) => d.isMain ? 'bold' : 'normal')
      .text((d: any) => d.label)
      .attr('class', 'node-label')
      .style('text-shadow', '0 1px 2px rgba(0,0,0,0.8)')
      .attr('pointer-events', 'none');

    // Add tooltips
    node.append('title')
      .text((d: any) => {
        const relationTypes: {[key: string]: string} = {
          'main': 'Main word',
          'synonym': 'Synonym',
          'antonym': 'Antonym',
          'hypernym': 'Hypernym (broader term)',
          'hyponym': 'Hyponym (more specific)',
          'related': 'Related word',
          'default': 'Related word'
        };
        
        const relationName = relationTypes[d.group] || 'Related word';
        return `${d.label} (${relationName})`;
      });

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

    // Add legend with better positioning
    const legendData = [
      { label: 'Main Word', color: getNodeColor('main') },
      { label: 'Synonym', color: getNodeColor('synonym') },
      { label: 'Antonym', color: getNodeColor('antonym') },
      { label: 'Hypernym', color: getNodeColor('hypernym') },
      { label: 'Hyponym', color: getNodeColor('hyponym') },
      { label: 'Related', color: getNodeColor('related') }
    ];

    const legendBg = svg.append('rect')
      .attr('x', width - 130) 
      .attr('y', 10)
      .attr('width', 120)
      .attr('height', legendData.length * 20 + 10)
      .attr('rx', 5)
      .attr('ry', 5)
      .attr('fill', theme === 'dark' ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.8)')
      .attr('stroke', theme === 'dark' ? '#444' : '#ddd')
      .attr('stroke-width', 1);
      
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 120}, 20)`);

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
  }, [wordNetwork, mainWord, onNodeClick, theme, getNodeColor, initialBreadth]);

  return (
    <div className="graph-container">
      <svg 
        ref={svgRef} 
        className="word-graph-svg"
        width="100%"
        height="100%"
      />
      {tooltipData && (
        <div 
          className="node-tooltip" 
          style={{
            position: 'absolute',
            left: `${tooltipData.x + 10}px`,
            top: `${tooltipData.y - 30}px`,
            backgroundColor: theme === 'dark' ? '#222' : '#fff',
            color: theme === 'dark' ? '#fff' : '#333',
            padding: '5px 8px',
            borderRadius: '4px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
            fontSize: '12px',
            pointerEvents: 'none',
            zIndex: 1000,
            border: `1px solid ${theme === 'dark' ? '#444' : '#ddd'}`
          }}
        >
          {tooltipData.text}
        </div>
      )}
    </div>
  );
};

export default WordGraph2;