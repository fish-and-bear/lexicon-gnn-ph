import React, { useEffect, useRef, useCallback } from "react";
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

  // Color scheme for different relation types
  const getNodeColor = useCallback((type: string): string => {
    const colors = {
      main: '#FF7700', // Bright Orange - Main word (more vibrant)
      synonym: theme === 'dark' ? '#55cf5a' : '#2e7d32', // Green - Synonyms
      antonym: theme === 'dark' ? '#ff5252' : '#d32f2f', // Red - Antonyms (brighter)
      hypernym: theme === 'dark' ? '#42a5f5' : '#1565c0', // Blue - Hypernyms (brighter)
      hyponym: theme === 'dark' ? '#ba68c8' : '#7b1fa2', // Purple - Hyponyms (brighter)
      related: theme === 'dark' ? '#ffb74d' : '#ef6c00', // Orange - Related (brighter)
      default: theme === 'dark' ? '#90a4ae' : '#607d8b' // Gray - Default (brighter)
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
        .attr('fill', theme === 'dark' ? '#fff' : '#000')
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
        .attr('stroke-width', 2);
        
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', theme === 'dark' ? '#fff' : '#000')
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
      .attr('transform', `translate(${width / 2}, ${height / 2})`)
      .attr('class', 'word-graph');

    // Define animated marker for links
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
    `);

    // Create arrowhead marker for directed edges
    svg.append("defs").selectAll("marker")
      .data(["arrow"])
      .enter().append("marker")
      .attr("id", d => d)
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 25)  // Position the arrow tip
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("fill", theme === 'dark' ? '#ddd' : '#666')
      .attr("d", "M0,-5L10,0L0,5");

    // Process nodes data
    const nodes = wordNetwork.nodes.map(node => {
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
      
      return {
        id: node.id,
        label: node.lemma,
        group: nodeType,
        isMain: node.lemma === mainWord,
        // Add these properties to satisfy d3's SimulationNodeDatum interface
        index: undefined,
        x: undefined,
        y: undefined,
        vx: undefined,
        vy: undefined,
        fx: undefined,
        fy: undefined
      };
    });

    // Process edges data
    const links = (wordNetwork.edges || [])
      .filter(edge => {
        // Only keep edges with valid source and target
        const source = typeof edge.source === 'number' ? edge.source : parseInt(String(edge.source), 10);
        const target = typeof edge.target === 'number' ? edge.target : parseInt(String(edge.target), 10);
        return !isNaN(source) && !isNaN(target);
      })
      .map(edge => {
        // Convert string IDs to numbers if needed
        const source = typeof edge.source === 'number' ? edge.source : parseInt(String(edge.source), 10);
        const target = typeof edge.target === 'number' ? edge.target : parseInt(String(edge.target), 10);
        
        return {
          id: `${source}-${target}`,
          source: source,
          target: target,
          type: edge.type || 'default'
        };
      });

    console.log(`Processed ${nodes.length} nodes and ${links.length} links`);

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(0, 0))
      .force('collision', d3.forceCollide().radius((d: any) => d.isMain ? 40 : 25))
      .alphaTarget(0.1)  // Keep simulation active longer
      .restart();

    // Create links first (so they appear behind nodes)
    const link = g.selectAll('.link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', d => `link ${Math.random() > 0.5 ? 'animated-link' : ''}`) // Animate some links
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
        return d.type === 'hypernym' || d.type === 'hyponym' ? 'url(#arrow)' : null;
      });

    // Create node groups
    const node = g.selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', d => `node ${(d as any).isMain ? 'main-node' : ''}`)
      .on('click', (event, d: any) => {
        event.preventDefault();
        event.stopPropagation();
        if (d.label && d.label !== mainWord) {
          console.log(`Node clicked: ${d.label}`);
          onNodeClick(d.label);
        }
      })
      .on('mouseover', function() {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', (d: any) => (d.isMain ? 32 : 22))
          .attr('stroke-width', 3);
      })
      .on('mouseout', function() {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', (d: any) => (d.isMain ? 30 : 20))
          .attr('stroke-width', (d: any) => (d.isMain ? 2 : 1.5));
      })
      .call(d3.drag<SVGGElement, any>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      );

    // Add circles to nodes
    node.append('circle')
      .attr('r', (d: any) => d.isMain ? 30 : 20)
      .attr('fill', (d: any) => getNodeColor(d.group))
      .attr('stroke', theme === 'dark' ? '#fff' : '#222')
      .attr('stroke-width', (d: any) => d.isMain ? 2 : 1.5)
      .attr('class', 'node-circle');

    // Add text labels to nodes
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('fill', (d: any) => {
        // Calculate text color based on background for better readability
        const nodeType = d.group;
        const isDark = ['antonym', 'hypernym', 'hyponym'].includes(nodeType) || theme === 'dark';
        return isDark ? '#ffffff' : '#000000';
      })
      .attr('font-size', (d: any) => d.isMain ? '14px' : '12px')
      .attr('font-weight', (d: any) => d.isMain ? 'bold' : 'normal')
      .text((d: any) => d.label)
      .attr('class', 'node-label')
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

    // Initial zoom level
    svg.call((zoom as any).transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(0.8));

    // Add legend
    const legendData = [
      { label: 'Main Word', color: getNodeColor('main') },
      { label: 'Synonym', color: getNodeColor('synonym') },
      { label: 'Antonym', color: getNodeColor('antonym') },
      { label: 'Hypernym', color: getNodeColor('hypernym') },
      { label: 'Hyponym', color: getNodeColor('hyponym') },
      { label: 'Related', color: getNodeColor('related') }
    ];

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
        .attr('font-size', '10px')
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
    </div>
  );
};

export default WordGraph2; 