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
    
    // Show loading or empty state if no data
    if (!wordNetwork || !mainWord || !wordNetwork.nodes || wordNetwork.nodes.length === 0) {
      // Get container dimensions
      const width = svg.node()?.parentElement?.clientWidth || 600;
      const height = svg.node()?.parentElement?.clientHeight || 400;
      
      // Create container group
      const g = svg.append('g')
        .attr('transform', `translate(${width / 2}, ${height / 2})`)
        .attr('class', 'word-graph');
      
      if (!wordNetwork || !mainWord) {
        // No data state
        g.append('text')
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .attr('fill', theme === 'dark' ? '#fff' : '#000')
          .attr('font-size', '16px')
          .text("Enter a word to explore connections");
      } else {
        // No connections state
        g.append('text')
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .attr('fill', theme === 'dark' ? '#fff' : '#000')
          .attr('font-size', '16px')
          .text(`No connections found for "${mainWord}"`);
      }
      
      return;
    }

    // Get container dimensions
    const width = svg.node()?.parentElement?.clientWidth || 600;
    const height = svg.node()?.parentElement?.clientHeight || 400;

    // Create a group for the entire visualization
    const g = svg.append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`)
      .attr('class', 'word-graph');

    console.log("Rendering graph with nodes:", wordNetwork.nodes.length);
    console.log("First few nodes:", wordNetwork.nodes.slice(0, 3));
    console.log("Edges:", wordNetwork.edges?.length || 0);
    
    // Process nodes data
    const nodes = wordNetwork.nodes.map(node => {
      // Determine node type based on relation
      let nodeType = 'default';
      
      if (node.lemma === mainWord) {
        nodeType = 'main';
      } else if (wordNetwork.edges) {
        const mainNodeId = wordNetwork.nodes.find(n => n.lemma === mainWord)?.id;
        
        if (mainNodeId) {
          // Find edge connecting to main word
          const edge = wordNetwork.edges.find(e => 
            (e.source === mainNodeId && e.target === node.id) || 
            (e.source === node.id && e.target === mainNodeId)
          );
          
          if (edge) {
            nodeType = edge.type || 'default';
          }
        }
      }
      
      return {
        id: node.id,
        label: node.lemma,
        group: nodeType,
        isMain: node.lemma === mainWord
      };
    });

    // Process links data - ensure we don't have null/undefined values
    const links = (wordNetwork.edges || []).map(edge => ({
      id: `${edge.source}-${edge.target}`,
      source: typeof edge.source === 'number' ? edge.source : parseInt(String(edge.source), 10),
      target: typeof edge.target === 'number' ? edge.target : parseInt(String(edge.target), 10),
      type: edge.type || 'default'
    })).filter(link => 
      // Filter out invalid links that would break the simulation
      !isNaN(link.source) && 
      !isNaN(link.target) && 
      typeof link.source === 'number' && 
      typeof link.target === 'number'
    );

    console.log("Processed nodes:", nodes.length);
    console.log("Processed links:", links.length);

    // Define animated marker for links
    svg.append("defs").append("style").text(`
      @keyframes dash {
        to {
          stroke-dashoffset: -20;
        }
      }
      
      .link {
        stroke-dasharray: 5, 5;
        animation: dash 5s linear infinite;
      }
    `);

    // Create arrowhead marker definitions for directed relationships
    svg.append("defs").selectAll("marker")
      .data(["arrow"])
      .enter().append("marker")
      .attr("id", d => d)
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("fill", theme === 'dark' ? '#aaa' : '#666')
      .attr("d", "M0,-5L10,0L0,5");

    // Create the links first so they appear behind nodes
    const link = g.selectAll('.link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'link')
      .attr('stroke-width', (d: any) => 3) // Thicker lines
      .attr('stroke-opacity', 0.9) // More opaque
      .attr('stroke', (d: any) => {
        // Enhanced link coloring based on relationship type
        switch (d.type) {
          case 'synonym': return theme === 'dark' ? '#4caf50' : '#2e7d32';
          case 'antonym': return theme === 'dark' ? '#f44336' : '#c62828';
          case 'hypernym': return theme === 'dark' ? '#2196f3' : '#1565c0';
          case 'hyponym': return theme === 'dark' ? '#9c27b0' : '#6a1b9a';
          case 'related': return theme === 'dark' ? '#ff9800' : '#ef6c00';
          default: return theme === 'dark' ? '#78909c' : '#546e7a';
        }
      })
      .attr('marker-end', (d: any) => {
        return d.type === 'hypernym' || d.type === 'hyponym' ? 'url(#arrow)' : null;
      });

    // Create the nodes
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
      });
      
    // Add drag behavior
    node.call(d3.drag<SVGGElement, any>()
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

    // Add circles to each node with enhanced styling
    node.append('circle')
      .attr('r', (d: any) => d.isMain ? 35 : 22) // Larger nodes
      .attr('fill', (d: any) => getNodeColor(d.group))
      .attr('stroke', theme === 'dark' ? '#ffffff' : '#000000')
      .attr('stroke-width', (d: any) => d.isMain ? 3 : 1.5)
      .attr('class', 'node-circle');

    // Add text labels to each node with better visibility
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', (d: any) => d.isMain ? '0.3em' : '0.3em')
      .attr('fill', (d: any) => {
        // Calculate text color based on background for better visibility
        const bgColor = getNodeColor(d.group);
        const isDark = theme === 'dark' || 
            ['hypernym', 'hyponym', 'antonym'].includes(d.group);
        return isDark ? '#ffffff' : '#000000';
      })
      .attr('font-size', (d: any) => d.isMain ? '14px' : '12px')
      .attr('font-weight', (d: any) => d.isMain ? 'bold' : 'normal')
      .text((d: any) => d.label || '')
      .attr('class', 'node-label')
      .attr('pointer-events', 'none'); // Ensure text doesn't interfere with click

    // Add tooltips to nodes
    node.append("title")
      .text((d: any) => {
        const nodeType = d.group === 'main' ? 'Main word' : 
                        d.group.charAt(0).toUpperCase() + d.group.slice(1);
        return `${d.label} (${nodeType})`;
      });

    // Set up force simulation - critical fix for connection issues
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links)
        .id((d: any) => d.id)
        .distance(120))  // Increased distance for better visibility
      .force('charge', d3.forceManyBody().strength(-300))  // Stronger repulsion
      .force('center', d3.forceCenter(0, 0))
      .force('collision', d3.forceCollide().radius((d: any) => d.isMain ? 45 : 30));

    // Tick function to update positions - ensure link positions match nodes
    simulation.on('tick', () => {
      // Update link positions
      link
        .attr('x1', (d: any) => typeof d.source === 'object' ? d.source.x : 0)
        .attr('y1', (d: any) => typeof d.source === 'object' ? d.source.y : 0)
        .attr('x2', (d: any) => typeof d.target === 'object' ? d.target.x : 0)
        .attr('y2', (d: any) => typeof d.target === 'object' ? d.target.y : 0);
        
      // Update node positions
      node.attr('transform', (d: any) => `translate(${d.x || 0}, ${d.y || 0})`);
    });

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.2, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Initial zoom to fit
    const initialScale = 0.8;
    svg.call((zoom as any).transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(initialScale));

    // Add a legend for relationship types
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
      .attr('transform', `translate(${width - 150}, 20)`);

    legendData.forEach((item, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      legendItem.append('circle')
        .attr('r', 6)
        .attr('fill', item.color)
        .attr('stroke', theme === 'dark' ? '#fff' : '#333')
        .attr('stroke-width', 1);
      
      legendItem.append('text')
        .attr('x', 12)
        .attr('y', 4)
        .attr('fill', theme === 'dark' ? '#eee' : '#333')
        .attr('font-size', '10px')
        .text(item.label);
    });

    // Add zoom controls
    const zoomControls = svg.append('g')
      .attr('class', 'zoom-controls')
      .attr('transform', `translate(20, ${height - 60})`);

    // Zoom in button
    zoomControls.append('circle')
      .attr('r', 15)
      .attr('fill', theme === 'dark' ? '#333' : '#fff')
      .attr('stroke', theme === 'dark' ? '#666' : '#ccc')
      .attr('stroke-width', 1)
      .attr('cx', 15)
      .attr('cy', 15)
      .attr('class', 'zoom-button')
      .style('cursor', 'pointer')
      .on('click', () => {
        const currentTransform = d3.zoomTransform(svg.node()!);
        svg.transition().duration(300).call(
          (zoom as any).transform, 
          currentTransform.scale(currentTransform.k * 1.3)
        );
      });

    zoomControls.append('text')
      .attr('x', 15)
      .attr('y', 19)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#fff' : '#333')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .text('+')
      .style('cursor', 'pointer')
      .style('pointer-events', 'none');

    // Zoom out button
    zoomControls.append('circle')
      .attr('r', 15)
      .attr('fill', theme === 'dark' ? '#333' : '#fff')
      .attr('stroke', theme === 'dark' ? '#666' : '#ccc')
      .attr('stroke-width', 1)
      .attr('cx', 50)
      .attr('cy', 15)
      .attr('class', 'zoom-button')
      .style('cursor', 'pointer')
      .on('click', () => {
        const currentTransform = d3.zoomTransform(svg.node()!);
        svg.transition().duration(300).call(
          (zoom as any).transform, 
          currentTransform.scale(currentTransform.k / 1.3)
        );
      });

    zoomControls.append('text')
      .attr('x', 50)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#fff' : '#333')
      .attr('font-size', '22px')
      .attr('font-weight', 'bold')
      .text('−')
      .style('cursor', 'pointer')
      .style('pointer-events', 'none');

    // Reset zoom button
    zoomControls.append('circle')
      .attr('r', 15)
      .attr('fill', theme === 'dark' ? '#333' : '#fff')
      .attr('stroke', theme === 'dark' ? '#666' : '#ccc')
      .attr('stroke-width', 1)
      .attr('cx', 85)
      .attr('cy', 15)
      .attr('class', 'zoom-button')
      .style('cursor', 'pointer')
      .on('click', () => {
        svg.transition().duration(500).call(
          (zoom as any).transform, 
          d3.zoomIdentity.translate(width / 2, height / 2).scale(0.8)
        );
      });

    zoomControls.append('text')
      .attr('x', 85)
      .attr('y', 19)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#fff' : '#333')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('⟳')
      .style('cursor', 'pointer')
      .style('pointer-events', 'none');

    // Schedule clean-up
    return () => {
      console.log("Cleaning up graph simulation");
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