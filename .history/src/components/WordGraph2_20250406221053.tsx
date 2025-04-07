import React, { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetworkResponse as WordNetwork } from "../types";
import { useTheme } from "../contexts/ThemeContext";

interface WordGraphProps {
  wordNetwork: WordNetwork | null;
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
      main: '#ff9500', // Orange - Main word
      synonym: theme === 'dark' ? '#4caf50' : '#2e7d32', // Green - Synonyms
      antonym: theme === 'dark' ? '#f44336' : '#c62828', // Red - Antonyms
      hypernym: theme === 'dark' ? '#2196f3' : '#1565c0', // Blue - Hypernyms
      hyponym: theme === 'dark' ? '#9c27b0' : '#6a1b9a', // Purple - Hyponyms
      related: theme === 'dark' ? '#ff9800' : '#ef6c00', // Orange - Related
      default: theme === 'dark' ? '#78909c' : '#546e7a' // Gray - Default
    };
    
    return colors[type as keyof typeof colors] || colors.default;
  }, [theme]);

  // Effect to update props to parent when depth changes
  useEffect(() => {
    onNetworkChange(initialDepth);
  }, [initialDepth, onNetworkChange]);

  useEffect(() => {
    if (!svgRef.current || !wordNetwork || !mainWord) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Get container dimensions
    const width = svg.node()?.parentElement?.clientWidth || 600;
    const height = svg.node()?.parentElement?.clientHeight || 400;

    // Create a group for the entire visualization
    const g = svg.append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`)
      .attr('class', 'word-graph');

    // Simple display if we have no or few nodes
    if (!wordNetwork.nodes || wordNetwork.nodes.length === 0) {
      // Just show main word in the center
      g.append('circle')
        .attr('r', 40)
        .attr('fill', '#ff9500')
        .attr('stroke', theme === 'dark' ? '#fff' : '#333')
        .attr('stroke-width', 2);

      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', theme === 'dark' ? '#fff' : '#000')
        .attr('font-weight', 'bold')
        .attr('font-size', '14px')
        .text(mainWord || 'No word');

      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('y', 70)
        .attr('fill', theme === 'dark' ? '#bbb' : '#666')
        .attr('font-size', '12px')
        .text('No connections available');
      
      return;
    }

    // Add a function to determine node type from relationships between nodes
    const nodes = wordNetwork.nodes.map(node => {
      // Determine the node type based on its relationship to the main word
      let nodeType = 'default';
      
      // If this is the main word
      if (node.lemma === mainWord) {
        nodeType = 'main';
      } 
      // Otherwise look for relationships in edges
      else if (wordNetwork.edges) {
        const mainNodeId = wordNetwork.nodes.find(n => n.lemma === mainWord)?.id;
        
        if (mainNodeId) {
          // Find any edge connecting this node to the main word
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

    // Add null check for edges
    const links = wordNetwork.edges ? wordNetwork.edges.map(edge => ({
      source: edge.source,
      target: edge.target,
      type: edge.type
    })) : [];

    // Simple force simulation
    const simulation = d3.forceSimulation()
      .nodes(nodes as any)
      .force('charge', d3.forceManyBody().strength(-250))
      .force('center', d3.forceCenter(0, 0))
      .force('collision', d3.forceCollide().radius(d => (d as any).isMain ? 35 : 25));
    
    // Only add link force if we have links
    if (links.length > 0) {
      simulation.force('link', d3.forceLink(links as any).id((d: any) => d.id).distance(100));
    }
    
    // Create the links only if we have any
    if (links.length > 0) {
      g.selectAll('.link')
        .data(links)
        .enter()
        .append('line')
        .attr('class', 'link')
        .attr('stroke', (d: any) => {
          // Color links based on relationship type
          switch (d.type) {
            case 'synonym': return theme === 'dark' ? '#4caf50' : '#2e7d32';
            case 'antonym': return theme === 'dark' ? '#f44336' : '#c62828';
            case 'hypernym': return theme === 'dark' ? '#2196f3' : '#1565c0';
            case 'hyponym': return theme === 'dark' ? '#9c27b0' : '#6a1b9a';
            case 'related': return theme === 'dark' ? '#ff9800' : '#ef6c00';
            default: return theme === 'dark' ? '#78909c' : '#546e7a';
          }
        })
        .attr('stroke-opacity', 0.7)
        .attr('stroke-width', 2);
    }

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

    // Add circles to each node
    node.append('circle')
      .attr('r', (d: any) => d.isMain ? 30 : 20)
      .attr('fill', (d: any) => getNodeColor(d.group))
      .attr('stroke', theme === 'dark' ? '#fff' : '#333')
      .attr('stroke-width', (d: any) => d.isMain ? 2 : 1.5)
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

    // Tick function to update positions
    simulation.on('tick', () => {
      if (links.length > 0) {
        g.selectAll('.link')
          .attr('x1', (d: any) => d.source.x)
          .attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x)
          .attr('y2', (d: any) => d.target.y);
      }

      node.attr('transform', (d: any) => `translate(${d.x}, ${d.y})`);
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