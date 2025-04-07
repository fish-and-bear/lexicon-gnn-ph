import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { WordNetworkResponse } from '../types';

interface SimpleWordGraphProps {
  wordNetwork: WordNetworkResponse;
  mainWord: string;
  onNodeClick: (word: string) => void;
}

const SimpleWordGraph: React.FC<SimpleWordGraphProps> = ({ wordNetwork, mainWord, onNodeClick }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !wordNetwork) return;

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
        .attr('stroke', '#444')
        .attr('stroke-width', 2);

      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#fff')
        .attr('font-weight', 'bold')
        .attr('font-size', '14px')
        .text(mainWord || 'No word');

      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('y', 70)
        .attr('fill', '#999')
        .attr('font-size', '12px')
        .text('No connections available');
      
      return;
    }

    // Create nodes and links from the data
    const nodes = wordNetwork.nodes.map(node => ({
      id: node.id,
      label: node.lemma,
      isMain: node.lemma === mainWord
    }));

    // Add null check for edges
    const links = wordNetwork.edges ? wordNetwork.edges.map(edge => ({
      source: edge.source,
      target: edge.target,
      type: edge.type
    })) : [];

    // Simple force simulation
    const simulation = d3.forceSimulation()
      .nodes(nodes as any)
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(0, 0))
      .force('collision', d3.forceCollide().radius(50));
    
    // Only add link force if we have links
    if (links.length > 0) {
      simulation.force('link', d3.forceLink(links as any).id((d: any) => d.id).distance(100));
    }
    
    simulation.on('tick', ticked);

    // Create the links only if we have any
    if (links.length > 0) {
      const link = g.selectAll('.link')
        .data(links)
        .enter()
        .append('line')
        .attr('class', 'link')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', 1.5);
    }

    // Create the nodes
    const node = g.selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', d => `node ${d.isMain ? 'main-node' : ''}`)
      .on('click', (event, d: any) => {
        if (d.label !== mainWord) {
          onNodeClick(d.label);
        }
      });

    // Add circles to each node
    node.append('circle')
      .attr('r', (d: any) => d.isMain ? 30 : 20)
      .attr('fill', (d: any) => d.isMain ? '#ff9500' : '#87CEEB')
      .attr('stroke', '#444')
      .attr('stroke-width', 1.5);

    // Add text labels to each node
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', (d: any) => d.isMain ? '0.3em' : '0.3em')
      .attr('fill', '#fff')
      .attr('font-size', (d: any) => d.isMain ? '14px' : '12px')
      .text((d: any) => d.label);

    // Tick function to update positions
    function ticked() {
      if (links.length > 0) {
        g.selectAll('.link')
          .attr('x1', (d: any) => d.source.x)
          .attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x)
          .attr('y2', (d: any) => d.target.y);
      }

      node.attr('transform', (d: any) => `translate(${d.x}, ${d.y})`);
    }

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Initial zoom to fit
    const initialScale = 0.8;
    svg.call((zoom as any).transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(initialScale));

    return () => {
      simulation.stop();
    };
  }, [wordNetwork, mainWord, onNodeClick]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <svg 
        ref={svgRef} 
        style={{ 
          width: '100%', 
          height: '100%', 
          backgroundColor: 'transparent',
          overflow: 'visible'
        }} 
      />
    </div>
  );
};

export default SimpleWordGraph; 