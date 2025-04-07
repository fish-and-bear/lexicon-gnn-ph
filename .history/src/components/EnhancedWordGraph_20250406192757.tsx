import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { WordNetworkResponse } from '../types';
import { useTheme } from '../contexts/ThemeContext';
import './EnhancedWordGraph.css';

interface WordGraphProps {
  wordNetwork: WordNetworkResponse;
  mainWord: string;
  onNodeClick: (word: string) => void;
}

interface CustomNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  group: string;
  isMain?: boolean;
  x?: number;
  y?: number;
}

interface CustomLink {
  source: string | CustomNode;
  target: string | CustomNode;
  type: string;
}

// Define color scheme based on relation types
const getNodeColor = (group: string, theme: string): string => {
  const colors = {
    main: '#ff9500', // Orange - Main word
    synonym: theme === 'dark' ? '#4caf50' : '#2e7d32', // Green - Synonyms
    antonym: theme === 'dark' ? '#f44336' : '#c62828', // Red - Antonyms
    hypernym: theme === 'dark' ? '#2196f3' : '#1565c0', // Blue - Hypernyms
    hyponym: theme === 'dark' ? '#9c27b0' : '#6a1b9a', // Purple - Hyponyms
    related: theme === 'dark' ? '#ff9800' : '#ef6c00', // Orange - Related
    root: theme === 'dark' ? '#ffeb3b' : '#f9a825', // Yellow - Roots
    default: theme === 'dark' ? '#78909c' : '#546e7a' // Gray - Default
  };
  
  return colors[group as keyof typeof colors] || colors.default;
};

// Get text color that contrasts with background
const getTextColor = (backgroundColor: string): string => {
  // Convert hex to RGB
  const r = parseInt(backgroundColor.slice(1, 3), 16);
  const g = parseInt(backgroundColor.slice(3, 5), 16);
  const b = parseInt(backgroundColor.slice(5, 7), 16);
  
  // Calculate luminance - simplified version
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  
  // Return white for dark backgrounds, black for light
  return luminance > 0.5 ? '#000000' : '#ffffff';
};

const EnhancedWordGraph: React.FC<WordGraphProps> = ({ wordNetwork, mainWord, onNodeClick }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const { theme } = useTheme();
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [tooltipInfo, setTooltipInfo] = useState<{show: boolean, content: string, x: number, y: number}>({
    show: false, content: '', x: 0, y: 0
  });
  const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);

  // Format node data from API response
  const nodes = useMemo(() => {
    if (!wordNetwork?.nodes) return [];
    
    return wordNetwork.nodes.map(node => ({
      id: node.id.toString(),
      label: node.lemma,
      // Determine group based on relationship - if not provided, use default or main
      group: node.lemma === mainWord ? 'main' : (node.type || 'default'),
      isMain: node.lemma === mainWord
    }));
  }, [wordNetwork, mainWord]);

  // Format link data from API response
  const links = useMemo(() => {
    if (!wordNetwork?.edges || !wordNetwork.nodes) return [];
    
    return wordNetwork.edges.map(edge => ({
      source: edge.source.toString(),
      target: edge.target.toString(),
      type: edge.type
    }));
  }, [wordNetwork]);

  // Setup visualization
  useEffect(() => {
    if (!svgRef.current || !nodes.length) return;
    
    // Clear existing visualization
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    // Get container dimensions
    const width = svg.node()?.parentElement?.clientWidth || 600;
    const height = svg.node()?.parentElement?.clientHeight || 400;
    
    // Add container group
    const g = svg.append('g')
      .attr('class', 'viz-container');
    
    // Create links first (so they're behind nodes)
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'link')
      .attr('stroke', (d: CustomLink) => {
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
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6)
      .attr('data-type', (d: CustomLink) => d.type);
    
    // Create nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', (d: CustomNode) => `node${d.isMain ? ' main-node' : ''}`)
      .on('click', (event, d: CustomNode) => {
        if (d.label !== mainWord) {
          onNodeClick(d.label);
        }
      })
      .on('mouseover', (event, d: CustomNode) => {
        // Show tooltip
        const [x, y] = d3.pointer(event, svg.node());
        setTooltipInfo({
          show: true,
          content: `${d.label} (${d.group})`,
          x: x,
          y: y
        });
        
        // Highlight node and connections
        svg.selectAll('.link')
          .attr('stroke-opacity', (link: any) => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return (sourceId === d.id || targetId === d.id) ? 1 : 0.2;
          })
          .attr('stroke-width', (link: any) => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return (sourceId === d.id || targetId === d.id) ? 3 : 1.5;
          });
        
        svg.selectAll('.node')
          .attr('opacity', (node: any) => {
            const isConnected = links.some(link => {
              const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
              const targetId = typeof link.target === 'object' ? link.target.id : link.target;
              return (sourceId === d.id && targetId === node.id) || 
                     (targetId === d.id && sourceId === node.id);
            });
            return node.id === d.id || isConnected ? 1 : 0.4;
          });
      })
      .on('mouseout', () => {
        // Hide tooltip
        setTooltipInfo({ ...tooltipInfo, show: false });
        
        // Reset highlighting
        svg.selectAll('.link')
          .attr('stroke-opacity', 0.6)
          .attr('stroke-width', 2);
        
        svg.selectAll('.node')
          .attr('opacity', 1);
      })
      .call(d3.drag<SVGGElement, CustomNode>()
        .on('start', (event, d) => {
          if (!event.active && simulationRef.current) 
            simulationRef.current.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active && simulationRef.current) 
            simulationRef.current.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      );
    
    // Add node circles
    node.append('circle')
      .attr('r', (d: CustomNode) => d.isMain ? 25 : 15)
      .attr('fill', (d: CustomNode) => getNodeColor(d.group, theme))
      .attr('stroke', (d: CustomNode) => d.isMain ? '#000' : '#fff')
      .attr('stroke-width', (d: CustomNode) => d.isMain ? 2 : 1);
    
    // Add node labels
    node.append('text')
      .attr('dy', '0.35em')
      .attr('text-anchor', 'middle')
      .attr('fill', (d: CustomNode) => getTextColor(getNodeColor(d.group, theme)))
      .attr('font-size', (d: CustomNode) => d.isMain ? '12px' : '10px')
      .attr('font-weight', (d: CustomNode) => d.isMain ? 'bold' : 'normal')
      .text((d: CustomNode) => d.label);
    
    // Setup force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink<CustomNode, CustomLink>(links)
        .id(d => d.id)
        .distance(80)
        .strength(0.7))
      .force('charge', d3.forceManyBody()
        .strength(-300)
        .distanceMax(500))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide(30).strength(0.8))
      .on('tick', () => {
        // Update link positions
        link
          .attr('x1', (d: any) => (typeof d.source === 'object' ? d.source.x || 0 : 0))
          .attr('y1', (d: any) => (typeof d.source === 'object' ? d.source.y || 0 : 0))
          .attr('x2', (d: any) => (typeof d.target === 'object' ? d.target.x || 0 : 0))
          .attr('y2', (d: any) => (typeof d.target === 'object' ? d.target.y || 0 : 0));
        
        // Update node positions
        node.attr('transform', (d: CustomNode) => `translate(${d.x || 0},${d.y || 0})`);
      });
    
    // Save simulation reference
    simulationRef.current = simulation;
    
    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom);
    
    // Initial zoom to show everything
    svg.call(zoom.transform, d3.zoomIdentity
      .translate(width / 2, height / 2)
      .scale(0.8)
      .translate(-width / 2, -height / 2));
    
    // Cleanup
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [nodes, links, theme, mainWord, onNodeClick]);

  // Create a legend for relationship types
  const renderLegend = useCallback(() => {
    const legendItems = [
      { label: 'Main Word', color: getNodeColor('main', theme) },
      { label: 'Synonym', color: getNodeColor('synonym', theme) },
      { label: 'Antonym', color: getNodeColor('antonym', theme) },
      { label: 'Hypernym', color: getNodeColor('hypernym', theme) },
      { label: 'Hyponym', color: getNodeColor('hyponym', theme) },
      { label: 'Related', color: getNodeColor('related', theme) }
    ];
    
    return (
      <div className="graph-legend">
        {legendItems.map((item, index) => (
          <div key={index} className="legend-item">
            <span 
              className="legend-color" 
              style={{ backgroundColor: item.color }}
            />
            <span className="legend-label">{item.label}</span>
          </div>
        ))}
      </div>
    );
  }, [theme]);

  // Add zoom controls
  const renderZoomControls = () => {
    const handleZoomIn = () => {
      if (svgRef.current) {
        const svg = d3.select(svgRef.current);
        const currentTransform = d3.zoomTransform(svg.node()!);
        const newScale = currentTransform.k * 1.3;
        
        svg.transition()
          .duration(300)
          .call(
            (d3.zoom<SVGSVGElement, unknown>() as any)
              .transform, 
            currentTransform.scale(newScale / currentTransform.k)
          );
      }
    };
    
    const handleZoomOut = () => {
      if (svgRef.current) {
        const svg = d3.select(svgRef.current);
        const currentTransform = d3.zoomTransform(svg.node()!);
        const newScale = currentTransform.k / 1.3;
        
        svg.transition()
          .duration(300)
          .call(
            (d3.zoom<SVGSVGElement, unknown>() as any)
              .transform, 
            currentTransform.scale(newScale / currentTransform.k)
          );
      }
    };
    
    const handleResetZoom = () => {
      if (svgRef.current) {
        const svg = d3.select(svgRef.current);
        const width = svg.node()?.parentElement?.clientWidth || 600;
        const height = svg.node()?.parentElement?.clientHeight || 400;
        
        svg.transition()
          .duration(500)
          .call(
            (d3.zoom<SVGSVGElement, unknown>() as any)
              .transform,
            d3.zoomIdentity
              .translate(width / 2, height / 2)
              .scale(0.8)
              .translate(-width / 2, -height / 2)
          );
      }
    };
    
    return (
      <div className="zoom-controls">
        <button className="zoom-button" onClick={handleZoomIn} title="Zoom In">+</button>
        <button className="zoom-button" onClick={handleZoomOut} title="Zoom Out">−</button>
        <button className="zoom-button" onClick={handleResetZoom} title="Reset View">⟳</button>
      </div>
    );
  };

  return (
    <div className="graph-container">
      <div className="graph-visualization">
        <svg 
          ref={svgRef} 
          className="word-graph-svg"
        />
        {renderLegend()}
        {renderZoomControls()}
        {tooltipInfo.show && (
          <div 
            className="node-tooltip"
            style={{
              left: tooltipInfo.x + 'px',
              top: tooltipInfo.y + 'px',
              display: tooltipInfo.show ? 'block' : 'none'
            }}
          >
            {tooltipInfo.content}
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedWordGraph; 