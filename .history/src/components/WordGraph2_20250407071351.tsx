import React, { useEffect, useRef, useCallback, useState, useMemo } from "react";
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
  
  // Performance optimization: Memoize color function to avoid recreating on each render
  const getNodeColor = useMemo(() => {
    return (type: string): string => {
      const colors = {
        main: '#FF7700', // Vibrant orange for main word
        synonym: theme === 'dark' ? '#8BC34A' : '#689F38', // Green for synonyms
        antonym: theme === 'dark' ? '#E53E3E' : '#C53030', // Red for antonyms
        hypernym: theme === 'dark' ? '#3182CE' : '#2B6CB0', // Blue for hypernyms
        hyponym: theme === 'dark' ? '#805AD5' : '#6B46C1', // Purple for hyponyms
        related: theme === 'dark' ? '#F6AD55' : '#DD6B20', // Orange for related words
        default: theme === 'dark' ? '#718096' : '#4A5568' // Gray for default/unknown
      };
      
      return colors[type as keyof typeof colors] || colors.default;
    };
  }, [theme]);

  // Effect to update props to parent when depth changes
  useEffect(() => {
    onNetworkChange(initialDepth);
  }, [initialDepth, onNetworkChange]);

  // Effect to handle window resize with debounce for better performance
  useEffect(() => {
    let resizeTimer: NodeJS.Timeout;
    
    const handleResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (containerRef.current && wordNetwork) {
          updateGraph();
        }
      }, 100);
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      clearTimeout(resizeTimer);
    };
  }, [wordNetwork]);

  // Main graph rendering function
  const updateGraph = useCallback(() => {
    if (!svgRef.current || !containerRef.current) return;
    
    // Get container dimensions for responsive sizing
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    
    // Clear any existing graph
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    if (!wordNetwork || !mainWord) {
      // Display a message for empty state
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
        .attr('stroke-width', 1.5);
        
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
        if (d === "arrow-synonym") return getNodeColor('synonym');
        if (d === "arrow-antonym") return getNodeColor('antonym');
        if (d === "arrow-hypernym") return getNodeColor('hypernym');
        if (d === "arrow-hyponym") return getNodeColor('hyponym');
        if (d === "arrow-related") return getNodeColor('related');
        return getNodeColor('default');
      })
      .attr("d", "M0,-5L10,0L0,5");

    // Add subtle animations for flow
    svg.append("defs").append("style").text(`
      .link {
        stroke-opacity: 0.7;
        transition: stroke-opacity 0.2s;
      }
      
      .link:hover {
        stroke-opacity: 1;
      }
      
      .animated-link {
        stroke-dasharray: 8, 4;
        animation: dash 15s linear infinite;
      }
      
      @keyframes dash {
        to {
          stroke-dashoffset: -100;
        }
      }
      
      .node {
        transition: transform 0.2s;
      }
      
      .node:hover {
        transform: scale(1.1);
      }
      
      .main-node {
        animation: pulse 3s infinite ease-in-out;
      }
      
      @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
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
        const mainNodeId = wordNetwork.nodes.find(n => String(n.lemma) === mainWord)?.id;
        
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
      
      // Improved initial position calculation for better initial layout
      const angle = (index / wordNetwork.nodes.length) * 2 * Math.PI;
      const radius = nodeLemma === mainWord ? 0 : Math.min(width, height) * 0.35;
      const x = width/2 + Math.cos(angle) * radius;
      const y = height/2 + Math.sin(angle) * radius;
      
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
        // Ensure source and target can be parsed
        const source = typeof edge.source === 'object' ? edge.source.id : edge.source;
        const target = typeof edge.target === 'object' ? edge.target.id : edge.target;
        
        // Only include edges where both source and target nodes exist
        return nodesById.has(source) && nodesById.has(target);
      })
      .map(edge => {
        // Get the source and target IDs
        const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
        const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
        
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
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100).strength(0.6))
      .force('charge', d3.forceManyBody().strength(-800))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => d.isMain ? 50 : 25))
      .force('x', d3.forceX(width / 2).strength(0.1))
      .force('y', d3.forceY(height / 2).strength(0.1))
      .alphaDecay(0.04); // Slower decay for smoother animation

    // Create links with cleaner styling
    const link = g.selectAll('.link')
      .data(links)
      .join('line')
      .attr('class', d => `link ${Math.random() > 0.5 ? 'animated-link' : ''}`)
      .attr('stroke', (d: any) => getNodeColor(d.type))
      .attr('stroke-width', 2)
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
          
        // Show tooltip with more information
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

    // Add drag behavior for interactivity
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

    // Add circles to nodes with cleaner styling
    node.append('circle')
      .attr('r', (d: any) => d.isMain ? 34 : 20)
      .attr('fill', (d: any) => getNodeColor(d.group))
      .attr('stroke', theme === 'dark' ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.2)')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer');

    // Add text labels with improved readability
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('fill', (d: any) => {
        // Calculate text color based on background for better readability
        const nodeType = d.group;
        return ['main', 'antonym', 'hypernym', 'hyponym'].includes(nodeType) || theme === 'dark' 
          ? '#ffffff' : '#000000';
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
    svg.call((zoom as any).transform, d3.zoomIdentity);
    
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
      .attr('width', 140)
      .attr('height', legendData.length * 22 + 10)
      .attr('rx', 8)
      .attr('ry', 8)
      .attr('fill', theme === 'dark' ? 'rgba(26, 32, 44, 0.8)' : 'rgba(255, 255, 255, 0.9)')
      .attr('stroke', theme === 'dark' ? '#4A5568' : '#E2E8F0')
      .attr('stroke-width', 1);
      
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(20, 20)`);

    legendData.forEach((item, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 22})`);
      
      legendItem.append('circle')
        .attr('r', 7)
        .attr('fill', item.color)
        .attr('stroke', theme === 'dark' ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.2)')
        .attr('stroke-width', 1);
      
      legendItem.append('text')
        .attr('x', 14)
        .attr('y', 4)
        .attr('fill', theme === 'dark' ? '#E2E8F0' : '#2D3748')
        .attr('font-size', '12px')
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
    <div ref={containerRef} className="graph-container">
      <svg 
        ref={svgRef} 
        className="word-graph-svg"
        width="100%"
        height="100%"
      />
      {tooltipData && (
        <div 
          className="graph-tooltip" 
          style={{
            position: 'absolute',
            left: `${tooltipData.x + 15}px`,
            top: `${tooltipData.y - 15}px`,
            backgroundColor: theme === 'dark' ? 'rgba(26, 32, 44, 0.92)' : 'rgba(255, 255, 255, 0.92)',
            color: theme === 'dark' ? '#E2E8F0' : '#2D3748',
            padding: '8px 12px',
            borderRadius: '8px',
            boxShadow: '0 4px 10px rgba(0, 0, 0, 0.15)',
            border: `1px solid ${theme === 'dark' ? '#4A5568' : '#E2E8F0'}`,
            zIndex: 1000,
            maxWidth: '250px',
            pointerEvents: 'none'
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{tooltipData.text}</div>
          <div style={{ 
            fontSize: '12px', 
            color: theme === 'dark' ? '#A0AEC0' : '#4A5568'
          }}>
            {getRelationDescription(tooltipData.group)}
          </div>
        </div>
      )}
    </div>
  );
};

export default WordGraph2;
export default WordGraph2;