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
  const [isDragging, setIsDragging] = useState(false);
  
  // Performance optimization: Memoize color function to use CSS variables
  const getNodeColor = useMemo(() => {
    return (type: string): string => {
      // Get computed styles to access CSS variables
      const isLightTheme = theme !== 'dark';
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
      .attr("refX", 30) // Increased to prevent arrow from overlapping the node
      .attr("refY", 0)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
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

    // Process nodes data with improved position initialization for centered layout
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

      // Calculate a balanced initial position with the main word in center
      // Using polar coordinates for better initial layout
      const angle = (index / wordNetwork.nodes.length) * 2 * Math.PI;
      const initialRadius = nodeLemma === mainWord ? 0 : Math.min(width, height) * 0.35;
      
      return {
        id: node.id,
        label: nodeLemma,
        group: nodeType,
        isMain: nodeLemma === mainWord,
        // Ensure main word starts in center
        x: width/2 + (nodeLemma === mainWord ? 0 : Math.cos(angle) * initialRadius),
        y: height/2 + (nodeLemma === mainWord ? 0 : Math.sin(angle) * initialRadius),
        // Pin the main word to the center
        fx: nodeLemma === mainWord ? width/2 : null,
        fy: nodeLemma === mainWord ? height/2 : null
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

    // Create force simulation with optimized forces for better layout
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100).strength(0.5))
      .force('charge', d3.forceManyBody().strength(-800))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => d.isMain ? 60 : 30))
      .force('x', d3.forceX(width / 2).strength(0.1))
      .force('y', d3.forceY(height / 2).strength(0.1))
      .alphaDecay(0.028) // Slower decay for smoother layout

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
      .style('cursor', 'pointer')
      .on('click', (event, d: any) => {
        if (isDragging) return; // Don't register clicks during drag
        
        event.preventDefault();
        event.stopPropagation();
        if (d.label && d.label !== mainWord) {
          onNodeClick(d.label);
        }
      })
      .on('mouseover', function(event, d: any) {
        if (!isDragging) { // Only show tooltip if not dragging
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
        }
      })
      .on('mouseout', function() {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', (d: any) => (d.isMain ? 34 : 20));
          
        // Clear tooltip
        setTooltipData(null);
      });

    // Add improved drag behavior for smoother interaction
    node.call(d3.drag<SVGGElement, any>()
      .on('start', function(event, d: any) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        setIsDragging(true);
        
        // Store current position
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', function(event, d: any) {
        // Update position during drag
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', function(event, d: any) {
        if (!event.active) simulation.alphaTarget(0);
        
        // Only keep main word fixed; release others
        if (!d.isMain) {
          d.fx = null;
          d.fy = null;
        }
        
        // Small delay to prevent click after drag
        setTimeout(() => setIsDragging(false), 100);
      }) as any);

    // Add circles to nodes with cleaner styling
    node.append('circle')
      .attr('r', (d: any) => d.isMain ? 34 : 20)
      .attr('fill', (d: any) => getNodeColor(d.group))
      .attr('stroke', theme === 'dark' ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.1)')
      .attr('stroke-width', 1.5);

    // Add text labels with improved readability
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('fill', d => {
        // Text is always white on dark nodes, dark on light nodes
        const type = (d as any).group;
        const darkNodeTypes = ['main', 'antonym', 'hypernym'];
        return darkNodeTypes.includes(type) || theme === 'dark' 
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
        .attr('stroke', theme === 'dark' ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.1)')
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
  }, [wordNetwork, mainWord, onNodeClick, theme, getNodeColor, isDragging]);

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
    <div ref={containerRef} className={`graph-container ${theme === 'dark' ? 'dark-theme' : ''}`}>
      <svg 
        ref={svgRef} 
        className="word-graph-svg"
        width="100%"
        height="100%"
      />
      {tooltipData && !isDragging && (
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