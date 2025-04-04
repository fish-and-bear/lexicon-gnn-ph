import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { EtymologyTree, EtymologyNode, EtymologyLink } from '../types'; // Assuming these types exist
import { useTheme } from '../contexts/ThemeContext';
import './EtymologyTreeGraph.css'; // Create this CSS file later for styling

interface EtymologyTreeGraphProps {
  etymologyTree: EtymologyTree | null;
  onEtymologyNodeClick: (nodeData: EtymologyNode) => void; // Handler for clicking a node
  width?: number; // Optional width
  height?: number; // Optional height
}

// Extend EtymologyNode for D3 hierarchy/tree layout if needed
interface D3HierarchyNode extends d3.HierarchyNode<EtymologyNode> {
  x?: number;
  y?: number;
  data: EtymologyNode; // Ensure data field holds the original EtymologyNode
}

const EtymologyTreeGraph: React.FC<EtymologyTreeGraphProps> = ({
  etymologyTree,
  onEtymologyNodeClick,
  width = 600, // Default width
  height = 400, // Default height
}) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement>(null); // Ref for the main group for zoom/pan

  useEffect(() => {
    if (!etymologyTree || !svgRef.current || !gRef.current) {
      // Clear previous render if data is missing
      if (gRef.current) {
        d3.select(gRef.current).selectAll('*').remove();
      }
      return;
    }

    const svg = d3.select(svgRef.current);
    const g = d3.select(gRef.current);

    // Clear previous render
    g.selectAll('*').remove();

    // --- Basic Placeholder ---
    // Replace this with actual D3 tree layout logic later
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#ccc' : '#555')
      .text('Etymology Tree Placeholder');
      
    // Example: Log nodes to console
    console.log('Etymology Tree Nodes:', etymologyTree.nodes);
    console.log('Etymology Tree Edges:', etymologyTree.edges);


    // --- D3 Tree Layout Logic (To be implemented) ---
    // 1. Prepare hierarchical data (using d3.stratify() if edges define parent-child)
    // 2. Create tree layout (d3.tree())
    // 3. Calculate node/link positions
    // 4. Draw links (paths)
    // 5. Draw nodes (circles/text)
    // 6. Add click handlers calling onEtymologyNodeClick(d.data)
    // 7. Implement zoom/pan

  }, [etymologyTree, width, height, theme, onEtymologyNodeClick]); // Dependencies

  return (
    <div className={`etymology-tree-container ${theme}-theme`}>
      <svg ref={svgRef} width={width} height={height} style={{ maxWidth: '100%', height: 'auto' }}>
        <g ref={gRef}></g> {/* Group for tree elements + zoom/pan */}
      </svg>
    </div>
  );
};

export default EtymologyTreeGraph;
