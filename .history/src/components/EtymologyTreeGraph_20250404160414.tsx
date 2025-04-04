import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
// Removed EtymologyNode, EtymologyLink imports - Assuming EtymologyTree defines the structure
import { EtymologyTree } from '../types'; 
import { useTheme } from '../contexts/ThemeContext';
import './EtymologyTreeGraph.css'; // Create this CSS file later for styling

interface EtymologyTreeGraphProps {
  etymologyTree: EtymologyTree | null;
  onEtymologyNodeClick: (nodeData: any) => void; // Use 'any' for now until node structure is clear
  width?: number; // Optional width
  height?: number; // Optional height
}

// The structure for D3 hierarchy might need adjustment based on the actual EtymologyTree structure.
// Using 'any' for data temporarily.
interface D3HierarchyNode extends d3.HierarchyNode<any> { 
  x?: number;
  y?: number;
  data: any; // Adjust this type based on the actual node structure within EtymologyTree
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
      
    // EtymologyTree type likely has a root node and children, not top-level nodes/edges.
    // Example: Log the root word (if available)
    if (etymologyTree.word) { 
      console.log('Etymology Tree Root Word:', etymologyTree.word);
    }
    // console.log('Etymology Tree Nodes:', etymologyTree.nodes); // Removed: Property 'nodes' likely doesn't exist
    // console.log('Etymology Tree Edges:', etymologyTree.edges); // Removed: Property 'edges' likely doesn't exist


    // --- D3 Tree Layout Logic (To be implemented) ---
    // 1. Prepare hierarchical data (likely using d3.hierarchy() on the root node of etymologyTree)
    // 2. Create tree layout (d3.tree() or d3.cluster())
    // 3. Calculate node/link positions based on the hierarchy
    // 4. Draw links (paths connecting parent/child nodes)
    // 5. Draw nodes (circles/text representing words/languages)
    // 6. Add click handlers calling onEtymologyNodeClick(d.data) (ensure d.data has the correct info)
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
