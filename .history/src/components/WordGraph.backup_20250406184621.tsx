// This is a backup of the original WordGraph component
// Keep it for reference in case we need to revert changes

import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { WordNetworkResponse } from '../types';
import { useTheme } from '../contexts/ThemeContext';
import './WordGraph.css';

interface WordGraphProps {
  wordNetwork: WordNetworkResponse | null;
  mainWord: string | null;
  onNodeClick: (word: string) => void;
  onNetworkChange: (depth: number) => void;
  initialDepth: number;
  initialBreadth?: number;
}

interface CustomNode extends d3.SimulationNodeDatum {
  id: string;
  word: string;
  label?: string;
  group: string;
  connections?: number;
  pinned?: boolean;
  originalId?: number;
  language?: string;
  definitions?: string[];
  path?: Array<{ type: string; word: string }>;
  has_baybayin?: boolean;
  baybayin_form?: string | null;
}

interface CustomLink {
  source: string | CustomNode;
  target: string | CustomNode;
  relationship: string;
}

const getTextColorForBackground = (hexColor: string): string => {
  // Logic to determine text color based on background color
  // For simplicity, returning white for now
  return '#ffffff';
};

const WordGraph: React.FC<WordGraphProps> = ({
  wordNetwork,
  mainWord,
  onNodeClick,
  onNetworkChange,
  initialDepth,
  initialBreadth = 15,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [transform, setTransform] = useState<d3.ZoomTransform | null>(null);

  // Memoize filtered nodes and links
  const { filteredNodes, filteredLinks } = useMemo(() => {
    if (!wordNetwork || !wordNetwork.nodes || !wordNetwork.edges) {
      return { filteredNodes: [], filteredLinks: [] };
    }

    // Create nodes from wordNetwork
    const nodes: CustomNode[] = wordNetwork.nodes.map((node: any) => ({
      id: node.id.toString(),
      word: node.lemma,
      label: node.lemma,
      group: node.lemma === mainWord ? 'main' : (node.group || 'default'),
      language: node.language_code,
      has_baybayin: node.has_baybayin,
      baybayin_form: node.baybayin_form,
      x: undefined,
      y: undefined,
      vx: undefined,
      vy: undefined,
      fx: undefined,
      fy: undefined
    }));

    // Create links from wordNetwork
    const links: CustomLink[] = wordNetwork.edges.map((edge: any) => ({
      source: edge.source.toString(),
      target: edge.target.toString(),
      relationship: edge.type
    }));

    return { filteredNodes: nodes, filteredLinks: links };
  }, [wordNetwork, mainWord]);

  // Simplified render method
  return (
    <div className="graph-container">
      <div className="graph-svg-container">
        <svg ref={svgRef} className="graph-svg"></svg>
      </div>
    </div>
  );
};

export default WordGraph; 