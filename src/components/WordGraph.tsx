import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import './WordGraph.css';
import { WordNetwork, NetworkWordInfo } from '../types';
import { useTheme } from '../contexts/ThemeContext';

interface WordGraphProps {
  wordNetwork: WordNetwork;
  mainWord: string;
  onNodeClick: (word: string) => void;
}

interface CustomNode extends d3.SimulationNodeDatum {
  id: string;
  group: string;
  info: NetworkWordInfo;
}

interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
  type: string;
}

const WordGraph: React.FC<WordGraphProps> = React.memo(({ wordNetwork, mainWord, onNodeClick }) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const nodes: CustomNode[] = useMemo(() => {
    return Object.entries(wordNetwork).map(([word, info]) => ({
      id: word,
      group: getNodeGroup(word, info, mainWord),
      info: info,
    }));
  }, [wordNetwork, mainWord]);

  const links: CustomLink[] = useMemo(() => {
    const tempLinks: CustomLink[] = [];
    Object.entries(wordNetwork).forEach(([word, info]) => {
      info.related_words.forEach((relatedWord) => {
        if (wordNetwork[relatedWord]) {
          tempLinks.push({ source: word, target: relatedWord, type: 'related' });
        }
      });
    });
    return tempLinks;
  }, [wordNetwork]);

  const updateGraph = useCallback(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const containerRect = svgRef.current.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;

    svg.attr('width', width).attr('height', height);

    const g = svg.append('g');

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-500))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius(50).strength(0.7));

    const link = g.selectAll('.link')
      .data(links)
      .join('line')
      .attr('class', 'link')
      .attr('stroke', (d) => getRelationColor(d.type));

    const node = g.selectAll('.node')
      .data(nodes)
      .join('g')
      .attr('class', (d) => `node ${d.id === selectedNodeId ? 'selected' : ''}`)
      .call(d3.drag<SVGGElement, CustomNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    node.append('circle')
      .attr('r', (d) => d.group === 'main' ? 35 : 25)
      .attr('fill', (d) => getNodeColor(d.group));

    node.append('text')
      .text((d) => d.id)
      .attr('dy', 40)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#e0e0e0' : '#333');

    node.on('click', (event, d) => {
      setSelectedNodeId(d.id);
      onNodeClick(d.id);
    })
    .on('mouseover', (event, d) => setHoveredNode(d))
    .on('mouseout', () => setHoveredNode(null));

    function ticked() {
      link
        .attr('x1', (d) => Math.max(50, Math.min(width - 50, (d.source as CustomNode).x!)))
        .attr('y1', (d) => Math.max(50, Math.min(height - 50, (d.source as CustomNode).y!)))
        .attr('x2', (d) => Math.max(50, Math.min(width - 50, (d.target as CustomNode).x!)))
        .attr('y2', (d) => Math.max(50, Math.min(height - 50, (d.target as CustomNode).y!)));

      node.attr('transform', (d) => `translate(${Math.max(50, Math.min(width - 50, d.x!))},${Math.max(50, Math.min(height - 50, d.y!))})`);
    }

    simulation.nodes(nodes).on('tick', ticked);
    simulation.force<d3.ForceLink<CustomNode, CustomLink>>('link')?.links(links);

    function dragstarted(event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    return () => {
      simulation.stop();
    };
  }, [nodes, links, theme, onNodeClick, selectedNodeId]);

  useEffect(() => {
    const cleanupFunction = updateGraph();
    window.addEventListener('resize', updateGraph);
    return () => {
      if (typeof cleanupFunction === 'function') {
        cleanupFunction();
      }
      window.removeEventListener('resize', updateGraph);
    };
  }, [updateGraph]);

  function getNodeGroup(word: string, info: NetworkWordInfo, mainWord: string): string {
    if (word === mainWord) return 'main';
    if (info.related_words.includes(mainWord)) return 'related';
    return 'other';
  }

  function getNodeColor(group: string): string {
    switch (group) {
      case 'main': return 'var(--color-main)';
      case 'related': return 'var(--color-associated)';
      default: return 'var(--color-other)';
    }
  }

  function getRelationColor(type: string): string {
    return 'var(--color-associated)';
  }

  const legendItems = [
    { key: 'main', label: 'Main Word' },
    { key: 'related', label: 'Related Word' },
  ];

  return (
    <div className={`graph-container ${theme}`}>
      <svg ref={svgRef}></svg>
      {hoveredNode && (
        <div className="node-tooltip" style={{ left: hoveredNode.x, top: hoveredNode.y }}>
          <h3>{hoveredNode.id}</h3>
          <p>{hoveredNode.info.definitions?.[0]?.meanings?.[0]?.definition || "No definition available"}</p>
          <p>Relation: {hoveredNode.group}</p>
        </div>
      )}
      <div className="legend">
        {legendItems.map(item => (
          <div key={item.key} className="legend-item">
            <div className={`color-box ${item.key}`}></div>
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

export default WordGraph;