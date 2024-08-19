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
  source: string | CustomNode;
  target: string | CustomNode;
  type: string;
}

const WordGraph: React.FC<WordGraphProps> = React.memo(({ wordNetwork, mainWord, onNodeClick }) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const getNodeRelation = useCallback((word: string, info: NetworkWordInfo, mainWord: string): string => {
    if (word === mainWord) return 'main';
    if (info.derivatives?.includes(mainWord)) return 'derivative';
    if (info.root_word === mainWord) return 'root';
    if (info.etymology?.parsed?.includes(mainWord)) return 'etymology';
    if (info.associated_words?.includes(mainWord)) return 'associated';
    if (info.synonyms?.includes(mainWord)) return 'synonym';
    if (info.antonyms?.includes(mainWord)) return 'antonym';
    return 'other';
  }, []);

  const nodes: CustomNode[] = useMemo(() => {
    return Object.entries(wordNetwork).map(([word, info]) => ({
      id: word,
      group: getNodeRelation(word, info, mainWord),
      info: info,
    }));
  }, [wordNetwork, mainWord, getNodeRelation]);

  const links: CustomLink[] = useMemo(() => {
    const tempLinks: CustomLink[] = [];
    Object.entries(wordNetwork).forEach(([word, info]) => {
      console.log(`Processing word: ${word}`, info);
      const addLink = (target: string, type: string) => {
        if (wordNetwork[target]) {
          tempLinks.push({ source: word, target, type });
        }
      };

      info.derivatives?.forEach(derivative => addLink(derivative, 'derivative'));
      if (info.root_word) addLink(info.root_word, 'root');
      info.etymology?.parsed?.forEach(etymWord => addLink(etymWord, 'etymology'));
      info.associated_words?.forEach(assocWord => addLink(assocWord, 'associated'));
      info.synonyms?.forEach(synonym => addLink(synonym, 'synonym'));
      info.antonyms?.forEach(antonym => addLink(antonym, 'antonym'));
    });
    console.log('Created links:', tempLinks);
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

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        g.attr('transform', event.transform.toString());
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    const simulation = d3.forceSimulation<CustomNode>(nodes)
      .force('link', d3.forceLink<CustomNode, CustomLink>(links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius(30).strength(0.7));

    const link = g.selectAll('.link')
      .data(links)
      .join('line')
      .attr('class', 'link')
      .attr('stroke', (d: CustomLink) => getNodeColor(d.type))
      .attr('stroke-width', 1);

    const node = g.selectAll('.node')
      .data(nodes)
      .join('g')
      .attr('class', (d: CustomNode) => `node ${d.id === selectedNodeId ? 'selected' : ''}`)
      .call(d3.drag<SVGGElement, CustomNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    node.append('circle')
      .attr('r', (d: CustomNode) => d.group === 'main' ? 25 : 15)
      .attr('fill', (d: CustomNode) => getNodeColor(d.group))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    node.append('text')
      .text((d: CustomNode) => d.id)
      .attr('dy', 25)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#e0e0e0' : '#333')
      .style('font-size', '10px')
      .style('font-weight', 'bold');

    node.on('click', (event: MouseEvent, d: CustomNode) => {
      setSelectedNodeId(d.id);
      onNodeClick(d.id);
    })
    .on('mouseover', (event: MouseEvent, d: CustomNode) => setHoveredNode(d))
    .on('mouseout', () => setHoveredNode(null));

    function ticked() {
      console.log('Tick');
      link
        .attr('x1', d => (d.source as CustomNode).x!)
        .attr('y1', d => (d.source as CustomNode).y!)
        .attr('x2', d => (d.target as CustomNode).x!)
        .attr('y2', d => (d.target as CustomNode).y!);

      node.attr('transform', (d: CustomNode) => `translate(${d.x!},${d.y!})`);
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

    // Center the graph on the main word
    const mainNode = nodes.find(n => n.id === mainWord);
    if (mainNode) {
      centerNode(mainNode);
    }

    return () => {
      simulation.stop();
    };
  }, [nodes, links, theme, onNodeClick, selectedNodeId, mainWord]);

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

  function getNodeColor(group: string): string {
    switch (group) {
      case 'main': return '#FF6B6B';
      case 'root': return '#4ECDC4';
      case 'associated': return '#45B7D1';
      case 'etymology': return '#F9C74F';
      case 'derivative': return '#9B5DE5';
      case 'synonym': return '#66BB6A';
      case 'antonym': return '#EF5350';
      default: return '#A0A0A0';
    }
  }

  const handleZoom = useCallback((scale: number) => {
    if (zoomRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(zoomRef.current.scaleBy, scale);
    }
  }, []);

  const handleResetZoom = useCallback(() => {
    if (zoomRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(zoomRef.current.transform, d3.zoomIdentity);
    }
  }, []);

  const centerNode = useCallback((node: CustomNode) => {
    if (zoomRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current);
      const width = Number(svg.attr('width'));
      const height = Number(svg.attr('height'));
      const scale = 0.8;
      const x = width / 2 - (node.x || 0) * scale;
      const y = height / 2 - (node.y || 0) * scale;
      svg.transition().duration(750).call(
        zoomRef.current.transform,
        d3.zoomIdentity.translate(x, y).scale(scale)
      );
    }
  }, []);

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
      <div className="zoom-controls">
        <button onClick={() => handleZoom(1.2)} className="zoom-button">+</button>
        <button onClick={() => handleZoom(1/1.2)} className="zoom-button">-</button>
        <button onClick={handleResetZoom} className="zoom-button">Reset</button>
      </div>
    </div>
  );
});

export default WordGraph;