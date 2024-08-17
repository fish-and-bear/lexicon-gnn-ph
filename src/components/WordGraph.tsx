import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import './WordGraph.css';
import { WordNetwork, WordInfo } from '../types';
import { useTheme } from '../contexts/ThemeContext';

interface WordGraphProps {
  wordNetwork: WordNetwork;
  mainWord: string;
  onNodeClick: (wordInfo: WordInfo) => void;
}

interface CustomNode extends d3.SimulationNodeDatum {
  id: string;
  group: string[];
  info: WordInfo;
}

interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
  type: string;
}

const WordGraph: React.FC<WordGraphProps> = ({ wordNetwork, mainWord, onNodeClick }) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement | null>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const nodes: CustomNode[] = useMemo(() => {
    return Object.entries(wordNetwork).map(([word, info]) => ({
      id: word,
      group: word === mainWord ? ['main'] : ['related'],
      info: info,
    }));
  }, [wordNetwork, mainWord]);

  const links: CustomLink[] = useMemo(() => {
    const tempLinks: CustomLink[] = [];
    Object.entries(wordNetwork).forEach(([word, info]) => {
      Object.keys(info.derivatives || {}).forEach(derivative => {
        if (wordNetwork[derivative]) {
          tempLinks.push({ source: word, target: derivative, type: 'derivative' });
        }
      });
      (info.associated_words || []).forEach(associatedWord => {
        if (wordNetwork[associatedWord]) {
          tempLinks.push({ source: word, target: associatedWord, type: 'associated' });
        }
      });
      (info.root_words || []).forEach(rootWord => {
        if (wordNetwork[rootWord]) {
          tempLinks.push({ source: word, target: rootWord, type: 'etymology' });
        }
      });
      if (info.root_word && wordNetwork[info.root_word]) {
        tempLinks.push({ source: word, target: info.root_word, type: 'root' });
      }
    });
    return tempLinks;
  }, [wordNetwork]);

  const updateGraph = useCallback(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const containerRect = svgRef.current.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;

    svg.attr('width', width).attr('height', height);

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 1]) // Prevent zooming by user
      .translateExtent([[0, 0], [width, height]]) // Constrain panning
      .on('zoom', (event) => {
        if (gRef.current) {
          d3.select(gRef.current).attr('transform', event.transform.toString());
        }
      });

    svg.call(zoom);

    const g = gRef.current || svg.append('g').node();
    gRef.current = g;

    const simulation = d3.forceSimulation<CustomNode>(nodes)
      .force('link', d3.forceLink<CustomNode, CustomLink>(links).id(d => d.id).distance(80)) 
      .force('charge', d3.forceManyBody().strength(-50)) 
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius(70).strength(0.7)) 
      .force('x', d3.forceX(width / 2).strength(0.05)) 
      .force('y', d3.forceY(height / 2).strength(0.05)) 
      .alphaDecay(0.02) 
      .velocityDecay(0.4); 

    const link = d3.select(g!).selectAll('.link')
      .data(links)
      .join('line')
      .attr('class', 'link')
      .attr('stroke', d => getRelationColor(d.type));

    const node = d3.select(g!).selectAll('.node')
      .data(nodes)
      .join('g')
      .attr('class', 'node')
      .call(d3.drag<SVGGElement, CustomNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    node.selectAll('circle')
      .data(d => [d])
      .join('circle')
      .attr('r', d => d.group.includes('main') ? 30 : 25)
      .attr('fill', d => getNodeColor(d.group));

    node.selectAll('text')
      .data(d => [d])
      .join('text')
      .text(d => d.id)
      .attr('dy', 35)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#e0e0e0' : '#333');

    node.on('click', (event, d) => {
      setSelectedNodeId(d.id);
      onNodeClick(d.info);

      const currentTransform = d3.zoomTransform(svg.node() as any);
      const translateX = width / 2 - d.x! * currentTransform.k;
      const translateY = height / 2 - d.y! * currentTransform.k;
      svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY));
    })
    .on('mouseover', (event, d) => setHoveredNode(d))
    .on('mouseout', () => setHoveredNode(null));

    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as CustomNode).x!)
        .attr('y1', d => (d.source as CustomNode).y!)
        .attr('x2', d => (d.target as CustomNode).x!)
        .attr('y2', d => (d.target as CustomNode).y!);
      
      node.attr('transform', d => `translate(${d.x},${d.y})`);

      nodes.forEach((d) => {
        d.x = Math.max(70, Math.min(width - 70, d.x!));
        d.y = Math.max(70, Math.min(height - 70, d.y!));
      });
    });

    node.classed('selected', d => d.id === selectedNodeId);

    function dragstarted(event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) {
      event.subject.fx = Math.max(70, Math.min(width - 70, event.x));
      event.subject.fy = Math.max(70, Math.min(height - 70, event.y));
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
  }, [nodes, links, theme, onNodeClick, selectedNodeId]);

  useEffect(() => {
    updateGraph();
    window.addEventListener('resize', updateGraph);
    return () => window.removeEventListener('resize', updateGraph);
  }, [updateGraph]);

  function getNodeColor(group: string[]): string {
    if (group.includes('main')) return '#ff4f00';
    if (group.includes('derivative')) return '#4a4a4a';
    if (group.includes('etymology')) return '#ffa500';
    if (group.includes('root')) return '#00ced1';
    return '#9370db'; 
  }

  function getRelationColor(type: string): string {
    switch (type) {
      case 'derivative': return '#4a4a4a';
      case 'etymology': return '#ffa500';
      case 'root': return '#00ced1';
      default: return '#9370db';
    }
  }

  return (
    <div className={`graph-container ${theme}`}>
      <svg ref={svgRef}></svg>
      {hoveredNode && (
        <div className="node-tooltip" style={{ left: hoveredNode.x, top: hoveredNode.y }}>
          <h3>{hoveredNode.id}</h3>
          <p>{hoveredNode.info?.definitions && hoveredNode.info.definitions.length > 0 ? hoveredNode.info.definitions[0].meanings[0] : 'No definition available'}</p>
          <p>Relations: {hoveredNode.group.join(', ')}</p>
        </div>
      )}
      <div className="legend">
        <div className="legend-item"><div className="color-box main"></div><span>Main Word</span></div>
        <div className="legend-item"><div className="color-box derivative"></div><span>Derivative</span></div>
        <div className="legend-item"><div className="color-box etymology"></div><span>Etymology</span></div>
        <div className="legend-item"><div className="color-box root"></div><span>Root</span></div>
        <div className="legend-item"><div className="color-box associated"></div><span>Associated Word</span></div>
      </div>
    </div>
  );
};

export default React.memo(WordGraph);
