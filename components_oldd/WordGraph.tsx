import React, { useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import { WordNode, WordLink } from '../../../src/types/wordTypes';
import { mockWordData } from '../../../src/data/mockWordData';

interface WordGraphProps {
  selectedWord: string | null;
  onWordSelect: (word: string) => void;
}

const WordGraph: React.FC<WordGraphProps> = ({ selectedWord, onWordSelect }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  const dragstarted = useCallback((event: d3.D3DragEvent<SVGGElement, WordNode, WordNode>, simulation: d3.Simulation<WordNode, undefined>) => {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }, []);

  const dragged = useCallback((event: d3.D3DragEvent<SVGGElement, WordNode, WordNode>) => {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }, []);

  const dragended = useCallback((event: d3.D3DragEvent<SVGGElement, WordNode, WordNode>, simulation: d3.Simulation<WordNode, undefined>) => {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }, []);

  useEffect(() => {
    if (selectedWord && svgRef.current) {
      const svg = d3.select(svgRef.current);
      const width = 600;
      const height = 400;

      svg.selectAll("*").remove();

      const nodes: WordNode[] = [{ id: selectedWord, group: 1 }];
      const links: WordLink[] = [];

      mockWordData[selectedWord].relatedWords.forEach(relatedWord => {
        nodes.push({ id: relatedWord, group: 2 });
        links.push({ source: selectedWord, target: relatedWord });
      });

      const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id((d: any) => d.id))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));

      const link = svg.append("g")
        .selectAll("line")
        .data(links)
        .enter().append("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", 2);

      const node = svg.append("g")
        .selectAll("g")
        .data(nodes)
        .enter().append("g")
        .call(d3.drag<SVGGElement, WordNode>()
          .on("start", event => dragstarted(event, simulation))
          .on("drag", dragged)
          .on("end", event => dragended(event, simulation)));

      node.append("circle")
        .attr("r", 5)
        .attr("fill", (d) => d.group === 1 ? "#ff4f00" : "#4a4a4a");

      node.append("text")
        .text((d) => d.id)
        .attr("x", 8)
        .attr("y", "0.31em")
        .style("font-family", "Inter, sans-serif")
        .style("font-size", "12px")
        .style("fill", "#4a4a4a");

      node.on("click", (event: MouseEvent, d: WordNode) => {
        onWordSelect(d.id);
      });

      simulation.on("tick", () => {
        link
          .attr("x1", (d: any) => d.source.x)
          .attr("y1", (d: any) => d.source.y)
          .attr("x2", (d: any) => d.target.x)
          .attr("y2", (d: any) => d.target.y);

        node
          .attr("transform", (d: any) => `translate(${d.x},${d.y})`);
      });
    }
  }, [selectedWord, onWordSelect, dragstarted, dragged, dragended]);

  return (
    <div className="graph-container">
      <svg ref={svgRef} width="600" height="400"></svg>
    </div>
  );
};

export default WordGraph;