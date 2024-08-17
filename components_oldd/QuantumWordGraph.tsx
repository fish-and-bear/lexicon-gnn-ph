import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import './QuantumWordGraph.css';

interface WordNode extends d3.SimulationNodeDatum {
  id: string;
  isRoot: boolean;
  definition: string;
}

interface WordLink extends d3.SimulationLinkDatum<WordNode> {
  source: string | WordNode;
  target: string | WordNode;
}

interface GraphData {
  nodes: WordNode[];
  links: WordLink[];
}

const QuantumWordGraph: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState<WordNode | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleSearch = () => {
    // Simulated API call to get word data
    const mockWordData: GraphData = {
      nodes: [
        { id: 'quantum', isRoot: true, definition: 'A discrete quantity of energy proportional in magnitude to the frequency of the radiation it represents' },
        { id: 'particle', isRoot: false, definition: 'A minute portion of matter' },
        { id: 'wave', isRoot: false, definition: 'A disturbance that propagates through a medium' },
        { id: 'entanglement', isRoot: false, definition: 'A quantum phenomenon where particles remain interconnected' },
        { id: 'superposition', isRoot: false, definition: 'The ability of a quantum system to be in multiple states at once' },
      ],
      links: [
        { source: 'quantum', target: 'particle' },
        { source: 'quantum', target: 'wave' },
        { source: 'quantum', target: 'entanglement' },
        { source: 'quantum', target: 'superposition' },
      ]
    };
    setGraphData(mockWordData);
    setSelectedNode(mockWordData.nodes.find(node => node.id === searchTerm) || mockWordData.nodes[0]);
  };

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
    if (graphData.nodes.length > 0 && svgRef.current && containerRef.current) {
      const svg = d3.select(svgRef.current);
      const container = containerRef.current;
      const width = container.clientWidth;
      const height = container.clientHeight;

      svg.attr("width", width).attr("height", height);
      svg.selectAll("*").remove();

      const simulation = d3.forceSimulation<WordNode>(graphData.nodes)
        .force("link", d3.forceLink<WordNode, WordLink>(graphData.links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-500))
        .force("center", d3.forceCenter(width / 2, height / 2));

      const link = svg.append("g")
        .selectAll<SVGLineElement, WordLink>("line")
        .data(graphData.links)
        .join("line")
        .attr("class", "link");

      const node = svg.append("g")
        .selectAll<SVGGElement, WordNode>("g")
        .data(graphData.nodes)
        .join("g")
        .attr("class", d => `node ${d.isRoot ? 'root' : 'derivative'}`)
        .call(d3.drag<SVGGElement, WordNode>()
          .on("start", event => dragstarted(event, simulation))
          .on("drag", dragged)
          .on("end", event => dragended(event, simulation)));

      node.append("text")
        .text(d => d.id)
        .attr("class", "node-text")
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central");

      node.on("mouseover", function(event, d) {
        d3.select(this).classed("hover", true);
        setSelectedNode(d);
      })
      .on("mouseout", function() {
        d3.select(this).classed("hover", false);
      });

      simulation.on("tick", () => {
        link
          .attr("x1", d => (d.source as WordNode).x!)
          .attr("y1", d => (d.source as WordNode).y!)
          .attr("x2", d => (d.target as WordNode).x!)
          .attr("y2", d => (d.target as WordNode).y!);

        node.attr("transform", d => `translate(${d.x},${d.y})`);
      });
    }
  }, [graphData, dragstarted, dragged, dragended]);

  return (
    <div className="quantum-word-graph">
      <div className="search-container">
        <input
          type="text"
          placeholder="Enter a word..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        <button onClick={handleSearch} className="search-button">Explore</button>
      </div>
      <div className="graph-container" ref={containerRef}>
        <svg ref={svgRef}></svg>
      </div>
      {selectedNode && (
        <div className="word-details">
          <h2>{selectedNode.id}</h2>
          <p>{selectedNode.definition}</p>
        </div>
      )}
    </div>
  );
};

export default QuantumWordGraph;