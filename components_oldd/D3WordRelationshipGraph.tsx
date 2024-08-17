import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

interface WordNode extends d3.SimulationNodeDatum {
  id: string;
  group: number;
  label: string;
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

const D3WordRelationshipGraph: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState<WordNode | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const handleSearch = () => {
    // Simulated API call to get word data
    const mockWordData: GraphData = {
      nodes: [
        { id: 'rely', group: 1, label: 'rely', definition: 'To depend on with full trust or confidence' },
        { id: 'reliable', group: 2, label: 'reliable', definition: 'Consistently good in quality or performance; able to be trusted' },
        { id: 'reliability', group: 2, label: 'reliability', definition: 'The quality of being reliable' },
        { id: 'unreliable', group: 2, label: 'unreliable', definition: 'Not able to be relied on' },
        { id: 'unreliability', group: 2, label: 'unreliability', definition: 'The quality of being unreliable' },
      ],
      links: [
        { source: 'rely', target: 'reliable' },
        { source: 'rely', target: 'reliability' },
        { source: 'rely', target: 'unreliable' },
        { source: 'rely', target: 'unreliability' },
      ]
    };
    setGraphData(mockWordData);
    setSelectedNode(mockWordData.nodes.find(node => node.id === searchTerm) || mockWordData.nodes[0]);
  };

  const dragstarted = (event: d3.D3DragEvent<SVGGElement, WordNode, WordNode>, simulation: d3.Simulation<WordNode, undefined>) => {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  };

  const dragged = (event: d3.D3DragEvent<SVGGElement, WordNode, WordNode>) => {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  };

  const dragended = (event: d3.D3DragEvent<SVGGElement, WordNode, WordNode>, simulation: d3.Simulation<WordNode, undefined>) => {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  };

  useEffect(() => {
    if (graphData.nodes.length > 0 && svgRef.current) {
      const svg = d3.select(svgRef.current);
      const width = 600;
      const height = 400;

      svg.selectAll("*").remove();

      const simulation = d3.forceSimulation<WordNode>(graphData.nodes)
        .force("link", d3.forceLink<WordNode, WordLink>(graphData.links).id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));

      const link = svg.append("g")
        .selectAll<SVGLineElement, WordLink>("line")
        .data(graphData.links)
        .join("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", 2);

      const node = svg.append("g")
        .selectAll<SVGGElement, WordNode>("g")
        .data(graphData.nodes)
        .join("g")
        .call(d3.drag<SVGGElement, WordNode>()
          .on("start", event => dragstarted(event, simulation))
          .on("drag", dragged)
          .on("end", event => dragended(event, simulation)));

      node.append("circle")
        .attr("r", 20)
        .attr("fill", d => d.group === 1 ? "#4CAF50" : "#2196F3");

      node.append("text")
        .text(d => d.label)
        .attr("x", 0)
        .attr("y", 30)
        .attr("text-anchor", "middle");

      node.on("click", (event: MouseEvent, d: WordNode) => {
        setSelectedNode(d);
        d3.selectAll("circle").attr("stroke", null);
        d3.select(event.currentTarget as SVGGElement).select("circle").attr("stroke", "#ff0000").attr("stroke-width", 2);
      });

      simulation.on("tick", () => {
        link
          .attr("x1", d => (d.source as WordNode).x!)
          .attr("y1", d => (d.source as WordNode).y!)
          .attr("x2", d => (d.target as WordNode).x!)
          .attr("y2", d => (d.target as WordNode).y!);

        node
          .attr("transform", d => `translate(${d.x},${d.y})`);
      });
    }
  }, [graphData]);

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-3xl font-bold mb-6 text-center">Interactive Word Relationship Explorer</h1>
      
      <div className="flex mb-6">
        <input
          type="text"
          placeholder="Enter a word..."
          value={searchTerm}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
          className="flex-grow mr-2 p-2 border rounded"
        />
        <button onClick={handleSearch} className="px-4 py-2 bg-blue-500 text-white rounded">
          Explore
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2">
          <div className="border rounded p-4">
            <h2 className="text-xl font-bold mb-2">Interactive Word Relationships</h2>
            <svg ref={svgRef} width="600" height="400" viewBox="0 0 600 400"></svg>
          </div>
        </div>
        
        <div className="border rounded p-4">
          <h2 className="text-xl font-bold mb-2">Word Details</h2>
          {selectedNode ? (
            <>
              <h3 className="text-lg font-semibold mb-2">{selectedNode.label}</h3>
              <p className="mb-4">{selectedNode.definition}</p>
              <p><strong>Related words:</strong></p>
              <ul className="list-disc pl-5">
                {graphData.links
                  .filter(link => 
                    (typeof link.source === 'object' && link.source.id === selectedNode.id) || 
                    (typeof link.target === 'object' && link.target.id === selectedNode.id) ||
                    link.source === selectedNode.id ||
                    link.target === selectedNode.id
                  )
                  .map(link => {
                    const relatedWordId = 
                      (typeof link.source === 'object' && link.source.id === selectedNode.id) ? 
                        (typeof link.target === 'object' ? link.target.id : link.target) :
                        (typeof link.source === 'object' ? link.source.id : link.source);
                    return <li key={relatedWordId}>{relatedWordId}</li>;
                  })}
              </ul>
            </>
          ) : (
            <p>Select a word from the graph to see details.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default D3WordRelationshipGraph;