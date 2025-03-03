import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetwork, NetworkWordInfo } from "../types";
import { useTheme } from "../contexts/ThemeContext";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import axios from 'axios';

interface WordGraphProps {
  wordNetwork: WordNetwork;
  mainWord: string;
  onNodeClick: (word: string) => void;
  onNetworkChange: (depth: number, breadth: number) => void;
  initialDepth: number;
  initialBreadth: number;
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

const WordGraph: React.FC<WordGraphProps> = ({
  wordNetwork,
  mainWord,
  onNodeClick,
  onNetworkChange,
  initialDepth,
  initialBreadth,
}) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(
    null
  );
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });

  const getNodeRelation = useCallback((word: string, info: NetworkWordInfo): string => {
    if (word === mainWord) return "main";
    
    // Check if this word is a root word for the main word
    if (info.relations.derived.some(rel => rel.word === mainWord)) return "root";
    
    // Check if this word is derived from the main word
    const mainWordNode = wordNetwork.nodes[mainWord];
    if (mainWordNode && mainWordNode.relations.derived.some(rel => rel.word === word)) return "derivative";
    
    // Check if this word is related to the main word
    if (info.relations.related.some(rel => rel.word === mainWord) || 
        (mainWordNode && mainWordNode.relations.related.some(rel => rel.word === word))) {
      return "associated";
    }
    
    // Check if this word is a synonym of the main word
    if (info.relations.synonyms.some(rel => rel.word === mainWord) || 
        (mainWordNode && mainWordNode.relations.synonyms.some(rel => rel.word === word))) {
      return "associated";
    }
    
    return "other";
  }, [wordNetwork, mainWord]);

  const getNodeColor = useCallback((group: string): string => {
    switch (group) {
      case "main":
        return "var(--color-main)";
      case "root":
        return "var(--color-root)";
      case "associated":
        return "var(--color-associated)";
      case "etymology":
        return "var(--color-etymology)";
      case "derivative":
        return "var(--color-derivative)";
      default:
        return "var(--color-default)";
    }
  }, []);

  const nodes: CustomNode[] = useMemo(() => {
    const nodeMap = new Map<string, CustomNode>();

    const addNode = (word: string, info: NetworkWordInfo) => {
      if (!nodeMap.has(word)) {
        nodeMap.set(word, {
          id: word,
          group: getNodeRelation(word, info),
          info: info,
        });
      }
    };

    // Add nodes from the network
    Object.entries(wordNetwork.nodes).forEach(([word, info]) => {
      addNode(word, info);
      
      // Add nodes from relations
      info.relations.related.forEach(rel => {
        if (wordNetwork.nodes[rel.word]) {
          addNode(rel.word, wordNetwork.nodes[rel.word]);
        }
      });
      
      info.relations.synonyms.forEach(rel => {
        if (wordNetwork.nodes[rel.word]) {
          addNode(rel.word, wordNetwork.nodes[rel.word]);
        }
      });
      
      info.relations.derived.forEach(rel => {
        if (wordNetwork.nodes[rel.word]) {
          addNode(rel.word, wordNetwork.nodes[rel.word]);
        }
      });
    });

    return Array.from(nodeMap.values());
  }, [wordNetwork, getNodeRelation]);

  const links: CustomLink[] = useMemo(() => {
    const linkSet = new Set<string>();
    const result: CustomLink[] = [];

    Object.entries(wordNetwork.nodes).forEach(([word, info]) => {
      // Add links for related words
      info.relations.related.forEach(rel => {
        const linkKey = [word, rel.word].sort().join('-');
        if (!linkSet.has(linkKey) && wordNetwork.nodes[rel.word]) {
          linkSet.add(linkKey);
          result.push({
            source: word,
            target: rel.word,
            type: 'related'
          });
        }
      });

      // Add links for synonyms
      info.relations.synonyms.forEach(rel => {
        const linkKey = [word, rel.word].sort().join('-');
        if (!linkSet.has(linkKey) && wordNetwork.nodes[rel.word]) {
          linkSet.add(linkKey);
          result.push({
            source: word,
            target: rel.word,
            type: 'synonym'
          });
        }
      });

      // Add links for derived words
      info.relations.derived.forEach(rel => {
        const linkKey = [word, rel.word].sort().join('-');
        if (!linkSet.has(linkKey) && wordNetwork.nodes[rel.word]) {
          linkSet.add(linkKey);
          result.push({
            source: word,
            target: rel.word,
            type: 'derived'
          });
        }
      });
    });

    return result;
  }, [wordNetwork]);

  const filteredNodes = useMemo(() => {
    const connectedNodes = new Set<string>([mainWord]);
    const queue: [string, number][] = [[mainWord, 0]];
    const visited = new Set<string>();

    while (queue.length > 0) {
      const [currentWord, currentDepth] = queue.shift()!;
      if (currentDepth >= depth) break;
      visited.add(currentWord);

      const relatedWords = links
        .filter(link => 
          (typeof link.source === 'string' ? link.source : link.source.id) === currentWord ||
          (typeof link.target === 'string' ? link.target : link.target.id) === currentWord
        )
        .map(link => {
          const otherWord = (typeof link.source === 'string' ? link.source : link.source.id) === currentWord
            ? (typeof link.target === 'string' ? link.target : link.target.id)
            : (typeof link.source === 'string' ? link.source : link.source.id);
          return otherWord;
        })
        .filter(word => !visited.has(word));

      const sortedWords = relatedWords.sort((a, b) => {
        const aNode = nodes.find(node => node.id === a);
        const bNode = nodes.find(node => node.id === b);
        const aGroup = aNode ? aNode.group : 'other';
        const bGroup = bNode ? bNode.group : 'other';
        const groupOrder = ['main', 'root', 'derivative', 'etymology', 'associated', 'other'];
        return groupOrder.indexOf(aGroup) - groupOrder.indexOf(bGroup);
      });

      sortedWords.slice(0, breadth).forEach(word => {
        connectedNodes.add(word);
        queue.push([word, currentDepth + 1]);
      });
    }

    return nodes.filter((node) => connectedNodes.has(node.id));
  }, [nodes, links, mainWord, depth, breadth]);

  const dragBehavior = d3.drag<SVGGElement, CustomNode>()
    .on("start", (event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) => {
      if (!event.active) event.subject.fx = event.subject.x;
      if (!event.active) event.subject.fy = event.subject.y;
    })
    .on("drag", (event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) => {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    })
    .on("end", (event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) => {
      if (!event.active) event.subject.fx = null;
      if (!event.active) event.subject.fy = null;
    });

  const updateGraph = useCallback(() => {
    if (!svgRef.current) return;

    setIsLoading(true);
    setError(null);

    try {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const { width, height } = setupSvgDimensions(svg);
      const g = svg.append("g");
      const zoom = setupZoom(svg, g);
      zoomRef.current = zoom;

      const simulation = setupSimulation(width, height);
      const link = createLinks(g);
      const node = createNodes(g);

      setupNodeInteractions(node, svg);

      simulation.nodes(filteredNodes);
      simulation.force<d3.ForceLink<CustomNode, CustomLink>>("link")?.links(links.filter(link => 
        filteredNodes.some(node => node.id === (typeof link.source === 'string' ? link.source : link.source.id)) &&
        filteredNodes.some(node => node.id === (typeof link.target === 'string' ? link.target : link.target.id))
      ));

      simulation.on("tick", () => {
        link
          .attr("x1", (d: CustomLink) => (d.source as CustomNode).x!)
          .attr("y1", (d: CustomLink) => (d.source as CustomNode).y!)
          .attr("x2", (d: CustomLink) => (d.target as CustomNode).x!)
          .attr("y2", (d: CustomLink) => (d.target as CustomNode).y!);

        node.attr("transform", (d: CustomNode) => `translate(${d.x!},${d.y!})`);
      });

      setIsLoading(false);

      return () => {
        simulation.stop();
      };
    } catch (err) {
      console.error("Error updating graph:", err);
      setError("An error occurred while updating the graph. Please try again.");
      setIsLoading(false);
    }
  }, [filteredNodes, links, theme, onNodeClick, selectedNodeId, getNodeColor, depth, breadth]);

  useEffect(() => {
    if (wordNetwork) {
      // Convert the network data to the format expected by the graph
      const nodes = Object.values(wordNetwork.nodes).map(node => ({
        id: node.id.toString(),
        label: node.word,
        title: node.word,
        group: getNodeGroup(node, wordNetwork.word),
        value: getNodeSize(node, wordNetwork.word),
        shape: getNodeShape(node, wordNetwork.word),
        color: getNodeColor(node, wordNetwork.word)
      }));

      // Create edges from the network data
      const edges = wordNetwork.edges.map(edge => ({
        from: edge.source.toString(),
        to: edge.target.toString(),
        label: edge.type,
        title: edge.type,
        arrows: 'to',
        color: getEdgeColor(edge.type),
        width: getEdgeWidth(edge.type)
      }));

      setGraphData({ nodes, edges });
      setIsLoading(false);
    }
  }, [wordNetwork]);

  const handleZoom = useCallback((scale: number) => {
    if (zoomRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(zoomRef.current.scaleBy, scale);
    }
  }, []);

  const handleResetZoom = useCallback(() => {
    if (zoomRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg
        .transition()
        .duration(300)
        .call(zoomRef.current.transform, d3.zoomIdentity);
    }
  }, []);

  const handleDepthChange = useCallback((event: Event, newValue: number | number[]) => {
    const newDepth = Array.isArray(newValue) ? newValue[0] : newValue;
    setDepth(newDepth);
    onNetworkChange(newDepth, breadth);
  }, [onNetworkChange, breadth]);

  const handleBreadthChange = useCallback((event: Event, newValue: number | number[]) => {
    const newBreadth = Array.isArray(newValue) ? newValue[0] : newValue;
    setBreadth(newBreadth);
    onNetworkChange(depth, newBreadth);
  }, [onNetworkChange, depth]);

  const legendItems = useMemo(
    () => [
      { key: "main", label: "Main Word" },
      { key: "derivative", label: "Derivative" },
      { key: "etymology", label: "Etymology" },
      { key: "root", label: "Root" },
      { key: "associated", label: "Associated Word" },
      { key: "other", label: "Other" },
    ],
    []
  );

  const setupSvgDimensions = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    svg.attr("width", width).attr("height", height);
    return { width, height };
  };

  const setupZoom = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        g.attr("transform", event.transform.toString());
      });

    svg.call(zoom);
    return zoom;
  };

  const setupSimulation = (width: number, height: number) => {
    const validLinks = links.filter(link => 
      filteredNodes.some(node => node.id === (typeof link.source === 'string' ? link.source : link.source.id)) &&
      filteredNodes.some(node => node.id === (typeof link.target === 'string' ? link.target : link.target.id))
    );

    const simulation = d3
      .forceSimulation<CustomNode>(filteredNodes)
      .force(
        "link",
        d3.forceLink<CustomNode, CustomLink>(validLinks).id((d) => d.id).distance(100)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force(
        "collide",
        d3.forceCollide<CustomNode>().radius(30).strength(0.7)
      );

    return simulation;
  };

  const createLinks = (g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const validLinks = links.filter(link => 
      filteredNodes.some(node => node.id === (typeof link.source === 'string' ? link.source : link.source.id)) &&
      filteredNodes.some(node => node.id === (typeof link.target === 'string' ? link.target : link.target.id))
    );

    return g
      .selectAll(".link")
      .data(validLinks)
      .join("line")
      .attr("class", "link")
      .attr("stroke", (d: CustomLink) =>
        getNodeColor((d.target as CustomNode).group)
      )
      .attr("stroke-width", 1);
  };

  const createNodes = (g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const node = g
      .selectAll<SVGGElement, CustomNode>(".node")
      .data(filteredNodes)
      .join("g")
      .attr(
        "class",
        (d: CustomNode) => `node ${d.id === selectedNodeId ? "selected" : ""}`
      )
      .call(dragBehavior as any);

    node
      .append("circle")
      .attr("r", (d: CustomNode) => (d.group === "main" ? 25 : 15))
      .attr("fill", (d: CustomNode) => getNodeColor(d.group));

    node
      .append("text")
      .text((d: CustomNode) => d.id)
      .attr("dy", 30)
      .attr("text-anchor", "middle")
      .attr("fill", theme === "dark" ? "#e0e0e0" : "#333")
      .style("font-size", "10px")
      .style("font-weight", "bold");

    return node;
  };

  const setupNodeInteractions = (node: d3.Selection<SVGGElement, CustomNode, SVGGElement, unknown>, svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    node
      .on("click", (event: MouseEvent, d: CustomNode) => {
        setSelectedNodeId(d.id);
        onNodeClick(d.id);
      })
      .on("mouseover", (event: MouseEvent, d: CustomNode) => {
        const [x, y] = d3.pointer(event, svg.node());
        setHoveredNode({ ...d, x, y });
      })
      .on("mouseout", () => setHoveredNode(null));
  };

  const handleNodeClick = useCallback((node: CustomNode) => {
    if (node.info.word !== mainWord) {
      onNodeClick(node.info.word);
    }
  }, [mainWord, onNodeClick]);

  const renderTooltip = useCallback(() => {
    if (!hoveredNode || typeof hoveredNode.x === 'undefined' || typeof hoveredNode.y === 'undefined') {
      return null;
    }
    
    return (
      <div
        className="tooltip"
        style={{
          position: "absolute",
          left: `${hoveredNode.x + 10}px`,
          top: `${hoveredNode.y + 10}px`,
        }}
      >
        <h3>{hoveredNode.info.word}</h3>
        {hoveredNode.info.definitions.map((def, index) => (
          <p key={index}>{def}</p>
        ))}
      </div>
    );
  }, [hoveredNode]);

  const getNodeGroup = (node: NetworkNode, mainWord: string): string => {
    if (node.word === mainWord) return 'main';
    
    // Check if node is in any cluster
    for (const [clusterName, nodeIds] of Object.entries(wordNetwork?.clusters || {})) {
      if (nodeIds.includes(node.id)) {
        return clusterName;
      }
    }
    
    return 'default';
  };

  const getNodeSize = (node: NetworkNode, mainWord: string): number => {
    if (node.word === mainWord) return 30;
    
    // Base size on node type
    if (node.type === 'root') return 25;
    if (node.type === 'related') return 20;
    
    return 15;
  };

  const getNodeShape = (node: NetworkNode, mainWord: string): string => {
    if (node.word === mainWord) return 'circle';
    
    // Shape based on node type
    if (node.type === 'root') return 'diamond';
    if (node.has_baybayin) return 'star';
    
    return 'dot';
  };

  const getNodeColor = (node: NetworkNode, mainWord: string): any => {
    if (node.word === mainWord) {
      return { background: '#ff9900', border: '#ff6600', highlight: { background: '#ffcc00', border: '#ff9900' } };
    }
    
    // Color based on language
    if (node.language !== 'tl') {
      return { background: '#9999ff', border: '#6666cc', highlight: { background: '#ccccff', border: '#9999ff' } };
    }
    
    // Default color
    return { background: '#66cc66', border: '#339933', highlight: { background: '#99ff99', border: '#66cc66' } };
  };

  const getEdgeColor = (type: string): string => {
    switch (type) {
      case 'synonym': return '#66cc66';
      case 'antonym': return '#cc6666';
      case 'derived': return '#9966cc';
      case 'etymology': return '#66cccc';
      case 'affix_prefix': return '#cc9966';
      case 'affix_suffix': return '#cc66cc';
      case 'affix_infix': return '#66ccff';
      default: return '#999999';
    }
  };

  const getEdgeWidth = (type: string): number => {
    switch (type) {
      case 'synonym': return 3;
      case 'antonym': return 3;
      case 'derived': return 2;
      case 'etymology': return 2;
      default: return 1;
    }
  };

  return (
    <div className="graph-container">
      <div className="graph-svg-container">
        {isLoading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Loading...</p>
          </div>
        )}
        {error && (
          <div className="error-overlay">
            <p className="error-message">{error}</p>
            <button onClick={updateGraph} className="retry-button">Retry</button>
          </div>
        )}
        <svg ref={svgRef} className={`graph-svg ${isLoading ? 'loading' : 'loaded'}`}>
          {/* SVG content will be rendered here by D3 */}
        </svg>
      </div>
      <div className="controls-container">
        <div className="zoom-controls">
          <button onClick={() => handleZoom(1.2)} className="zoom-button">
            +
          </button>
          <button onClick={() => handleZoom(1 / 1.2)} className="zoom-button">
            -
          </button>
          <button onClick={handleResetZoom} className="zoom-button">
            Reset
          </button>
        </div>
        <div className="graph-controls">
          <div className="slider-container">
            <Typography variant="caption">Depth: {depth}</Typography>
            <Slider
              value={depth}
              onChange={handleDepthChange}
              aria-labelledby="depth-slider"
              step={1}
              marks
              min={1}
              max={5}
              size="small"
            />
          </div>
          <div className="slider-container">
            <Typography variant="caption">Breadth: {breadth}</Typography>
            <Slider
              value={breadth}
              onChange={handleBreadthChange}
              aria-labelledby="breadth-slider"
              step={1}
              marks
              min={5}
              max={20}
              size="small"
            />
          </div>
        </div>
      </div>
      <div className={`legend ${theme}`}>
        {legendItems.map((item) => (
          <div key={item.key} className="legend-item">
            <div
              className="color-box"
              style={{ backgroundColor: getNodeColor(item.key) }}
            ></div>
            <span>{item.label}</span>
          </div>
        ))}
      </div>
      {hoveredNode && (
        <div
          className="node-tooltip"
          style={{ left: hoveredNode.x, top: hoveredNode.y }}
        >
          <h3 style={{ color: getNodeColor(hoveredNode.group) }}>
            {hoveredNode.id}
          </h3>
          <p>{hoveredNode.info.definitions[0] || "No definition available"}</p>
          <p>Relation: {hoveredNode.group}</p>
        </div>
      )}
    </div>
  );
};

export default React.memo(WordGraph);
