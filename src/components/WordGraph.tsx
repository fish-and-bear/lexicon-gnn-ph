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
  const [isLoading, setIsLoading] = useState(true);

  const getNodeRelation = useCallback((word: string, info: NetworkWordInfo): string => {
    if (word === mainWord) return "main";
    if (info.derivatives?.includes(mainWord)) return "root";
    if (wordNetwork[mainWord]?.derivatives?.includes(word)) return "derivative";
    if (info.etymology?.parsed?.includes(mainWord) || wordNetwork[mainWord]?.etymology?.parsed?.includes(word)) return "etymology";
    if (info.associated_words?.includes(mainWord) || wordNetwork[mainWord]?.associated_words?.includes(word)) return "associated";
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
    return Object.entries(wordNetwork).map(([word, info]) => ({
      id: word,
      group: getNodeRelation(word, info),
      info: info,
    }));
  }, [wordNetwork, getNodeRelation]);

  const links: CustomLink[] = useMemo(() => {
    const tempLinks: CustomLink[] = [];
    const addLink = (source: string, target: string, type: string) => {
      if (wordNetwork[target]) {
        tempLinks.push({ source, target, type });
      }
    };

    nodes.forEach((node) => {
      const { id: word, info } = node;
      info.derivatives?.forEach((derivative) =>
        addLink(word, derivative, "derivative")
      );
      if (info.root_word) addLink(word, info.root_word, "root");
      info.etymology?.parsed?.forEach((etymWord) =>
        addLink(word, etymWord, "etymology")
      );
      info.associated_words?.forEach((assocWord) =>
        addLink(word, assocWord, "associated")
      );
    });

    return tempLinks;
  }, [nodes, wordNetwork]);

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
        .filter(word => !visited.has(word))
        .slice(0, breadth);

      relatedWords.forEach(word => {
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

    setIsLoading(false);

    return () => {
      simulation.stop();
    };

    function setupSvgDimensions(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
      const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
      const width = containerRect ? containerRect.width : 800;
      const height = containerRect ? containerRect.height : 600;
      svg.attr("width", width).attr("height", height);
      return { width, height };
    }

    function setupZoom(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, g: d3.Selection<SVGGElement, unknown, null, undefined>) {
      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
          g.attr("transform", event.transform.toString());
        });

      svg.call(zoom);
      return zoom;
    }

    function setupSimulation(width: number, height: number) {
      const validLinks = links.filter(link => 
        filteredNodes.some(node => node.id === (typeof link.source === 'string' ? link.source : link.source.id)) &&
        filteredNodes.some(node => node.id === (typeof link.target === 'string' ? link.target : link.target.id))
      );

      return d3
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
        )
        .on("tick", () => {
          link
            .attr("x1", (d) => (d.source as CustomNode).x!)
            .attr("y1", (d) => (d.source as CustomNode).y!)
            .attr("x2", (d) => (d.target as CustomNode).x!)
            .attr("y2", (d) => (d.target as CustomNode).y!);

          node.attr("transform", (d) => `translate(${d.x!},${d.y!})`);
        });
    }

    function createLinks(g: d3.Selection<SVGGElement, unknown, null, undefined>) {
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
    }

    function createNodes(g: d3.Selection<SVGGElement, unknown, null, undefined>) {
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
    }

    function setupNodeInteractions(node: d3.Selection<SVGGElement, CustomNode, SVGGElement, unknown>, svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
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
    }
  }, [filteredNodes, links, theme, onNodeClick, selectedNodeId, getNodeColor, depth, breadth]);

  useEffect(() => {
    const cleanupFunction = updateGraph();
    window.addEventListener("resize", updateGraph);
    return () => {
      if (typeof cleanupFunction === "function") {
        cleanupFunction();
      }
      window.removeEventListener("resize", updateGraph);
    };
  }, [updateGraph, wordNetwork, mainWord, depth, breadth]);

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

  return (
    <div className="graph-container">
      <div className="graph-svg-container">
        {isLoading && <div className="loading-overlay">Loading...</div>}
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
          <p>{hoveredNode.info.definition || "No definition available"}</p>
          <p>Relation: {hoveredNode.group}</p>
        </div>
      )}
    </div>
  );
};

export default WordGraph;
