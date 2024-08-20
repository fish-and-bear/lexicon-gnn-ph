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
import Box from "@mui/material/Box";

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

const WordGraph: React.FC<WordGraphProps> = React.memo(
  ({
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

    const getNodeRelation = useCallback(
      (word: string, info: NetworkWordInfo): string => {
        if (word === mainWord) return "main";
        if (info.derivatives?.includes(mainWord)) return "root";
        if (wordNetwork[mainWord]?.derivatives?.includes(word))
          return "derivative";
        if (
          info.etymology?.parsed?.includes(mainWord) ||
          wordNetwork[mainWord]?.etymology?.parsed?.includes(word)
        )
          return "etymology";
        if (
          info.associated_words?.includes(mainWord) ||
          wordNetwork[mainWord]?.associated_words?.includes(word)
        )
          return "associated";
        return "other";
      },
      [wordNetwork, mainWord]
    );

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
      links.forEach((link) => {
        connectedNodes.add(
          typeof link.source === "string" ? link.source : link.source.id
        );
        connectedNodes.add(
          typeof link.target === "string" ? link.target : link.target.id
        );
      });
      return nodes.filter((node) => connectedNodes.has(node.id));
    }, [nodes, links, mainWord]);

    const updateGraph = useCallback(() => {
      if (!svgRef.current) return;

      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const containerRect =
        svgRef.current.parentElement?.getBoundingClientRect();
      const width = containerRect ? containerRect.width : 800;
      const height = containerRect ? containerRect.height : 600;

      svg.attr("width", width).attr("height", height);

      const g = svg.append("g");

      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
          g.attr("transform", event.transform.toString());
        });

      svg.call(zoom);
      zoomRef.current = zoom;

      const simulation = d3
        .forceSimulation<CustomNode>(filteredNodes)
        .force(
          "link",
          d3
            .forceLink<CustomNode, CustomLink>(links)
            .id((d) => d.id)
            .distance(100)
        )
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force(
          "collide",
          d3.forceCollide<CustomNode>().radius(30).strength(0.7)
        );

      const link = g
        .selectAll(".link")
        .data(links)
        .join("line")
        .attr("class", "link")
        .attr("stroke", (d: CustomLink) =>
          getNodeColor((d.target as CustomNode).group)
        )
        .attr("stroke-width", 1);

      const borderColor = getComputedStyle(document.documentElement)
        .getPropertyValue("--selected-node-border-color")
        .trim();

      const node = g
        .selectAll(".node")
        .data(filteredNodes)
        .join("g")
        .attr(
          "class",
          (d: CustomNode) => `node ${d.id === selectedNodeId ? "selected" : ""}`
        )
        .call(
          d3
            .drag<SVGGElement, CustomNode>()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended) as any
        );

      node
        .append("circle")
        .attr("r", (d: CustomNode) => (d.group === "main" ? 25 : 15))
        .attr("fill", (d: CustomNode) => getNodeColor(d.group))

      node
        .append("text")
        .text((d: CustomNode) => d.id)
        .attr("dy", 30)
        .attr("text-anchor", "middle")
        .attr("fill", theme === "dark" ? "#e0e0e0" : "#333")
        .style("font-size", "10px")
        .style("font-weight", "bold");

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

      function ticked() {
        link
          .attr("x1", (d) => (d.source as CustomNode).x!)
          .attr("y1", (d) => (d.source as CustomNode).y!)
          .attr("x2", (d) => (d.target as CustomNode).x!)
          .attr("y2", (d) => (d.target as CustomNode).y!);

        node.attr("transform", (d: CustomNode) => `translate(${d.x!},${d.y!})`);
      }

      simulation.nodes(filteredNodes).on("tick", ticked);
      simulation
        .force<d3.ForceLink<CustomNode, CustomLink>>("link")
        ?.links(links);

      return () => {
        simulation.stop();
      };
    }, [
      filteredNodes,
      links,
      theme,
      onNodeClick,
      selectedNodeId,
      getNodeColor,
    ]);

    useEffect(() => {
      const cleanupFunction = updateGraph();
      window.addEventListener("resize", updateGraph);
      return () => {
        if (typeof cleanupFunction === "function") {
          cleanupFunction();
        }
        window.removeEventListener("resize", updateGraph);
      };
    }, [updateGraph]);

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

    const handleDepthChange = (event: Event, newValue: number | number[]) => {
      const newDepth = Array.isArray(newValue) ? newValue[0] : newValue;
      setDepth(newDepth);
      onNetworkChange(newDepth, breadth);
    };

    const handleBreadthChange = (event: Event, newValue: number | number[]) => {
      const newBreadth = Array.isArray(newValue) ? newValue[0] : newValue;
      setBreadth(newBreadth);
      onNetworkChange(depth, newBreadth);
    };

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
          <svg ref={svgRef}></svg>
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
  }
);

function dragstarted(
  event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>
) {
  if (!event.active) event.subject.fx = event.subject.x;
  if (!event.active) event.subject.fy = event.subject.y;
}

function dragged(event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) {
  event.subject.fx = event.x;
  event.subject.fy = event.y;
}

function dragended(event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>) {
  if (!event.active) event.subject.fx = null;
  if (!event.active) event.subject.fy = null;
}

export default WordGraph;
