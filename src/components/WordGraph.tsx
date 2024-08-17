import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetwork, WordInfo } from "../types";
import { useTheme } from "../contexts/ThemeContext";

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

const WordGraph: React.FC<WordGraphProps> = ({
  wordNetwork,
  mainWord,
  onNodeClick,
}) => {
  const { theme } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const nodes: CustomNode[] = useMemo(() => {
    return Object.entries(wordNetwork).map(([word, info]) => ({
      id: word,
      group: word === mainWord ? ["main"] : ["related"],
      info: info,
    }));
  }, [wordNetwork, mainWord]);

  const links: CustomLink[] = useMemo(() => {
    const tempLinks: CustomLink[] = [];
    Object.entries(wordNetwork).forEach(([word, info]) => {
      Object.keys(info.derivatives || {}).forEach((derivative) => {
        if (wordNetwork[derivative]) {
          tempLinks.push({
            source: word,
            target: derivative,
            type: "derivative",
          });
        }
      });
      (info.associated_words || []).forEach((associatedWord) => {
        if (wordNetwork[associatedWord]) {
          tempLinks.push({
            source: word,
            target: associatedWord,
            type: "associated",
          });
        }
      });
      (info.root_words || []).forEach((rootWord) => {
        if (wordNetwork[rootWord]) {
          tempLinks.push({ source: word, target: rootWord, type: "etymology" });
        }
      });
      if (info.root_word && wordNetwork[info.root_word]) {
        tempLinks.push({ source: word, target: info.root_word, type: "root" });
      }
    });
    return tempLinks;
  }, [wordNetwork]);

  const updateGraph = useCallback(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear existing graph

    const containerRect = svgRef.current.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;

    svg.attr("width", width).attr("height", height);

    const g = svg.append("g");

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink(links)
          .id((d: any) => d.id)
          .distance(120)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius(60).strength(0.7))
      .force("x", d3.forceX(width / 2).strength(0.1))
      .force("y", d3.forceY(height / 2).strength(0.1))

    const link = g
      .selectAll(".link")
      .data(links)
      .join("line")
      .attr("class", "link")
      .attr("stroke", (d) => getRelationColor(d.type));

    const node = g
      .selectAll(".node")
      .data(nodes)
      .join("g")
      .attr("class", (d) => `node ${d.id === selectedNodeId ? "selected" : ""}`)
      .call(
        d3
          .drag<SVGGElement, CustomNode>()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended) as any
      );

    node
      .append("circle")
      .attr("r", (d) => (d.group.includes("main") ? 30 : 25))
      .attr("fill", (d) => getNodeColor(d.group));

    node
      .append("text")
      .text((d) => d.id)
      .attr("dy", 40)
      .attr("text-anchor", "middle")
      .attr("fill", theme === "dark" ? "#e0e0e0" : "#333");

    node
      .on("click", (event, d) => {
        setSelectedNodeId(d.id);
        onNodeClick(d.info);
      })
      .on("mouseover", (event, d) => setHoveredNode(d))
      .on("mouseout", () => setHoveredNode(null));

    function ticked() {
      link
        .attr("x1", (d) =>
          Math.max(50, Math.min(width - 50, (d.source as CustomNode).x!))
        )
        .attr("y1", (d) =>
          Math.max(50, Math.min(height - 50, (d.source as CustomNode).y!))
        )
        .attr("x2", (d) =>
          Math.max(50, Math.min(width - 50, (d.target as CustomNode).x!))
        )
        .attr("y2", (d) =>
          Math.max(50, Math.min(height - 50, (d.target as CustomNode).y!))
        );

      node.attr(
        "transform",
        (d) =>
          `translate(${Math.max(50, Math.min(width - 50, d.x!))},${Math.max(
            50,
            Math.min(height - 50, d.y!)
          )})`
      );
    }

    simulation.nodes(nodes).on("tick", ticked);
    simulation
      .force<d3.ForceLink<CustomNode, CustomLink>>("link")
      ?.links(links);

    function dragstarted(
      event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>
    ) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(
      event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>
    ) {
      const x = Math.max(50, Math.min(width - 50, event.x));
      const y = Math.max(50, Math.min(height - 50, event.y));

      // Add playful movement
      const dx = x - event.subject.x!;
      const dy = y - event.subject.y!;
      const angle = Math.atan2(dy, dx);
      const distance = Math.sqrt(dx * dx + dy * dy);
      const springForce = 0.1;

      event.subject.vx = Math.cos(angle) * distance * springForce;
      event.subject.vy = Math.sin(angle) * distance * springForce;

      event.subject.fx = x;
      event.subject.fy = y;
    }

    function dragended(
      event: d3.D3DragEvent<SVGGElement, CustomNode, CustomNode>
    ) {
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
    window.addEventListener("resize", updateGraph);
    return () => {
      if (typeof cleanupFunction === "function") {
        cleanupFunction();
      }
      window.removeEventListener("resize", updateGraph);
    };
  }, [updateGraph, wordNetwork]);

  function getNodeColor(group: string[]): string {
    if (group.includes("main")) return "var(--color-main)";
    if (group.includes("derivative")) return "var(--color-derivative)";
    if (group.includes("etymology")) return "var(--color-etymology)";
    if (group.includes("root")) return "var(--color-root)";
    return "var(--color-associated)";
  }

  function getRelationColor(type: string): string {
    switch (type) {
      case "derivative":
        return "var(--color-derivative)";
      case "etymology":
        return "var(--color-etymology)";
      case "root":
        return "var(--color-root)";
      default:
        return "var(--color-associated)";
    }
  }

  return (
    <div className={`graph-container ${theme}`}>
      <svg ref={svgRef}></svg>
      {hoveredNode && (
        <div
          className="node-tooltip"
          style={{ left: hoveredNode.x, top: hoveredNode.y }}
        >
          <h3>{hoveredNode.id}</h3>
          <p>
            {hoveredNode.info?.definitions &&
            hoveredNode.info.definitions.length > 0
              ? hoveredNode.info.definitions[0].meanings[0]
              : "No definition available"}
          </p>
          <p>Relations: {hoveredNode.group.join(", ")}</p>
        </div>
      )}
      <div className="legend">
        <div className="legend-item">
          <div className="color-box main"></div>
          <span>Main Word</span>
        </div>
        <div className="legend-item">
          <div className="color-box derivative"></div>
          <span>Derivative</span>
        </div>
        <div className="legend-item">
          <div className="color-box etymology"></div>
          <span>Etymology</span>
        </div>
        <div className="legend-item">
          <div className="color-box root"></div>
          <span>Root</span>
        </div>
        <div className="legend-item">
          <div className="color-box associated"></div>
          <span>Associated Word</span>
        </div>
      </div>
    </div>
  );
};

export default React.memo(WordGraph);
