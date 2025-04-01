import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetwork } from "../types";
import { useTheme } from "../contexts/ThemeContext";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import axios from 'axios';

interface WordGraphProps {
  wordNetwork: WordNetwork | null;
  mainWord: string | null;
  onNodeClick: (word: string) => void;
  onNetworkChange: (depth: number, breadth: number) => void;
  initialDepth: number;
  initialBreadth: number;
}

interface CustomNode extends d3.SimulationNodeDatum {
  id: string;
  word: string;
  label?: string;
  group: string;
  connections?: number;
  pinned?: boolean;
  originalId?: number;
  language?: string;
  definitions?: string[];
  path?: Array<{ type: string; word: string }>;
}

interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
  relationship: string;
  source: string | CustomNode;
  target: string | CustomNode;
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
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(mainWord);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isValidNetwork, setIsValidNetwork] = useState(true);
  const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);

  const isDraggingRef = useRef(false);
  const isTransitioningRef = useRef(false);
  const lastClickTimeRef = useRef(0);
  const prevMainWordRef = useRef<string | null>(null);

  useEffect(() => {
    if (!wordNetwork || !wordNetwork.nodes || !Array.isArray(wordNetwork.nodes) || 
        !wordNetwork.edges || !Array.isArray(wordNetwork.edges)) {
      console.error("Invalid wordNetwork structure:", wordNetwork);
      setIsValidNetwork(false);
    } else {
      setIsValidNetwork(true);
    }
  }, [wordNetwork]);

  useEffect(() => {
    setSelectedNodeId(mainWord);
  }, [mainWord]);

  const mapRelationshipToGroup = useCallback((relationship?: string): string => {
    if (!relationship) return 'associated';
    const relLower = relationship.toLowerCase();
    switch (relLower) {
      case 'main': return 'main';
      case 'synonym': return 'synonym';
      case 'antonym': return 'antonym';
      case 'related': return 'related';
      case 'kaugnay': return 'related';
      case 'variant':
      case 'spelling_variant':
      case 'regional_variant': return 'variant';
      case 'derived':
      case 'derived_from':
      case 'root_of': return 'derived';
      case 'root': return 'root';
      case 'hypernym':
      case 'hyponym': return 'taxonomic';
      case 'meronym':
      case 'holonym': return 'part_whole';
      case 'etymology': return 'etymology';
      case 'component_of': return 'component_of';
      case 'cognate': return 'cognate';
      case 'see_also':
      case 'compare_with': return 'usage';
      default: return 'associated';
    }
  }, []);

  const getNodeColor = useCallback((group: string): string => {
    switch (group) {
      case "main": return "#1d3557";
      case "root": return "#e63946";
      case "derived": return "#2a9d8f";
      case "synonym": return "#457b9d";
      case "antonym": return "#f77f00";
      case "variant": return "#f4a261";
      case "related": return "#fcbf49";
      case "taxonomic": return "#8338ec";
      case "part_whole": return "#3a86ff";
      case "usage": return "#0ead69";
      case "etymology": return "#3d5a80";
      case "component_of": return "#ffb01f";
      case "cognate": return "#9381ff";
      case "associated": return "#adb5bd";
      default: return "#6c757d";
    }
  }, []);

  const baseLinks = useMemo<{ source: string; target: string; relationship: string }[]>(() => {
    if (!wordNetwork?.nodes || !wordNetwork.edges) return [];

    return wordNetwork.edges
      .map(edge => {
        const sourceNode = wordNetwork.nodes.find(n => n.id === edge.source);
        const targetNode = wordNetwork.nodes.find(n => n.id === edge.target);
        
        if (!sourceNode || !targetNode) {
          console.warn(`Could not find nodes for edge: ${edge.source} -> ${edge.target}`);
          return null;
        }
        
        return {
          source: sourceNode.label,
          target: targetNode.label,
          relationship: edge.type
        };
      })
      .filter((link): link is { source: string; target: string; relationship: string; } => link !== null);
  }, [wordNetwork]);

  const baseNodes = useMemo<CustomNode[]>(() => {
    if (!wordNetwork?.nodes || !mainWord) return [];

    return wordNetwork.nodes.map(node => {
      let calculatedGroup = 'associated';
      if (node.label === mainWord) {
        calculatedGroup = 'main';
      } else {
        const connectingLink = baseLinks.find(link =>
          (link.source === mainWord && link.target === node.label) ||
          (link.source === node.label && link.target === mainWord)
        );
        calculatedGroup = mapRelationshipToGroup(connectingLink?.relationship);
      }
      
      return {
        id: node.label,
        word: node.label,
        label: node.label,
        group: calculatedGroup,
        originalId: node.id,
        language: node.language,
        definitions: node.definitions,
        path: node.path,
        has_baybayin: node.has_baybayin,
        baybayin_form: node.baybayin_form,
        index: undefined, x: undefined, y: undefined, vx: undefined, vy: undefined, fx: undefined, fy: undefined
      };
    });
  }, [wordNetwork, mainWord, baseLinks, mapRelationshipToGroup]);

  const filteredNodes = useMemo<CustomNode[]>(() => {
    if (!mainWord || baseNodes.length === 0) {
      return [];
    }
    
    const nodeMap = new Map(baseNodes.map(n => [n.id, n]));
    const connectedNodeIds = new Set<string>([mainWord]);
    const queue: [string, number][] = [[mainWord, 0]];
    const visited = new Set<string>();

    while (queue.length > 0) {
      const [currentWordId, currentDepth] = queue.shift()!;

      if (currentDepth >= depth || visited.has(currentWordId)) continue;
      visited.add(currentWordId);

      const relatedLinks = baseLinks.filter(link =>
        link.source === currentWordId || link.target === currentWordId
      );

      const relatedWordIds = relatedLinks.map(link => {
        return link.source === currentWordId ? link.target : link.source;
      }).filter(id => !visited.has(id));

      const sortedWords = relatedWordIds.sort((aId, bId) => {
         const aNode = nodeMap.get(aId);
         const bNode = nodeMap.get(bId);
         const aGroup = aNode ? aNode.group : 'associated';
         const bGroup = bNode ? bNode.group : 'associated';
         const groupOrder = [
            'main', 'root', 'root_of', 'synonym', 'antonym', 'derived',
            'variant', 'related', 'kaugnay', 'component_of', 'cognate',
            'etymology', 'derivative', 'associated', 'other'
          ];
         return groupOrder.indexOf(aGroup) - groupOrder.indexOf(bGroup);
      });

      sortedWords.slice(0, breadth).forEach(wordId => {
         if (nodeMap.has(wordId)) {
             connectedNodeIds.add(wordId);
             queue.push([wordId, currentDepth + 1]);
         }
      });
    }

    return baseNodes.filter((node) => connectedNodeIds.has(node.id));
  }, [baseNodes, baseLinks, mainWord, depth, breadth]);

  const getNodeRadius = useCallback((node: CustomNode) => {
    if (node.id === mainWord) return 25;
    if (node.group === 'root') return 20;
    if (node.connections && node.connections > 3) return 18;
    return 14;
  }, [mainWord]);

  const setupZoom = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 3])
      .interpolate(d3.interpolateZoom)
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        if (!isDraggingRef.current) {
          g.attr("transform", event.transform.toString());
        }
      })
      .filter(event => !isDraggingRef.current && !isTransitioningRef.current && !event.ctrlKey && !event.button);
    svg.call(zoom);
    const initialTransform = d3.zoomIdentity.translate(width / 2, height / 2);
    svg.transition().duration(150).ease(d3.easeExpOut).call(zoom.transform, initialTransform);
    return zoom;
  }, []);

  const ticked = useCallback(() => {
      if (!svgRef.current) return;
      const svg = d3.select(svgRef.current);
      const nodeSelection = svg.selectAll<SVGGElement, CustomNode>(".node");
      const linkSelection = svg.selectAll<SVGLineElement, CustomLink>(".link");

      nodeSelection.attr("transform", d => `translate(${d.x ?? 0}, ${d.y ?? 0})`);

      linkSelection
        .attr("x1", d => (typeof d.source === 'object' ? d.source.x ?? 0 : 0))
        .attr("y1", d => (typeof d.source === 'object' ? d.source.y ?? 0 : 0))
        .attr("x2", d => (typeof d.target === 'object' ? d.target.x ?? 0 : 0))
        .attr("y2", d => (typeof d.target === 'object' ? d.target.y ?? 0 : 0));
  }, []);

  const setupSimulation = useCallback((nodes: CustomNode[], links: CustomLink[]) => {
      simulationRef.current = d3.forceSimulation<CustomNode>(nodes)
        .alphaDecay(0.015)
        .velocityDecay(0.55)
        .force("link", d3.forceLink<CustomNode, CustomLink>()
          .id(d => d.id)
          .links(links)
          .distance(link => {
            switch (link.relationship) {
              case "synonym": return 80; case "antonym": return 120;
              case "derived": return 100; default: return 90;
            }
          })
          .strength(0.6))
        .force("charge", d3.forceManyBody<CustomNode>().strength(-120).distanceMax(300).theta(0.9))
        .force("collide", d3.forceCollide<CustomNode>().radius(d => getNodeRadius(d) + 2).strength(0.7).iterations(2))
        .force("center", d3.forceCenter(0, 0))
        .force("x", d3.forceX(0).strength(0.05))
        .force("y", d3.forceY(0).strength(0.05))
        .on("tick", ticked);

        return simulationRef.current;
  }, [getNodeRadius, ticked]);

  const createDragBehavior = useCallback((simulation: d3.Simulation<CustomNode, CustomLink>) => {
      return d3.drag<SVGGElement, CustomNode>()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
          isDraggingRef.current = true;
          d3.select(event.sourceEvent.target.closest(".node")).classed("dragging", true);
          d3.selectAll(".link").filter((l: any) => l.source.id === d.id || l.target.id === d.id)
             .classed("connected-link", true)
             .raise();
        })
        .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          if (!d.pinned) { d.fx = null; d.fy = null; }
          isDraggingRef.current = false;
          d3.select(event.sourceEvent.target.closest(".node")).classed("dragging", false);
          d3.selectAll(".link.connected-link").classed("connected-link", false);
        });
  }, []);

  const createLinks = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>, linksData: CustomLink[]) => {
      return g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(linksData)
        .join("line")
        .attr("class", "link")
        .attr("stroke", d => {
             switch (d.relationship.toLowerCase()) {
                 case "synonym": return theme === "dark" ? "#4CAF50" : "#2E7D32";
                 case "antonym": return theme === "dark" ? "#F44336" : "#C62828";
                 case "derived": return theme === "dark" ? "#2196F3" : "#1565C0";
                 case "related": return theme === "dark" ? "#9C27B0" : "#6A1B9A";
                 default: return theme === "dark" ? "#78909C" : "#546E7A";
             }
         })
        .attr("stroke-opacity", 0.3)
        .attr("stroke-width", 1.5);
  }, [theme]);

  const createNodes = useCallback((
      g: d3.Selection<SVGGElement, unknown, null, undefined>,
      nodesData: CustomNode[],
      simulation: d3.Simulation<CustomNode, CustomLink> | null
      ) => {
      const drag = simulation ? createDragBehavior(simulation) : null;
      const nodeGroups = g.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(nodesData)
        .join("g")
        .attr("class", d => `node ${d.id === mainWord ? "main-node" : ""} ${d.group}`);

      if (drag) nodeGroups.call(drag as any);

      nodeGroups.append("circle")
        .attr("r", d => getNodeRadius(d))
        .attr("fill", d => getNodeColor(d.group))
        .attr("stroke", theme === "dark" ? "#DDD" : "#333")
        .attr("stroke-width", d => d.pinned ? 2 : 1)
        .attr("stroke-dasharray", d => d.pinned ? "3,2" : "none");

      nodeGroups.append("text")
        .attr("dy", ".3em")
        .attr("text-anchor", "middle")
        .text(d => d.word)
        .attr("font-size", d => `${Math.min(16, 10 + getNodeRadius(d) / 2)}px`)
        .attr("font-weight", d => d.id === mainWord ? "bold" : "normal")
        .attr("fill", theme === "dark" ? "#FFF" : "#000")
        .style("pointer-events", "none");

      nodeGroups.append("title").text(d => `${d.word}\nGroup: ${d.group}`);
      return nodeGroups;
  }, [createDragBehavior, getNodeRadius, getNodeColor, theme, mainWord]);

  const setupNodeInteractions = useCallback((
      nodeSelection: d3.Selection<d3.BaseType, CustomNode, SVGGElement, unknown>
    ) => {
       nodeSelection
        .on("click", (event, d) => {
          event.stopPropagation();
          if (isDraggingRef.current) return;
          setSelectedNodeId(d.id);

          d3.selectAll(".node").classed("selected", false).select("circle").attr("r", (n: unknown) => getNodeRadius(n as CustomNode));
          d3.selectAll(".link").classed("highlighted", false).attr("stroke-opacity", 0.3).attr("stroke-width", 1.5);

          const targetNode = d3.select(event.currentTarget as Element);
          targetNode.classed("selected", true).select("circle").attr("r", getNodeRadius(d) * 1.2);

          d3.selectAll<SVGLineElement, CustomLink>(".link")
            .filter(l => l.source === d.id || l.target === d.id)
            .classed("highlighted", true)
            .attr("stroke-opacity", 0.8)
            .attr("stroke-width", 2)
            .raise();

           const connectedIds = new Set<string>();
           d3.selectAll<SVGLineElement, CustomLink>(".link.highlighted")
              .each(l => {
                 if (l.source === d.id) connectedIds.add(l.target as string);
                 if (l.target === d.id) connectedIds.add(l.source as string);
              });

           d3.selectAll<SVGGElement, CustomNode>(".node")
             .filter(n => connectedIds.has(n.id))
             .classed("connected", true);

          if (onNodeClick) onNodeClick(d.id);
        })
        .on("mouseover", (event, d) => {
            if (isDraggingRef.current) return;
            d3.select(event.currentTarget as Element).select("circle").attr("r", getNodeRadius(d) * 1.1);
            setHoveredNode({ ...d });
        })
        .on("mouseout", (event, d) => {
            if (isDraggingRef.current) return;
            if (d.id !== selectedNodeId) {
                d3.select(event.currentTarget as Element).select("circle").attr("r", getNodeRadius(d));
            }
            setHoveredNode(null);
        })
        .on("dblclick", (event, d) => {
            if (onNodeClick) onNodeClick(d.id);
        });
  }, [selectedNodeId, onNodeClick, getNodeRadius]);

  const centerOnMainWord = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, nodesToSearch: CustomNode[]) => {
      if (!zoomRef.current || isDraggingRef.current || !mainWord) return;
      const mainNodeData = nodesToSearch.find(n => n.id === mainWord);
      const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
      const width = containerRect ? containerRect.width : 800;
      const height = containerRect ? containerRect.height : 600;

      if (mainNodeData && mainNodeData.x !== undefined && mainNodeData.y !== undefined) {
        const dist = Math.sqrt(mainNodeData.x**2 + mainNodeData.y**2);
        if (dist > 20) {
            isTransitioningRef.current = true;
            const newTransform = d3.zoomIdentity.translate(width/2 - mainNodeData.x, height/2 - mainNodeData.y).scale(1);
            svg.transition().duration(600).ease(d3.easeCubicInOut)
               .call(zoomRef.current.transform, newTransform)
               .on("end", () => { isTransitioningRef.current = false; });
        }
      }
  }, [mainWord]);

  const setupSvgDimensions = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);
    return { width, height };
  }, []);

  useEffect(() => {
    if (!svgRef.current || !wordNetwork || !mainWord || filteredNodes.length === 0) {
      if (svgRef.current) d3.select(svgRef.current).selectAll("*").remove();
      if (simulationRef.current) simulationRef.current.stop();
      setError(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    if (simulationRef.current) {
      simulationRef.current.stop();
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const { width, height } = setupSvgDimensions(svg);
    const g = svg.append("g").attr("class", "graph-content").attr("transform", `translate(${width / 2}, ${height / 2})`);
    const zoom = setupZoom(svg, g);
    zoomRef.current = zoom;

    const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
    const currentFilteredLinks = baseLinks.filter(link =>
      filteredNodeIds.has(link.source) &&
      filteredNodeIds.has(link.target)
    ) as CustomLink[];

    if(currentFilteredLinks.length === 0 && filteredNodes.length > 1) {
         console.warn("Graph has nodes but no links connect them within the current depth/breadth.");
    }

    const currentSim = setupSimulation(filteredNodes, currentFilteredLinks);

    createLinks(g, currentFilteredLinks);
    const nodeElements = createNodes(g, filteredNodes, currentSim);
    setupNodeInteractions(nodeElements);

    if (currentSim) {
      currentSim.nodes(filteredNodes);
      const linkForce = currentSim.force<d3.ForceLink<CustomNode, CustomLink>>("link");
      if (linkForce) {
        linkForce.links(currentFilteredLinks);
      }
      currentSim.alpha(1).restart();
    }

    setTimeout(() => centerOnMainWord(svg, filteredNodes), 500);

    const legendContainer = svg.append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${width - 150}, 20)`);

    const allRelationGroups = [
        { type: 'main', label: 'Main Word' },
        { type: 'root', label: 'Root' },
        { type: 'derived', label: 'Derived/Root Of' },
        { type: 'synonym', label: 'Synonym' },
        { type: 'antonym', label: 'Antonym' },
        { type: 'variant', label: 'Variant' },
        { type: 'related', label: 'Related/Kaugnay' },
        { type: 'taxonomic', label: 'Taxonomic' },
        { type: 'part_whole', label: 'Part/Whole' },
        { type: 'usage', label: 'Usage Note' },
        { type: 'etymology', label: 'Etymology' },
        { type: 'component_of', label: 'Component Of' },
        { type: 'cognate', label: 'Cognate' },
        { type: 'associated', label: 'Associated' }
    ];

    const legendWidth = 140;
    legendContainer.append("rect")
      .attr("width", legendWidth)
      .attr("height", allRelationGroups.length * 18 + 30)
      .attr("rx", 5).attr("ry", 5)
      .attr("fill", theme === "dark" ? "rgba(0, 0, 0, 0.7)" : "rgba(255, 255, 255, 0.9)")
      .attr("stroke", theme === "dark" ? "rgba(255, 255, 255, 0.2)" : "rgba(0, 0, 0, 0.1)");

    legendContainer.append("text")
      .attr("x", legendWidth / 2)
      .attr("y", 15)
      .attr("text-anchor", "middle")
      .attr("font-weight", "bold").attr("font-size", "10px")
      .attr("fill", theme === "dark" ? "#fff" : "#333")
      .text("Relationship Types");

    allRelationGroups.forEach((item, i) => {
      const legendEntry = legendContainer.append("g")
        .attr("transform", `translate(10, ${i * 18 + 30})`);

      legendEntry.append("circle")
        .attr("r", 5)
        .attr("fill", getNodeColor(item.type));

      legendEntry.append("text")
        .attr("x", 12).attr("y", 4)
        .attr("font-size", "9px")
        .attr("fill", theme === "dark" ? "#fff" : "#333")
        .text(item.label);
    });

    setIsLoading(false);

    return () => {
      if (currentSim) currentSim.stop();
    };
  }, [
     wordNetwork,
     mainWord,
     depth,
     breadth,
     theme,
     mapRelationshipToGroup,
     getNodeColor,
     getNodeRadius,
     setupZoom,
     ticked,
     setupSimulation,
     createDragBehavior,
     createLinks,
     createNodes,
     setupNodeInteractions,
     centerOnMainWord,
     setupSvgDimensions,
     filteredNodes,
     baseLinks
  ]);

  useEffect(() => {
    if (prevMainWordRef.current && prevMainWordRef.current !== mainWord && svgRef.current) {
        setTimeout(() => {
            if(svgRef.current) centerOnMainWord(d3.select(svgRef.current), filteredNodes);
        }, 500);
    }
    prevMainWordRef.current = mainWord;
  }, [mainWord, centerOnMainWord, filteredNodes]);

  const handleDepthChange = useCallback((event: Event, newValue: number | number[]) => {
    const newDepth = Array.isArray(newValue) ? newValue[0] : newValue;
    setDepth(newDepth);
  }, []);

  const handleBreadthChange = useCallback((event: Event, newValue: number | number[]) => {
    const newBreadth = Array.isArray(newValue) ? newValue[0] : newValue;
    setBreadth(newBreadth);
  }, []);

  const handleZoom = useCallback((scale: number) => {
      if (zoomRef.current && svgRef.current) {
         d3.select(svgRef.current).transition().duration(300).ease(d3.easeCubicInOut)
           .call(zoomRef.current.scaleBy, scale);
       }
  }, []);

  const handleResetZoom = useCallback(() => {
      if (zoomRef.current && svgRef.current) {
         const svg = d3.select(svgRef.current);
         const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
         const width = containerRect ? containerRect.width : 800;
         const height = containerRect ? containerRect.height : 600;
         svg.transition().duration(500).ease(d3.easeCubicInOut)
           .call(zoomRef.current.transform, d3.zoomIdentity.translate(width/2, height/2));
         setTimeout(() => { if(svgRef.current) centerOnMainWord(svg, filteredNodes) }, 600);
       }
  }, [centerOnMainWord, filteredNodes]);

  const renderTooltip = useCallback(() => {
    if (!hoveredNode?.x || !hoveredNode?.y) return null;
    return (
        <div
          className="node-tooltip"
          style={{
            position: "absolute", left: `${hoveredNode.x + 10}px`, top: `${hoveredNode.y + 10}px`,
            background: theme === "dark" ? "rgba(13, 17, 23, 0.95)" : "rgba(255, 255, 255, 0.95)",
            border: `2px solid ${getNodeColor(hoveredNode.group)}`, borderRadius: "6px",
            padding: "10px 12px", maxWidth: "250px", zIndex: 1000, pointerEvents: "none",
            fontFamily: "system-ui, -apple-system, sans-serif"
          }}
        >
           <h4 style={{ margin: 0, color: theme === "dark" ? "#ffffff" : "#333333" }}>{hoveredNode.id}</h4>
           <div style={{ display: "flex", alignItems: "center", gap: "6px", padding: "4px 0" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: getNodeColor(hoveredNode.group) }}></span>
              <span style={{ fontSize: "13px", color: getNodeColor(hoveredNode.group), fontWeight: "500" }}>
                  {hoveredNode.group.charAt(0).toUpperCase() + hoveredNode.group.slice(1).replace(/_/g, ' ')}
              </span>
           </div>
           <div style={{ fontSize: "11px", marginTop: "6px", color: theme === "dark" ? "#8b949e" : "#666666", fontStyle: "italic" }}>
               Click to make main word
           </div>
        </div>
      );
  }, [hoveredNode, theme, getNodeColor]);

  if (!isValidNetwork) {
    return (
      <div className="graph-container">
        <div className="error-overlay">
          <p className="error-message">Invalid network data structure. Please try again.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="graph-container">
      <div className="graph-svg-container">
        {isLoading && (
          <div className="loading-overlay"><div className="spinner"></div><p>Loading...</p></div>
        )}
        {error && (
          <div className="error-overlay">
            <p className="error-message">{error}</p>
          </div>
        )}
         {(!wordNetwork || !mainWord || filteredNodes.length === 0 && !isLoading && !error) && (
           <div className="empty-graph-message">Enter a word to see its network.</div>
         )}
        <svg ref={svgRef} className={`graph-svg ${isLoading ? 'loading' : 'loaded'}`}>
        </svg>
      </div>
      <div className="controls-container">
         <div className="zoom-controls">
           <button onClick={() => handleZoom(1.2)} className="zoom-button">+</button>
           <button onClick={() => handleZoom(1 / 1.2)} className="zoom-button">-</button>
           <button onClick={handleResetZoom} className="zoom-button">Reset</button>
         </div>
         <div className="graph-controls">
           <div className="slider-container">
             <Typography variant="caption">Depth: {depth}</Typography>
             <Slider value={depth} onChange={handleDepthChange} aria-labelledby="depth-slider" step={1} marks min={1} max={5} size="small" />
           </div>
           <div className="slider-container">
             <Typography variant="caption">Breadth: {breadth}</Typography>
             <Slider value={breadth} onChange={handleBreadthChange} aria-labelledby="breadth-slider" step={1} marks min={5} max={20} size="small" />
           </div>
         </div>
      </div>
      {renderTooltip()}
    </div>
  );
};

export default React.memo(WordGraph);
