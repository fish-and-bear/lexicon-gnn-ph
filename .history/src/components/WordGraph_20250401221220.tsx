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
  has_baybayin?: boolean;
  baybayin_form?: string | null;
}

interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
  relationship: string;
  source: string | CustomNode;
  target: string | CustomNode;
}

// Helper to get luminance and decide text color
const getTextColorForBackground = (hexColor: string): string => {
  try {
    const color = d3.color(hexColor);
    if (!color) return '#111'; // Default dark text
    const rgb = color.rgb();
    // Calculate luminance using the standard formula
    const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
    return luminance > 0.5 ? '#111111' : '#f8f8f8'; // Dark text on light bg, light text on dark bg
  } catch (e) {
    console.error("Error parsing color for text:", hexColor, e);
    return '#111'; // Fallback
  }
};

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

  // State for tooltip delay
  const [tooltipTimeoutId, setTooltipTimeoutId] = useState<NodeJS.Timeout | null>(null);

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
    // More vibrant, fun color palette
    switch (group) {
      case "main": return "#4361ee"; // Brighter blue
      case "root": return "#e63946"; // Vibrant red
      case "derived": return "#06d6a0"; // Teal
      case "synonym": return "#118ab2"; // Blue
      case "antonym": return "#ff7b00"; // Orange
      case "variant": return "#ffd166"; // Yellow
      case "related": return "#7209b7"; // Purple
      case "taxonomic": return "#8338ec"; // Lavender
      case "part_whole": return "#3a86ff"; // Light blue
      case "usage": return "#0ead69"; // Green
      case "etymology": return "#3d5a80"; // Navy blue
      case "component_of": return "#ffb01f"; // Amber
      case "cognate": return "#9381ff"; // Light purple
      case "associated": return "#adb5bd"; // Gray
      default: return "#6c757d"; // Dark gray
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
      
      // Count connections for potential sizing later
       const connections = baseLinks.filter(l => l.source === node.label || l.target === node.label).length;

      return {
        id: node.label,
        word: node.label,
        label: node.label,
        group: calculatedGroup,
        connections: connections, // Store connection count
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

  // Memoize nodeMap for use in multiple callbacks
  const nodeMap = useMemo(() => {
      return new Map(filteredNodes.map(n => [n.id, n]));
  }, [filteredNodes]);

  const getNodeRadius = useCallback((node: CustomNode) => {
    if (node.id === mainWord) return 22;
    if (node.group === 'root') return 18;
    const connFactor = node.connections ? Math.min(node.connections, 5) : 0; // Cap connection influence
    return 12 + connFactor; // Base size + connection bonus
  }, [mainWord]);

  const setupZoom = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 8]) // Increased max zoom slightly
      .interpolate(d3.interpolateZoom)
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        if (!isDraggingRef.current) g.attr("transform", event.transform.toString());
      })
      .filter(event => !isDraggingRef.current && !isTransitioningRef.current && !event.ctrlKey && !event.button);
    svg.call(zoom);
    const initialTransform = d3.zoomIdentity.translate(width / 2, height / 2);
    svg.call(zoom.transform, initialTransform);
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
        .alphaDecay(0.022)
        .velocityDecay(0.4)
        .force("link", d3.forceLink<CustomNode, CustomLink>()
          .id(d => d.id)
          .links(links)
          .distance(110)
          .strength(0.4))
        .force("charge", d3.forceManyBody<CustomNode>().strength(-300).distanceMax(300))
        .force("collide", d3.forceCollide<CustomNode>().radius(d => getNodeRadius(d) + 6).strength(1.0))
        .force("center", d3.forceCenter(0, 0))
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

  const createLinks = useCallback((
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    linksData: CustomLink[]
  ) => {
    // Categorize relationships for visual distinction
    const getRelationshipCategory = (type: string): string => {
      const synonymLike = ["synonym", "antonym", "variant"];
      const semanticRelated = ["related", "taxonomic", "part_whole", "component_of", "usage", "associated"];
      const etymological = ["derived", "etymology", "cognate"];
      
      if (synonymLike.includes(type)) return "synonym-like";
      if (semanticRelated.includes(type)) return "semantic";
      if (etymological.includes(type)) return "etymology";
      return "other";
    };
    
    // Create marker defs for arrows if they don't exist
    const svg = d3.select(svgRef.current);
    
    // Remove existing defs to prevent duplicates
    svg.selectAll("defs").remove();
    
    const defs = svg.append("defs");
    
    // Create curved line generator
    const line = d3.line<[number, number]>()
      .curve(d3.curveBasis);
    
    // Create animated dashed lines
    defs.append("filter")
      .attr("id", "link-glow")
      .attr("height", "300%")
      .attr("width", "300%")
      .attr("x", "-100%")
      .attr("y", "-100%")
      .append("feGaussianBlur")
      .attr("stdDeviation", "2")
      .attr("result", "blur");
    
    // Create link gradient for each relationship category
    ["synonym-like", "semantic", "etymology", "other"].forEach(category => {
      // Determine colors based on category
      let startColor, endColor;
      
      switch(category) {
        case "synonym-like":
          startColor = "#4cc9f0";
          endColor = "#4895ef";
          break;
        case "semantic":
          startColor = "#f72585";
          endColor = "#b5179e";
          break;
        case "etymology":
          startColor = "#e9c46a";
          endColor = "#f4a261";
          break;
        default:
          startColor = "#adb5bd";
          endColor = "#6c757d";
      }
      
      const gradient = defs.append("linearGradient")
        .attr("id", `link-gradient-${category}`)
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "100%");
      
      gradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", startColor);
        
      gradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", endColor);
    });
    
    // Define marker for each category
    ["synonym-like", "semantic", "etymology", "other"].forEach(category => {
      let color;
      switch(category) {
        case "synonym-like": color = "#4895ef"; break;
        case "semantic": color = "#b5179e"; break;
        case "etymology": color = "#f4a261"; break;
        default: color = "#6c757d";
      }
      
      defs.append("marker")
        .attr("id", `arrowhead-${category}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 15)
        .attr("refY", 0)
        .attr("orient", "auto")
        .attr("markerWidth", 5)
        .attr("markerHeight", 5)
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", color);
    });
    
    const linkGroups = g.append("g")
      .attr("class", "links")
      .selectAll("path")
      .data(linksData, (d: any) => `${(typeof d.source === 'object' ? d.source.id : d.source)}_${(typeof d.target === 'object' ? d.target.id : d.target)}`)
      .join(
        enter => {
          // Get the relationship category for styling
          const getCategory = (d: CustomLink) => {
            const relType = d.relationship.toLowerCase();
            return getRelationshipCategory(relType);
          };
          
          // Create paths with curved lines
          const path = enter.append("path")
            .attr("class", d => `link link-${getCategory(d)}`)
            .attr("stroke", d => `url(#link-gradient-${getCategory(d)})`)
            .attr("stroke-width", 1.5)
            .attr("fill", "none")
            .attr("opacity", 0)
            .attr("marker-end", d => `url(#arrowhead-${getCategory(d)})`)
            .style("stroke-dasharray", d => {
              const category = getCategory(d);
              switch(category) {
                case "synonym-like": return "none"; // Solid
                case "semantic": return "5,3"; // Dashed
                case "etymology": return "2,2"; // Dotted
                default: return "3,3,1,3"; // Dash-dot
              }
            });
          
          // Apply enter animation with path drawing effect
          path.call(enter => enter.transition()
            .delay((_, i) => i * 10) // Stagger for visual interest
            .duration(600)
            .attrTween("stroke-dashoffset", function() {
              const length = this.getTotalLength();
              return d3.interpolate(length, 0);
            })
            .attr("opacity", 0.6)
          );
          
          return path;
        },
        update => update,
        exit => exit.call(exit => exit.transition()
          .duration(300)
          .attr("opacity", 0)
          .remove()
        )
      );
    
    return linkGroups;
  }, []);

  const createNodes = useCallback((
      g: d3.Selection<SVGGElement, unknown, null, undefined>,
      nodesData: CustomNode[],
      simulation: d3.Simulation<CustomNode, CustomLink> | null
      ) => {
    const drag = simulation ? createDragBehavior(simulation) : null;
    
    // Add a gradient definition for each node type
    const defs = g.append("defs");
    
    // Create gradients for each node group
    Object.entries({
      "main": getNodeColor("main"),
      "root": getNodeColor("root"),
      "derived": getNodeColor("derived"),
      "synonym": getNodeColor("synonym"),
      "antonym": getNodeColor("antonym"),
      "variant": getNodeColor("variant"),
      "related": getNodeColor("related"),
      "taxonomic": getNodeColor("taxonomic"),
      "part_whole": getNodeColor("part_whole"),
      "usage": getNodeColor("usage"),
      "etymology": getNodeColor("etymology"),
      "component_of": getNodeColor("component_of"),
      "cognate": getNodeColor("cognate"),
      "associated": getNodeColor("associated"),
    }).forEach(([group, color]) => {
      const gradId = `node-gradient-${group}`;
      const grad = defs.append("radialGradient")
        .attr("id", gradId)
        .attr("cx", "30%")
        .attr("cy", "30%")
        .attr("r", "70%");
      
      // Lighter center, original color at edge
      const c = d3.color(color);
      if (c) {
        grad.append("stop")
          .attr("offset", "0%")
          .attr("stop-color", c.brighter(0.7).formatHex());
          
        grad.append("stop")
          .attr("offset", "100%")
          .attr("stop-color", color);
      }
    });
    
    // Add pulsing animation for main node
    const pulseAnimation = defs.append("filter")
      .attr("id", "pulse-animation")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");
      
    // Create the animation components
    const animator = pulseAnimation.append("feGaussianBlur")
      .attr("in", "SourceGraphic")
      .attr("stdDeviation", "0")
      .attr("result", "blur");
      
    const animateValues = "0;1;2;2;1;0";
    const animKeyTimes = "0;0.2;0.4;0.6;0.8;1";
    
    animator.append("animate")
      .attr("attributeName", "stdDeviation")
      .attr("values", "0;0.5;1;1;0.5;0")
      .attr("keyTimes", animKeyTimes)
      .attr("dur", "4s")
      .attr("repeatCount", "indefinite");
    
    // Create node groups
    const nodeGroups = g.append("g")
      .attr("class", "nodes")
      .selectAll("g")
        .data(nodesData, (d: any) => (d as CustomNode).id)
      .join(
          enter => {
              const nodeGroup = enter.append("g")
                  .attr("class", d => `node node-group-${d.group} ${d.id === mainWord ? "main-node" : ""}`)
                  .attr("transform", d => `translate(${d.x ?? 0}, ${d.y ?? 0})`)
                  .style("opacity", 0);

              // Add a subtle shadow/glow effect
              nodeGroup.append("circle")
                .attr("class", "node-shadow")
                .attr("r", d => getNodeRadius(d) + 2)
                .attr("fill", d => d.id === mainWord 
                  ? `url(#node-gradient-${d.group})` 
                  : `url(#node-gradient-${d.group})`)
                .attr("filter", "drop-shadow(0 0 3px rgba(0,0,0,0.2))") 
                .attr("opacity", 0.3)
                .attr("stroke", "none");
                
              // Main circle
              nodeGroup.append("circle")
                .attr("class", "node-circle")
                .attr("r", d => getNodeRadius(d))
                .attr("fill", d => `url(#node-gradient-${d.group})`)
                .attr("stroke", d => d3.color(getNodeColor(d.group))?.darker(0.6).formatHex() ?? (theme === "dark" ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)"))
                .attr("stroke-width", 1.5);

              // Add a filter to the main word node for the pulsing effect
              if (mainWord) {
                nodeGroup.filter(d => d.id === mainWord)
                  .select(".node-shadow")
                  .attr("filter", "url(#pulse-animation)")
                  .attr("opacity", 0.4);
              }

              // Text group with halo
              const textGroup = nodeGroup.append("text")
                  .attr("dy", ".35em")
                  .attr("text-anchor", "middle")
                  .attr("font-size", d => d.id === mainWord ? "11px" : "9px")
                  .attr("font-weight", d => d.id === mainWord ? "600" : "400")
                  .text(d => d.word)
                  .style("pointer-events", "none");
      
              // Halo
              textGroup.clone(true)
                  .lower()
                  .attr("fill", "none")
                  .attr("stroke", theme === "dark" ? "rgba(0,0,0,0.8)" : "rgba(255,255,255,0.8)")
                  // Make main word halo slightly thicker
                  .attr("stroke-width", d => d.id === mainWord ? 4 : 3.5)
                  .attr("stroke-linejoin", "round");

              // Main text fill
              nodeGroup.select("text:not([stroke])")
                  .attr("fill", d => getTextColorForBackground(getNodeColor(d.group)));

              // Add tooltips
              nodeGroup.append("title").text(d => `${d.word}\nGroup: ${d.group}\nConnections: ${d.connections ?? 0}`);

              // Apply fade-in transition with a slight bounce effect
              nodeGroup.call(enter => enter.transition()
                .duration(500)
                .style("opacity", 1)
                .attrTween("transform", function(d) {
                  const startX = d.x ?? 0;
                  const startY = d.y ?? 0;
                  return function(t) {
                    // Add a small bounce effect
                    const bounce = Math.sin(t * Math.PI) * 5 * (1-t);
                    return `translate(${startX},${startY + bounce})`;
                  };
                })
              );
              
              return nodeGroup;
          },
          update => update,
          exit => exit
              .call(exit => exit.transition()
                .duration(300)
                .style("opacity", 0)
                .attrTween("transform", function(d) {
                  const startX = d.x ?? 0;
                  const startY = d.y ?? 0;
                  return function(t) {
                    return `translate(${startX},${startY - 10 * t})`;
                  };
                })
              )
              .remove()
      );

      if (drag) nodeGroups.call(drag as any);
    return nodeGroups;
  }, [createDragBehavior, getNodeRadius, getNodeColor, theme, mainWord]);

  const setupNodeInteractions = useCallback((
    nodeGroups: d3.Selection<SVGGElement, CustomNode, SVGGElement, unknown>,
    linkGroups: d3.Selection<SVGPathElement, CustomLink, SVGGElement, unknown>
  ) => {
    // Helper to get connecting links for a node
    const getConnectedLinks = (nodeId: string) => {
      return linkGroups.filter(d => {
        const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
        const targetId = typeof d.target === 'object' ? d.target.id : d.target;
        return sourceId === nodeId || targetId === nodeId;
      });
    };
    
    // Helper to get connecting nodes for a node
    const getConnectedNodes = (nodeId: string) => {
      const connectedNodes = new Set<string>();
      
      linkGroups.each(d => {
        const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
        const targetId = typeof d.target === 'object' ? d.target.id : d.target;
        
        if (sourceId === nodeId) {
          connectedNodes.add(targetId);
        } else if (targetId === nodeId) {
          connectedNodes.add(sourceId);
        }
      });
      
      return nodeGroups.filter(d => connectedNodes.has(d.id));
    };

    // Add click for pinning
    nodeGroups.on("click", (event, d) => {
      event.stopPropagation();
      
      // Toggle pinned state with visual effect
      d.fx = d.fx ? null : d.x;
      d.fy = d.fy ? null : d.y;
      
      // Update visual appearance for pinned state
      d3.select(event.currentTarget)
        .classed("pinned", !!d.fx)
        .select(".node-circle")
        .transition()
        .duration(300)
        .attr("stroke-width", d.fx ? 3 : 1.5)
        .attr("stroke-dasharray", d.fx ? "3,2" : "none");
        
      // Show brief indication of pinned/unpinned
      const statusText = d3.select(event.currentTarget)
        .append("text")
        .attr("class", "status-text")
        .attr("dy", -20)
        .attr("text-anchor", "middle")
        .attr("fill", theme === "dark" ? "#fff" : "#000")
        .style("pointer-events", "none")
        .style("font-size", "9px")
        .style("opacity", 0)
        .text(d.fx ? "Pinned" : "Unpinned");
        
      statusText.transition()
        .duration(300)
        .style("opacity", 1)
        .transition()
        .delay(1000)
        .duration(300)
        .style("opacity", 0)
        .remove();
    });

    // Hover effects with fun animations
    nodeGroups.on("mouseenter", (event, d) => {
      const currentNode = d3.select(event.currentTarget);
      
      // Highlight current node
      currentNode
        .raise() // Bring to front
        .transition()
        .duration(200)
        .select(".node-circle")
        .attr("stroke-width", 3)
        .attr("r", getNodeRadius(d) * 1.1); // Slightly enlarge
      
      // Add a subtle pulse effect
      currentNode.select(".node-shadow")
        .transition()
        .duration(300)
        .attr("r", getNodeRadius(d) + 5)
        .attr("opacity", 0.5);
      
      // Highlight connected links
      const connectedLinks = getConnectedLinks(d.id);
      connectedLinks
        .raise() // Bring to front
        .transition()
        .duration(200)
        .attr("stroke-width", 2.5)
        .attr("opacity", 1);
        
      // Highlight connected nodes
      const connectedNodes = getConnectedNodes(d.id);
      connectedNodes
        .raise() // Bring to front
        .transition()
        .duration(200)
        .select(".node-circle")
        .attr("stroke-width", 2);
      
      // Dim other nodes
      nodeGroups.filter(n => n.id !== d.id && !connectedNodes.filter(cn => cn.id === n.id).empty())
        .transition()
        .duration(200)
        .style("opacity", 0.7);
      
      // Dim other links
      linkGroups.filter(l => {
        const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
        const targetId = typeof l.target === 'object' ? l.target.id : l.target;
        return sourceId !== d.id && targetId !== d.id;
      })
      .transition()
      .duration(200)
      .style("opacity", 0.2);
      
      // Show tooltip with animation
      if (d.word) {
        setTooltipContent(`
          <div style="font-weight: bold">${d.word}</div>
          <div>Group: ${d.group}</div>
          <div>Connections: ${d.connections ?? 0}</div>
        `);
        setHoveredNode(d);
      }
    });

    // Mouse leave effect
    nodeGroups.on("mouseleave", (event, d) => {
      const currentNode = d3.select(event.currentTarget);
      
      // Reset current node
      currentNode
        .transition()
        .duration(300)
        .select(".node-circle")
        .attr("stroke-width", d.fx ? 3 : 1.5) // Maintain pinned style if pinned
        .attr("r", getNodeRadius(d));
      
      // Reset shadow
      currentNode.select(".node-shadow")
        .transition()
        .duration(300)
        .attr("r", getNodeRadius(d) + 2)
        .attr("opacity", 0.3);
      
      // Reset all nodes
      nodeGroups
        .transition()
        .duration(300)
        .style("opacity", 1);
      
      // Reset all links
      linkGroups
        .transition()
        .duration(300)
        .attr("stroke-width", 1.5)
        .style("opacity", 0.6);
      
      // Hide tooltip
      setHoveredNode(null);
    });
  }, [getNodeRadius, theme]);

  // Define handleResetZoom before centerOnMainWord
  const handleResetZoom = useCallback(() => {
    if (zoomRef.current && svgRef.current) {
       const svg = d3.select(svgRef.current);
       const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
       const width = containerRect ? containerRect.width : 800;
       const height = containerRect ? containerRect.height : 600;
       const resetTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
       svg.transition().duration(600).ease(d3.easeCubicInOut)
         .call(zoomRef.current.transform, resetTransform);
     }
  }, []);

  const centerOnMainWord = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, nodesToSearch: CustomNode[]) => {
    if (!zoomRef.current || isDraggingRef.current || !mainWord) return;
    const mainNodeData = nodesToSearch.find(n => n.id === mainWord);
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    if (mainNodeData && mainNodeData.x !== undefined && mainNodeData.y !== undefined) {
      const currentTransform = d3.zoomTransform(svg.node()!);
      const targetScale = Math.max(0.5, Math.min(2, currentTransform.k));
      const targetX = width / 2 - mainNodeData.x * targetScale;
      const targetY = height / 2 - mainNodeData.y * targetScale;
      const newTransform = d3.zoomIdentity.translate(targetX, targetY).scale(targetScale);
      svg.transition().duration(750).ease(d3.easeCubicInOut)
         .call(zoomRef.current.transform, newTransform);
    } else {
        handleResetZoom();
    }
  }, [mainWord]); // Removed handleResetZoom from dependencies

  const setupSvgDimensions = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
    const width = containerRect ? containerRect.width : 800;
    const height = containerRect ? containerRect.height : 600;
    svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);
    return { width, height };
  }, []);

  useEffect(() => {
    if (!svgRef.current || !wordNetwork || !mainWord || baseNodes.length === 0) {
      if (svgRef.current) d3.select(svgRef.current).selectAll("*").remove();
      if (simulationRef.current) simulationRef.current.stop();
      setError(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    if (simulationRef.current) simulationRef.current.stop();
      
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

    const { width, height } = setupSvgDimensions(svg);
    const g = svg.append("g").attr("class", "graph-content");
      const zoom = setupZoom(svg, g);
      zoomRef.current = zoom;

    console.log("[Graph Effect] Base links count:", baseLinks.length);
    const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
    console.log("[Graph Effect] Filtered node IDs:", Array.from(filteredNodeIds));

    console.log("[Graph Effect] Base links sample before filter:", JSON.stringify(baseLinks.slice(0, 10).map(l => ({ s: l.source, t: l.target, r: l.relationship }))));
    console.log("[Graph Effect] Filtered node IDs Set content:", JSON.stringify(Array.from(filteredNodeIds)));

    const currentFilteredLinks = baseLinks.filter(link => {
      const sourceId = typeof link.source === 'object' && link.source !== null ? (link.source as CustomNode).id : link.source as string;
      const targetId = typeof link.target === 'object' && link.target !== null ? (link.target as CustomNode).id : link.target as string;
      return filteredNodeIds.has(sourceId) && filteredNodeIds.has(targetId);
    }) as CustomLink[];
    console.log("[Graph Effect] Filtered links count AFTER filtering:", currentFilteredLinks.length);
    if(currentFilteredLinks.length > 0 && currentFilteredLinks.length < 10) {
         console.log("[Graph Effect] Filtered links sample AFTER filtering:", JSON.stringify(currentFilteredLinks.map(l => ({s: l.source, t: l.target}))));
      }

    if(currentFilteredLinks.length === 0 && filteredNodes.length > 1) {
         console.warn("Graph has nodes but no links connect them within the current depth/breadth.");
    }

    const currentSim = setupSimulation(filteredNodes, currentFilteredLinks);

    const linkGroups = createLinks(g, currentFilteredLinks);
    const nodeElements = createNodes(g, filteredNodes, currentSim);
    setupNodeInteractions(nodeElements, linkGroups);

    if (currentSim) {
      currentSim.nodes(filteredNodes);
      const linkForce = currentSim.force<d3.ForceLink<CustomNode, CustomLink>>("link");
      if (linkForce) {
        linkForce.links(currentFilteredLinks);
      }
      currentSim.alpha(1).restart();

      // Pin the main node after a short delay to allow initial positioning
      const mainNodeData = filteredNodes.find(n => n.id === mainWord);
      if (mainNodeData) {
          mainNodeData.fx = 0; // Pin to center (since forceCenter is 0,0)
          mainNodeData.fy = 0;
      }
    }

    setTimeout(() => centerOnMainWord(svg, filteredNodes), 800);

      const legendContainer = svg.append("g")
        .attr("class", "legend")
      .attr("transform", `translate(${width - 160}, 25)`);
      
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
    const legendItemHeight = 20;
    const legendWidth = 150;
    const legendPadding = 15;
    const legendX = width - legendWidth - legendPadding;

      legendContainer.append("rect")
        .attr("width", legendWidth)
        .attr("height", allRelationGroups.length * legendItemHeight + legendPadding * 2 + 10)
        .attr("rx", 6).attr("ry", 6)
        .attr("fill", theme === "dark" ? "rgba(30, 30, 30, 0.85)" : "rgba(255, 255, 255, 0.85)")
        .attr("stroke", theme === "dark" ? "rgba(255, 255, 255, 0.15)" : "rgba(0, 0, 0, 0.1)");

      legendContainer.append("text")
        .attr("x", legendWidth / 2)
        .attr("y", legendPadding + 2)
        .attr("text-anchor", "middle")
        .attr("font-weight", "600").attr("font-size", "11px")
        .attr("fill", theme === "dark" ? "#eee" : "#222")
        .text("Relationship Types");
      
     allRelationGroups.forEach((item, i) => {
        const legendEntry = legendContainer.append("g")
            .attr("class", "legend-item")
            .attr("transform", `translate(${legendPadding}, ${i * legendItemHeight + legendPadding + 25})`);

        legendEntry.append("circle")
            .attr("class", "legend-color")
            .attr("cx", 0)
            .attr("cy", 0)
            .attr("r", 6)
          .attr("fill", getNodeColor(item.type));
        
        legendEntry.append("text")
            .attr("class", "legend-label")
            .attr("x", 15)
            .attr("y", 0)
            .attr("dy", ".35em")
            .attr("font-size", "10px")
            .attr("fill", theme === "dark" ? "#ddd" : "#333")
          .text(item.label);

        // Add tooltip to legend item group
        legendEntry.append("title").text(`Relationship Type: ${item.label}`);
      });

      setIsLoading(false);
      
    // Tooltip depends on state now, so keep it outside useEffect cleanup?
    const centerTimeout = setTimeout(() => {
         if (svgRef.current) centerOnMainWord(svg, filteredNodes);
     }, 800);

      return () => {
      if (currentSim) currentSim.stop();
       clearTimeout(centerTimeout);
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
        const recenterTimeout = setTimeout(() => {
            if(svgRef.current) centerOnMainWord(d3.select(svgRef.current), filteredNodes);
        }, 800);
         return () => clearTimeout(recenterTimeout);
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

  const renderTooltip = useCallback(() => {
    // Tooltip visibility is now handled by hoveredNode state directly (set via timeout)
    if (!hoveredNode?.id || !hoveredNode?.x || !hoveredNode?.y || !svgRef.current) return null;

    const svgNode = svgRef.current;
    const transform = d3.zoomTransform(svgNode);

    const [screenX, screenY] = transform.apply([hoveredNode.x, hoveredNode.y]);

    const offsetX = (screenX > window.innerWidth / 2) ? -20 - 250 : 20;
    const offsetY = (screenY > window.innerHeight / 2) ? -20 - 80 : 20;
      
      return (
        <div
          className="node-tooltip"
          style={{
            position: "absolute",
            left: `${screenX + offsetX}px`,
            top: `${screenY + offsetY}px`,
            background: theme === "dark" ? "rgba(30, 30, 30, 0.9)" : "rgba(250, 250, 250, 0.9)",
            border: `1.5px solid ${getNodeColor(hoveredNode.group)}`, borderRadius: "8px",
            padding: "10px 14px", maxWidth: "280px", zIndex: 1000, pointerEvents: "none",
            fontFamily: "system-ui, -apple-system, sans-serif",
            transition: "left 0.1s ease-out, top 0.1s ease-out, opacity 0.1s ease-out",
            boxShadow: theme === "dark" ? "0 4px 15px rgba(0,0,0,0.4)" : "0 4px 15px rgba(0,0,0,0.15)",
            opacity: 1,
          }}
        >
           <h4 style={{ margin: 0, marginBottom: '6px', color: getNodeColor(hoveredNode.group), fontSize: '15px' }}>{hoveredNode.id}</h4>
           <div style={{ display: "flex", alignItems: "center", gap: "6px", paddingBottom: "4px" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: getNodeColor(hoveredNode.group), flexShrink: 0 }}></span>
              <span style={{ fontSize: "13px", color: theme === 'dark' ? '#ccc' : '#555', fontWeight: "500" }}>
                  {hoveredNode.group.charAt(0).toUpperCase() + hoveredNode.group.slice(1).replace(/_/g, ' ')}
              </span>
          </div>
           {hoveredNode.definitions && hoveredNode.definitions.length > 0 && (
                <p style={{ fontSize: '12px', color: theme === 'dark' ? '#bbb' : '#666', margin: '6px 0 0 0', fontStyle: 'italic' }}>
                    {hoveredNode.definitions[0].length > 100 ? hoveredNode.definitions[0].substring(0, 97) + '...' : hoveredNode.definitions[0]}
            </p>
          )}
           <div style={{ fontSize: "11px", marginTop: "8px", color: theme === "dark" ? "#8b949e" : "#777777" }}>
               Click to focus | Double-click to pin/unpin
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
           <button onClick={() => handleZoom(1.3)} className="zoom-button" title="Zoom In">+</button>
           <button onClick={() => handleZoom(1 / 1.3)} className="zoom-button" title="Zoom Out">-</button>
           <button onClick={handleResetZoom} className="zoom-button" title="Reset View">Reset</button>
        </div>
        <div className="graph-controls">
          <div className="slider-container">
             <Typography variant="caption" sx={{ mr: 1 }}>Depth: {depth}</Typography>
             <Slider value={depth} onChange={handleDepthChange} onChangeCommitted={() => onNetworkChange(depth, breadth)} aria-labelledby="depth-slider" step={1} marks min={1} max={5} size="small" sx={{ width: 100 }}
                title={`Set relationship depth (Current: ${depth})`}/>
          </div>
          <div className="slider-container">
             <Typography variant="caption" sx={{ mr: 1 }}>Breadth: {breadth}</Typography>
             <Slider value={breadth} onChange={handleBreadthChange} onChangeCommitted={() => onNetworkChange(depth, breadth)} aria-labelledby="breadth-slider" step={1} marks min={5} max={20} size="small" sx={{ width: 100 }}
                title={`Set max connections per node (Current: ${breadth})`}/>
          </div>
        </div>
      </div>
      {renderTooltip()}
    </div>
  );
};

export default React.memo(WordGraph);
