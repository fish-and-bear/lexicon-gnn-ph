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
  const containerRef = useRef<HTMLDivElement>(null); // Ref for container div
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);

  const isDraggingRef = useRef(false);
  const isTransitioningRef = useRef(false);
  const lastClickTimeRef = useRef(0);
  const prevMainWordRef = useRef<string | null>(mainWord); // Keep track of previous main word

  // State for tooltip delay
  const [tooltipTimeoutId, setTooltipTimeoutId] = useState<NodeJS.Timeout | null>(null);

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
          // console.warn(`Could not find nodes for edge: ${edge.source} -> ${edge.target}`);
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
    if (!wordNetwork?.nodes || !mainWord) {
        return [];
    }

    const mappedNodes = wordNetwork.nodes.map(node => {
      let calculatedGroup = 'associated'; // Default
      // Use mainWord prop to identify the main node
      if (node.label === mainWord) {
        calculatedGroup = 'main';
      } else {
        // Determine group based on direct relationship to mainWord if possible
        const connectingLink = baseLinks.find(link =>
          (link.source === mainWord && link.target === node.label) ||
          (link.source === node.label && link.target === mainWord)
        );
        if (connectingLink) {
            calculatedGroup = mapRelationshipToGroup(connectingLink.relationship);
        } else {
            // Fallback: find *any* connecting link to infer a group? Or keep 'associated'?
            // Let's keep associated for nodes not directly connected to mainWord in the current flat list
            // The filtering logic below will determine actual displayed nodes and links.
        }
      }
      
      const connections = baseLinks.filter(l => l.source === node.label || l.target === node.label).length;

      return {
        id: node.label, // Use label as the unique ID for D3 simulation
        word: node.label,
        label: node.label,
        group: calculatedGroup,
        connections: connections,
        originalId: node.id,
        language: node.language,
        definitions: node.definitions,
        path: node.path,
        has_baybayin: node.has_baybayin,
        baybayin_form: node.baybayin_form,
        // D3 simulation properties initialized as undefined
        index: undefined, x: undefined, y: undefined, vx: undefined, vy: undefined, fx: undefined, fy: undefined
      };
    });

    // Filter out duplicate nodes based on id (label)
    const uniqueNodesMap = new Map<string, CustomNode>();
    for (const node of mappedNodes) {
        if (!uniqueNodesMap.has(node.id)) {
            uniqueNodesMap.set(node.id, node);
        }
        // Optional: If duplicates found, maybe prioritize the one marked as 'main'?
        else if (node.group === 'main') {
             uniqueNodesMap.set(node.id, node);
        }
    }
    return Array.from(uniqueNodesMap.values());
  }, [wordNetwork, mainWord, baseLinks, mapRelationshipToGroup]);

  const filteredNodes = useMemo<CustomNode[]>(() => {
    if (!mainWord || baseNodes.length === 0) {
      return [];
    }
    
    const nodeMap = new Map(baseNodes.map(n => [n.id, n]));
    const nodesToShow = new Set<string>([mainWord]);
    let queue: [string, number][] = [[mainWord, 0]];
    const visitedEdges = new Set<string>(); // To limit breadth
    const nodesAddedAtDepth = new Map<string, number>(); // Track depth node was added
    nodesAddedAtDepth.set(mainWord, 0);

    while (queue.length > 0) {
        const [currentWordId, currentDepth] = queue.shift()!;

        if (currentDepth >= depth) continue;

        const relatedLinks = baseLinks.filter(link =>
            (link.source === currentWordId || link.target === currentWordId)
        );

        let neighborsAdded = 0;
        for (const link of relatedLinks) {
            const neighborId = link.source === currentWordId ? link.target : link.source;
            const edgeId = [currentWordId, neighborId].sort().join('--');

            if (!nodesToShow.has(neighborId) && neighborsAdded < breadth && !visitedEdges.has(edgeId)) {
                // Check if neighbor exists in baseNodes before adding
                const neighborNode = nodeMap.get(neighborId);
                if (neighborNode) { 
                    nodesToShow.add(neighborId);
                    queue.push([neighborId, currentDepth + 1]);
                    nodesAddedAtDepth.set(neighborId, currentDepth + 1);
                    neighborsAdded++;
                    visitedEdges.add(edgeId); // Mark edge as used for breadth limiting
                }
            }
        }
    }

    // Include nodes that are part of the path even if beyond depth/breadth?
    // This might be complex, revisit if needed.

    return baseNodes.filter(node => nodesToShow.has(node.id));
}, [baseNodes, baseLinks, mainWord, depth, breadth]);

  const filteredLinks = useMemo<CustomLink[]>(() => {
    if (!filteredNodes || filteredNodes.length === 0) return [];

    const nodeIds = new Set(filteredNodes.map(n => n.id));

    return baseLinks
      .filter(link => nodeIds.has(link.source) && nodeIds.has(link.target))
      .map(link => ({
        ...link,
        // D3 expects source/target to be node objects or IDs. Use IDs here.
        source: link.source,
        target: link.target,
      }));
  }, [filteredNodes, baseLinks]);

  // --- D3 Simulation and Rendering --- 
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || filteredNodes.length === 0) return;

    const svgElement = svgRef.current;
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;
    svgElement.setAttribute("width", width.toString());
    svgElement.setAttribute("height", height.toString());
    svgElement.setAttribute("viewBox", `${-width / 2} ${-height / 2} ${width} ${height}`);

    const svg = d3.select(svgElement);
    // Clear previous elements
    svg.selectAll("*").remove();

    const g = svg.append("g"); // Main group for zoom/pan

    // Initialize or update simulation
    if (!simulationRef.current) {
      simulationRef.current = d3.forceSimulation<CustomNode, CustomLink>()
        .force("link", d3.forceLink<CustomNode, CustomLink>(filteredLinks).id(d => d.id).distance(link => link.relationship === 'synonym' || link.relationship === 'antonym' ? 50 : 100).strength(0.6))
        .force("charge", d3.forceManyBody().strength(-250))
        .force("center", d3.forceCenter(0, 0))
        .force("collide", d3.forceCollide().radius(35)); // Add collision force
    } else {
        simulationRef.current.nodes(filteredNodes);
        const linkForce = simulationRef.current.force<d3.ForceLink<CustomNode, CustomLink>>("link");
        if(linkForce) {
            linkForce.links(filteredLinks);
        }
        simulationRef.current.alpha(0.3).restart(); // Reheat simulation
    }

    simulationRef.current.nodes(filteredNodes);
    const linkForce = simulationRef.current.force<d3.ForceLink<CustomNode, CustomLink>>("link");
    if(linkForce) {
       linkForce.links(filteredLinks);
    }

    // Setup zoom behavior
    const handleZoom = (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
      g.attr("transform", event.transform.toString());
    };

    zoomRef.current = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", handleZoom);

    svg.call(zoomRef.current);

    // Apply initial transform (optional - maybe center on mainWord)
    // Find main node position if simulation has run?
    // svg.call(zoomRef.current.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(1));


    // Draw links (lines)
    const link = g.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(filteredLinks)
      .join("line")
      .attr("stroke", theme === 'dark' ? "#555" : "#aaa")
      .attr("stroke-opacity", 0.6)
      // Use a constant stroke-width as CustomLink has no 'value' property
      .attr("stroke-width", 1.5); 

    // Draw nodes (circles)
    const node = g.append("g")
      .attr("class", "nodes")
      .selectAll<SVGCircleElement, CustomNode>("circle") // Specify types here
      .data(filteredNodes, d => d.id) // Use id as key
      .join("circle")
      .attr("r", d => (d.id === mainWord ? 15 : 10)) // Larger radius for main word
      .attr("fill", d => getNodeColor(d.group))
      .attr("stroke", theme === 'dark' ? "#fff" : "#333")
      .attr("stroke-width", d => (d.id === mainWord ? 2 : 1)); // Thicker stroke for main word

    // Node hover effects and tooltip
    node.on("mouseover", (event, d) => {
        if (isDraggingRef.current || isTransitioningRef.current) return;

        // Clear any existing timeout
        if (tooltipTimeoutId) {
          clearTimeout(tooltipTimeoutId);
          setTooltipTimeoutId(null);
        }
        
        // Set a new timeout to show tooltip after delay
        const timeoutId = setTimeout(() => {
            setHoveredNode(d);
             // Highlight node and neighbors
            node.attr('opacity', n => (n.id === d.id || filteredLinks.some(l => (l.source as CustomNode).id === d.id && (l.target as CustomNode).id === n.id) || filteredLinks.some(l => (l.target as CustomNode).id === d.id && (l.source as CustomNode).id === n.id)) ? 1 : 0.3);
            link.attr('stroke-opacity', l => ((l.source as CustomNode).id === d.id || (l.target as CustomNode).id === d.id) ? 1 : 0.2);
            label.attr('opacity', lbl => (lbl.id === d.id || filteredLinks.some(l => (l.source as CustomNode).id === d.id && (l.target as CustomNode).id === lbl.id) || filteredLinks.some(l => (l.target as CustomNode).id === d.id && (l.source as CustomNode).id === lbl.id)) ? 1 : 0.5);
        }, 300); // 300ms delay

        setTooltipTimeoutId(timeoutId);
        
        d3.select(event.currentTarget).transition().duration(150).attr("r", d.id === mainWord ? 20 : 15);
      })
      .on("mouseout", (event, d) => {
         // Clear timeout if mouse moves out before delay finishes
        if (tooltipTimeoutId) {
          clearTimeout(tooltipTimeoutId);
          setTooltipTimeoutId(null);
        }
        
        setHoveredNode(null); // Hide tooltip immediately
        d3.select(event.currentTarget).transition().duration(150).attr("r", d.id === mainWord ? 15 : 10);
        // Restore normal opacity
        node.attr('opacity', 1);
        link.attr('stroke-opacity', 0.6);
        label.attr('opacity', 1);
      });

    // Click handler - modified to call onNodeClick prop
    const handleNodeClick = (event: MouseEvent, d: CustomNode) => {
      const now = Date.now();
      const doubleClickThreshold = 300; // ms

      if (now - lastClickTimeRef.current < doubleClickThreshold) {
        // Double-click: Toggle pin
        d.pinned = !d.pinned;
        d.fx = d.pinned ? d.x : null;
        d.fy = d.pinned ? d.y : null;
        // Use standard function scope and 'this' with 'as any' workaround
        const circleElement = event.currentTarget as SVGCircleElement; // Cast target to circle
        d3.select(circleElement).attr("stroke", d.pinned ? "#ff0000" : (theme === 'dark' ? "#fff" : "#333"));
      } else {
        // Single-click (check if not dragging)
        if (!isDraggingRef.current) {
          onNodeClick(d.word); // Call the prop function
           // Optional: Maybe center view on clicked node?
           // centerViewOnNode(d);
        }
      }
      lastClickTimeRef.current = now;
    };
    
    node.on("click", handleNodeClick); 

    // Add drag capabilities
    // Use standard function syntax for correct 'this' binding
    function dragstarted(this: SVGCircleElement, event: d3.D3DragEvent<SVGCircleElement, CustomNode, any>, d: CustomNode) {
      isDraggingRef.current = true;
      if (!event.active && simulationRef.current) simulationRef.current.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
      // Use d3.select(this as any) as workaround for persistent linter error
      d3.select(this as any).raise().attr("stroke", "black"); 
    };

    const dragged = (event: d3.D3DragEvent<SVGCircleElement, CustomNode, any>, d: CustomNode) => {
      // Dragged function can remain arrow function as it doesn't need 'this'
      d.fx = event.x;
      d.fy = event.y;
    };

    // Use standard function syntax for correct 'this' binding
    function dragended(this: SVGCircleElement, event: d3.D3DragEvent<SVGCircleElement, CustomNode, any>, d: CustomNode) {
      if (!event.active && simulationRef.current) simulationRef.current.alphaTarget(0);
      if (!d.pinned) { // Only release if not pinned by double-click
         d.fx = null;
         d.fy = null;
      }
      // Use d3.select(this as any) as workaround for persistent linter error
      d3.select(this as any).attr("stroke", d.pinned ? "#ff0000" : (theme === 'dark' ? "#fff" : "#333"));
      
       // Use setTimeout to reset dragging flag after a short delay
      // This helps distinguish between drag end and click
      setTimeout(() => {
        isDraggingRef.current = false;
      }, 50); 
    };

    node.call(d3.drag<SVGCircleElement, CustomNode>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add labels
    const label = g.append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(filteredNodes)
      .join("text")
      .text(d => d.label || d.word)
      .attr("x", 14) // Offset from node center
      .attr("y", 5) // Vertical alignment offset
      .attr("font-size", "10px")
      .attr("fill", theme === 'dark' ? "#eee" : "#333")
      .attr("pointer-events", "none") // Prevent labels interfering with mouse events on nodes
      .attr("text-anchor", "start"); // Align text start to the offset point


    // Tooltip Div (ensure it's appended to the container, not svg)
     const tooltip = d3.select(containerRef.current).append("div")
      .attr("class", "graph-tooltip")
      .style("opacity", 0)
      .style("position", "absolute")
      .style("background-color", theme === 'dark' ? "rgba(40,40,40,0.9)" : "rgba(255,255,255,0.9)")
      .style("color", theme === 'dark' ? "#eee" : "#333")
      .style("border", `1px solid ${theme === 'dark' ? "#555" : "#ccc"}`)
      .style("padding", "8px")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("pointer-events", "none") // Important!
      .style("max-width", "250px")
      .style("line-height", "1.4");

    // Update tooltip content and position
    const updateTooltip = (nodeData: CustomNode | null, event?: MouseEvent) => {
       if (nodeData && event) {
            const [mouseX, mouseY] = d3.pointer(event, containerRef.current); // Get pointer relative to container
            const pathHtml = nodeData.path ? 
                `<div style="margin-top: 5px; padding-top: 5px; border-top: 1px dashed #888;"><strong>Path:</strong> ${nodeData.path.map(p => `${p.word} (${p.type})`).join(' → ')}</div>` 
                : '';
            const baybayinHtml = nodeData.has_baybayin && nodeData.baybayin_form ?
                 `<div style="margin-top: 5px;"><strong style="display: block; font-size: 0.9em; margin-bottom: 2px;">Baybayin:</strong><span style="font-family: 'Noto Sans Baybayin', sans-serif; font-size: 1.3em;">${nodeData.baybayin_form}</span></div>`
                 : '';
             const definitionHtml = nodeData.definitions && nodeData.definitions.length > 0 ?
                 `<div style="margin-top: 5px;"><strong>Def:</strong> ${nodeData.definitions[0]}${nodeData.definitions.length > 1 ? '...' : ''}</div>`
                 : '';    

            tooltip
                .html(`
                    <div style="font-weight: bold; font-size: 1.1em; margin-bottom: 4px;">${nodeData.label || nodeData.word}</div>
                    <div>Group: ${nodeData.group}</div>
                    <div>Lang: ${nodeData.language || 'N/A'}</div>
                    ${definitionHtml}
                    ${baybayinHtml}
                    ${pathHtml}
                `)
                .style("left", `${mouseX + 15}px`)
                .style("top", `${mouseY + 15}px`)
                .transition().duration(100)
                .style("opacity", 1);
        } else {
            tooltip.transition().duration(150).style("opacity", 0);
        }
    };

    // Update tooltip on hover state change
    // Need to pass the event to updateTooltip
    node.on("mousemove", (event, d) => {
        if (hoveredNode && hoveredNode.id === d.id) { // Update position if already shown
           updateTooltip(d, event);
        }
    });
    // Re-bind mouseover/mouseout to handle tooltip visibility via hoveredNode state
     node.on("mouseover.tooltip", (event, d) => {
        if (isDraggingRef.current || isTransitioningRef.current) return;
         if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);
          const timeoutId = setTimeout(() => {
              setHoveredNode(d); 
               updateTooltip(d, event); // Show tooltip with data
              // ... rest of hover effects ...
          }, 300);
         setTooltipTimeoutId(timeoutId);
         // ... immediate visual feedback ...
      })
      .on("mouseout.tooltip", (event, d) => {
         if (tooltipTimeoutId) clearTimeout(tooltipTimeoutId);
         setTooltipTimeoutId(null);
         setHoveredNode(null);
         updateTooltip(null); // Hide tooltip
         // ... rest of mouseout effects ...
      });

    // Tick function to update positions
    simulationRef.current.on("tick", () => {
      link
        .attr("x1", d => (d.source as CustomNode).x || 0)
        .attr("y1", d => (d.source as CustomNode).y || 0)
        .attr("x2", d => (d.target as CustomNode).x || 0)
        .attr("y2", d => (d.target as CustomNode).y || 0);

      node
        .attr("cx", d => d.x || 0)
        .attr("cy", d => d.y || 0);

      label
        .attr("x", d => (d.x || 0) + 14) // Position labels relative to nodes
        .attr("y", d => (d.y || 0) + 5);
    });

    // Cleanup function
    return () => {
      simulationRef.current?.stop();
      svg.selectAll("*").remove(); // Clear SVG contents
      d3.select(containerRef.current).select(".graph-tooltip").remove(); // Remove tooltip
       if (tooltipTimeoutId) {
         clearTimeout(tooltipTimeoutId);
       }
    };
  }, [filteredNodes, filteredLinks, mainWord, theme, getNodeColor, onNodeClick]); // Added onNodeClick dependency

   // Center view logic
  const centerViewOnNode = useCallback((nodeToCenter: CustomNode) => {
    if (!svgRef.current || !zoomRef.current || !nodeToCenter || nodeToCenter.x === undefined || nodeToCenter.y === undefined) return;

    const svg = d3.select(svgRef.current);
    const width = svg.node()!.clientWidth;
    const height = svg.node()!.clientHeight;
    
    isTransitioningRef.current = true; // Prevent hover effects during transition

    // Get current transform
     const currentTransform = d3.zoomTransform(svg.node()!);
     const targetScale = Math.max(0.5, Math.min(2, currentTransform.k)); // Keep scale reasonable

    // Calculate the translation needed to center the node
     const targetX = width / 2 - nodeToCenter.x * targetScale;
     const targetY = height / 2 - nodeToCenter.y * targetScale;

    const newTransform = d3.zoomIdentity.translate(targetX, targetY).scale(targetScale);

    svg.transition().duration(750)
        .call(zoomRef.current.transform, newTransform)
        .on("end", () => { isTransitioningRef.current = false; });
  }, []); // Dependencies are refs and geometry, should be stable

  // Effect to center view when mainWord changes
  useEffect(() => {
     if (mainWord && mainWord !== prevMainWordRef.current && simulationRef.current) {
       // Allow simulation to settle briefly before centering
       const timer = setTimeout(() => {
          const nodeToCenter = filteredNodes.find(n => n.id === mainWord);
           if (nodeToCenter) {
              centerViewOnNode(nodeToCenter);
           }
       }, 100); // Delay might need adjustment
       
       prevMainWordRef.current = mainWord; // Update previous word
       return () => clearTimeout(timer);
     }
     prevMainWordRef.current = mainWord; // Ensure it's updated even if no centering happens
   }, [mainWord, filteredNodes, centerViewOnNode]); // Add dependencies

  // Handlers for sliders - modified to call onNetworkChange prop
  const handleDepthChange = (event: Event, newValue: number | number[]) => {
    const newDepth = newValue as number;
    setDepth(newDepth);
    // Call the prop function
    onNetworkChange(newDepth, breadth);
  };

  const handleBreadthChange = (event: Event, newValue: number | number[]) => {
    const newBreadth = newValue as number;
    setBreadth(newBreadth);
    // Call the prop function
    onNetworkChange(depth, newBreadth);
  };

  // Return graph container and controls
  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
      <div ref={containerRef} style={{ flexGrow: 1, position: "relative" }}>
        <svg ref={svgRef}></svg>
        {/* Tooltip will be appended here by D3 */} 
      </div>
      <div style={{ padding: "10px", backgroundColor: theme === 'dark' ? '#222' : '#f0f0f0' }}>
        <Typography gutterBottom>Depth: {depth}</Typography>
        <Slider
          value={depth}
          onChange={handleDepthChange}
          aria-labelledby="depth-slider"
          valueLabelDisplay="auto"
          step={1}
          marks
          min={1}
          max={5}
        />
        <Typography gutterBottom>Breadth: {breadth}</Typography>
        <Slider
          value={breadth}
          onChange={handleBreadthChange}
          aria-labelledby="breadth-slider"
          valueLabelDisplay="auto"
          step={1}
          marks
          min={1}
          max={10}
        />
      </div>
    </div>
  );
};

export default WordGraph;
