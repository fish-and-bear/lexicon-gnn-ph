/* eslint-disable @typescript-eslint/no-this-alias */
import React, {
    useEffect,
    useRef,
    useState,
    useCallback,
    useMemo,
  } from "react";
  import * as d3 from "d3";
  import "./WordGraph.css";
  import { WordNetwork as ImportedWordNetwork, NetworkNode as ImportedNetworkNode, NetworkLink as ImportedNetworkLink } from "../types";
  import { useAppTheme } from "../contexts/ThemeContext";
  import Slider from '@mui/material/Slider';
  import Typography from "@mui/material/Typography";
  import axios from 'axios';
  import NetworkControls from './NetworkControls';
  import { createPortal } from 'react-dom';
  // Import color utility functions
  import { 
    mapRelationshipToGroup, 
    getNodeColor, 
    getTextColorForBackground,
    getRelationshipTypeLabel
  } from '../utils/colorUtils';
  
  // MUI imports
  import Drawer from '@mui/material/Drawer';
  import IconButton from '@mui/material/IconButton';
  import Box from '@mui/material/Box';
  import TuneIcon from '@mui/icons-material/Tune';
  import List from '@mui/material/List';
  import ListItem from '@mui/material/ListItem';
  import ListItemIcon from '@mui/material/ListItemIcon';
  import ListItemText from '@mui/material/ListItemText';
  import Checkbox from '@mui/material/Checkbox';
  import ListSubheader from '@mui/material/ListSubheader';
  import Divider from '@mui/material/Divider';
  import { useTheme as useMuiTheme } from '@mui/material/styles';
  import { alpha } from '@mui/material/styles';
  import Paper from '@mui/material/Paper';
  import { CircularProgress } from '@mui/material';
  
  interface WordGraphProps {
    wordNetwork: ImportedWordNetwork | null;
    mainWord: string | null;
    onNodeClick: (word: string) => void;
    onNetworkChange: (depth: number, breadth: number) => void;
    initialDepth: number;
    initialBreadth: number;
    isMobile: boolean;
    isLoading: boolean;
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
    has_baybayin?: boolean;
    baybayin_form?: string | null;
    relationshipToMain?: string;
    pathToMain?: string[];
    x?: number;
    y?: number;
    fx?: number | null;
    fy?: number | null;
  }
  
  interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
    relationship: string;
    source: string | CustomNode;
    target: string | CustomNode;
    metadata: Record<string, any> | null | undefined; // Reverted: metadata required (can be null/undefined)
  }
  
  // Define interfaces for relationship categories
  interface RelationshipTypeInfo {
    category: string;
    label: string;
    color: string;
  }
  
  interface RelationshipLabelInfo {
    label: string;
    color: string;
    types: string[]; // Original types mapping to this label
  }
  
  interface RelationshipGroups {
    uniqueTypes: Record<string, RelationshipTypeInfo>; // Keep original mapping for lookups
    categories: Array<{ name: string; labels: RelationshipLabelInfo[] }>;
  }
  
  // NodePeekCard Component
  const NodePeekCard = ({ peekedNodeData, onClose }: { peekedNodeData: { node: CustomNode, x: number, y: number } | null, onClose: () => void }) => {
    const cardRef = useRef<HTMLDivElement>(null);
    const muiTheme = useMuiTheme();
    const [isVisible, setIsVisible] = useState(false);
  
    useEffect(() => {
      if (peekedNodeData) {
        // Delay visibility slightly to allow transition
        const timer = setTimeout(() => setIsVisible(true), 10);
        return () => clearTimeout(timer);
      } else {
        setIsVisible(false);
      }
    }, [peekedNodeData]);
  
    useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
        if (isVisible && cardRef.current && !cardRef.current.contains(event.target as Node)) {
          const targetElement = event.target as Element;
          if (!targetElement.closest('.node')) {
             onClose();
          }
        }
      };
      const handleEscape = (event: KeyboardEvent) => {
        if (isVisible && event.key === 'Escape') {
          onClose();
        }
      };
      // Add listeners only when visible
      if (isVisible) {
          document.addEventListener('mousedown', handleClickOutside);
          document.addEventListener('keydown', handleEscape);
      } else {
          document.removeEventListener('mousedown', handleClickOutside);
          document.removeEventListener('keydown', handleEscape);
      }
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
        document.removeEventListener('keydown', handleEscape);
      };
    }, [isVisible, onClose]);
  
    if (!peekedNodeData) return null;
  
    const { node, x, y } = peekedNodeData;
    const groupLabel = node.group.charAt(0).toUpperCase() + node.group.slice(1).replace(/_/g, ' ');
    const definition = node.definitions && node.definitions.length > 0
      ? (node.definitions[0].length > 80 ? node.definitions[0].substring(0, 77) + '...' : node.definitions[0])
      : 'No definition available.';
  
    // Basic positioning logic (needs refinement for edge cases)
    const cardWidth = 220; // Increased width slightly
    const cardHeight = 120; // Estimate
    let offsetX = 15; // Default offset to the right
    let offsetY = 15; // Default offset below
  
    // Adjust position based on click coordinates relative to window boundaries
    if (x + cardWidth + offsetX > window.innerWidth) {
      offsetX = -cardWidth - 15; // Place to the left
    }
    if (y + cardHeight + offsetY > window.innerHeight) {
      offsetY = -cardHeight - 15; // Place above
    }
  
    // Use Portal for rendering
    return createPortal(
      <Paper
        ref={cardRef}
        elevation={5} // Slightly higher elevation than tooltip
        sx={{
          position: 'fixed', // Use fixed position relative to viewport
          left: `${x + offsetX}px`,
          top: `${y + offsetY}px`,
          zIndex: 1100, // Above tooltip (tooltip is 1000)
          padding: '12px 16px',
          width: `${cardWidth}px`,
          pointerEvents: 'auto',
          backgroundColor: 'var(--card-bg-color-elevated, background.paper)', // Use elevated background
          border: `1px solid ${alpha(muiTheme.palette.divider, 0.4)}`,
          borderRadius: '10px',
          backdropFilter: 'blur(5px)', // Add blur for modern feel
          transition: 'opacity 0.2s ease-out, transform 0.2s ease-out',
          opacity: isVisible ? 1 : 0, // Control visibility
          transform: isVisible ? 'scale(1)' : 'scale(0.95)', // Control visibility
          animation: isVisible ? 'fadeInTooltip 0.2s ease-out' : 'none', // Reuse tooltip animation
          boxShadow: muiTheme.palette.mode === 'dark' ? '0 6px 20px rgba(0,0,0,0.3)' : '0 6px 20px rgba(0,0,0,0.1)',
        }}
      >
        <Typography variant="h6" sx={{ fontSize: '1rem', mb: 0.5, color: getNodeColor(node.group), fontWeight: 600 }}>
          {node.word}
        </Typography>
        <Typography variant="caption" display="block" sx={{ color: 'text.secondary', mb: 1, fontSize: '0.75rem' }}>
          {groupLabel}
        </Typography>
        <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'text.primary', lineHeight: 1.4 }}>
          {definition}
        </Typography>
      </Paper>,
      document.body // Render in body
    );
  };
  
  const WordGraph: React.FC<WordGraphProps> = ({
    wordNetwork,
    mainWord,
    onNodeClick,
    onNetworkChange,
    initialDepth,
    initialBreadth,
    isMobile,
    isLoading
  }) => {
    const { themeMode } = useAppTheme();
    const muiTheme = useMuiTheme();
    const svgRef = useRef<SVGSVGElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const gRef = useRef<SVGGElement | null>(null);
    const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
    const linkGroupRef = useRef<SVGGElement>(null);
    const nodeGroupRef = useRef<SVGGElement>(null);
    const legendContainerRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null); // Ref for legend container
    const legendWidthRef = useRef<number>(0); // Ref for legend width
    const resizeTimeoutRef = useRef<NodeJS.Timeout | null>(null); // Ref for debounce timer
  
    // Add state variables to store the graph elements
    const [graphElement, setGraphElement] = useState<SVGGElement | null>(null);
    const [zoomBehavior, setZoomBehavior] = useState<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  
    // Log the received mainWord prop
    useEffect(() => {
      console.log(`[WordGraph] Received mainWord prop: '${mainWord}'`);
    }, [mainWord]);
  
    const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
    const [peekedNode, setPeekedNode] = useState<{ node: CustomNode, x: number, y: number } | null>(null);
    const [depth, setDepth] = useState<number>(initialDepth);
    const [breadth, setBreadth] = useState<number>(initialBreadth);
    const [error, setError] = useState<string | null>(null);
    const [isValidNetwork, setIsValidNetwork] = useState(true);
    const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);
    const [filteredRelationships, setFilteredRelationships] = useState<string[]>([]);
    const [forceUpdate, setForceUpdate] = useState<number>(0);
  
    // Drawer state
    const [controlsOpen, setControlsOpen] = useState(false);
  
    const isDraggingRef = useRef(false);
    const isTransitioningRef = useRef(false);
    const prevMainWordRef = useRef<string | null>(null);
  
    // Create a key that changes whenever filtered relationships change
    // This will force the graph to completely rebuild
    const filterUpdateKey = useMemo(() => {
      return filteredRelationships.join(',');
    }, [filteredRelationships]);
  
    // Check if network data is valid
    useEffect(() => {
      if (!wordNetwork || !wordNetwork.nodes || !Array.isArray(wordNetwork.nodes) || 
          !wordNetwork.links || !Array.isArray(wordNetwork.links)) {
        console.error("Invalid wordNetwork structure:", wordNetwork);
        setIsValidNetwork(false);
      } else {
        setIsValidNetwork(true);
      }
    }, [wordNetwork]);
    
    // Toggle control drawer
    const toggleControlsDrawer = () => {
      setControlsOpen(!controlsOpen);
    };
  
    // Add handleToggleRelationshipFilter function
    const handleToggleRelationshipFilter = useCallback((typeOrTypes: string | string[]) => {
      const typesToToggle = Array.isArray(typeOrTypes) ? typeOrTypes : [typeOrTypes];
      const typesToToggleLower = typesToToggle.map(t => t.toLowerCase());
      
      console.log(`[FILTER] Toggling filter for type(s): '${typesToToggleLower.join(', ')}'`);
      
      setFilteredRelationships(prevFilters => {
        // Check if *all* types in the group are currently filtered
        const allAreFiltered = typesToToggleLower.every(t => prevFilters.includes(t));
        let newFilters;
        
        if (allAreFiltered) {
          // Remove all types in this group from filters
          console.log(`[FILTER] Removing group '${typesToToggleLower.join(', ')}' from filters`);
          newFilters = prevFilters.filter(f => !typesToToggleLower.includes(f));
        } else {
          // Add any missing types from this group to filters
          console.log(`[FILTER] Adding group '${typesToToggleLower.join(', ')}' to filters`);
          newFilters = Array.from(new Set([...prevFilters, ...typesToToggleLower]));
        }
        
        console.log(`[FILTER] New filters:`, newFilters);
        return newFilters;
      });
      
      // Force complete graph rebuild
      setForceUpdate(prev => prev + 1);
    }, []);
  
    // Define getUniqueRelationshipGroups function
    const getUniqueRelationshipGroups = useCallback((): RelationshipGroups => {
      const typeMappings: Record<string, RelationshipTypeInfo> = {
        main: { category: "Core", label: "Main Word", color: getNodeColor("main") },
        // Origin Category
        root_of: { category: "Origin", label: "Root Of", color: getNodeColor("root_of") }, // Consolidated root
        derived_from: { category: "Origin", label: "Derived From", color: getNodeColor("derived_from") }, 
        etymology: { category: "Origin", label: "Etymology", color: getNodeColor("etymology") },
        cognate: { category: "Origin", label: "Cognate", color: getNodeColor("cognate") },
        // Derivation Category
        derived: { category: "Derivation", label: "Derived", color: getNodeColor("derived") }, 
        derivative: { category: "Derivation", label: "Derived", color: getNodeColor("derived") }, // Map derivative here too
        sahod: { category: "Derivation", label: "Derived", color: getNodeColor("derived") }, 
        isahod: { category: "Derivation", label: "Derived", color: getNodeColor("derived") }, 
        affix: { category: "Derivation", label: "Affix", color: getNodeColor("derived") }, // Add affix here
        // Meaning Category
        synonym: { category: "Meaning", label: "Synonym", color: getNodeColor("synonym") },
        antonym: { category: "Meaning", label: "Antonym", color: getNodeColor("antonym") },
        kasalungat: { category: "Meaning", label: "Antonym", color: getNodeColor("antonym") },
        related: { category: "Meaning", label: "Related", color: getNodeColor("related") },
        similar: { category: "Meaning", label: "Related", color: getNodeColor("related") },
        kaugnay: { category: "Meaning", label: "Related", color: getNodeColor("related") },
        kahulugan: { category: "Meaning", label: "Related", color: getNodeColor("related") },
        has_translation: { category: "Meaning", label: "Translation", color: getNodeColor("related") },
        translation_of: { category: "Meaning", label: "Translation", color: getNodeColor("related") },
        // Form Category
        variant: { category: "Form", label: "Variant", color: getNodeColor("variant") },
        spelling_variant: { category: "Form", label: "Variant", color: getNodeColor("variant") },
        regional_variant: { category: "Form", label: "Variant", color: getNodeColor("variant") },
        abbreviation: { category: "Form", label: "Variant", color: getNodeColor("variant") },
        form_of: { category: "Form", label: "Variant", color: getNodeColor("variant") },
        itapat: { category: "Form", label: "Variant", color: getNodeColor("variant") },
        atapat: { category: "Form", label: "Variant", color: getNodeColor("variant") },
        inatapat: { category: "Form", label: "Variant", color: getNodeColor("variant") },
        // Structure Category
        hypernym: { category: "Structure", label: "Taxonomic", color: getNodeColor("taxonomic") },
        hyponym: { category: "Structure", label: "Taxonomic", color: getNodeColor("taxonomic") },
        taxonomic: { category: "Structure", label: "Taxonomic", color: getNodeColor("taxonomic") }, // Keep taxonomic if needed
        meronym: { category: "Structure", label: "Components/Parts", color: getNodeColor("part_whole") },
        holonym: { category: "Structure", label: "Components/Parts", color: getNodeColor("part_whole") },
        part_whole: { category: "Structure", label: "Components/Parts", color: getNodeColor("part_whole") }, // Keep part_whole?
        component: { category: "Structure", label: "Components/Parts", color: getNodeColor("part_whole") },
        component_of: { category: "Structure", label: "Components/Parts", color: getNodeColor("part_whole") },
        // Info Category - MERGED into Meaning/Related
        usage: { category: "Info", label: "Usage / Info", color: getNodeColor("related") }, // Keep category Info, use Related color
        associated: { category: "Info", label: "Usage / Info", color: getNodeColor("related") }, 
        see_also: { category: "Info", label: "Usage / Info", color: getNodeColor("related") }, 
        compare_with: { category: "Info", label: "Usage / Info", color: getNodeColor("related") }, 
        other: { category: "Info", label: "Other", color: "#adb5bd" } // Keep Other label here for now
      };
  
      const groupedByCategoryAndLabel: Record<string, Record<string, RelationshipLabelInfo>> = {};
  
      Object.entries(typeMappings).forEach(([type, info]) => {
        if (!groupedByCategoryAndLabel[info.category]) {
          groupedByCategoryAndLabel[info.category] = {};
        }
        if (!groupedByCategoryAndLabel[info.category][info.label]) {
          groupedByCategoryAndLabel[info.category][info.label] = {
            label: info.label,
            color: info.color,
            types: []
          };
        }
        groupedByCategoryAndLabel[info.category][info.label].types.push(type);
      });
  
      // Define the order of categories in the legend
      const categoryOrder = ["Core", "Origin", "Derivation", "Meaning", "Form", "Structure", "Info"]; // Keep Info here
  
      const finalCategories: Array<{ name: string; labels: RelationshipLabelInfo[] }> = categoryOrder
        .filter(categoryName => categoryName !== 'Info') // Filter out the Info category HERE
        .filter(categoryName => groupedByCategoryAndLabel[categoryName])
        .map(categoryName => ({
          name: categoryName,
          labels: Object.values(groupedByCategoryAndLabel[categoryName])
        }));
  
      return {
        uniqueTypes: typeMappings,
        categories: finalCategories
      };
    }, []);
    
    // Memoize base links processing to avoid recomputing on every render
    const baseLinks = useMemo((): CustomLink[] => { // Explicit return type
      if (!wordNetwork?.nodes || !wordNetwork.links) return [];
      console.log("[BASE] Processing wordNetwork links:", wordNetwork.links.length);
      const links = wordNetwork.links
        .map(link => {
          // Ensure source and target are strings (IDs) if they are objects
          const sourceId = typeof link.source === 'object' && link.source !== null ? link.source.id : link.source;
          const targetId = typeof link.target === 'object' && link.target !== null ? link.target.id : link.target;
          
          // Validate that sourceId and targetId are defined and are strings
          if (typeof sourceId !== 'string' || typeof targetId !== 'string') {
              console.warn("Invalid link structure detected, skipping:", link);
              return null; // Skip this link if structure is invalid
          }
          
          return {
            source: sourceId,
            target: targetId,
            relationship: link.relationship,
            metadata: link.metadata // Keep metadata if it exists
          } as CustomLink; // Cast to CustomLink here
        })
        // Filter out nulls introduced by mapping invalid links
        .filter((link): link is CustomLink => link !== null); 
        
      console.log("[BASE] Processed links (after filtering nulls):", links.length);
      if (links.length > 0) {
        console.log("[BASE] Sample links:");
        links.slice(0, 5).forEach(link => {
          // No need for null check here as links array is filtered
          console.log(`  ${link.source} -[${link.relationship}${link.metadata ? ' (meta)' : ''}]-> ${link.target}`); 
        });
      }
      return links; // Return the filtered array
    }, [wordNetwork]);
  
    // Fix the createLinks function to match the old_src_2 implementation exactly
    const createLinks = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>, linksData: CustomLink[]) => {
      // Draw links first (behind nodes and labels)
      const linkGroup = g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(linksData, (d: any) => `${(typeof d.source === 'object' ? d.source.id : d.source)}_${(typeof d.target === 'object' ? d.target.id : d.target)}`)
        .join(
          enter => enter.append("line")
            .attr("class", "link")
            .attr("stroke", themeMode === "dark" ? "#666" : "#ccc") // Consistent neutral color
            .attr("stroke-opacity", 0) // Start transparent
            .attr("stroke-width", 1.5) // Consistent width
            .attr("stroke-linecap", "round")
            .attr("x1", d => (typeof d.source === 'object' ? d.source.x ?? 0 : 0))
            .attr("y1", d => (typeof d.source === 'object' ? d.source.y ?? 0 : 0))
            .attr("x2", d => (typeof d.target === 'object' ? d.target.x ?? 0 : 0))
            .attr("y2", d => (typeof d.target === 'object' ? d.target.y ?? 0 : 0))
            // Add title element for link tooltip
            .call(enter => enter.append("title").text((d: CustomLink) => d.relationship))
            .call(enter => enter.transition().duration(300).attr("stroke-opacity", 0.6)), // Default opacity slightly higher
          update => update
            // Ensure updates reset to default style before transitions
            .attr("stroke", themeMode === "dark" ? "#666" : "#ccc")
            .attr("stroke-width", 1.5)
            .call(update => update.transition().duration(300)
                  .attr("stroke-opacity", 0.6)),
          exit => exit
            .call(exit => exit.transition().duration(300).attr("stroke-opacity", 0))
            .remove()
        );
      return linkGroup;
    }, [themeMode]);
  
    // Fix baseNodes calculation to exactly match old_src_2 implementation
    const baseNodes = useMemo<CustomNode[]>(() => {
      // Ensure wordNetwork and mainWord exist before proceeding
      if (!wordNetwork?.nodes || !mainWord) {
        return []; // Return empty array if prerequisites are missing
      }
  
      console.log("[BASE] Processing wordNetwork nodes:", wordNetwork.nodes.length);
      console.log("[BASE] Main word:", mainWord);
  
      const mappedNodes = wordNetwork.nodes.map((node: ImportedNetworkNode) => {
        let calculatedGroup = 'related'; // Default group to 'related' (light blue) instead of 'associated'
        let relationshipToMainWord: string | undefined = undefined;
        
        if (node.label === mainWord) {
          calculatedGroup = 'main';
          relationshipToMainWord = 'main';
        } else {
          // Find direct connection to main word first using baseLinks
          const connectingLink = baseLinks.find(link => // Use baseLinks
            (link.source === mainWord && link.target === node.label) ||
            (link.source === node.label && link.target === mainWord)
          );
          
          // If direct connection exists, use its relationship type to determine group
          if (connectingLink) { // Check if connectingLink was found
            relationshipToMainWord = connectingLink.relationship;
            // Use mapRelationshipToGroup to get the correct group name
            calculatedGroup = mapRelationshipToGroup(connectingLink.relationship);
            // More detailed log
            console.log(`[BASE] Node ${node.label}: Found direct link [${connectingLink.relationship}]. Assigned group: ${calculatedGroup}`);
          } else {
            // Explicitly keep 'related' if no direct link is found
            calculatedGroup = 'related';
            console.log(`[BASE] Node ${node.label}: No direct link to main word found. Assigned group: ${calculatedGroup}`);
          }
        }
        
        // Count connections for potential sizing later using baseLinks
        const connections = baseLinks.filter(l => l.source === node.label || l.target === node.label).length; // Use baseLinks
  
        // Create the node with proper attributes
        return {
          id: node.label,
          word: node.word || node.label,
          label: node.label,
          group: calculatedGroup, // Ensure the determined group is used
          connections: connections, // Store connection count
          relationshipToMain: relationshipToMainWord,
          pathToMain: node.label === mainWord ? [node.label] : undefined,
              pinned: false,
          originalId: node.id,
              language: node.language || undefined,
          definitions: (node as any).definitions?.map((def: any) => def.text || def.definition_text).filter(Boolean) || [],
              has_baybayin: node.has_baybayin || false,
              baybayin_form: node.baybayin_form || null,
          index: undefined, x: undefined, y: undefined, vx: undefined, vy: undefined, fx: undefined, fy: undefined
        };
      });
  
      // Filter out duplicate nodes based on id (label), keeping the first occurrence
      const uniqueNodes: CustomNode[] = [];
      const seenIds = new Set<string>();
      for (const node of mappedNodes) {
        if (!seenIds.has(node.id)) {
          uniqueNodes.push(node);
          seenIds.add(node.id);
        }
      }
      console.log("[BASE] Final unique nodes:", uniqueNodes.length);
      
      if (uniqueNodes.length > 0) {
        console.log("[BASE] Sample nodes:");
        uniqueNodes.slice(0, 5).forEach(node => {
          console.log(`  ${node.id} (${node.group}) - relationshipToMain: ${node.relationshipToMain}, connections: ${node.connections}`);
        });
      }
      
      return uniqueNodes;
    }, [wordNetwork, mainWord, baseLinks, mapRelationshipToGroup]); // Use baseLinks dependency
  
    // Add the getNodeRadius function that was missing from the setupNodeInteractions dependency array
    const getNodeRadius = useCallback((node: CustomNode) => {
      // Exact sizing from old_src_2 implementation
      if (node.id === mainWord) return 22; // Slightly larger main node
      if (node.group === 'root') return 17;
      return 14; // Standard size for other nodes
    }, [mainWord]);
  
    // Create a map from filteredNodes for quick lookups (used in setupNodeInteractions)
    const nodeMap = useMemo(() => {
      return new Map(baseNodes.map(n => [n.id, n]));
    }, [baseNodes]);
  
    // Add a ref for tracking drag start time
    const dragStartTimeRef = useRef<number>(0);
  
    // Function to update node and link positions on tick
    const ticked = useCallback(() => {
      if (!svgRef.current) return;
        
      const svg = d3.select(svgRef.current);
      const nodeSelection = svg.selectAll<SVGGElement, CustomNode>(".node");
      const linkSelection = svg.selectAll<SVGLineElement, CustomLink>(".link");
      const labelSelection = svg.selectAll<SVGTextElement, CustomNode>(".node-label"); // Select external labels
  
      // Update node group positions with safety checks
      nodeSelection.attr("transform", d => 
          (typeof d.x === 'number' && typeof d.y === 'number') 
          ? `translate(${d.x}, ${d.y})` 
          : `translate(0, 0)` // Default position if coords are invalid
      );
  
      // Update link line coordinates with safety checks
      linkSelection
          .attr("x1", d => (typeof d.source === 'object' && typeof d.source.x === 'number') ? d.source.x : 0)
          .attr("y1", d => (typeof d.source === 'object' && typeof d.source.y === 'number') ? d.source.y : 0)
          .attr("x2", d => (typeof d.target === 'object' && typeof d.target.x === 'number') ? d.target.x : 0)
          .attr("y2", d => (typeof d.target === 'object' && typeof d.target.y === 'number') ? d.target.y : 0);
  
      // Update external text label positions with safety checks
      labelSelection
          .attr("x", d => typeof d.x === 'number' ? d.x : 0)
          .attr("y", d => (typeof d.y === 'number' ? d.y : 0) + getNodeRadius(d) + 12); // Adjust offset as needed
    }, [getNodeRadius]);
  
    // Update node colors to match the old_src_2 visualization
    const setupNodeInteractions = useCallback((nodeSelection: d3.Selection<d3.BaseType, CustomNode, SVGGElement, unknown>) => {
      // Clear all event handlers first
      nodeSelection.on(".click", null)
                  .on(".dblclick", null)
                  .on(".contextmenu", null) // Clear context menu too
                  .on("mousedown", null)
                  .on("mouseup", null)
                  .on("mouseenter", null)
                  .on("mouseleave", null)
                  .on("mouseover", null)
                  .on("mouseout", null);
  
      // --- Peek Interaction (Right-Click) ---
      nodeSelection.on("contextmenu", (event: MouseEvent, d: CustomNode) => {
        event.preventDefault(); // Prevent browser context menu
        console.log(`Peek triggered for node: ${d.word}`);
        // Use clientX/clientY for viewport coordinates
        const xPos = event.clientX;
        const yPos = event.clientY;
        setPeekedNode({ node: d, x: xPos, y: yPos });
        // Also hide the main tooltip if it was visible
        setHoveredNode(null);
      });
      // --- End Peek Interaction ---
  
      // Enhanced hover effect - EXACT match to the latest screenshots + Path Highlighting
      nodeSelection.on("mouseenter", (event, d) => {
        const currentTargetElement = event.currentTarget as SVGGElement;
        if (isDraggingRef.current || !mainWord) return; // Exit if dragging or no mainWord
        setPeekedNode(null);
  
        // --- Find Path to Main Word (BFS Backwards) --- 
        const pathNodeIds = new Set<string>();
        const pathLinkIds = new Set<string>(); // Store link IDs (sourceId_targetId)
        const queue: [string, string[]][] = [[d.id, [d.id]]]; // [currentNodeId, currentPath]
        const visited = new Set<string>([d.id]);
        let foundPath = false;
  
        if (d.id !== mainWord) { // No need to search if hovering main word
          while (queue.length > 0 && !foundPath) {
            const [currentId, currentPath] = queue.shift()!;
  
            // Find links connected TO the current node using baseLinks
            const incomingLinks = baseLinks.filter(l => { // Use baseLinks
              const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
              return targetId === currentId;
            });
            
            // Also check links FROM the current node (in case of bidirectional) using baseLinks
            const outgoingLinks = baseLinks.filter(l => { // Use baseLinks
               const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
              return sourceId === currentId;
            });
            
            const potentialLinks = [...incomingLinks, ...outgoingLinks];
  
            for (const link of potentialLinks) {
              const sourceId = typeof link.source === 'object' ? (link.source as CustomNode).id : link.source as string;
              const targetId = typeof link.target === 'object' ? (link.target as CustomNode).id : link.target as string;
              const neighborId = sourceId === currentId ? targetId : sourceId;
  
              if (!visited.has(neighborId)) {
                  visited.add(neighborId);
                const newPath = [...currentPath, neighborId];
                const linkId = `${sourceId}_${targetId}`;
                const reverseLinkId = `${targetId}_${sourceId}`;
  
                if (neighborId === mainWord) {
                  // Path found!
                  newPath.forEach(id => pathNodeIds.add(id));
                  // Add links along the path
                  for(let i = 0; i < newPath.length - 1; i++) {
                    const pathLinkId = `${newPath[i]}_${newPath[i+1]}`;
                    const pathReverseLinkId = `${newPath[i+1]}_${newPath[i]}`;
                    // Find the actual link and add its ID using baseLinks
                     const actualLink = baseLinks.find(fl => // Use baseLinks
                         (`${(fl.source as any).id || fl.source}_${(fl.target as any).id || fl.target}` === pathLinkId) || 
                         (`${(fl.source as any).id || fl.source}_${(fl.target as any).id || fl.target}` === pathReverseLinkId)
                     );
                     if (actualLink) {
                         const actualSourceId = (actualLink.source as any).id || actualLink.source;
                         const actualTargetId = (actualLink.target as any).id || actualLink.target;
                         pathLinkIds.add(`${actualSourceId}_${actualTargetId}`);
                     }
                  }
                  foundPath = true;
                  break; // Exit inner loop
                }
                queue.push([neighborId, newPath]);
              }
            }
          }
          } else {
          pathNodeIds.add(mainWord); // Path for main word is just itself
        }
        // --- End Path Finding --- 
  
        // Find direct connections (neighbors) using baseLinks
        const directNeighborIds = new Set<string>();
        baseLinks.forEach(l => { // Use baseLinks (Already Corrected)
            const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
            const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
            if (sourceId === d.id) directNeighborIds.add(targetId);
            if (targetId === d.id) directNeighborIds.add(sourceId);
        });
  
        // Combine path nodes and direct neighbors for highlighting (manual combine for TS compatibility)
        const highlightNodeIds = new Set<string>([d.id]);
        pathNodeIds.forEach(id => highlightNodeIds.add(id));
        directNeighborIds.forEach(id => highlightNodeIds.add(id));
  
        // --- Dim non-highlighted elements more subtly --- 
        d3.selectAll<SVGGElement, CustomNode>(".node")
          .filter(n => !highlightNodeIds.has(n.id))
          .transition().duration(250)
          .style("opacity", 0.7); // More subtle dimming
  
        d3.selectAll<SVGLineElement, CustomLink>(".link")
          .filter((l: CustomLink) => {
            const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
            const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
            const linkId = `${sourceId}_${targetId}`;
            // Dim if NOT directly connected to hovered OR part of the path
            return !( (sourceId === d.id && directNeighborIds.has(targetId)) || 
                      (targetId === d.id && directNeighborIds.has(sourceId)) ||
                      pathLinkIds.has(linkId) || pathLinkIds.has(`${targetId}_${sourceId}`) ); 
          })
          .transition().duration(250)
          .style("stroke-opacity", 0.5); // More subtle dimming
  
        d3.selectAll<SVGTextElement, CustomNode>(".node-label")
          .filter(n => !highlightNodeIds.has(n.id))
          .transition().duration(250)
          .style("opacity", 0.7); // More subtle dimming
  
        // --- Highlight elements in the path and direct connections --- 
        // Nodes
        d3.selectAll<SVGGElement, CustomNode>(".node")
          .filter(n => highlightNodeIds.has(n.id))
          .raise()
          .transition().duration(300) // Slightly longer for smoother effect
          .style("opacity", 1)
          // Apply more natural highlighting without scale - better for UX
          .attr("transform", n => {
              return `translate(${n.x || 0},${n.y || 0})`;
          })
          .select("circle")
            // Apply enhanced shine effect on hover - improved for flat design
            .attr("stroke-width", n => {
              if (n.id === d.id) return 1.5; // Slightly thicker for hovered
              if (n.id === mainWord) return 1; // Medium for main word
              return 0.5; // Thin for other connected nodes
            })
            .attr("stroke-opacity", n => n.id === d.id ? 0.9 : 0.7) // Improved opacity
            .attr("stroke", n => {
              const baseColor = d3.color(getNodeColor(n.group)) || d3.rgb("#888");
              // Color adjustments for better visibility
              if (n.id === d.id) {
                return baseColor.brighter(0.8).toString(); // Brighter for hovered
              } else if (n.id === mainWord) {
                return baseColor.brighter(0.5).toString(); // Medium for main
              } else {
                // More subtle color adjustment for connected nodes
                return themeMode === 'dark' 
                  ? baseColor.brighter(0.3).toString() 
                  : baseColor.brighter(0.4).toString();
              }
            })
            // Improved brightness adjustment - more subtle and flat
            .attr("filter", n => {
              if (n.id === d.id) {
                // Enhanced brightness but no drop shadow for flat design
                return `url(#apple-node-shadow) brightness(1.18)`;
              } else if (pathNodeIds.has(n.id)) {
                // Highlight nodes in path slightly
                return `url(#apple-node-shadow) brightness(1.12)`;
              } else if (n.id === mainWord) {
                return `url(#apple-node-shadow) brightness(1.1)`;
              } else {
                return `url(#apple-node-shadow)`;
              }
            });
  
        // Links (Directly connected OR on the path)
        d3.selectAll<SVGLineElement, CustomLink>(".link")
          .filter((l: CustomLink) => { 
            const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
            const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
            const linkId = `${sourceId}_${targetId}`;
            return ( (sourceId === d.id && directNeighborIds.has(targetId)) || 
                     (targetId === d.id && directNeighborIds.has(sourceId)) ||
                     pathLinkIds.has(linkId) || pathLinkIds.has(`${targetId}_${sourceId}`) );
          })
          .raise()
          .transition().duration(200)
          .style("stroke-opacity", 0.9) // Highlighted link opacity
          .attr("stroke-width", 2.5)
          // Apply color using .each
          .each(function(l: CustomLink) {
            const sourceId = typeof l.source === 'object' ? (l.source as CustomNode).id : l.source as string;
            const targetId = typeof l.target === 'object' ? (l.target as CustomNode).id : l.target as string;
            const linkId = `${sourceId}_${targetId}`;
            const reverseLinkId = `${targetId}_${sourceId}`;
            let connectedNodeId: string;
  
            if (pathLinkIds.has(linkId) || pathLinkIds.has(reverseLinkId)) {
              // Simple approach: Color based on non-hovered node.
              connectedNodeId = sourceId === d.id ? targetId : sourceId;
        } else {
              // Standard coloring for direct neighbors not on path
              connectedNodeId = sourceId === d.id ? targetId : sourceId;
            }
            const connectedNode = nodeMap.get(connectedNodeId);
            const color = connectedNode ? getNodeColor(connectedNode.group) : (themeMode === "dark" ? "#aaa" : "#666");
            d3.select(this).style("stroke", color);
          });
  
        // Labels (Directly connected OR on the path)
        d3.selectAll<SVGTextElement, CustomNode>(".node-label")
          .filter(n => highlightNodeIds.has(n.id))
          .transition().duration(200)
          .style("opacity", 1)
          .style("font-weight", "bold");
      });
  
      nodeSelection.on("mouseleave", (event, d) => {
        // Reset logic needs to ensure stroke width/opacity/transform/color are reset correctly based on theme
         const currentTargetElement = event.currentTarget as SVGGElement;
        if (isDraggingRef.current) return;
  
        // Reset all nodes, links, and labels to default appearance with transitions
        d3.selectAll<SVGGElement, CustomNode>(".node")
          .transition().duration(200) // Add transition for smooth reset
          .style("opacity", n => n.id === mainWord ? 1 : 0.8) 
          .attr("transform", n => `translate(${n.x || 0},${n.y || 0})`) // Ensure transform is reset
          .select("circle")
            .attr("stroke-width", 0.5) // Reset to thin highlight ring
            .attr("stroke-opacity", 0.6) // Reset to subtle opacity
            .attr("filter", n => n.id === mainWord ? `url(#apple-node-shadow) brightness(1.15)` : `url(#apple-node-shadow)`) // Reset filter
            .attr("stroke", n => { // Reset stroke color to highlight
              const baseColor = d3.color(getNodeColor(n.group)) || d3.rgb("#888");
              return themeMode === 'dark' 
                ? baseColor.brighter(0.3).toString()
                : baseColor.brighter(0.5).toString();
            }); 
            
        d3.selectAll<SVGLineElement, CustomLink>(".link")
          .transition().duration(200) // Add transition for smooth reset
          .style("stroke-opacity", 0.6)
          .attr("stroke-width", 1.5)
          .style("stroke", themeMode === "dark" ? "#666" : "#ccc"); // Reset stroke color using style
            
        d3.selectAll<SVGTextElement, CustomNode>(".node-label")
          .transition().duration(200) // Add transition for smooth reset
          .style("opacity", 0.9) // Reset opacity
          .style("font-weight", n => n.id === mainWord ? "bold" : "normal"); // Reset font weight
      });
      
      // Double-click to navigate with improved visual feedback
      nodeSelection.on("dblclick", (event, d) => {
        const currentTargetElement = event.currentTarget as SVGGElement;
        event.preventDefault();
        event.stopPropagation();
        
        if (isDraggingRef.current) return;
        // Dismiss peek card on double click
        setPeekedNode(null);
        
        // Enhanced visual feedback for navigation
        const nodeGroup = d3.select(currentTargetElement);
        const circleElement = nodeGroup.select("circle");
        const textLabel = d3.select(`.node-label[data-id="${d.id}"]`);
        
        // Add a ripple effect
        const radius = parseFloat(circleElement.attr("r"));
        const ripple = nodeGroup.append("circle")
          .attr("class", "ripple")
          .attr("r", radius)
          .attr("fill", "none")
          .attr("stroke", (d3.color(getNodeColor(d.group))?.brighter(0.8).toString() || "#fff"))
          .attr("stroke-width", 2)
          .attr("opacity", 1);
          
        // Animate ripple
        ripple.transition()
          .duration(400)
          .attr("r", radius * 2.5)
          .attr("opacity", 0)
          .remove();
          
        // Pulse the node itself
        circleElement
          .transition()
          .duration(200)
          .attr("fill-opacity", 0.8)
          .attr("r", radius * 1.2)
          .transition()
          .duration(200)
          .attr("fill-opacity", 1)
          .attr("r", radius);
          
        // Highlight text briefly
        if (textLabel.size() > 0) {
          textLabel
            .transition()
            .duration(200)
            .style("font-weight", "bold")
            .style("opacity", 1);
        }
        
        // Show a brief toast notification to indicate navigation
        const toast = d3.select(svgRef.current?.parentNode as HTMLElement)
          .append("div")
          .attr("class", "navigation-toast")
          .style("position", "absolute")
          .style("bottom", "20px")
          .style("left", "50%")
          .style("transform", "translateX(-50%)")
          .style("background", themeMode === "dark" ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.9)")
          .style("color", themeMode === "dark" ? "#fff" : "#333")
          .style("padding", "8px 16px")
          .style("border-radius", "20px")
          .style("font-size", "13px")
          .style("box-shadow", "0 2px 8px rgba(0,0,0,0.2)")
          .style("opacity", "0")
          .style("z-index", "1000")
          .text(`Navigating to "${d.word}"...`);
        
        // Animate toast
        toast.transition()
          .duration(200)
          .style("opacity", "1")
          .transition()
          .delay(600)
          .duration(200)
          .style("opacity", "0")
          .remove();
        
        // Navigate after visual feedback
        setTimeout(() => {
          console.log(`Double-click on node: ${d.word} - Navigating`);
          if (onNodeClick) {
            onNodeClick(d.word);
          }
        }, 350); // Slightly delayed but feels more responsive
      });
      
      // Tooltip display logic (remains arrow function)
      nodeSelection.on("mouseover", (event, d) => {
          if (isDraggingRef.current) return;
  
        // Check if this is directly connected to main word using baseLinks
        let relationshipToMain = ""; // Initialize
        if (d.id !== mainWord) {
            const link = baseLinks.find(l => { // Use baseLinks (Already Corrected)
                return (l.source === mainWord && l.target === d.id) || 
                       (l.source === d.id && l.target === mainWord);
            });
            if (link) { // Check if a link was found
                relationshipToMain = link.relationship;
            }
        }
        
        // Pass relationship to tooltip only if peek isn't active
        if (!peekedNode || peekedNode.node.id !== d.id) {
             setHoveredNode({ 
                 ...d, 
                 relationshipToMain 
             });
             
             // Set tooltip position
             const { clientX, clientY } = event;
             setTooltipPosition({ x: clientX, y: clientY });
        }
      });
      
      nodeSelection.on("mouseout", (event, d) => {
          if (isDraggingRef.current) return;
          setHoveredNode(null);
      });
  
    }, [mainWord, onNodeClick, getNodeColor, themeMode, nodeMap, baseLinks, getNodeRadius, peekedNode]);
  
    // Function to create nodes in D3 - adjusted for exact styling 
    const createNodes = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>, nodesData: CustomNode[], simulation: d3.Simulation<CustomNode, CustomLink>) => {
      // Create node group
      const nodeGroup = g.append("g")
        .attr("class", "nodes")
        .selectAll("g.node") // More specific selector
        .data(nodesData, d => (d as CustomNode).id)
        .join(
            enter => {
                const nodeGroup = enter.append("g")
                    .attr("class", d => `node node-group-${d.group} ${d.id === mainWord ? "main-node" : ""}`)
                    .attr("data-id", d => d.id)
                    .style("opacity", 0); // Start transparent
  
                // Create defs for gradients if they don't exist
                if (!d3.select(svgRef.current).select("defs").size()) {
                  d3.select(svgRef.current).append("defs");
                }
                
                // Create very subtle flat shadow filter
                const shadowId = "apple-node-shadow";
                if (!d3.select(svgRef.current).select(`#${shadowId}`).size()) {
                  const filter = d3.select(svgRef.current).select("defs").append("filter")
                    .attr("id", shadowId)
                    .attr("x", "-50%")
                    .attr("y", "-50%")
                    .attr("width", "200%")
                    .attr("height", "200%");
                  
                  filter.append("feGaussianBlur")
                    .attr("in", "SourceAlpha")
                    .attr("stdDeviation", 0.8)
                    .attr("result", "blur");
                  
                  filter.append("feOffset")
                    .attr("in", "blur")
                    .attr("dx", 0)
                    .attr("dy", 0.3)
                    .attr("result", "offsetBlur");
                  
                  const feMerge = filter.append("feMerge");
                  feMerge.append("feMergeNode")
                    .attr("in", "offsetBlur");
                  feMerge.append("feMergeNode")
                    .attr("in", "SourceGraphic");
                }
                
                // Apply the nodes with simplified styling (from old_src_2)
                nodeGroup.append("circle")
                  .attr("r", getNodeRadius)
        .attr("fill", d => getNodeColor(d.group))
                  .attr("fill-opacity", 1)
                    .attr("stroke", d => d3.color(getNodeColor(d.group))?.darker(0.8).formatHex() ?? "#888")
                  .attr("stroke-width", d => d.id === mainWord ? 2.5 : 1.5)
                  .attr("stroke-opacity", 0.7)
              .attr("shape-rendering", "geometricPrecision");
  
                nodeGroup.call(enter => enter.transition().duration(300).style("opacity", d => d.id === mainWord ? 1 : 0.8));
                return nodeGroup;
            },
            update => update,
            exit => exit
                .call(exit => exit.transition().duration(300).style("opacity", 0))
                .remove()
        );
  
      // External Labels (rendered separately)
      const labelGroup = g.append("g")
          .attr("class", "labels")
        .selectAll("text.node-label")
        .data(nodesData, d => (d as CustomNode).id)
          .join(
              enter => {
                  const textElement = enter.append("text")
                      .attr("class", "node-label")
                      .attr("data-id", d => d.id)
        .attr("text-anchor", "middle")
                      .attr("font-size", d => d.id === mainWord ? "12px" : "10px") // Match the screenshot font sizes
                      .attr("font-weight", d => d.id === mainWord ? "bold" : "normal") // Match the screenshot font weights
        .text(d => d.word)
              .attr("x", d => d.x ?? 0)
                      .attr("y", d => (d.y ?? 0) + getNodeRadius(d) + 12)
                      .style("opacity", 0) // Start transparent
                      .style("pointer-events", "none") // Prevent blocking interactions
                      .style("user-select", "none");
  
                  // Halo for contrast against background - matches the screenshots
                  textElement.clone(true)
                      .lower()
                      .attr("fill", "none")
                      .attr("stroke", themeMode === "dark" ? "rgba(0,0,0,0.8)" : "rgba(255,255,255,0.8)")
                      .attr("stroke-width", 3)
                      .attr("stroke-linejoin", "round");
  
                  // Set main text fill color based on theme - matches the screenshots
                  textElement.attr("fill", themeMode === "dark" ? "#eee" : "#222");
  
                  textElement.call(enter => enter.transition().duration(300).style("opacity", d => d.id === mainWord ? 1 : 0.9));
                  return textElement;
              },
              update => update,
              exit => exit
                  .call(exit => exit.transition().duration(300).style("opacity", 0))
                  .remove()
          );
  
      // Setup drag behavior per old_src_2
      // Create custom drag behavior functions
      function dragStarted(this: SVGGElement, event: d3.D3DragEvent<SVGGElement, CustomNode, any>, d: CustomNode) {
        // Record drag start time to distinguish from clicks
        dragStartTimeRef.current = Date.now();
        
        // Dismiss peek card on drag start
        setPeekedNode(null);
        
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        isDraggingRef.current = true;
        
        // Mark as dragging - visual feedback
        d3.select(this)
          .classed("dragging", true)
          .select("circle")
          .attr("stroke-dasharray", "3,2");
      }
  
      function dragged(this: SVGGElement, event: d3.D3DragEvent<SVGGElement, CustomNode, any>, d: CustomNode) {
        // Just update fixed position and let the simulation handle the rest
        d.fx = event.x;
        d.fy = event.y;
      }
  
      function dragEnded(this: SVGGElement, event: d3.D3DragEvent<SVGGElement, CustomNode, any>, d: CustomNode) {
        // Calculate how long the drag lasted
        const dragEndTime = Date.now();
        const dragDuration = dragEndTime - dragStartTimeRef.current;
        
        if (!event.active) simulation.alphaTarget(0);
        
        // Release position constraint for non-main nodes
        if (d.id !== mainWord) {
          d.fx = null;
          d.fy = null;
        }
        
        // Reset visual state
        d3.select(this)
          .classed("dragging", false)
          .select("circle")
          .attr("stroke-dasharray", null);
        
        // If drag was very short, don't interfere with click events
        if (dragDuration < 150) {
          isDraggingRef.current = false;
        } else {
          // For longer drags, delay clearing the flag
          setTimeout(() => {
            isDraggingRef.current = false;
          }, 150);
        }
      }
      
      // Define a D3 drag behavior with the functions
      const drag = d3.drag<SVGGElement, CustomNode>()
        .filter(event => !event.ctrlKey && event.button === 0) // Only primary mouse button, no CTRL
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded);
      
      // Apply drag to node group
      (nodeGroup as any).call(drag);
      
      // Apply other node interactions
      setupNodeInteractions(nodeGroup as any);
      
      // Handle tick updates for node and label positions
      simulation.on("tick", ticked);
      
      return nodeGroup;
    }, [mainWord, getNodeRadius, getNodeColor, themeMode, setupNodeInteractions, ticked, svgRef, dragStartTimeRef, setPeekedNode]);
  
    // Add improved zoom setup function with better UX
    const setupZoom = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
      const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
      const width = containerRect ? containerRect.width : 800;
      const height = containerRect ? containerRect.height : 600;
      
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4]) // Match old scaleExtent (or adjust as needed)
        .wheelDelta((event) => {
          // Smoother wheel zooming - less jerky
          return -event.deltaY * (event.ctrlKey ? 0.01 : 0.002); // Adjusted delta
        })
        .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
          // Apply smooth transform
          g.attr("transform", event.transform.toString());
          // Optionally adjust label visibility based on zoom (k = event.transform.k)
          // ... (keep existing label logic if desired)
        })
        .filter(event => !event.ctrlKey && event.button === 0); // Standard filter, allow wheel zoom
      
      svg.call(zoom)
         .on("dblclick.zoom", null); // Disable double-click zoom
         
      // Set initial transform to center the view on the (0,0) simulation center
      const initialTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
      svg.call(zoom.transform, initialTransform);
      
      return zoom;
    }, []); // Removed mainWord dependency as centering is handled separately
  
    // --- START: Dedicated Legend Rendering Function ---
    const renderOrUpdateLegend = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, width: number) => {
      if (isMobile) return; // Don't render SVG legend on mobile
  
      // Remove previous legend if it exists
      svg.select(".legend").remove();
  
      const legendPadding = 16; // Original padding
      const legendItemHeight = 24; // Original height
      const dotRadius = 4; // Original radius
      const textPadding = 12; // Original padding
      const categorySpacing = 12; // Original spacing
      const maxLabelWidth = 120;
  
      // Use theme for styling
      const bgColor = themeMode === 'dark' 
        ? 'var(--card-bg-color-elevated)' // Use var for dark, looks better
        : muiTheme.palette.background.paper; // Use MUI paper for light mode
      const textColorPrimary = muiTheme.palette.text.primary;
      const textColorSecondary = muiTheme.palette.text.secondary;
      const dividerColor = alpha(muiTheme.palette.divider, 0.5); // Use alpha for divider
  
      const legendContainer = svg.append("g").attr("class", "legend");
  
      const { categories: legendCategories } = getUniqueRelationshipGroups();
  
      // --- Text Measurement ---
      const tempText = svg.append("text")
          .style("font-family", muiTheme.typography.fontFamily || "system-ui, -apple-system, sans-serif")
          .style("font-size", muiTheme.typography.pxToRem(11))
        .style("opacity", 0);
      let maxTextWidth = 0;
      let maxCategoryWidth = 0;
      legendCategories.forEach((category) => {
        tempText.style("font-weight", 600).text(category.name);
        const categoryWidth = tempText.node()?.getBBox().width || 0;
        maxCategoryWidth = Math.max(maxCategoryWidth, categoryWidth);
        category.labels.forEach(labelInfo => {
          tempText.style("font-weight", 400).text(labelInfo.label);
          const textWidth = tempText.node()?.getBBox().width || 0;
          maxTextWidth = Math.max(maxTextWidth, Math.min(textWidth, maxLabelWidth));
        });
      });
      tempText.remove();
  
      // --- Calculate Legend Dimensions --- 
      const legendWidth = Math.max(
          maxCategoryWidth,
          maxTextWidth + dotRadius * 2 + textPadding // Dot + padding + text
      ) + (legendPadding * 2);
      legendWidthRef.current = legendWidth; // Store legend width in ref
  
      // Position container top-right 
      legendContainer.attr("transform", `translate(${width - legendWidth - 20}, 20)`);
      legendContainerRef.current = legendContainer; // Store legend container selection in ref
  
      // Calculate height dynamically 
      let calculatedHeight = legendPadding;
      calculatedHeight += 24; // Space for title
      calculatedHeight += 18; // Space for subtitle
      calculatedHeight += categorySpacing;
      legendCategories.forEach(category => {
        calculatedHeight += legendItemHeight; // Space for category header
        calculatedHeight += category.labels.length * legendItemHeight; // Space for items
        calculatedHeight += categorySpacing;
      });
      calculatedHeight += legendPadding;
      const legendHeight = calculatedHeight - categorySpacing;
  
      // --- Render Legend Elements --- 
      // Background Rectangle
      legendContainer.append("rect")
        .attr("width", legendWidth).attr("height", legendHeight)
        .attr("rx", 10).attr("ry", 10)
        .attr("fill", bgColor)
        .attr("stroke", dividerColor).attr("stroke-width", 0.5);
  
      // Title & Subtitle
      legendContainer.append("text") // Title
        .attr("x", legendWidth / 2).attr("y", legendPadding + 10).attr("text-anchor", "middle")
        .style("font-size", muiTheme.typography.pxToRem(13)).style("font-weight", 600)
        .attr("fill", textColorPrimary).text("Relationship Types");
      legendContainer.append("text") // Subtitle
        .attr("x", legendWidth / 2).attr("y", legendPadding + 26).attr("text-anchor", "middle")
        .style("font-size", muiTheme.typography.pxToRem(10)).attr("fill", textColorSecondary)
        .text("Click to filter");
  
      let yPos = legendPadding + 40 + categorySpacing;
  
      legendCategories.forEach((category) => {
        // Category Header
        legendContainer.append("text")
          .attr("x", legendPadding).attr("y", yPos + legendItemHeight / 2).attr("dy", ".35em")
          .style("font-weight", 600).style("font-size", muiTheme.typography.pxToRem(11))
          .attr("fill", textColorPrimary).text(category.name);
        yPos += legendItemHeight;
  
        // Category Items (Labels)
        category.labels.forEach(labelInfo => {
          const allOriginalTypesFiltered = labelInfo.types.every(t =>
            filteredRelationships.includes(t.toLowerCase())
          );
          const itemOpacity = allOriginalTypesFiltered ? 0.5 : 1;
  
          const entry = legendContainer.append("g")
            .attr("transform", `translate(${legendPadding}, ${yPos + legendItemHeight / 2})`)
            .attr("class", "legend-item").style("cursor", "pointer").style("opacity", itemOpacity)
            .classed("filtered", allOriginalTypesFiltered)
            .on("mouseover", function(this: SVGGElement) {
              if (itemOpacity === 1) {
                  d3.select(this).select("circle").transition().duration(150).attr("r", dotRadius * 1.3);
                  d3.select(this).select("text").transition().duration(150).style("font-weight", 600);
              }
            })
            .on("mouseout", function(this: SVGGElement) {
               d3.select(this).select("circle").transition().duration(150).attr("r", dotRadius);
               d3.select(this).select("text").transition().duration(150).style("font-weight", 400);
            })
            .on("click", function(this: SVGGElement) {
              handleToggleRelationshipFilter(labelInfo.types);
            });
  
            // Click Target (Invisible Rect)
            entry.append("rect")
              .attr("x", -legendPadding / 2).attr("y", -legendItemHeight / 2)
              .attr("width", legendWidth - legendPadding).attr("height", legendItemHeight)
              .attr("fill", "transparent");
  
            // Color Dot
            entry.append("circle")
              .attr("cx", dotRadius).attr("cy", 0)
              .attr("r", dotRadius)
              .attr("fill", labelInfo.color)
              .attr("stroke", alpha(labelInfo.color, 0.5)).attr("stroke-width", 0.5);
  
            // Label Text
            entry.append("text")
              .attr("x", dotRadius * 2 + textPadding).attr("y", 0).attr("dy", ".35em")
              .style("font-size", muiTheme.typography.pxToRem(11))
              .style("font-weight", 400).attr("fill", textColorPrimary)
              .text(labelInfo.label);
  
          yPos += legendItemHeight;
        });
        yPos += categorySpacing;
      });
    }, [
        isMobile,
        themeMode,
        muiTheme,
        getUniqueRelationshipGroups,
        filteredRelationships,
        handleToggleRelationshipFilter,
        // Refs are stable, no need to include svgRef, legendContainerRef, legendWidthRef
    ]);
    // --- END: Dedicated Legend Rendering Function ---
  
    // Improved SVG dimensions setup with responsive design + Debounced Legend Update
    const setupSvgDimensions = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
      const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
      let width = containerRect ? containerRect.width : 800;
      let height = containerRect ? containerRect.height : 600;
      
      // Use SVG viewBox for responsive scaling
      svg.attr("width", "100%")
         .attr("height", "100%")
         .attr("viewBox", `0 0 ${width} ${height}`)
         .attr("preserveAspectRatio", "xMidYMid meet");
      
      // Debounced resize handler
      const handleResize = () => {
        // Clear the previous timeout if it exists
        if (resizeTimeoutRef.current) {
          clearTimeout(resizeTimeoutRef.current);
        }
  
        // Set a new timeout
        resizeTimeoutRef.current = setTimeout(() => {
            if (svgRef.current) {
              const newContainerRect = svgRef.current.parentElement?.getBoundingClientRect();
              if (newContainerRect) {
                const newWidth = newContainerRect.width;
                const newHeight = newContainerRect.height;
                // Update viewBox on resize
                svg.attr("viewBox", `0 0 ${newWidth} ${newHeight}`);
                
                // --- Rerender Legend with new width --- 
                renderOrUpdateLegend(svg, newWidth);
                // --- End Rerender Legend ---
  
                // Recenter graph after resize (optional, keep if desired)
                if (zoomRef.current && mainWord) {
                    const currentTransform = d3.zoomTransform(svgRef.current);
                    // Simple recentering logic - adjust scale if needed or keep current
                    const transform = d3.zoomIdentity
                        .translate(newWidth / 2, newHeight / 2)
                        .scale(currentTransform.k); // Keep current scale
                    d3.select(svgRef.current).transition()
                       .duration(300)
                       .call(zoomRef.current.transform, transform);
                }
              }
            }
          }, 300); // Debounce timeout (300ms)
      };
      
      // Add resize event listener
      window.addEventListener('resize', handleResize);
  
      // Return cleanup and initial dimensions
      return { 
        width, 
        height,
        cleanup: () => {
        window.removeEventListener('resize', handleResize);
          if (resizeTimeoutRef.current) {
            clearTimeout(resizeTimeoutRef.current); // Clear timeout on unmount
          }
        }
      };
    }, [mainWord, renderOrUpdateLegend]); // Add renderOrUpdateLegend dependency
  
    // Center on main word function
    const centerOnMainWord = useCallback((svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, nodesToSearch: CustomNode[]) => {
      if (!zoomRef.current || isDraggingRef.current || !mainWord) return;
      
      // Main node is now pinned at (0,0) in the simulation space
      const mainNodeData = nodesToSearch.find(n => n.id === mainWord); 
      const containerRect = svg.node()?.parentElement?.getBoundingClientRect();
      const width = containerRect ? containerRect.width : 800;
      const height = containerRect ? containerRect.height : 600;
      
      if (mainNodeData) {
        const currentTransform = d3.zoomTransform(svg.node()!); 
        // Target scale can remain based on current zoom or be reset
        const targetScale = Math.max(0.5, Math.min(2, currentTransform.k)); 
        // Target translation centers the view on simulation coords (0,0)
        const targetX = width / 2; 
        const targetY = height / 2;
        
        const newTransform = d3.zoomIdentity.translate(targetX, targetY).scale(targetScale);
        svg.transition()
           .duration(750)
           .ease(d3.easeCubicInOut)
           .call(zoomRef.current.transform, newTransform);
      } else {
        // Reset view if main word somehow not found
        const resetTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
        svg.transition()
           .duration(600)
           .ease(d3.easeCubicInOut)
           .call(zoomRef.current.transform, resetTransform);
      }
    }, [mainWord]);
  
    // Create nodes and links directly based on filter state
    const { filteredNodes, filteredLinks } = useMemo<{ filteredNodes: CustomNode[], filteredLinks: CustomLink[] }>(() => {
      if (!mainWord || baseNodes.length === 0) {
        return { filteredNodes: [], filteredLinks: [] };
      }
      
      console.log("[FILTER] Applying depth/breadth and relationship filters");
      console.log("[FILTER] Main word:", mainWord);
      console.log("[FILTER] Base nodes:", baseNodes.length);
      console.log("[FILTER] Base links:", baseLinks.length); // Use baseLinks for logging (Already Corrected)
      console.log("[FILTER] Depth limit:", depth);
      console.log("[FILTER] Breadth limit:", breadth);
      
      // First, verify if the main word exists in our node set
      const mainWordNode = baseNodes.find(n => n.id === mainWord);
      if (!mainWordNode) {
        console.error(`[FILTER] ERROR: Main word "${mainWord}" not found in baseNodes!`);
        // Return just the single mainWord node if we can construct it
        return {
          filteredNodes: [{ 
            id: mainWord, 
            word: mainWord, 
            label: mainWord, 
            group: 'main',
            index: undefined, x: undefined, y: undefined, vx: undefined, vy: undefined, fx: undefined, fy: undefined 
          }],
          filteredLinks: []
        };
      }
      
      // Step 1: First collect ALL connected nodes based on depth/breadth limits (ignoring relationship filters)
      const nodeMap = new Map(baseNodes.map(n => [n.id, n]));
      const connectedNodeIds = new Set<string>([mainWord]); // Start with main word
      const queue: [string, number][] = [[mainWord, 0]]; // [nodeId, depth]
      const visited = new Set<string>();
  
      // Log some debug information
      console.log("[FILTER] Starting BFS with mainWord:", mainWord);
      console.log("[FILTER] Node map size:", nodeMap.size);
      
      // Do BFS traversal to find all nodes within depth/breadth
      while (queue.length > 0) {
        const [currentWordId, currentDepth] = queue.shift()!;
  
        if (currentDepth >= depth) {
          continue;
        }
        
        if (visited.has(currentWordId)) {
          continue;
        }
        
        visited.add(currentWordId);
  
        // Find all links connected to this node using baseLinks
        const relatedLinks = baseLinks.filter(link => { // Use baseLinks (Already Corrected)
          const sourceId = typeof link.source === 'string' ? link.source : (link.source as any)?.id || link.source;
          const targetId = typeof link.target === 'string' ? link.target : (link.target as any)?.id || link.target;
          return sourceId === currentWordId || targetId === currentWordId;
        });
  
        // Get all connected nodes
        const relatedWordIds = relatedLinks.map(link => {
          const sourceId = typeof link.source === 'string' ? link.source : (link.source as any)?.id || link.source;
          const targetId = typeof link.target === 'string' ? link.target : (link.target as any)?.id || link.target;
          return sourceId === currentWordId ? targetId : sourceId;
        }).filter(id => !visited.has(id)); // Parameter 'id' implicitly has 'any' - Add type
        // .filter((id: string) => !visited.has(id)); // Tentative fix for implicit any
  
        // Sort nodes by relationship type for consistent breadth application
        const sortedWords = [...relatedWordIds].sort((aId, bId) => {
           const aNode = nodeMap.get(aId);
           const bNode = nodeMap.get(bId);
          
          if (!aNode) return 1;
          if (!bNode) return -1;
          
          const aGroup = aNode.group.toLowerCase();
          const bGroup = bNode.group.toLowerCase();
          
          const groupOrder = [
              'main', 'root', 'root_of', 'synonym', 'antonym', 'derived',
              'variant', 'related', 'kaugnay', 'component_of', 'cognate',
              'etymology', 'derivative', 'associated', 'other'
          ];
          
          return groupOrder.indexOf(aGroup) - groupOrder.indexOf(bGroup);
        });
  
        // Apply breadth limit and add to traversal queue
        const wordsToAdd = sortedWords.slice(0, breadth);
        
        wordsToAdd.forEach(wordId => {
           if (nodeMap.has(wordId)) {
               connectedNodeIds.add(wordId);
               queue.push([wordId, currentDepth + 1]);
          }
        });
      }
  
      // Step 2: Create lists of depth-limited nodes and links
      const depthLimitedNodes = baseNodes.filter(node => connectedNodeIds.has(node.id));
      
      // Find links where both source and target are in our depth-limited node set using baseLinks
      const depthLimitedLinks = baseLinks.map(link => { // Use baseLinks (Already Corrected)
          const sourceId = typeof link.source === 'string' ? link.source : (link.source as any)?.id || link.source;
          const targetId = typeof link.target === 'string' ? link.target : (link.target as any)?.id || link.target;
        
        return {
          source: sourceId,
          target: targetId,
          relationship: link.relationship
        } as CustomLink;
      }).filter(link => { // Filter based on nodes (parameter 'link' implicitly has 'any' - Add type)
      // }).filter((link: CustomLink) => { // Tentative fix for implicit any
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as any)?.id || link.source;
        const targetId = typeof link.target === 'string' ? link.target : (link.target as any)?.id || link.target;
        
        return connectedNodeIds.has(sourceId) && connectedNodeIds.has(targetId);
      });
      
      console.log(`[FILTER] After depth/breadth: ${depthLimitedNodes.length}/${baseNodes.length} nodes and ${depthLimitedLinks.length}/${baseLinks.length} links`);
      
      // Step 3: Apply relationship filtering if any filters are active
      if (filteredRelationships.length === 0) {
        // No relationship filters, so return all depth-limited nodes and links
        console.log(`[FILTER] No relationship filters active - showing all nodes and links`);
        return { 
          filteredNodes: depthLimitedNodes, 
          filteredLinks: depthLimitedLinks 
        };
      }
      
      // Filter nodes by relationship type
      const relationshipFilteredNodes = depthLimitedNodes.filter(node => {
        // Always include the main word node
        if (node.id === mainWord) {
          return true;
        }
        
        // Check if this node's group is in the filtered list
        const nodeGroup = node.group.toLowerCase();
        const isGroupFiltered = filteredRelationships.includes(nodeGroup);
        
        // Keep nodes whose group is NOT in the filtered list
        return !isGroupFiltered;
      });
      
      // Only include links where both source and target remain in the filtered node set
      const relationshipFilteredNodeIds = new Set(relationshipFilteredNodes.map(n => n.id));
      const relationshipFilteredLinks = depthLimitedLinks.filter(link => {
        // Ensure both nodes are still included
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as any)?.id || link.source;
        const targetId = typeof link.target === 'string' ? link.target : (link.target as any)?.id || link.target;
        
        const sourceIncluded = relationshipFilteredNodeIds.has(sourceId);
        const targetIncluded = relationshipFilteredNodeIds.has(targetId);
        
        // If either endpoint is filtered out, don't include the link
        if (!sourceIncluded || !targetIncluded) {
          return false;
        }
        
        return true;
      });
      
      console.log(`[FILTER] After relationship filtering: ${relationshipFilteredNodes.length}/${depthLimitedNodes.length} nodes and ${relationshipFilteredLinks.length}/${depthLimitedLinks.length} links`);
      
      return {
        filteredNodes: relationshipFilteredNodes,
        filteredLinks: relationshipFilteredLinks
      };
    }, [baseNodes, baseLinks, mainWord, depth, breadth, filteredRelationships]);
  
    // Define functions needed for the simulation
    const setupSimulation = useCallback((nodes: CustomNode[], links: CustomLink[], width: number, height: number) => {
        // Restore simulation parameters from old_src_2 version
        simulationRef.current = d3.forceSimulation<CustomNode>(nodes)
          .alphaDecay(0.025) // From old_src_2
          .velocityDecay(0.4) // Standard value, matches old_src_2
          .force("link", d3.forceLink<CustomNode, CustomLink>()
            .id(d => d.id)
            .links(links)
            .distance(110) // From old_src_2
            .strength(0.4)) // From old_src_2
          .force("charge", d3.forceManyBody<CustomNode>()
            .strength(-300) // From old_src_2
            .distanceMax(350)) // From old_src_2
          .force("collide", d3.forceCollide<CustomNode>()
            // Simplified collision radius from old_src_2
            .radius(d => getNodeRadius(d) + 25) 
            .strength(1.0))
          // Center force from old_src_2 - IMPORTANT CHANGE
          .force("center", d3.forceCenter(0, 0)) 
          .on("tick", ticked); // Ensure ticked is correctly assigned
  
        // Pin main word - adjust fx/fy since center is now (0,0)
        const mainNode = nodes.find(node => node.id === mainWord);
        if (mainNode) {
          mainNode.fx = 0; // Pin main word to the simulation center (0,0)
          mainNode.fy = 0;
        }
  
        // No need to pre-position nodes if main is pinned at 0,0 and using forceCenter(0,0)
  
        return simulationRef.current;
    }, [mainWord, getNodeRadius, ticked]); // Dependencies updated
  
    // --- Main graph construction useEffect ---
    useEffect(() => {
      // --- START: Error Handling & Prerequisites Check ---
      if (!svgRef.current || !wordNetwork || !mainWord) {
        if (svgRef.current) d3.select(svgRef.current).selectAll("*").remove();
        if (simulationRef.current) simulationRef.current.stop();
        setError(null); // Reset error state
        setIsValidNetwork(true); // Assume valid until proven otherwise
        return;
      }
  
      if (baseNodes.length === 0 && !isLoading) {
          if (svgRef.current) d3.select(svgRef.current).selectAll("*").remove();
          if (simulationRef.current) simulationRef.current.stop();
          setError("No nodes found for the current network."); // Specific error
          setIsValidNetwork(false);
          return; // Added return statement to exit early
      }
      
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();
      setError(null);
  
      const { width, height, cleanup } = setupSvgDimensions(svg);
      const g = svg.append("g")
        .attr("class", "graph-content")
        .attr("data-filter-key", filteredRelationships.join(','));
      gRef.current = g.node() as SVGGElement;
  
      const zoom = setupZoom(svg, g);
      zoomRef.current = zoom;
  
      const currentSim = setupSimulation(filteredNodes, filteredLinks, width, height);
  
      const linkElements = createLinks(g, filteredLinks);
      const nodeElements = createNodes(g, filteredNodes, currentSim);
      setupNodeInteractions(nodeElements);
  
      // --- Render Initial Legend --- 
      renderOrUpdateLegend(svg, width);
      // --- End Render Initial Legend ---
  
      if (currentSim) {
        const mainNodeData = filteredNodes.find(n => n.id === mainWord);
        if (mainNodeData) {
            mainNodeData.fx = 0;
            mainNodeData.fy = 0;
        }
        currentSim.alpha(1).restart();
      }
  
      const centerTimeout = setTimeout(() => centerOnMainWord(svg, filteredNodes), 800);
  
      return () => {
        if (currentSim) currentSim.stop();
        clearTimeout(centerTimeout);
        if (cleanup) cleanup(); // Call the cleanup function from setupSvgDimensions
        if (svgRef.current) {
          const svg = d3.select(svgRef.current);
          // Clean up D3 event listeners specifically
          svg.selectAll(".node").on(".drag", null).on(".click", null).on(".dblclick", null).on(".mouseover", null).on(".mouseout", null).on(".contextmenu", null);
          svg.selectAll(".graph-legend-svg .legend-item-svg").on(".click", null).on(".mouseover", null).on(".mouseout", null);
          svg.on(".zoom", null);
          console.log("[Cleanup] D3 listeners removed.");
        } else {
          console.log("[Cleanup] SVG ref not found, skipping listener removal.");
        }
      };
    }, [
      // Keep existing dependencies, ensure renderOrUpdateLegend and setupSvgDimensions are included if defined inside
      // OR ensure their own dependency arrays are correct if defined outside/with useCallback
      wordNetwork, 
      mainWord, 
      filteredNodes, 
      filteredLinks, 
      depth, 
      breadth, 
      themeMode, // Needed by renderOrUpdateLegend
      isMobile, // Needed by renderOrUpdateLegend
      baseNodes,
      isLoading,
      getUniqueRelationshipGroups, // Needed by renderOrUpdateLegend
      handleToggleRelationshipFilter, // Needed by renderOrUpdateLegend
      setupSvgDimensions, // Now depends on renderOrUpdateLegend
      setupZoom, 
      setupSimulation, 
      createLinks, 
      createNodes, 
      setupNodeInteractions, 
      ticked, 
      centerOnMainWord, 
      getNodeRadius, 
      filterUpdateKey, 
      forceUpdate,
      renderOrUpdateLegend // Added dependency
    ]);
  
    // Re-center when main word changes
    useEffect(() => {
      if (prevMainWordRef.current && prevMainWordRef.current !== mainWord && svgRef.current) {
        const recenterTimeout = setTimeout(() => {
          if (svgRef.current) centerOnMainWord(d3.select(svgRef.current), filteredNodes);
        }, 800);
        return () => clearTimeout(recenterTimeout);
      }
      prevMainWordRef.current = mainWord;
    }, [mainWord, centerOnMainWord, filteredNodes]);
  
    // Improved tooltip with relationship info directly from old_src_2
    const renderTooltip = useCallback(() => {
      // Hide tooltip if peek card is showing for the same node
      if (!hoveredNode?.id || !hoveredNode?.x || !hoveredNode?.y || !svgRef.current || (peekedNode && peekedNode.node.id === hoveredNode.id)) {
           return null;
      }
  
      const svgNode = svgRef.current;
      const transform = d3.zoomTransform(svgNode);
  
      const [screenX, screenY] = transform.apply([hoveredNode.x, hoveredNode.y]);
  
      const offsetX = (screenX > window.innerWidth / 2) ? -20 - 250 : 20;
      const offsetY = (screenY > window.innerHeight / 2) ? -20 - 80 : 20;
  
      // Find the specific link causing this relationship using baseLinks
      let posContext = "";
      const connectingLink = baseLinks.find(l => { // Use baseLinks (Already Corrected)
          return (l.source === mainWord && l.target === hoveredNode.id && l.relationship === hoveredNode.relationshipToMain) || 
                 (l.target === mainWord && l.source === hoveredNode.id && l.relationship === hoveredNode.relationshipToMain);
      });
      if (connectingLink?.metadata?.english_pos_context) { // Use optional chaining for connectingLink
          posContext = ` (as ${connectingLink.metadata.english_pos_context})`;
      }
  
      return (
        <div
          className="node-tooltip"
            style={{
            position: "absolute",
            left: `${screenX + offsetX}px`,
            top: `${screenY + offsetY}px`,
            background: themeMode === "dark" ? "rgba(30, 30, 30, 0.95)" : "rgba(250, 250, 250, 0.95)",
            border: `1.5px solid ${getNodeColor(hoveredNode.group)}`, 
            borderRadius: "8px",
            padding: "10px 14px", 
            maxWidth: "280px", 
            zIndex: 1000, // Ensure tooltip is below peek card (1100)
            pointerEvents: "none",
            fontFamily: "system-ui, -apple-system, sans-serif",
            transition: "opacity 0.15s ease-out, transform 0.15s ease-out",
            boxShadow: themeMode === "dark" ? "0 4px 15px rgba(0,0,0,0.4)" : "0 4px 15px rgba(0,0,0,0.15)",
            opacity: 1,
            transform: "translateY(0)",
            animation: "fadeInTooltip 0.2s ease-out",
          }}
        >
           <h4 style={{ margin: 0, marginBottom: '6px', color: getNodeColor(hoveredNode.group), fontSize: '15px' }}>{hoveredNode.id}</h4>
           
           {/* Relationship to main word */}
           {hoveredNode.id !== mainWord && hoveredNode.relationshipToMain && (
             <div style={{ 
               display: "flex", 
               alignItems: "center", 
               gap: "6px", 
               paddingBottom: "4px",
               background: themeMode === "dark" ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.03)",
               padding: "5px 8px",
               borderRadius: "4px",
               marginBottom: "5px"
             }}>
               <span style={{ 
                 fontSize: "11px", 
                 color: themeMode === "dark" ? "#aaa" : "#666", 
                 fontWeight: "500",
                 whiteSpace: "nowrap"
               }}>
                 {mainWord} 
                 <span style={{ margin: "0 4px", opacity: 0.7 }}>→</span> 
                 <span style={{ 
                   fontStyle: "italic", 
                   color: themeMode === "dark" ? "#ddd" : "#333",
                   fontWeight: "600" 
                 }}>
                   {hoveredNode.relationshipToMain}{posContext} { /* <<< Display POS context here */}
                 </span>
               </span>
            </div>
          )}
          
           <div style={{ display: "flex", alignItems: "center", gap: "6px", paddingBottom: "4px" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: getNodeColor(hoveredNode.group), flexShrink: 0 }}></span>
              <span style={{ fontSize: "13px", color: themeMode === 'dark' ? '#ccc' : '#555', fontWeight: "500" }}>
                  {hoveredNode.group.charAt(0).toUpperCase() + hoveredNode.group.slice(1).replace(/_/g, ' ')}
              </span>
            </div>
           {hoveredNode.definitions && hoveredNode.definitions.length > 0 && (
                <p style={{ fontSize: '12px', color: themeMode === 'dark' ? '#bbb' : '#666', margin: '6px 0 0 0', fontStyle: 'italic' }}>
                    {hoveredNode.definitions[0].length > 100 ? hoveredNode.definitions[0].substring(0, 97) + '...' : hoveredNode.definitions[0]}
            </p>
          )}
           <div style={{ 
             fontSize: "11px", 
             marginTop: "8px", 
             color: themeMode === "dark" ? "#8b949e" : "#777777",
             display: "flex",
             justifyContent: "center",
             gap: "12px",
             borderTop: themeMode === "dark" ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.06)",
             paddingTop: "6px"
           }}>
             <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
               <span style={{ 
                 fontSize: "10px", 
                 background: themeMode === "dark" ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.06)", 
                 borderRadius: "3px", 
                 padding: "1px 4px"
               }}>Double-click</span>
               <span>Navigate</span>
             </div>
             {/* Add hint for Peek interaction */}
             <div style={{ display: "flex", alignItems: "center", gap: "4px", opacity: 0.7 }}>
               <span style={{ 
                 fontSize: "10px", 
                 background: themeMode === "dark" ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.06)", 
                 borderRadius: "3px", 
                 padding: "1px 4px"
               }}>Right-click</span>
               <span>Peek</span>
             </div>
          </div>
        </div>
      );
    }, [hoveredNode, themeMode, getNodeColor, mainWord, peekedNode, svgRef, baseLinks]); // Corrected to baseLinks
  
    // Make sure to return the JSX at the end
    return (
      <div ref={containerRef} className="graph-container">
        {/* SVG Container for the Graph */}
        <div className="graph-wrapper">
          <div className="graph-svg-container">
            {isLoading && (
              <div className="loading-overlay"><div className="spinner"></div><p>Loading Network...</p></div>
            )}
            {error && (
              <div className="error-overlay">
                <p className="error-message">{error}</p>
              </div>
            )}
            {(!wordNetwork || !mainWord || filteredNodes.length === 0 && !error) && (
              <div className="empty-graph-message">Enter a word to see its network.</div>
            )}
            <svg 
              ref={svgRef} 
              className={`graph-svg ${isLoading ? 'loading' : 'loaded'}`}
              key={`graph-${mainWord}-${depth}-${breadth}-${filteredRelationships.join('.')}-${forceUpdate}-${filterUpdateKey}`}
            >
            </svg>
          </div>
        </div>
          
        {/* Bottom Controls - Slider Area (fixed at bottom) */}
              <div className="controls-container">
          {/* Zoom Controls */} 
                <div className="zoom-controls">
            <button onClick={() => { if (zoomRef.current && svgRef.current) d3.select(svgRef.current).call(zoomRef.current.scaleBy, 1.3); }} className="zoom-button" title="Zoom In">+</button>
            <button onClick={() => { if (zoomRef.current && svgRef.current) d3.select(svgRef.current).call(zoomRef.current.scaleBy, 1/1.3); }} className="zoom-button" title="Zoom Out">-</button>
            <button 
              onClick={() => { 
                if (zoomRef.current && svgRef.current) {
                  const containerRect = svgRef.current.parentElement?.getBoundingClientRect();
                  const width = containerRect ? containerRect.width : 800;
                  const height = containerRect ? containerRect.height : 600;
                  const resetTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
                  d3.select(svgRef.current).transition().duration(600).ease(d3.easeCubicInOut)
                    .call(zoomRef.current.transform, resetTransform);
                }
              }} 
              className="reset-zoom-button" 
              title="Reset View"
            >
              Reset
            </button>
                </div>
  
                <NetworkControls 
                  depth={depth}
                  breadth={breadth}
            onDepthChange={(newDepth) => {
              setDepth(newDepth);
              onNetworkChange(newDepth, breadth);
            }}
            onBreadthChange={(newBreadth) => {
              setBreadth(newBreadth); 
              onNetworkChange(depth, newBreadth);
            }}
                  className="network-controls"
                />
              </div>
              
        {/* Peek Card for Node Detail View */}
              {peekedNode && (
                <NodePeekCard 
                  peekedNodeData={peekedNode} 
                  onClose={() => setPeekedNode(null)} 
                />
              )}
              
              {/* Tooltip */}
        {renderTooltip()}
      </div>
    );
  };
  
  export default WordGraph;
  