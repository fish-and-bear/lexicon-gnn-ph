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
    id: string; // Keep as string ID from BFS calculation
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
    relationshipToMain?: string; // ADDED BACK
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
    console.log("[WordGraph] Received wordNetwork prop:", wordNetwork); // <-- ADD THIS LOG
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
      // *** ADD LOGGING HERE ***
      console.log("[BASELINKS] Recalculating baseLinks. Input wordNetwork:", wordNetwork);
      if (!wordNetwork?.nodes || !wordNetwork.links) {
        console.log("[BASELINKS] Prerequisites missing or links empty. Returning empty array.");
        return [];
      }
      console.log("[BASE] Processing wordNetwork links:", wordNetwork.links.length);
      // Log the first few raw links if available
      if (wordNetwork.links.length > 0) {
        console.log("[BASELINKS] Raw input links (first 5):", wordNetwork.links.slice(0, 5));
      }
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
  
    // Fix baseNodes calculation to use stringified numeric ID for D3
    const baseNodes = useMemo<CustomNode[]>(() => {
      // *** ADD LOGGING HERE ***
      console.log("[BASENODES] Recalculating baseNodes. Input wordNetwork:", wordNetwork);
      console.log("[BASENODES] Input mainWord:", mainWord);
      console.log("[BASENODES] Input baseLinks count:", baseLinks?.length); // Log dependency state
  
      // Ensure wordNetwork and mainWord exist before proceeding
      if (!wordNetwork?.nodes || !mainWord || !baseLinks) {
        console.log("[BASE] Prerequisites missing (wordNetwork, mainWord, or baseLinks). Returning empty nodes.");
        return []; // Return empty array if prerequisites are missing
      }
  
      console.log("[BASE] Processing wordNetwork nodes:", wordNetwork.nodes.length);
      console.log("[BASE] Main word:", mainWord);
      console.log("[BASE] Base links count:", baseLinks.length); // Log link count
  
      // Find the main word node data first to get its STRING ID
      // *** CHANGE: Use node.word for matching ***
      let mainWordNodeData = wordNetwork.nodes.find((n: ImportedNetworkNode) => n.word === mainWord);
      let mainWordIdStr: string | null = null; // Initialize as null
      let mainWordMatchMethod = "";
  
      if (mainWordNodeData) {
        mainWordIdStr = String(mainWordNodeData.id);
        mainWordMatchMethod = "word";
      } else {
         console.warn(`[BASE] Could not find mainWord node data for '${mainWord}' using node.word. Falling back to node.label.`);
         mainWordNodeData = wordNetwork.nodes.find((n: ImportedNetworkNode) => n.label === mainWord);
         if (mainWordNodeData) {
             mainWordIdStr = String(mainWordNodeData.id);
             mainWordMatchMethod = "label";
         } else {
             console.error(`[BASE] CRITICAL: Could not find mainWord node data for '${mainWord}' using word or label matching. Cannot determine relationships correctly.`);
             return []; // Cannot proceed without main word ID
         }
      }
  
      console.log(`[BASE] Found mainWordIdStr: ${mainWordIdStr} (for word: ${mainWord}, matched via node.${mainWordMatchMethod})`);
  
      const mappedNodes = wordNetwork.nodes.map((node: ImportedNetworkNode, index: number) => {
          let calculatedGroup = 'related'; // Default group
          let relationshipToMainWord: string | undefined = undefined;
          const currentNodeIdStr = String(node.id); // Current node's string ID
          let connectingLink: CustomLink | undefined = undefined; // Store the link if found
          const isMainNode = currentNodeIdStr === mainWordIdStr; // Check if this is the main node
  
          if (isMainNode) {
            calculatedGroup = 'main';
            relationshipToMainWord = 'main';
            // Log main node identification
            if (index < 20 || isMainNode) { // Log more nodes initially or if it's the main word
                console.log(`[BASE MAP #${index}] Node: '${node.word || node.label}' (ID: ${currentNodeIdStr}) - IDENTIFIED AS MAIN WORD. Group: ${calculatedGroup}`);
            }
          } else if (mainWordIdStr) { // Only search for links if mainWordId was found and it's not the main node
            // Find direct connection using STRING IDs
            connectingLink = baseLinks.find(link => {
              // Explicitly get string IDs from link source/target, ensuring they are treated as strings
              const sourceId = typeof link.source === 'object' && link.source !== null ? String(link.source.id) : String(link.source);
              const targetId = typeof link.target === 'object' && link.target !== null ? String(link.target.id) : String(link.target);
              return (sourceId === mainWordIdStr && targetId === currentNodeIdStr) ||
                     (targetId === mainWordIdStr && sourceId === currentNodeIdStr);
            });
  
            if (connectingLink) {
              relationshipToMainWord = connectingLink.relationship;
              calculatedGroup = mapRelationshipToGroup(connectingLink.relationship); // Get group from mapping
              // Log direct connection found
              if (index < 20) {
                   console.log(`[BASE MAP #${index}] Node: '${node.word || node.label}' (ID: ${currentNodeIdStr}) - DIRECT LINK FOUND to main (${mainWordIdStr}). Rel: '${relationshipToMainWord}', RawGroup: '${calculatedGroup}'`);
              }
            } else {
              calculatedGroup = 'related'; // Default if no direct link
              // Log no direct connection found
              if (index < 20) {
                  console.log(`[BASE MAP #${index}] Node: '${node.word || node.label}' (ID: ${currentNodeIdStr}) - NO DIRECT LINK to main (${mainWordIdStr}). Group: ${calculatedGroup}`);
              }
            }
          } else {
            // Log if mainWordIdStr was missing (shouldn't happen if initial check passed, but safety net)
            if (index < 20) {
                console.warn(`[BASE MAP #${index}] Node: '${node.word || node.label}' (ID: ${currentNodeIdStr}) - Cannot determine relationship because mainWordIdStr is missing.`);
            }
            calculatedGroup = 'related'; // Default if main word ID is missing
          }
  
          // Count connections (using string IDs for safety)
          const connections = baseLinks.filter(l => {
              const sourceId = typeof l.source === 'object' && l.source !== null ? String(l.source.id) : String(l.source);
              const targetId = typeof l.target === 'object' && l.target !== null ? String(l.target.id) : String(l.target);
              return sourceId === currentNodeIdStr || targetId === currentNodeIdStr;
          }).length;
  
          // Create the node object
          return {
              id: currentNodeIdStr, // Use the stringified numeric ID
              word: node.word || node.label, // Prefer 'word', fallback to 'label'
              label: node.label, // Keep original label if needed
              group: calculatedGroup, // Assigned group
              connections: connections,
              relationshipToMain: relationshipToMainWord, // Assigned relationship string
              // *** CHANGE: Use word property for path check consistent with main word check ***
              pathToMain: (node.word === mainWord || (mainWordMatchMethod === 'label' && node.label === mainWord)) ? [mainWord] : undefined, // Path based on how main word was matched
              pinned: false,
              originalId: node.id, // Keep original numeric ID
              language: node.language || undefined,
              definitions: (node as any).definitions?.map((def: any) => def.text || def.definition_text).filter(Boolean) || [],
              has_baybayin: node.has_baybayin || false,
              baybayin_form: node.baybayin_form || null,
              // D3 simulation properties initialized as undefined
              index: undefined, x: undefined, y: undefined, vx: undefined, vy: undefined, fx: undefined, fy: undefined
          };
      });
  
      // Deduplication based on string ID (should be rare if original nodes are unique)
      const uniqueNodes: CustomNode[] = [];
      const seenIds = new Set<string>();
      for (const node of mappedNodes) {
          if (!seenIds.has(node.id)) {
              uniqueNodes.push(node);
              seenIds.add(node.id);
          } else {
               console.warn(`[BASE] Skipping duplicate node during final mapping with ID: ${node.id} (Word: ${node.word || node.label})`);
          }
      }
      console.log("[BASE] Final unique nodes count:", uniqueNodes.length);
      if (uniqueNodes.length > 0) {
        console.log("[BASE] Sample final nodes (first 5):");
        uniqueNodes.slice(0, 5).forEach(n => console.log(`  Node Word: '${n.word}', ID: ${n.id}, Group: ${n.group}, RelToMain: ${n.relationshipToMain}`));
      }
  
      return uniqueNodes;
  
    }, [wordNetwork, mainWord, baseLinks, mapRelationshipToGroup]); // Dependencies include mapRelationshipToGroup
  
    // >>> ADD LOGS HERE <<<
    // REMOVE OLD LOGS
    // console.log("[DEBUG] baseLinks:", baseLinks);
    // console.log("[DEBUG] baseNodes (before filtering):", baseNodes);
  
    // Add the getNodeRadius function that was missing from the setupNodeInteractions dependency array
    const getNodeRadius = useCallback((node: CustomNode) => {
      // Exact sizing from old_src_2 implementation
      // <<< CHANGE: Check node word/label against mainWord string for main node check >>>
      if (node.word === mainWord) return 22; 
      if (node.group === 'root') return 17;
      return 14; // Standard size for other nodes
    }, [mainWord]);
  
    // Create a map from filteredNodes for quick lookups (used in setupNodeInteractions)
    // <<< CHANGE: nodeMap key is now the string numeric ID >>>
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
      // <<< CHANGE: Access source/target node objects via link properties >>>
      linkSelection
        .attr("x1", d => {
            const sourceNode = d.source as CustomNode; // Links source/target are resolved to nodes by D3
            return (typeof sourceNode === 'object' && typeof sourceNode.x === 'number') ? sourceNode.x : 0;
        })
        .attr("y1", d => {
            const sourceNode = d.source as CustomNode;
            return (typeof sourceNode === 'object' && typeof sourceNode.y === 'number') ? sourceNode.y : 0;
        })
        .attr("x2", d => {
            const targetNode = d.target as CustomNode;
            return (typeof targetNode === 'object' && typeof targetNode.x === 'number') ? targetNode.x : 0;
        })
        .attr("y2", d => {
            const targetNode = d.target as CustomNode;
            return (typeof targetNode === 'object' && typeof targetNode.y === 'number') ? targetNode.y : 0;
        });
  
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
        // <<< CHANGE: Log uses word for readability >>>
        console.log(`Peek triggered for node: ${d.word} [ID: ${d.id}]`); 
        const xPos = event.clientX;
        const yPos = event.clientY;
        setPeekedNode({ node: d, x: xPos, y: yPos });
        setHoveredNode(null);
      });
      // --- End Peek Interaction ---
  
      // Enhanced hover effect - EXACT match to the latest screenshots + Path Highlighting
      nodeSelection.on("mouseenter", (event, d) => {
        const currentTargetElement = event.currentTarget as SVGGElement;
        // <<< CHANGE: Use mainWord string for check, find main node ID via nodeMap >>>
        const mainWordNodeEntry = Array.from(nodeMap.entries()).find(([id, node]) => node.word === mainWord);
        const mainWordIdString = mainWordNodeEntry ? mainWordNodeEntry[0] : null;
        if (isDraggingRef.current || !mainWordIdString) return; 
        setPeekedNode(null);
  
        // --- Find Path to Main Word (BFS Backwards) --- 
        const pathNodeIds = new Set<string>();
        const pathLinkIds = new Set<string>(); 
        // <<< CHANGE: Start BFS with the node's string numeric ID >>>
        const queue: [string, string[]][] = [[d.id, [d.id]]]; 
        const visited = new Set<string>([d.id]);
        let foundPath = false;
  
        // <<< CHANGE: Check against mainWordIdString >>>
        if (d.id !== mainWordIdString) { 
          while (queue.length > 0 && !foundPath) {
            const [currentId, currentPath] = queue.shift()!;
  
            // Find links connected TO the current node using baseLinks
            const incomingLinks = baseLinks.filter(l => { 
              // <<< CHANGE: baseLinks source/target are already strings >>>
              return l.target === currentId; 
            });
            
            // Also check links FROM the current node using baseLinks
            const outgoingLinks = baseLinks.filter(l => { 
              return l.source === currentId;
            });
            
            const potentialLinks = [...incomingLinks, ...outgoingLinks];
  
            for (const link of potentialLinks) {
              // <<< CHANGE: source/target are strings from baseLinks >>>
              const sourceId = link.source as string; 
              const targetId = link.target as string;
              const neighborId = sourceId === currentId ? targetId : sourceId;
  
              if (!visited.has(neighborId)) {
                  visited.add(neighborId);
                const newPath = [...currentPath, neighborId];
                const linkId = `${sourceId}_${targetId}`;
                const reverseLinkId = `${targetId}_${sourceId}`;
  
                // <<< CHANGE: Check neighbor against mainWordIdString >>>
                if (neighborId === mainWordIdString) { 
                  // Path found!
                  newPath.forEach(id => pathNodeIds.add(id));
                  // Add links along the path
                  for(let i = 0; i < newPath.length - 1; i++) {
                    const pathLinkId = `${newPath[i]}_${newPath[i+1]}`;
                    const pathReverseLinkId = `${newPath[i+1]}_${newPath[i]}`;
                      // Find the actual link and add its ID using baseLinks
                       const actualLink = baseLinks.find(fl => 
                         // <<< CHANGE: Source/target are strings >>>
                         (`${fl.source}_${fl.target}` === pathLinkId) || 
                         (`${fl.source}_${fl.target}` === pathReverseLinkId)
                     );
                     if (actualLink) {
                         // <<< CHANGE: Source/target are strings >>>
                         pathLinkIds.add(`${actualLink.source}_${actualLink.target}`); 
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
            // <<< CHANGE: Path for main word is its string ID >>>
            pathNodeIds.add(mainWordIdString); 
        }
        // --- End Path Finding --- 
  
        // Find direct connections (neighbors) using baseLinks
        const directNeighborIds = new Set<string>();
        baseLinks.forEach(l => { // <<< CHANGE: Use string IDs >>>
            const sourceId = l.source as string;
            const targetId = l.target as string;
            if (sourceId === d.id) directNeighborIds.add(targetId);
            if (targetId === d.id) directNeighborIds.add(sourceId);
        });
  
        // Combine path nodes and direct neighbors for highlighting (using string IDs)
        const highlightNodeIds = new Set<string>([d.id]);
        pathNodeIds.forEach(id => highlightNodeIds.add(id));
        directNeighborIds.forEach(id => highlightNodeIds.add(id));
  
        // --- Dim non-highlighted elements --- 
        d3.selectAll<SVGGElement, CustomNode>(".node")
          // <<< CHANGE: Filter using string ID >>>
          .filter(n => !highlightNodeIds.has(n.id)) 
          .transition().duration(250)
          .style("opacity", 0.7);
  
        d3.selectAll<SVGLineElement, CustomLink>(".link")
          .filter((l: CustomLink) => {
            // <<< CHANGE: Use string IDs >>>
            const sourceId = l.source as string; 
            const targetId = l.target as string;
            const linkId = `${sourceId}_${targetId}`;
            // Dim if NOT directly connected to hovered OR part of the path
            return !( (sourceId === d.id && directNeighborIds.has(targetId)) || 
                      (targetId === d.id && directNeighborIds.has(sourceId)) ||
                      pathLinkIds.has(linkId) || pathLinkIds.has(`${targetId}_${sourceId}`) ); 
          })
          .transition().duration(250)
          .style("stroke-opacity", 0.5);
  
        d3.selectAll<SVGTextElement, CustomNode>(".node-label")
          // <<< CHANGE: Filter using string ID >>>
          .filter(n => !highlightNodeIds.has(n.id)) 
          .transition().duration(250)
          .style("opacity", 0.7);
  
        // --- Highlight elements in the path and direct connections --- 
        // Nodes
        d3.selectAll<SVGGElement, CustomNode>(".node")
          // <<< CHANGE: Filter using string ID >>>
          .filter(n => highlightNodeIds.has(n.id)) 
          .raise()
          .transition().duration(300)
          .style("opacity", 1)
          .attr("transform", n => `translate(${n.x || 0},${n.y || 0})`)
          .select("circle")
            .attr("stroke-width", n => {
              // <<< CHANGE: Compare string IDs >>>
              if (n.id === d.id) return 1.5; 
              if (n.id === mainWordIdString) return 1; 
              return 0.5;
            })
            // <<< CHANGE: Compare string IDs >>>
            .attr("stroke-opacity", n => n.id === d.id ? 0.9 : 0.7) 
            .attr("stroke", n => {
              const baseColor = d3.color(getNodeColor(n.group)) || d3.rgb("#888");
              // <<< CHANGE: Compare string IDs >>>
              if (n.id === d.id) return baseColor.brighter(0.8).toString(); 
              if (n.id === mainWordIdString) return baseColor.brighter(0.5).toString(); 
              return themeMode === 'dark' ? baseColor.brighter(0.3).toString() : baseColor.brighter(0.4).toString();
            })
            .attr("filter", n => {
              // <<< CHANGE: Compare string IDs >>>
              if (n.id === d.id) return `url(#apple-node-shadow) brightness(1.18)`; 
              if (pathNodeIds.has(n.id)) return `url(#apple-node-shadow) brightness(1.12)`;
              if (n.id === mainWordIdString) return `url(#apple-node-shadow) brightness(1.1)`; 
              return `url(#apple-node-shadow)`;
            });
  
        // Links (Directly connected OR on the path)
        d3.selectAll<SVGLineElement, CustomLink>(".link")
          .filter((l: CustomLink) => { 
            // <<< CHANGE: Use string IDs >>>
            const sourceId = l.source as string; 
            const targetId = l.target as string;
            const linkId = `${sourceId}_${targetId}`;
            return ( (sourceId === d.id && directNeighborIds.has(targetId)) || 
                     (targetId === d.id && directNeighborIds.has(sourceId)) ||
                     pathLinkIds.has(linkId) || pathLinkIds.has(`${targetId}_${sourceId}`) );
          })
          .raise()
          .transition().duration(200)
          .style("stroke-opacity", 0.9) 
          .attr("stroke-width", 2.5)
          .each(function(l: CustomLink) {
            // <<< CHANGE: Use string IDs >>>
            const sourceId = l.source as string; 
            const targetId = l.target as string;
            const linkId = `${sourceId}_${targetId}`;
            const reverseLinkId = `${targetId}_${sourceId}`;
            let connectedNodeId: string;
  
            if (pathLinkIds.has(linkId) || pathLinkIds.has(reverseLinkId)) {
              connectedNodeId = sourceId === d.id ? targetId : sourceId;
            } else {
              connectedNodeId = sourceId === d.id ? targetId : sourceId;
            }
            // <<< CHANGE: Use nodeMap with string ID >>>
            const connectedNode = nodeMap.get(connectedNodeId); 
            const color = connectedNode ? getNodeColor(connectedNode.group) : (themeMode === "dark" ? "#aaa" : "#666");
            d3.select(this).style("stroke", color);
          });
  
        // Labels (Directly connected OR on the path)
        d3.selectAll<SVGTextElement, CustomNode>(".node-label")
          // <<< CHANGE: Filter using string ID >>>
          .filter(n => highlightNodeIds.has(n.id)) 
          .transition().duration(200)
          .style("opacity", 1)
          .style("font-weight", "bold");
      });
  
      nodeSelection.on("mouseleave", (event, d) => {
        const currentTargetElement = event.currentTarget as SVGGElement;
        // <<< CHANGE: Use mainWord string for check, find main node ID via nodeMap >>>
        const mainWordNodeEntry = Array.from(nodeMap.entries()).find(([id, node]) => node.word === mainWord);
        const mainWordIdString = mainWordNodeEntry ? mainWordNodeEntry[0] : null;
        if (isDraggingRef.current) return;
  
        d3.selectAll<SVGGElement, CustomNode>(".node")
          .transition().duration(200) 
          // <<< CHANGE: Compare string IDs for main node opacity >>>
          .style("opacity", n => n.id === mainWordIdString ? 1 : 0.8) 
          .attr("transform", n => `translate(${n.x || 0},${n.y || 0})`) 
          .select("circle")
            .attr("stroke-width", 0.5) 
            .attr("stroke-opacity", 0.6) 
            // <<< CHANGE: Compare string IDs for main node filter >>>
            .attr("filter", n => n.id === mainWordIdString ? `url(#apple-node-shadow) brightness(1.15)` : `url(#apple-node-shadow)`) 
            .attr("stroke", n => { 
              const baseColor = d3.color(getNodeColor(n.group)) || d3.rgb("#888");
              return themeMode === 'dark' ? baseColor.brighter(0.3).toString() : baseColor.brighter(0.5).toString();
            }); 
            
        d3.selectAll<SVGLineElement, CustomLink>(".link")
          .transition().duration(200) 
          .style("stroke-opacity", 0.6)
          .attr("stroke-width", 1.5)
          .style("stroke", themeMode === "dark" ? "#666" : "#ccc"); 
          
        d3.selectAll<SVGTextElement, CustomNode>(".node-label")
          .transition().duration(200) 
          .style("opacity", 0.9) 
          // <<< CHANGE: Compare string IDs for main node font weight >>>
          .style("font-weight", n => n.id === mainWordIdString ? "bold" : "normal"); 
      });
      
      // Double-click to navigate with improved visual feedback
      nodeSelection.on("dblclick", (event, d) => {
        const currentTargetElement = event.currentTarget as SVGGElement;
        event.preventDefault();
        event.stopPropagation();
        
        if (isDraggingRef.current) return;
        setPeekedNode(null);
        
        const nodeGroup = d3.select(currentTargetElement);
        const circleElement = nodeGroup.select("circle");
        // <<< CHANGE: Select label using string ID >>>
        const textLabel = d3.select(`.node-label[data-id="${d.id}"]`); 
        
        const radius = parseFloat(circleElement.attr("r"));
        const ripple = nodeGroup.append("circle")
          .attr("class", "ripple")
          .attr("r", radius)
          .attr("fill", "none")
          .attr("stroke", (d3.color(getNodeColor(d.group))?.brighter(0.8).toString() || "#fff"))
          .attr("stroke-width", 2)
          .attr("opacity", 1);
          
        ripple.transition()
          .duration(400)
          .attr("r", radius * 2.5)
          .attr("opacity", 0)
          .remove();
          
        circleElement
          .transition()
          .duration(200)
          .attr("fill-opacity", 0.8)
          .attr("r", radius * 1.2)
          .transition()
          .duration(200)
          .attr("fill-opacity", 1)
          .attr("r", radius);
          
        if (textLabel.size() > 0) {
          textLabel
            .transition()
            .duration(200)
            .style("font-weight", "bold")
            .style("opacity", 1);
        }
        
        const toast = d3.select(svgRef.current?.parentNode as HTMLElement)
          .append("div")
          .attr("class", "navigation-toast")
          // Styling remains the same
          // ...
          // <<< CHANGE: Use word for toast text >>>
          .text(`Navigating to "${d.word}"...`); 
        
        toast.transition()
          // Timing remains the same
          // ...
          .remove();
        
        setTimeout(() => {
          // <<< CHANGE: Log uses word, callback uses word >>>
          console.log(`Double-click on node: ${d.word} [ID: ${d.id}] - Navigating`); 
          if (onNodeClick) {
            onNodeClick(d.word); 
          }
        }, 350); 
      });
      
      // Tooltip display logic (remains arrow function)
      nodeSelection.on("mouseover", (event, d) => {
          if (isDraggingRef.current) return;
          // <<< CHANGE: Use mainWord string for check, find main node ID via nodeMap >>>
          const mainWordNodeEntry = Array.from(nodeMap.entries()).find(([id, node]) => node.word === mainWord);
          const mainWordIdString = mainWordNodeEntry ? mainWordNodeEntry[0] : null;
  
          // Check if this is directly connected to main word using baseLinks and string IDs
          let relationshipToMain = ""; 
          // <<< CHANGE: Compare string IDs >>>
          if (d.id !== mainWordIdString) { 
              const link = baseLinks.find(l => { 
                  // <<< CHANGE: Source/target are strings >>>
                  return (l.source === mainWordIdString && l.target === d.id) || 
                         (l.target === mainWordIdString && l.source === d.id); 
              });
              if (link) { 
                relationshipToMain = link.relationship;
            }
        }
        
        // Pass relationship to tooltip only if peek isn't active
        if (!peekedNode || peekedNode.node.id !== d.id) {
             setHoveredNode({ 
                 ...d, 
                 // relationshipToMain is removed, relationshipFromParent is already in d 
             });
             
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
        .selectAll("g.node") 
        // <<< CHANGE: Use string ID as key >>>
        .data(nodesData, d => (d as CustomNode).id) 
        .join(
            enter => {
                // <<< CHANGE: Use mainWord string for main node class check >>>
                const nodeGroup = enter.append("g")
                    .attr("class", d => `node node-group-${d.group} ${d.word === mainWord ? "main-node" : ""}`) 
                    // <<< CHANGE: Use string ID for data-id >>>
                    .attr("data-id", d => d.id) 
                    .style("opacity", 0);
  
                // Defs creation remains the same
                // ...
                
                // Node circle creation remains the same
                nodeGroup.append("circle")
                  .attr("r", getNodeRadius)
                  .attr("fill", d => getNodeColor(d.group))
                  .attr("fill-opacity", 1)
                  .attr("stroke", d => d3.color(getNodeColor(d.group))?.darker(0.8).formatHex() ?? "#888")
                  // <<< CHANGE: Use mainWord string for stroke width check >>>
                  .attr("stroke-width", d => d.word === mainWord ? 2.5 : 1.5) 
                  .attr("stroke-opacity", 0.7)
                  .attr("shape-rendering", "geometricPrecision");
  
                // <<< CHANGE: Use mainWord string for opacity check >>>
                nodeGroup.call(enter => enter.transition().duration(300).style("opacity", d => d.word === mainWord ? 1 : 0.8)); 
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
          // <<< CHANGE: Use string ID as key >>>
          .data(nodesData, d => (d as CustomNode).id) 
          .join(
              enter => {
                  const textElement = enter.append("text")
                      .attr("class", "node-label")
                      // <<< CHANGE: Use string ID for data-id >>>
                      .attr("data-id", d => d.id) 
                      .attr("text-anchor", "middle")
                      // <<< CHANGE: Use mainWord string for font size/weight check >>>
                      .attr("font-size", d => d.word === mainWord ? "12px" : "10px") 
                      .attr("font-weight", d => d.word === mainWord ? "bold" : "normal") 
                      // <<< CHANGE: Use word for text content >>>
                      .text(d => d.word) 
                      .attr("x", d => d.x ?? 0)
                      .attr("y", d => (d.y ?? 0) + getNodeRadius(d) + 12)
                      .style("opacity", 0) 
                      .style("pointer-events", "none") 
                      .style("user-select", "none");
  
                  // Halo creation remains the same
                  textElement.clone(true)
                      // ...
                      .attr("stroke", themeMode === "dark" ? "rgba(0,0,0,0.8)" : "rgba(255,255,255,0.8)");
  
                  // Text fill remains the same
                  textElement.attr("fill", themeMode === "dark" ? "#eee" : "#222");
  
                  // <<< CHANGE: Use mainWord string for opacity check >>>
                  textElement.call(enter => enter.transition().duration(300).style("opacity", d => d.word === mainWord ? 1 : 0.9)); 
                  return textElement;
              },
              update => update,
              exit => exit
                  .call(exit => exit.transition().duration(300).style("opacity", 0))
                  .remove()
          );
  
      // Setup drag behavior
      function dragStarted(this: SVGGElement, event: d3.D3DragEvent<SVGGElement, CustomNode, any>, d: CustomNode) {
        dragStartTimeRef.current = Date.now();
        setPeekedNode(null);
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        isDraggingRef.current = true;
        d3.select(this).classed("dragging", true).select("circle").attr("stroke-dasharray", "3,2");
      }
  
      function dragged(this: SVGGElement, event: d3.D3DragEvent<SVGGElement, CustomNode, any>, d: CustomNode) {
        d.fx = event.x;
        d.fy = event.y;
      }
  
      function dragEnded(this: SVGGElement, event: d3.D3DragEvent<SVGGElement, CustomNode, any>, d: CustomNode) {
        const dragEndTime = Date.now();
        const dragDuration = dragEndTime - dragStartTimeRef.current;
        if (!event.active) simulation.alphaTarget(0);
        // <<< CHANGE: Use mainWord string for check >>>
        if (d.word !== mainWord) { 
          d.fx = null;
          d.fy = null;
        }
        d3.select(this).classed("dragging", false).select("circle").attr("stroke-dasharray", null);
        if (dragDuration < 150) {
          isDraggingRef.current = false;
        } else {
          setTimeout(() => { isDraggingRef.current = false; }, 150);
        }
      }
      
      const drag = d3.drag<SVGGElement, CustomNode>()
        .filter(event => !event.ctrlKey && event.button === 0) 
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded);
      
      (nodeGroup as any).call(drag);
      
      // Apply other node interactions
      setupNodeInteractions(nodeGroup as any);
      
      // Handle tick updates for node and label positions
      simulation.on("tick", ticked);
      
      return nodeGroup;
      // <<< CHANGE: Dependencies include mainWord string >>>
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
      const mainNodeData = nodesToSearch.find(n => n.word === mainWord); 
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
      // Ensure prerequisites are met
      if (!mainWord || baseNodes.length === 0 || !wordNetwork?.nodes) {
        return { filteredNodes: [], filteredLinks: [] };
      }
  
      console.log("[FILTER] Applying depth/breadth and relationship filters");
      console.log("[FILTER] Main word:", mainWord);
      console.log("[FILTER] Base nodes:", baseNodes.length);
      console.log("[FILTER] Base links:", baseLinks.length);
      console.log("[FILTER] Depth limit:", depth);
      console.log("[FILTER] Breadth limit:", breadth);
  
      // --- Perform BFS to limit depth and breadth using String IDs ---
      const connectedNodeIds = new Set<string>();
      const queue: { nodeId: string; currentDepth: number }[] = [];
      const nodeMap = new Map<string, CustomNode>(baseNodes.map(n => [n.id, n])); // Map ID string -> Node
  
      // Find the main word's Node ID string
      const mainWordNode = baseNodes.find(node => node.group === 'main');
      const mainWordIdString = mainWordNode ? mainWordNode.id : null;
  
      if (mainWordIdString) {
        console.log(`[FILTER] Starting BFS with mainWord ID: ${mainWordIdString}`);
        queue.push({ nodeId: mainWordIdString, currentDepth: 0 });
        connectedNodeIds.add(mainWordIdString);
      } else {
        console.warn('[FILTER] Cannot start BFS: Main word node not found in baseNodes.');
      }
  
      console.log('[FILTER] Node map size:', nodeMap.size);
  
      let head = 0;
      const nodesAddedAtDepth: { [key: number]: number } = {}; // Track nodes added per depth level for breadth limit
  
      while (head < queue.length) {
        const { nodeId, currentDepth } = queue[head++];
  
        if (currentDepth >= depth) continue; // Stop if max depth reached
  
        // Initialize counter for the current depth if not present
        if (nodesAddedAtDepth[currentDepth] === undefined) {
          nodesAddedAtDepth[currentDepth] = 0;
        }
  
        // Find neighbors (both incoming and outgoing links)
        baseLinks.forEach(link => {
          let neighborId: string | null = null;
          const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
          const targetId = typeof link.target === 'object' ? link.target.id : link.target;
  
          // --- FIX: Explicit type assertion for check --- 
          if (sourceId === nodeId && !connectedNodeIds.has(targetId as string)) {
            neighborId = targetId;
          } else if (targetId === nodeId && !connectedNodeIds.has(sourceId as string)) {
            neighborId = sourceId;
          }
  
          // If a new neighbor is found AND the node actually exists in our map
          if (neighborId && nodeMap.has(neighborId)) {
             // Check breadth limit for the *neighbor's* depth level (currentDepth + 1)
             const nextDepth = currentDepth + 1;
             if (nodesAddedAtDepth[nextDepth] === undefined) {
               nodesAddedAtDepth[nextDepth] = 0;
             }
  
             // Add neighbor only if breadth limit for *its* depth level is not exceeded
             // (Allow unlimited at level 0/direct connections if breadth > 0)
             if (breadth === 0 || nodesAddedAtDepth[nextDepth] < breadth) {
               connectedNodeIds.add(neighborId);
               queue.push({ nodeId: neighborId, currentDepth: nextDepth });
               nodesAddedAtDepth[nextDepth]++;
             }
          }
        });
      }
      console.log(`[FILTER-BFS] BFS finished. Found ${connectedNodeIds.size} connected node IDs within limits.`);
  
      // --- Filter nodes and links based ONLY on BFS results FIRST ---
      const bfsFilteredNodes = baseNodes.filter(node => connectedNodeIds.has(node.id));
      // CRITICAL FIX: Filter links to ONLY include those connecting nodes within the connectedNodeIds set
      const bfsFilteredLinks = baseLinks.filter(link => {
          const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
          const targetId = typeof link.target === 'object' ? link.target.id : link.target;
          // --- FIX: Explicit type assertion for check --- 
          return connectedNodeIds.has(sourceId as string) && connectedNodeIds.has(targetId as string);
      });
  
      console.log(`[FILTER] After depth/breadth (using String IDs): ${bfsFilteredNodes.length}/${baseNodes.length} nodes and ${bfsFilteredLinks.length}/${baseLinks.length} links`);
  
      // --- Now apply relationship type filters if any are active ---
      let finalFilteredNodes = bfsFilteredNodes;
      let finalFilteredLinks = bfsFilteredLinks;
  
      if (filteredRelationships.length > 0) {
          console.log('[FILTER] Applying relationship type filters:', filteredRelationships);
  
          // Filter links further based on the selected relationship types
          finalFilteredLinks = bfsFilteredLinks.filter(link =>
              filteredRelationships.includes(link.relationship)
          );
  
          // Filter nodes: Keep main node + nodes connected by the selected link types
          // Recalculate the set of nodes involved in the *final* filtered links
          const nodesConnectedByFinalLinks = new Set<string>();
          if (mainWordIdString) nodesConnectedByFinalLinks.add(mainWordIdString); // Always keep main word
  
          finalFilteredLinks.forEach(link => {
              const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
              const targetId = typeof link.target === 'object' ? link.target.id : link.target;
              // --- FIX: Explicit type assertion for check --- 
              nodesConnectedByFinalLinks.add(sourceId as string);
              nodesConnectedByFinalLinks.add(targetId as string);
          });
  
          finalFilteredNodes = bfsFilteredNodes.filter(node =>
              nodesConnectedByFinalLinks.has(node.id)
          );
          console.log(`[FILTER] After relationship filters: ${finalFilteredNodes.length} nodes, ${finalFilteredLinks.length} links`);
  
      } else {
        console.log('[FILTER] No relationship filters active - showing all depth/breadth limited nodes and links');
      }
  
      // Ensure the main word node is *always* included if it was originally present
      if (mainWordNode && !finalFilteredNodes.some(n => n.id === mainWordIdString)) {
          console.warn("[FILTER] Main word node was filtered out, re-adding it.");
          finalFilteredNodes.push(mainWordNode); // Add it back if filters removed it
      }
  
  
      return { filteredNodes: finalFilteredNodes, filteredLinks: finalFilteredLinks };
  
    }, [baseNodes, baseLinks, depth, breadth, filteredRelationships, mainWord, getRelationshipTypeLabel]); // Ensure all dependencies are listed
  
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
    }, [mainWord, getNodeRadius, ticked]);
  
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
  
      // Need mainWordIdStr for POS context check
      const mainWordNodeEntry = Array.from(nodeMap.entries()).find(([id, node]) => node.word === mainWord);
      const mainWordIdStr = mainWordNodeEntry ? mainWordNodeEntry[0] : null;
  
      const svgNode = svgRef.current;
      const transform = d3.zoomTransform(svgNode);
  
      const [screenX, screenY] = transform.apply([hoveredNode.x, hoveredNode.y]);
  
      const offsetX = (screenX > window.innerWidth / 2) ? -20 - 250 : 20;
      const offsetY = (screenY > window.innerHeight / 2) ? -20 - 80 : 20;
  
      // Find the specific link causing this relationship using baseLinks and relationshipToMain
      let posContext = "";
      const connectingLink = baseLinks.find(l => 
          (l.source === mainWordIdStr && l.target === hoveredNode.id && l.relationship === hoveredNode.relationshipToMain) || 
          (l.target === mainWordIdStr && l.source === hoveredNode.id && l.relationship === hoveredNode.relationshipToMain)
      );
      if (connectingLink?.metadata?.english_pos_context) { 
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
           <h4 style={{ margin: 0, marginBottom: '6px', color: getNodeColor(hoveredNode.group), fontSize: '15px' }}>{hoveredNode.word}</h4> {/* Use word */}
           
           {/* Relationship to main word */}
           {hoveredNode.id !== mainWordIdStr && hoveredNode.relationshipToMain && (
             <div style={{ 
               display: "flex", 
               alignItems: "center", 
               gap: "6px", 
               paddingBottom: "4px",
               background: themeMode === "dark" ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.03)",
               padding: "5px 8px",
               borderRadius: "4px",
               marginBottom: "5px"
             }}
             // title={`This relationship determined the node's color based on the path from '${mainWord}'`} // Remove title or update
             >
               <span style={{ 
                 fontSize: "11px", 
                 color: themeMode === "dark" ? "#aaa" : "#666", 
                 fontWeight: "500",
                 whiteSpace: "nowrap"
               }}>
                 {mainWord} {/* Show main word */} 
                 <span style={{ margin: "0 4px", opacity: 0.7 }}>→</span> {/* Arrow */} 
                 <span style={{ 
                   fontStyle: "italic", 
                   color: themeMode === "dark" ? "#ddd" : "#333",
                   fontWeight: "600",
                   // marginLeft: "4px" // No extra margin needed here
                 }}>
                   {getRelationshipTypeLabel(hoveredNode.relationshipToMain || '').label} {/* Use relationshipToMain */}
                   {posContext} {/* Display POS context here */}
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
    }, [hoveredNode, themeMode, getNodeColor, mainWord, peekedNode, svgRef, baseLinks, nodeMap, getRelationshipTypeLabel]); // Keep dependencies
  
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
              // <<< CHANGE: Key uses mainWord string >>>
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
  
                {/* Network Controls */}
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
  