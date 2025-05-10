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
  import { CircularProgress, Dialog, DialogTitle, DialogContent, DialogActions, Button } from '@mui/material'; // Added Dialog imports
  import InfoIcon from '@mui/icons-material/Info'; // Icon for legend button
  import ListItemButton from '@mui/material/ListItemButton'; // Added ListItemButton import
  
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
    id: string; // Will be node.label
    word: string;
    label: string; // CHANGED: Made label required
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
    const [depth, setDepth] = useState<number>(initialDepth);
    const [breadth, setBreadth] = useState<number>(initialBreadth);
    const [error, setError] = useState<string | null>(null);
    const [isValidNetwork, setIsValidNetwork] = useState(true);
    const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);
    const [filteredRelationships, setFilteredRelationships] = useState<string[]>([]);
    const [forceUpdate, setForceUpdate] = useState<number>(0);
    const [mobileLegendOpen, setMobileLegendOpen] = useState(false); // New state for mobile legend
    const [selectedMobileNodeId, setSelectedMobileNodeId] = useState<string | null>(null); // <-- ADD THIS LINE
  
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
  
    const handleToggleMobileLegend = () => {
      setMobileLegendOpen(!mobileLegendOpen);
    };
  
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
        derived: { category: "Derivation", label: "Constructions", color: getNodeColor("derived") }, 
        derivative: { category: "Derivation", label: "Constructions", color: getNodeColor("derived") }, 
        sahod: { category: "Derivation", label: "Constructions", color: getNodeColor("derived") }, 
        isahod: { category: "Derivation", label: "Constructions", color: getNodeColor("derived") }, 
        affix: { category: "Derivation", label: "Affix", color: getNodeColor("affix") },
        // Meaning Category
        synonym: { category: "Meaning", label: "Synonym", color: getNodeColor("synonym") },
        antonym: { category: "Meaning", label: "Antonym", color: getNodeColor("antonym") },
        kasalungat: { category: "Meaning", label: "Antonym", color: getNodeColor("antonym") },
        related: { category: "Meaning", label: "Related", color: getNodeColor("related") },
        similar: { category: "Meaning", label: "Related", color: getNodeColor("related") },
        kaugnay: { category: "Meaning", label: "Related", color: getNodeColor("related") },
        kahulugan: { category: "Meaning", label: "Related", color: getNodeColor("related") },
        has_translation: { category: "Meaning", label: "Translation", color: getNodeColor("translation") },
        translation_of: { category: "Meaning", label: "Translation", color: getNodeColor("translation") },
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
    
    // Memoize base links processing - USE LABEL AS ID
    const baseLinks = useMemo((): CustomLink[] => {
      if (!wordNetwork?.nodes || !wordNetwork.links) return [];
      console.log("[BASELINKS-LABEL] Processing links using LABEL as ID");

      // --- START: Create map from numeric ID string/number to label --- 
      const nodeIdToLabelMap = new Map<string, string>(); // Key is now string
      wordNetwork.nodes.forEach((node, index) => {
        let nodeIdKey: string | null = null;
        // Handle both numeric and string IDs from raw data
        if (typeof node.id === 'number') {
            nodeIdKey = String(node.id);
        } else if (typeof node.id === 'string' && node.id) { // Check if string and non-empty, remove trim
            nodeIdKey = node.id; // Removed trim
        }

        // Ensure we have a valid key and a non-empty label
        if (nodeIdKey && typeof node.label === 'string' && node.label) {
          nodeIdToLabelMap.set(nodeIdKey, node.label);
        } else if (index < 5) { // Log why first few nodes might fail
             console.warn(`[BASELINKS-LABEL MAP DEBUG] Skipping node #${index}: Invalid id ('${node.id}', type: ${typeof node.id}) or label ('${node.label}', type: ${typeof node.label})`);
        }
      });
      console.log(`[BASELINKS-LABEL] Created nodeIdToLabelMap with ${nodeIdToLabelMap.size} entries`);
      // --- END: Map creation ---

      const links = wordNetwork.links
        .map(link => {
            // --- MODIFIED: Look up LABEL using numeric ID string from map --- 
            let sourceLabel: string | null = null;
            let targetLabel: string | null = null;

            // Get string key for source ID
            let sourceIdKey: string | null = null;
            if (typeof link.source === 'string') {
                sourceIdKey = link.source; // Assume it's the string ID
            } else if (typeof link.source === 'number') {
                sourceIdKey = String(link.source);
            }

            // Get string key for target ID
            let targetIdKey: string | null = null;
            if (typeof link.target === 'string') {
                targetIdKey = link.target; // Assume it's the string ID
            } else if (typeof link.target === 'number') {
                targetIdKey = String(link.target);
            }

            // Look up labels using string ID keys
            if (sourceIdKey && nodeIdToLabelMap.has(sourceIdKey)) {
                sourceLabel = nodeIdToLabelMap.get(sourceIdKey)!;
            }
            if (targetIdKey && nodeIdToLabelMap.has(targetIdKey)) {
                targetLabel = nodeIdToLabelMap.get(targetIdKey)!;
            }
           // --- END MODIFICATION ---

            if (!sourceLabel || !targetLabel) {
                 console.warn(`[BASELINKS-LABEL] Could not determine label for link source='${link.source}' (Key: ${sourceIdKey}) or target='${link.target}' (Key: ${targetIdKey}). Skipping link.`);
                 return null; // Skip if labels couldn't be found
          }
          
          return {
              source: sourceLabel, // Use looked-up LABEL
              target: targetLabel, // Use looked-up LABEL
            relationship: link.relationship,
              metadata: link.metadata
            } as CustomLink;
        })
        .filter((link): link is CustomLink => link !== null); 
        console.log("[BASELINKS-LABEL] Processed links:", links.length);
        return links;
    }, [wordNetwork]); // Dependency is only on the raw network data
  
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
            .call(enter => enter.transition().duration(300).attr("stroke-opacity", 1)), // MODIFIED: Default opacity 1
          update => update
            // Ensure updates reset to default style before transitions
            .attr("stroke", themeMode === "dark" ? "#666" : "#ccc")
            .attr("stroke-width", 1.5)
            .call(update => update.transition().duration(300)
                  .attr("stroke-opacity", 1)), // MODIFIED: Default opacity 1
          exit => exit
            .call(exit => exit.transition().duration(300).attr("stroke-opacity", 0))
            .remove()
        );
      return linkGroup;
    }, [themeMode]);
  
    // Fix baseNodes calculation to use stringified numeric ID for D3
    const baseNodes = useMemo<CustomNode[]>(() => {
      if (!wordNetwork?.nodes || !mainWord) return [];
      console.log("[BASENODES-LABEL] Processing nodes using LABEL as ID");

      const mappedNodes = wordNetwork.nodes.map((node: ImportedNetworkNode) => {
          // --- MODIFIED: Use label as the primary ID --- 
          const nodeIdLabel = node.label; 
          if (!nodeIdLabel) {
               console.warn("Node missing label, cannot use as ID. Skipping node:", node);
               return null; // Skip nodes without labels
          }
          // --- END MODIFICATION ---

          let calculatedGroup = 'related';
          let relationshipToMainWord: string | undefined = undefined;
          const isMainNode = nodeIdLabel === mainWord;
  
          if (isMainNode) {
            calculatedGroup = 'main';
            relationshipToMainWord = 'main';
          } else {
            // Find direct connection using LABEL IDs from baseLinks
            const connectingLink = baseLinks.find(link => 
              (link.source === mainWord && link.target === nodeIdLabel) ||
              (link.source === nodeIdLabel && link.target === mainWord)
            );
            if (connectingLink) {
              relationshipToMainWord = connectingLink.relationship;
              calculatedGroup = mapRelationshipToGroup(connectingLink.relationship);
            } else {
              calculatedGroup = 'related';
            }
          }
          
          // Count connections using LABEL IDs
          const connections = baseLinks.filter(l => 
              l.source === nodeIdLabel || l.target === nodeIdLabel
          ).length;

          return {
            id: nodeIdLabel, // Use LABEL as the ID
            word: node.word || node.label, // Keep word property distinct if needed
            label: node.label, // Store label explicitly
            group: calculatedGroup, 
              connections: connections,
            relationshipToMain: relationshipToMainWord,
            pathToMain: isMainNode ? [mainWord] : undefined,
              pinned: false,
            originalId: node.id, // Keep original numeric ID if needed elsewhere
              language: node.language || undefined,
              definitions: (node as any).definitions?.map((def: any) => def.text || def.definition_text).filter(Boolean) || [],
              has_baybayin: node.has_baybayin || false,
              baybayin_form: node.baybayin_form || null,
            // D3 properties
              index: undefined, x: undefined, y: undefined, vx: undefined, vy: undefined, fx: undefined, fy: undefined
          };
      })
      .filter(node => node !== null);
  
      // Deduplication based on label ID
      const uniqueNodes: CustomNode[] = [];
      const seenIds = new Set<string>();
      // MappedNodes now guaranteed to not contain nulls
      for (const node of mappedNodes) {
          if (!seenIds.has(node.id)) {
              uniqueNodes.push(node);
              seenIds.add(node.id);
        }
      }
      console.log("[BASENODES-LABEL] Final unique nodes:", uniqueNodes.length);
      return uniqueNodes;
    }, [wordNetwork, mainWord, baseLinks, mapRelationshipToGroup]);
  
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
  
    // --- START: Define mainWordIdString at component scope ---
    const mainWordNodeEntry = useMemo(() => {
      if (!mainWord || nodeMap.size === 0) return undefined;
      return Array.from(nodeMap.entries()).find(([/*id*/, node]) => node.word === mainWord);
    }, [nodeMap, mainWord]);
  
    const mainWordIdString = useMemo(() => {
      return mainWordNodeEntry ? mainWordNodeEntry[0] : null;
    }, [mainWordNodeEntry]);
    // --- END: Define mainWordIdString at component scope ---
  
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
        // REMOVED: setPeekedNode({ node: d, x: xPos, y: yPos }); // This line was causing the error as setPeekedNode is removed
        setHoveredNode(null); // Ensure tooltip is hidden when peek is activated
      });
      // --- End Peek Interaction ---
  
      // Enhanced hover effect - EXACT match to the latest screenshots + Path Highlighting
      nodeSelection.on("mouseenter", (event, d) => {
        const currentTargetElement = event.currentTarget as SVGGElement;
        // const mainWordNodeEntry = Array.from(nodeMap.entries()).find(([id, node]) => node.word === mainWord);
        // const mainWordIdString = mainWordNodeEntry ? mainWordNodeEntry[0] : null;
        if (isDraggingRef.current || !mainWordIdString) return; 

        console.log(`[HOVER ENTER] Node: '${d.word}' (ID: ${d.id})`);
  
        // --- Find Path to Main Word (BFS Backwards) --- 
        let foundPath = false;
        let calculatedPath: string[] | undefined = undefined; // Use a different name for clarity
        const pathNodeIds = new Set<string>(); // Initialize here
        const pathLinkIds = new Set<string>(); // Initialize here
  
        console.log(`[BFS Start] Starting BFS for node: ${d.id} to find main word: ${mainWordIdString}`);

        if (d.id !== mainWordIdString) { 
          const queue: [string, string[]][] = [[d.id, [d.id]]]; 
          const visited = new Set<string>([d.id]);

          while (queue.length > 0 && !foundPath) {
            const [currentId, currentPath] = queue.shift()!;
            console.log(`[BFS Loop] Processing: ${currentId}, Current Path: ${currentPath.join('->')}`);

            const incomingLinks = baseLinks.filter(l => l.target === currentId);
            const outgoingLinks = baseLinks.filter(l => l.source === currentId);
            const potentialLinks = [...incomingLinks, ...outgoingLinks];
  
            for (const link of potentialLinks) {
              const sourceId = link.source as string; 
              const targetId = link.target as string;
              const neighborId = sourceId === currentId ? targetId : sourceId;
  
              if (!visited.has(neighborId)) {
                  visited.add(neighborId);
                  const newPathForThisBranch = [...currentPath, neighborId]; // Create path first
  
                if (neighborId === mainWordIdString) { 
                    console.log(`[BFS PATH FOUND] Found main word '${mainWordIdString}' via neighbor '${neighborId}' from node '${currentId}'`);
                    calculatedPath = newPathForThisBranch.reverse(); // Assign to outer scope variable
                  foundPath = true;
                  break; // Exit inner loop
                }
                  queue.push([neighborId, newPathForThisBranch]); // Use the created path
              }
            }
            if (foundPath) break; // Exit outer loop if path found
          }
          console.log(`[BFS End] For node ${d.id}, foundPath = ${foundPath}`);
        } else {
            calculatedPath = [mainWordIdString];
            foundPath = true; // Mark as found for main node
        }
        // --- End Path Finding --- 

        // Assign the found path to the node's data
        d.pathToMain = calculatedPath; // Assign the final calculated path
        console.log(`[HOVER PATH ASSIGN] Node: '${d.word}' (ID: ${d.id}), Assigned Path:`, d.pathToMain);
  
        // Populate pathNodeIds and pathLinkIds *after* BFS completes if path was found
        if (d.pathToMain) { 
            d.pathToMain.forEach((id: string) => pathNodeIds.add(id));
            for (let i = 0; i < d.pathToMain.length - 1; i++) { 
                const pathStartNodeId = d.pathToMain[i];
                const pathEndNodeId = d.pathToMain[i+1];
                const actualLink = baseLinks.find(fl => 
                    ((fl.source === pathStartNodeId && fl.target === pathEndNodeId) || 
                     (fl.source === pathEndNodeId && fl.target === pathStartNodeId))
                );
                if (actualLink) {
                    pathLinkIds.add(`${actualLink.source as string}_${actualLink.target as string}`);
                }
            }
        }
        // --- End populating path sets --- 
  
        // Find direct connections (neighbors) using baseLinks
        const directNeighborIds = new Set<string>();
        baseLinks.forEach((l) => { 
            // --- REVERTED: Treat link source/target as STRINGS --- 
            const sourceId = l.source as string;
            const targetId = l.target as string;
            if (sourceId === d.id) directNeighborIds.add(targetId);
            if (targetId === d.id) directNeighborIds.add(sourceId);
            // --- END REVERT ---
        });
  
        // Combine path nodes and direct neighbors for highlighting (using string IDs)
        // --- ADDED: Detailed logging for neighbor finding ---
        console.log(`[HOVER NEIGHBOR] Finding neighbors for hovered node ID: ${d.id} (Type: ${typeof d.id})`);
        baseLinks.forEach((l, index) => { 
            const sourceId = l.source as string;
            const targetId = l.target as string;
            // Log the comparison attempt
            if (index < 15) { // Log first 15 link comparisons
                 console.log(`[HOVER NEIGHBOR #${index}] Comparing d.id='${d.id}'(type:${typeof d.id}) with link src='${sourceId}'(type:${typeof sourceId}), tgt='${targetId}'(type:${typeof targetId})`);
            }
            // Explicitly check types before comparison
            let matchFound = false;
            if (typeof d.id === 'string' && typeof sourceId === 'string' && sourceId === d.id) {
                 if (typeof targetId === 'string') {
                      directNeighborIds.add(targetId);
                      matchFound = true;
                 }
            }
            if (typeof d.id === 'string' && typeof targetId === 'string' && targetId === d.id) {
                 if (typeof sourceId === 'string') {
                     directNeighborIds.add(sourceId);
                     matchFound = true;
                 }
            }
            if (index < 15 && matchFound) {
                 console.log(`[HOVER NEIGHBOR #${index}]   MATCH FOUND! Added neighbor.`);
            }
        });
        console.log('[HOVER NEIGHBOR] Finished loop. Found neighbors:', Array.from(directNeighborIds));
        // --- END: Detailed logging ---
  
        // Combine path nodes and direct neighbors for highlighting (using string IDs)
        const highlightNodeIds = new Set<string>([d.id]);
        pathNodeIds.forEach(id => highlightNodeIds.add(id));
        directNeighborIds.forEach(id => highlightNodeIds.add(id));
  
        // --- Diagnostic Logging (Keep this as well) ---
        console.log(`[HOVER ENTER] Node: '${d.word}' (ID: ${d.id})`);
        console.log('[HOVER ENTER] Main Word ID:', mainWordIdString);
        console.log('[HOVER ENTER] Path Node IDs:', Array.from(pathNodeIds)); // Now populated correctly
        console.log('[HOVER ENTER] Path Link IDs:', Array.from(pathLinkIds)); // Now populated correctly
        console.log('[HOVER ENTER] Direct Neighbor IDs:', Array.from(directNeighborIds)); // Log the final result
        console.log('[HOVER ENTER] Highlight Node IDs:', Array.from(highlightNodeIds));
        // --- END DIAGNOSTIC LOGGING --- 
  
        // --- Dim non-highlighted elements --- 
        d3.selectAll<SVGGElement, CustomNode>(".node")
          // <<< CHANGE: Filter using string ID >>>
          .filter(n => !highlightNodeIds.has(n.id)) 
          .transition().duration(250)
          .style("opacity", 0.6); // MODIFIED: Dim non-highlighted nodes to 0.6
  
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
          .style("stroke-opacity", 0.5); // MODIFIED: Dim non-highlighted links to 0.5
  
        d3.selectAll<SVGTextElement, CustomNode>(".node-label")
          // <<< CHANGE: Filter using string ID >>>
          .filter(n => !highlightNodeIds.has(n.id)) 
          .transition().duration(250)
          .style("opacity", 0.6); // MODIFIED: Dim non-highlighted labels to 0.6
  
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
            // Correctly access source and target IDs from CustomNode objects
            const sourceNodeId = (l.source as CustomNode).id;
            const targetNodeId = (l.target as CustomNode).id;

            const linkIdForward = `${sourceNodeId}_${targetNodeId}`;
            const linkIdBackward = `${targetNodeId}_${sourceNodeId}`;

            // Highlight if:
            // 1. Directly connected to the hovered node (d)
            // 2. Part of the path to the main word (pathLinkIds)
            return ( (sourceNodeId === d.id && directNeighborIds.has(targetNodeId)) ||
                     (targetNodeId === d.id && directNeighborIds.has(sourceNodeId)) ||
                     pathLinkIds.has(linkIdForward) || pathLinkIds.has(linkIdBackward) );
          })
          .raise()
          .style("stroke-opacity", 0.9) 
          .attr("stroke-width", 2.5)
          .each(function(l: CustomLink) {
            let determinedColor: string;

            // Log the link being processed, accessing IDs from CustomNode objects
            console.log(`[HOVER LINK COLOR] Processing link: source=${(l.source as CustomNode).id}, target=${(l.target as CustomNode).id}, relationship=${l.relationship}`);

            if (l.relationship) {
              // Color the link based on its own relationship type
              determinedColor = getNodeColor(l.relationship);
              console.log(`[HOVER LINK COLOR]   Link relationship: '${l.relationship}', Determined color: ${determinedColor}`);
            } else {
              // Fallback for links with no relationship type
              determinedColor = themeMode === 'dark' ? "#999" : "#777";
              console.warn(`[HOVER LINK COLOR]   Link has no relationship type. Using fallback color.`);
            }

            d3.select(this).style("stroke", determinedColor);
          });
          /* REMOVED .each() block: // Keep comment showing previous state if desired
          .each(function(l: CustomLink) {
            // ... old coloring logic ...
            d3.select(this).style("stroke", color);
          });
          */
  
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
        // const mainWordNodeEntry = Array.from(nodeMap.entries()).find(([id, node]) => node.word === mainWord); // ALREADY REMOVED/USING COMPONENT SCOPE
        // const mainWordIdString = mainWordNodeEntry ? mainWordNodeEntry[0] : null; // ALREADY REMOVED/USING COMPONENT SCOPE
  
        // --- REMOVED TRANSITIONS for reset ---
        d3.selectAll<SVGGElement, CustomNode>(".node")
          //.transition().duration(200) // REMOVED TRANSITION
          .style("opacity", 1) // Reset all nodes to opacity 1
          .attr("transform", n => `translate(${n.x || 0},${n.y || 0})`) 
          .select("circle")
            .attr("stroke-width", 0.5) 
            .attr("stroke-opacity", 0.6) // This is for the circle's own stroke, not the node group opacity
            // <<< CHANGE: Compare string IDs for main node filter >>>
            .attr("filter", n => n.id === mainWordIdString ? `url(#apple-node-shadow) brightness(1.15)` : `url(#apple-node-shadow)`) 
            .attr("stroke", n => { 
              const baseColor = d3.color(getNodeColor(n.group)) || d3.rgb("#888");
              return themeMode === 'dark' ? baseColor.brighter(0.3).toString() : baseColor.brighter(0.5).toString();
            }); 
            
        d3.selectAll<SVGLineElement, CustomLink>(".link")
          //.transition().duration(200) // REMOVED TRANSITION
          .style("stroke-opacity", 1) // MODIFIED: Reset to new default 1
          .attr("stroke-width", 1.5)
          .style("stroke", themeMode === "dark" ? "#666" : "#ccc"); 
          
        d3.selectAll<SVGTextElement, CustomNode>(".node-label")
          //.transition().duration(200) // REMOVED TRANSITION
          .style("opacity", 1) // Reset labels to opacity 1
          // <<< CHANGE: Compare string IDs for main node font weight >>>
          .style("font-weight", n => n.id === mainWordIdString ? "bold" : "normal"); 
      });
      
      // Double-click to navigate with improved visual feedback
      nodeSelection.on("dblclick", (event, d) => {
        const currentTargetElement = event.currentTarget as SVGGElement;
        event.preventDefault();
        event.stopPropagation();
        
        if (isDraggingRef.current) return;
        // REMOVED: setPeekedNode(null);
        
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
          // const mainWordNodeEntry = Array.from(nodeMap.entries()).find(([id, node]) => node.word === mainWord); // ALREADY REMOVED/USING COMPONENT SCOPE
          // const mainWordIdString = mainWordNodeEntry ? mainWordNodeEntry[0] : null; // ALREADY REMOVED/USING COMPONENT SCOPE
  
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
        
        // MODIFIED: Always set hovered node for tooltip, as peek is removed
             setHoveredNode({ 
                 ...d, 
                 // relationshipToMain is removed, relationshipFromParent is already in d 
             });
             
             const { clientX, clientY } = event;
             setTooltipPosition({ x: clientX, y: clientY });
      });
      
      nodeSelection.on("mouseout", (event, d) => {
          if (isDraggingRef.current) return;
          // MODIFIED: Always hide tooltip on mouseout, as peek is removed
          setHoveredNode(null);
      });
  
    }, [mainWord, onNodeClick, getNodeColor, themeMode, nodeMap, getNodeRadius, baseLinks, mainWordIdString]); // MODIFIED: Removed peekedNode from dependencies, ADDED isMobile
  
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
                    .attr("class", d => `node node-group-${d.group} ${d.id === mainWord ? "main-node" : ""}`) 
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
                  .attr("stroke-width", d => d.id === mainWord ? 2.5 : 1.5) 
                  .attr("stroke-opacity", 0.7)
                  .attr("shape-rendering", "geometricPrecision");
  
                // <<< CHANGE: Use mainWord string for opacity check >>>
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
                      .attr("font-size", d => d.id === mainWord ? "12px" : "10px") 
                      .attr("font-weight", d => d.id === mainWord ? "bold" : "normal") 
                      // <<< CHANGE: Use word for text content >>>
                      .text(d => d.word) 
                      .attr("x", d => d.x ?? 0)
                      .attr("y", d => (d.y ?? 0) + getNodeRadius(d) + 12)
                      .style("opacity", 0) 
                      .style("pointer-events", "none") 
                      .style("user-select", "none");
  
                  // Halo creation remains the same
                  textElement.clone(true)
                      .attr("class", "node-label-halo") // Ensure this class is defined or handled
                      .attr("fill", "none")
                      .attr("stroke", themeMode === "dark" ? "rgba(0,0,0,0.8)" : "rgba(255,255,255,0.8)")
                      .attr("stroke-width", 3)
                      .attr("stroke-linejoin", "round");
  
                  // Text fill remains the same
                  textElement.attr("fill", themeMode === "dark" ? "#f5f5f7" : "#000000"); // MODIFIED: Light mode to #000000, dark to #f5f5f7
  
                  // <<< CHANGE: Use mainWord string for opacity check >>>
                  textElement.call(enter => enter.transition().duration(300).style("opacity", 1)); // Default opacity 1
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
        // REMOVED: setPeekedNode(null);
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
        if (d.id !== mainWord) { 
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
    }, [mainWord, getNodeRadius, getNodeColor, themeMode, setupNodeInteractions, ticked, svgRef, dragStartTimeRef]); 
  
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
      if (isMobile) return; // Ensure this line is present to hide legend on mobile
  
      // Remove previous legend if it exists
      svg.select(".legend").remove();
  
      // --- ADJUST SIZES BASED ON isMobile ---
      const pcScaleFactor = 0.75; // Ensure PC scale factor is 0.75
      const legendPadding = isMobile ? 16 : Math.round(16 * pcScaleFactor);
      const legendItemHeight = isMobile ? 24 : Math.round(24 * pcScaleFactor);
      const dotRadius = isMobile ? 4 : Math.round(4 * pcScaleFactor);
      const textPadding = isMobile ? 12 : Math.round(12 * pcScaleFactor);
      const categorySpacing = isMobile ? 12 : Math.round(12 * pcScaleFactor);
      const maxLabelWidth = isMobile ? 120 : Math.round(120 * pcScaleFactor);

      const titleFontSize = isMobile ? 13 : 10; // Ensure PC titleFontSize is 10
      const subtitleFontSize = isMobile ? 10 : 8; // Ensure PC subtitleFontSize is 8
      const categoryHeaderFontSize = isMobile ? 11 : 9; // Ensure PC categoryHeaderFontSize is 9
      const itemLabelFontSize = isMobile ? 11 : 9; // Ensure PC itemLabelFontSize is 9
      // --- END ADJUST SIZES ---
  
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
          .style("font-size", muiTheme.typography.pxToRem(itemLabelFontSize)) // Use adjusted itemLabelFontSize
        .style("opacity", 0);
      let maxTextWidth = 0;
      let maxCategoryWidth = 0;
      legendCategories.forEach((category) => {
        tempText.style("font-weight", 600).style("font-size", muiTheme.typography.pxToRem(categoryHeaderFontSize)).text(category.name); // Use adjusted categoryHeaderFontSize
        const categoryWidth = tempText.node()?.getBBox().width || 0;
        maxCategoryWidth = Math.max(maxCategoryWidth, categoryWidth);
        category.labels.forEach(labelInfo => {
          tempText.style("font-weight", 400).style("font-size", muiTheme.typography.pxToRem(itemLabelFontSize)).text(labelInfo.label); // Use adjusted itemLabelFontSize
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
      calculatedHeight += isMobile ? 24 : Math.round(24 * pcScaleFactor); // Space for title (adjusted)
      calculatedHeight += isMobile ? 18 : Math.round(18 * pcScaleFactor); // Space for subtitle (adjusted)
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
        .attr("x", legendWidth / 2).attr("y", legendPadding + (isMobile ? 10 : Math.round(10 * pcScaleFactor))).attr("text-anchor", "middle")
        .style("font-size", muiTheme.typography.pxToRem(titleFontSize)).style("font-weight", 600)
        .attr("fill", textColorPrimary).text("Relationship Types");
      legendContainer.append("text") // Subtitle
        .attr("x", legendWidth / 2).attr("y", legendPadding + (isMobile ? 26 : Math.round(26 * pcScaleFactor))).attr("text-anchor", "middle")
        .style("font-size", muiTheme.typography.pxToRem(subtitleFontSize)).attr("fill", textColorSecondary)
        .text("Click to filter");
  
      let yPos = legendPadding + (isMobile ? 40 : Math.round(40 * pcScaleFactor)) + categorySpacing; // Adjusted starting yPos
  
      legendCategories.forEach((category) => {
        // Category Header
        legendContainer.append("text")
          .attr("x", legendPadding).attr("y", yPos + legendItemHeight / 2).attr("dy", ".35em")
          .style("font-weight", 600).style("font-size", muiTheme.typography.pxToRem(categoryHeaderFontSize))
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
              .style("font-size", muiTheme.typography.pxToRem(itemLabelFontSize))
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
    }, [mainWord, renderOrUpdateLegend, isMobile]); // Add isMobile dependency
  
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
      // Ensure prerequisites are met
      if (!mainWord || baseNodes.length === 0 || !wordNetwork?.nodes) {
        console.log("[FILTER DEBUG] Prerequisites not met, returning empty arrays."); // <-- ADDED DEBUG
        return { filteredNodes: [], filteredLinks: [] };
      }
  
      console.log("[FILTER DEBUG] Applying depth/breadth and relationship filters"); // <-- ADDED DEBUG
      console.log("[FILTER DEBUG] Main word:", mainWord); // <-- ADDED DEBUG
      console.log("[FILTER DEBUG] Base nodes count:", baseNodes.length); // <-- ADDED DEBUG
      console.log("[FILTER DEBUG] Base links count:", baseLinks.length); // <-- ADDED DEBUG
      console.log("[FILTER DEBUG] Depth limit:", depth); // <-- ADDED DEBUG
      console.log("[FILTER DEBUG] Breadth limit:", breadth); // <-- ADDED DEBUG
      console.log("[FILTER DEBUG] Active relationship filters:", filteredRelationships); // <-- ADDED DEBUG
  
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
  
      // console.log(`[FILTER DEBUG BFS Start] Queue:`, JSON.stringify(queue)); // Optional: Log initial queue state
  
      while (head < queue.length) {
        const { nodeId, currentDepth } = queue[head++];
  
        // console.log(`[FILTER DEBUG BFS Loop] Processing nodeId: ${nodeId} at depth ${currentDepth}`); // Optional
  
        if (currentDepth >= depth) continue; // Stop if max depth reached
  
        if (nodesAddedAtDepth[currentDepth] === undefined) {
          nodesAddedAtDepth[currentDepth] = 0;
        }
  
        baseLinks.forEach((link, linkIndex) => {
          let neighborId: string | null = null;
          const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
          const targetId = typeof link.target === 'object' ? link.target.id : link.target;
  
          // --- DETAILED LOGS FOR NEIGHBOR FINDING ---
          // Log first few links, and any link that involves the current nodeId, to reduce log spam
          const shouldLogThisLink = linkIndex < 5 || sourceId === nodeId || targetId === nodeId;
          if (shouldLogThisLink) {
            console.log(`[FILTER DEBUG BFS Detail] Current BFS Node ID: '${nodeId}' (Depth: ${currentDepth})`);
            console.log(`[FILTER DEBUG BFS Detail] Iterating Link #${linkIndex}: source='${link.source}' (becomes '${sourceId}'), target='${link.target}' (becomes '${targetId}')`);
            console.log(`[FILTER DEBUG BFS Detail]   Comparing with source: ('${sourceId}' === '${nodeId}') is ${sourceId === nodeId}`);
            console.log(`[FILTER DEBUG BFS Detail]   Comparing with target: ('${targetId}' === '${nodeId}') is ${targetId === nodeId}`);
          }
          // --- END DETAILED LOGS ---
  
          if (sourceId === nodeId && !connectedNodeIds.has(targetId as string)) {
            neighborId = targetId;
            // if (shouldLogThisLink) console.log(`[FILTER DEBUG BFS Detail]     Found potential neighbor via source: ${neighborId}`);
          } else if (targetId === nodeId && !connectedNodeIds.has(sourceId as string)) {
            neighborId = sourceId;
            // if (shouldLogThisLink) console.log(`[FILTER DEBUG BFS Detail]     Found potential neighbor via target: ${neighborId}`);
          }
  
          if (neighborId && shouldLogThisLink) {
             console.log(`[FILTER DEBUG BFS Detail]   Neighbor ID determined for Link #${linkIndex}: ${neighborId}`);
          }
  
          if (neighborId && nodeMap.has(neighborId)) {
             const nextDepth = currentDepth + 1;
             if (nodesAddedAtDepth[nextDepth] === undefined) {
               nodesAddedAtDepth[nextDepth] = 0;
             }
             if (breadth === 0 || nodesAddedAtDepth[nextDepth] < breadth) {
               if (!connectedNodeIds.has(neighborId)) { // Double check before adding to queue
               connectedNodeIds.add(neighborId);
               queue.push({ nodeId: neighborId, currentDepth: nextDepth });
               nodesAddedAtDepth[nextDepth]++;
                    // if (shouldLogThisLink) console.log(`[FILTER DEBUG BFS Detail]       Added neighbor '${neighborId}' to queue. connectedNodeIds size: ${connectedNodeIds.size}`);
               }
             }
          }
        });
      }
      console.log(`[FILTER DEBUG] BFS finished. Found ${connectedNodeIds.size} connected node IDs.`); // <-- ADDED DEBUG
  
      // --- Filter nodes and links based ONLY on BFS results FIRST ---
      const bfsFilteredNodes = baseNodes.filter(node => connectedNodeIds.has(node.id));
      const bfsFilteredLinks = baseLinks.filter(link => {
          const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
          const targetId = typeof link.target === 'object' ? link.target.id : link.target;
          return connectedNodeIds.has(sourceId as string) && connectedNodeIds.has(targetId as string);
      });
  
      console.log(`[FILTER DEBUG] After BFS: ${bfsFilteredNodes.length} nodes, ${bfsFilteredLinks.length} links`); // <-- ADDED DEBUG
  
      // --- Now apply relationship type filters if any are active ---
      let finalFilteredNodes = bfsFilteredNodes;
      let finalFilteredLinks = bfsFilteredLinks;
  
      if (filteredRelationships.length > 0) {
          console.log('[FILTER DEBUG] Applying relationship type filters:', filteredRelationships); // <-- ADDED DEBUG
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
          console.log(`[FILTER DEBUG] After relationship filters: ${finalFilteredNodes.length} nodes, ${finalFilteredLinks.length} links`); // <-- ADDED DEBUG
  
      } else {
        console.log('[FILTER DEBUG] No relationship filters active.'); // <-- ADDED DEBUG
      }
  
      // Ensure the main word node is *always* included if it was originally present
      // const mainWordNode = baseNodes.find(node => node.group === 'main'); // Reuse mainWordNode - REMOVED REDECLARATION
      // const mainWordIdString = mainWordNode ? mainWordNode.id : null; // REMOVED REDECLARATION
      if (mainWordNode && mainWordIdString && !finalFilteredNodes.some(n => n.id === mainWordIdString)) { // Check mainWordIdString exists too
          console.warn("[FILTER DEBUG] Main word node was filtered out, re-adding it."); // <-- ADDED DEBUG
          finalFilteredNodes.push(mainWordNode); // Add it back if filters removed it
      }
  
      console.log(`[FILTER DEBUG] Returning: ${finalFilteredNodes.length} nodes, ${finalFilteredLinks.length} links`); // <-- ADDED DEBUG
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
  
      // --- MODIFIED: Create copies for D3 simulation/rendering --- 
      // This prevents D3 from mutating the original arrays used by React logic
      console.log("[D3 INIT] Creating copies of nodes/links for simulation.");
      console.log(`[D3 INIT] Filtered nodes count: ${filteredNodes.length}, Filtered links count: ${filteredLinks.length}`);
      const nodesCopy = filteredNodes.map(n => ({ ...n })); 
      const linksCopy = filteredLinks.map(l => ({ ...l })); 
      // --- END MODIFICATION ---

      // --- Pass COPIES to D3 functions --- 
      const currentSim = setupSimulation(nodesCopy, linksCopy, width, height);
      const linkElements = createLinks(g, linksCopy); 
      const nodeElements = createNodes(g, nodesCopy, currentSim);
      // --- END --- 

      setupNodeInteractions(nodeElements);
  
      // --- Render Initial Legend --- 
      renderOrUpdateLegend(svg, width);
  
      if (currentSim) {
        // --- MODIFIED: Find main node in the COPY --- 
        const mainNodeData = nodesCopy.find(n => n.id === mainWord); // Use label ID
        // --- END MODIFICATION ---
        if (mainNodeData) {
            mainNodeData.fx = 0;
            mainNodeData.fy = 0;
        }
        currentSim.alpha(1).restart();
      }
  
      // --- MODIFIED: Pass original filteredNodes to centerOnMainWord ---
      const centerTimeout = setTimeout(() => centerOnMainWord(svg, filteredNodes), 800);
      // --- END MODIFICATION ---
  
      // --- Cleanup --- 
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
          // --- MODIFIED: Pass original filteredNodes to centerOnMainWord ---
          if (svgRef.current) centerOnMainWord(d3.select(svgRef.current), filteredNodes);
          // --- END MODIFICATION ---
        }, 800);
        return () => clearTimeout(recenterTimeout);
      }
      prevMainWordRef.current = mainWord;
    }, [mainWord, centerOnMainWord, filteredNodes]); // Keep filteredNodes dependency
  
    // Improved tooltip with relationship info directly from old_src_2
    const renderTooltip = useCallback(() => {
      console.log('[WordGraph] Checking getRelationshipTypeLabel:', typeof getRelationshipTypeLabel, getRelationshipTypeLabel);
      // MODIFIED: Simplified condition, as peekedNode is removed
      if (!hoveredNode?.id || !hoveredNode?.x || !hoveredNode?.y || !svgRef.current) {
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
           {/* Path to Main Info */}
           {hoveredNode.pathToMain && hoveredNode.pathToMain.length > 1 && (
             <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: `1px solid ${themeMode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)'}`, fontSize: '11px', color: themeMode === 'dark' ? '#c9d1d9' : '#57606a' }}>
               {/* Path Display */}
               <div style={{ marginBottom: '4px' }}>
                 <span style={{ fontWeight: '600', color: themeMode === 'dark' ? '#8b949e' : '#6e7781', marginRight: '5px' }}>Path:</span>
                 <span>{hoveredNode.pathToMain.map((nodeId, index) => (
                   <React.Fragment key={nodeId}>
                     {index > 0 && <span style={{ margin: '0 3px', opacity: 0.6 }}>→</span>}
                     <span style={{ fontStyle: index === 0 ? 'italic' : 'normal' }}>{nodeId}</span>
                   </React.Fragment>
                 ))}</span>
               </div>
               {/* Degrees and Via */}
               <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                 <div>
                   <span style={{ fontWeight: '600', color: themeMode === 'dark' ? '#8b949e' : '#6e7781', marginRight: '5px' }}>Degrees:</span>
                   <span style={{ fontWeight: '500' }}>{hoveredNode.pathToMain.length - 1}</span>
                 </div>
                 {hoveredNode.pathToMain.length > 2 && (
                   <div style={{ borderLeft: `1px solid ${themeMode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)'}`, paddingLeft: '10px' }}>
                     <span style={{ fontWeight: '600', color: themeMode === 'dark' ? '#8b949e' : '#6e7781', marginRight: '5px' }}>Via:</span>
                     <span style={{ fontWeight: '500', fontStyle: 'italic', color: getNodeColor(hoveredNode.group) }}>{hoveredNode.pathToMain[hoveredNode.pathToMain.length - 2]}</span>
                   </div>
                 )}
               </div>
             </div>
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
             {/* <div style={{ display: "flex", alignItems: "center", gap: "4px", opacity: 0.7 }}>
               <span style={{ 
                 fontSize: "10px", 
                 background: themeMode === "dark" ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.06)", 
                 borderRadius: "3px", 
                 padding: "1px 4px"
               }}>Right-click</span>
               <span>Peek</span>
             </div> */}
          </div>
        </div>
      );
    }, [hoveredNode, themeMode, getNodeColor, mainWord, svgRef, baseLinks, nodeMap, getRelationshipTypeLabel, isMobile]); // MODIFIED: Removed peekedNode from dependencies, ADDED isMobile
  
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
                {isMobile && (
                  <IconButton
                    onClick={handleToggleMobileLegend}
                    color="primary"
                    aria-label="view legend"
                    size="small" // Make button smaller for mobile
                    sx={{
                      mr: 0.5, // Adjust margin
                      padding: '6px',
                      border: `1px solid ${alpha(muiTheme.palette.primary.main, 0.3)}`,
                      backgroundColor: alpha(muiTheme.palette.primary.main, 0.05),
                       '&:hover': {
                          backgroundColor: alpha(muiTheme.palette.primary.main, 0.1),
                       }
                    }}
                  >
                    <InfoIcon fontSize="medium" /> {/* Slightly larger icon */}
                  </IconButton>
                )}
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
              
              {/* Tooltip */}
        {renderTooltip()}

        {/* Mobile Legend Dialog */}
        <Dialog open={mobileLegendOpen} onClose={handleToggleMobileLegend} fullWidth maxWidth="xs" scroll="paper">
          <DialogTitle sx={{ pb: 1, fontSize: '1.1rem', borderBottom: `1px solid ${muiTheme.palette.divider}` }}>
            Graph Legend
          </DialogTitle>
          <DialogContent dividers sx={{ p: 0, '& .MuiListSubheader-root': { lineHeight: '32px', py: 0.25 } }}>
            <List dense disablePadding>
              {getUniqueRelationshipGroups().categories.map((category, catIndex) => (
                <React.Fragment key={category.name}>
                  <ListSubheader
                    disableSticky
                    sx={{
                      bgcolor: alpha(muiTheme.palette.background.default, 0.95), // Use default for slight transparency
                      fontSize: '0.75rem',
                      fontWeight: 'bold',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      color: muiTheme.palette.text.secondary,
                      borderTop: catIndex > 0 ? `1px solid ${muiTheme.palette.divider}` : 'none',
                      // borderBottom: `1px solid ${muiTheme.palette.divider}`, // Removed bottom border from subheader
                      py: 0.75,
                      px: 2, // Add padding to subheader
                    }}
                  >
                    {category.name}
                  </ListSubheader>
                  {category.labels.map((labelInfo) => {
                    const isFiltered = labelInfo.types.every(t => filteredRelationships.includes(t.toLowerCase()));
                    return (
                      <ListItemButton
                        key={labelInfo.label}
                        dense
                        onClick={() => {
                          handleToggleRelationshipFilter(labelInfo.types);
                        }}
                        sx={{
                          pl: 2, // Indent items under subheader
                          opacity: isFiltered ? 0.45 : 1,
                          transition: 'opacity 0.2s ease-in-out, background-color 0.15s linear',
                          '&:hover': {
                            backgroundColor: alpha(labelInfo.color, 0.12),
                          },
                          borderBottom: `1px solid ${muiTheme.palette.divider}`, // Add border to each item
                           '&:last-child': { // Remove border from last item in group (if needed, but might look okay)
                             // borderBottom: 'none', 
                           },
                        }}
                      >
                        <ListItemIcon sx={{ minWidth: 28, pl: 0 }}>
                          <Box
                            sx={{
                              width: 12,
                              height: 12,
                              borderRadius: '50%',
                              backgroundColor: labelInfo.color,
                              border: `1px solid ${alpha(labelInfo.color, isFiltered ? 0.3 : 0.7)}`,
                              boxShadow: `0 0 4px ${alpha(labelInfo.color, isFiltered ? 0.2 : 0.4)}`,
                              transition: 'all 0.2s ease-in-out',
                            }}
                          />
                        </ListItemIcon>
                        <ListItemText
                          primary={labelInfo.label}
                          primaryTypographyProps={{
                            fontSize: '0.875rem',
                            fontWeight: isFiltered ? 300 : 400,
                            color: isFiltered ? muiTheme.palette.text.disabled : muiTheme.palette.text.primary,
                            style: {
                               textDecoration: isFiltered ? 'line-through' : 'none',
                               textDecorationColor: alpha(muiTheme.palette.text.disabled, 0.7)
                            }
                          }}
                        />
                      </ListItemButton>
                    );
                  })}
                </React.Fragment>
              ))}
            </List>
          </DialogContent>
          <DialogActions sx={{ pt: 1.5, pb: 1.5, pr: 2, borderTop: `1px solid ${muiTheme.palette.divider}` }}>
            <Button onClick={handleToggleMobileLegend} variant="outlined" size="small">Close</Button>
          </DialogActions>
        </Dialog>
      </div>
    );
  };
  
  export default WordGraph;
  