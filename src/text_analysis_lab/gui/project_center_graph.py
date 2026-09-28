"""Graph layout, interaction, and UI-state persistence for Project Center."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_LAYOUT_VERSION = 1
_LAYOUT_MODES = ("lineage", "provenance")


def _empty_layout_state() -> dict[str, Any]:
    return {
        "version": _LAYOUT_VERSION,
        "graph_layouts": {mode: {"positions": {}} for mode in _LAYOUT_MODES},
    }


def project_center_layout_path(manifest_path: str | Path) -> Path:
    """Return the project-local Project Center UI state path."""
    return Path(manifest_path).parent / "ui" / "project_center.json"


def load_project_center_graph_layout(manifest_path: str | Path) -> dict[str, Any]:
    """Load normalized graph layout state, or an empty state when none exists."""
    path = project_center_layout_path(manifest_path)
    if not path.is_file():
        return _empty_layout_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_layout_state()
    if not isinstance(payload, Mapping):
        return _empty_layout_state()
    return _normalize_layout_state(payload)


def save_project_center_graph_layout(
    manifest_path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically persist normalized Project Center graph UI state."""
    normalized = _normalize_layout_state(payload)
    path = project_center_layout_path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".project_center-", suffix=".json.tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return normalized


def _normalize_layout_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    layouts = payload.get("graph_layouts", {})
    if not isinstance(layouts, Mapping):
        layouts = {}
    normalized = _empty_layout_state()
    for mode in _LAYOUT_MODES:
        raw_layout = layouts.get(mode, {})
        if not isinstance(raw_layout, Mapping):
            continue
        raw_positions = raw_layout.get("positions", {})
        if not isinstance(raw_positions, Mapping):
            continue
        positions: dict[str, dict[str, Any]] = {}
        for node_id, raw_position in raw_positions.items():
            if not isinstance(node_id, str) or not node_id:
                continue
            if not isinstance(raw_position, Mapping):
                continue
            x = raw_position.get("x")
            y = raw_position.get("y")
            if isinstance(x, bool) or isinstance(y, bool):
                continue
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            x_float = float(x)
            y_float = float(y)
            if not math.isfinite(x_float) or not math.isfinite(y_float):
                continue
            positions[node_id] = {
                "x": x_float,
                "y": y_float,
                "pinned": bool(raw_position.get("pinned", False)),
            }
        normalized["graph_layouts"][mode]["positions"] = positions
    return normalized


PROJECT_CENTER_GRAPH_SCRIPT = r"""
function graphEdges() {
  return $('graphMode').value === 'lineage' ? app.graph.lineage_edges : app.graph.provenance_edges;
}
function graphNodes() {
  const nodes = $('graphMode').value === 'lineage' ? app.graph.artifact_nodes : app.graph.nodes;
  if ($('showDeletedArtifacts')?.checked) return nodes;
  return nodes.filter(node => node.node_type !== 'artifact' || !node.deleted);
}
function graphSelectedSeeds(edges) {
  if (app.selectedArtifact) return [app.selectedArtifact];
  if (!app.selectedOperation) return [];
  if ($('graphMode').value === 'provenance') return [app.selectedOperation];
  const seeds = new Set();
  for (const edge of edges) {
    if (edge.operation_id === app.selectedOperation) {
      seeds.add(edge.source); seeds.add(edge.target);
    }
  }
  return Array.from(seeds);
}
function graphNeighborhood(nodes, edges, seeds, radius = 2) {
  const ids = new Set(nodes.map(node => node.id));
  const adjacent = new Map(nodes.map(node => [node.id, new Set()]));
  for (const edge of edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target)) continue;
    adjacent.get(edge.source).add(edge.target);
    adjacent.get(edge.target).add(edge.source);
  }
  const seen = new Set(seeds.filter(id => ids.has(id)));
  let frontier = Array.from(seen);
  for (let step = 0; step < radius; step++) {
    const next = [];
    for (const id of frontier) {
      for (const neighbor of adjacent.get(id) || []) {
        if (seen.has(neighbor)) continue;
        seen.add(neighbor); next.push(neighbor);
      }
    }
    frontier = next;
    if (!frontier.length) break;
  }
  return seen;
}
function graphData() {
  let nodes = graphNodes();
  let edges = graphEdges();
  if (app.graphFocused) {
    const seeds = graphSelectedSeeds(edges);
    if (seeds.length) {
      const keep = graphNeighborhood(nodes, edges, seeds, 2);
      nodes = nodes.filter(node => keep.has(node.id));
      edges = edges.filter(edge => keep.has(edge.source) && keep.has(edge.target));
    }
  }

  const query = $('artifactSearch').value.trim().toLowerCase();
  if (query) {
    const matched = new Set();
    for (const node of nodes) {
      const fields = [node.id, node.label, node.node_type, node.artifact_type, node.operation_type, node.operator_id, node.status];
      if (fields.some(value => String(value || '').toLowerCase().includes(query))) matched.add(node.id);
    }
    for (const edge of edges) {
      const fields = [edge.label, edge.operation_label, edge.operation_id, edge.source_label, edge.output_label];
      if (fields.some(value => String(value || '').toLowerCase().includes(query))) {
        matched.add(edge.source); matched.add(edge.target);
      }
    }
    const keep = graphNeighborhood(nodes, edges, Array.from(matched), 1);
    nodes = nodes.filter(node => keep.has(node.id));
    edges = edges.filter(edge => keep.has(edge.source) && keep.has(edge.target));
  }
  return {nodes, edges};
}
function renderGraphControls() {
  const hasSelection = Boolean(app.selectedArtifact || ($('graphMode').value === 'provenance' && app.selectedOperation));
  $('focusSelected').disabled = !hasSelection;
  $('focusSelected').textContent = app.graphFocused ? 'Show all' : 'Focus selected';
  $('focusSelected').classList.toggle('active', app.graphFocused);
  $('focusNote').textContent = app.graphFocused && hasSelection ? 'Showing local neighborhood' : '';
  const lineage = $('graphMode').value === 'lineage';
  $('legendOperation').hidden = lineage;
  $('unfreezeGraphNodes').disabled = graphManualPositions().size === 0;
}
function provenanceEdgeLabel(edge, nodeById) {
  const value = String(edge.label || '').trim();
  if (!value || ['source', 'output', 'input'].includes(value.toLowerCase())) return '';
  const from = nodeById.get(edge.source); const to = nodeById.get(edge.target);
  if (value === String(from?.label || '') || value === String(to?.label || '')) return '';
  return truncate(value, 24);
}
function graphManualPositions() {
  const mode = $('graphMode').value;
  if (!app.graphManualPositions[mode]) app.graphManualPositions[mode] = new Map();
  return app.graphManualPositions[mode];
}
function graphLivePositions() {
  const mode = $('graphMode').value;
  if (!app.graphLivePositions[mode]) app.graphLivePositions[mode] = new Map();
  return app.graphLivePositions[mode];
}
function graphModeData(mode) {
  if (!app.graph) return {nodes: [], edges: []};
  return mode === 'lineage'
    ? {nodes: app.graph.artifact_nodes || [], edges: app.graph.lineage_edges || []}
    : {nodes: app.graph.nodes || [], edges: app.graph.provenance_edges || []};
}
function graphHorizontalGap(mode) {
  return mode === 'lineage' ? 110 : 80;
}
function restoreGraphLayoutState(payload) {
  const layouts = payload?.graph_layouts || {};
  for (const mode of ['lineage', 'provenance']) {
    const live = new Map();
    const manual = new Map();
    const saved = layouts?.[mode]?.positions || {};
    for (const [nodeId, position] of Object.entries(saved)) {
      const x = Number(position?.x); const y = Number(position?.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const point = {x, y};
      live.set(nodeId, point);
      if (position?.pinned) manual.set(nodeId, {...point});
    }
    app.graphLivePositions[mode] = live;
    app.graphManualPositions[mode] = manual;
  }
}
function buildGraphLayoutPayload() {
  const graphLayouts = {};
  for (const mode of ['lineage', 'provenance']) {
    const {nodes} = graphModeData(mode);
    const live = app.graphLivePositions[mode] || new Map();
    const manual = app.graphManualPositions[mode] || new Map();
    const positions = {};
    for (const node of nodes) {
      const point = live.get(node.id);
      if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
      positions[node.id] = {x: point.x, y: point.y, pinned: manual.has(node.id)};
    }
    graphLayouts[mode] = {positions};
  }
  return {version: 1, graph_layouts: graphLayouts};
}
function scheduleGraphLayoutSave(delay = 300) {
  if (!app.graphLayoutLoaded || !app.graph) return;
  if (app.graphLayoutSaveTimer !== null) clearTimeout(app.graphLayoutSaveTimer);
  app.graphLayoutSaveTimer = setTimeout(() => {
    app.graphLayoutSaveTimer = null;
    saveGraphLayoutNow();
  }, Math.max(0, delay));
}
function saveGraphLayoutNow() {
  if (!app.graphLayoutLoaded || !app.graph) return Promise.resolve();
  if (app.graphLayoutSaveTimer !== null) {
    clearTimeout(app.graphLayoutSaveTimer);
    app.graphLayoutSaveTimer = null;
  }
  const payload = buildGraphLayoutPayload();
  const save = async () => {
    try {
      await api('/api/graph-layout', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
    } catch (error) {
      if (!app.sessionEnding) setStatus(`Could not save graph layout: ${error.message}`, true);
    }
  };
  const queue = app.graphLayoutSaveQueue || Promise.resolve();
  app.graphLayoutSaveQueue = queue.then(save, save);
  return app.graphLayoutSaveQueue;
}
function flushGraphLayoutWithBeacon() {
  if (!app.graphLayoutLoaded || !app.graph || !navigator.sendBeacon) return;
  try {
    const body = new Blob([JSON.stringify(buildGraphLayoutPayload())], {type: 'application/json'});
    navigator.sendBeacon('/api/graph-layout', body);
  } catch (_) { /* best-effort page-exit persistence */ }
}
function graphPositionOverlaps(node, candidate, positions, nodeById, ignoreId = null) {
  const metrics = graphNodeMetrics(node);
  const marginX = 18; const marginY = 14;
  for (const [otherId, otherPosition] of positions) {
    if (otherId === ignoreId) continue;
    const otherNode = nodeById.get(otherId);
    if (!otherNode) continue;
    const other = graphNodeMetrics(otherNode);
    const separated = candidate.x + metrics.width + marginX <= otherPosition.x
      || otherPosition.x + other.width + marginX <= candidate.x
      || candidate.y + metrics.height + marginY <= otherPosition.y
      || otherPosition.y + other.height + marginY <= candidate.y;
    if (!separated) return true;
  }
  return false;
}
function graphOpenVerticalSlot(node, preferred, positions, nodeById) {
  const step = graphNodeMetrics(node).height + 22;
  for (let ring = 0; ring < 80; ring++) {
    const offsets = ring === 0 ? [0] : [ring * step, -ring * step];
    for (const offset of offsets) {
      const candidate = {x: Math.max(18, preferred.x), y: Math.max(18, preferred.y + offset)};
      if (!graphPositionOverlaps(node, candidate, positions, nodeById)) return candidate;
    }
  }
  return {x: Math.max(18, preferred.x), y: Math.max(18, preferred.y + 80 * step)};
}
function placeIncrementalGraphNodes(mode, nodes, edges, positions) {
  const nodeById = new Map(nodes.map(node => [node.id, node]));
  const ids = new Set(nodeById.keys());
  const incoming = new Map(nodes.map(node => [node.id, []]));
  for (const edge of edges) {
    if (ids.has(edge.source) && ids.has(edge.target)) incoming.get(edge.target).push(edge.source);
  }
  const seed = graphFastLayout(nodes, edges, mode);
  const placed = new Set(Array.from(positions.keys()).filter(id => ids.has(id)));
  const visiting = new Set();
  const gap = graphHorizontalGap(mode);

  const place = nodeId => {
    if (placed.has(nodeId)) return;
    const node = nodeById.get(nodeId);
    if (!node) return;
    if (visiting.has(nodeId)) {
      const fallback = graphOpenVerticalSlot(node, seed.get(nodeId) || {x: 35, y: 35}, positions, nodeById);
      positions.set(nodeId, fallback); placed.add(nodeId); return;
    }
    visiting.add(nodeId);
    const parentIds = (incoming.get(nodeId) || []).filter(id => nodeById.has(id));
    for (const parentId of parentIds) place(parentId);
    const parents = parentIds
      .map(id => ({node: nodeById.get(id), position: positions.get(id)}))
      .filter(item => item.node && item.position);
    let preferred;
    if (parents.length) {
      const rightEdge = Math.max(...parents.map(item => item.position.x + graphNodeMetrics(item.node).width));
      const centerY = parents.reduce((sum, item) => {
        const metrics = graphNodeMetrics(item.node);
        return sum + item.position.y + metrics.height / 2;
      }, 0) / parents.length;
      preferred = {
        x: rightEdge + gap,
        y: centerY - graphNodeMetrics(node).height / 2,
      };
    } else {
      preferred = seed.get(nodeId) || {x: 35, y: 35};
    }
    positions.set(nodeId, graphOpenVerticalSlot(node, preferred, positions, nodeById));
    visiting.delete(nodeId);
    placed.add(nodeId);
  };

  for (const node of nodes) place(node.id);
}
function initializeGraphLayoutMode(mode) {
  const {nodes, edges} = graphModeData(mode);
  const ids = new Set(nodes.map(node => node.id));
  const live = app.graphLivePositions[mode] || (app.graphLivePositions[mode] = new Map());
  const manual = app.graphManualPositions[mode] || (app.graphManualPositions[mode] = new Map());
  let changed = false;
  for (const id of Array.from(live.keys())) {
    if (!ids.has(id)) { live.delete(id); changed = true; }
  }
  for (const id of Array.from(manual.keys())) {
    if (!ids.has(id)) { manual.delete(id); changed = true; }
  }
  if (!nodes.length) return changed;
  const missing = nodes.filter(node => !live.has(node.id));
  if (!missing.length) return changed;
  if (!live.size) {
    const seed = graphFastLayout(nodes, edges, mode);
    for (const node of nodes) {
      const point = seed.get(node.id);
      if (point) live.set(node.id, {...point});
    }
  } else {
    placeIncrementalGraphNodes(mode, nodes, edges, live);
  }
  app.graphPhysicsPending[mode] = nodes.length > 1;
  return true;
}
function initializeGraphLayouts() {
  let changed = false;
  for (const mode of ['lineage', 'provenance']) changed = initializeGraphLayoutMode(mode) || changed;
  if (changed) scheduleGraphLayoutSave(500);
}
function releaseGraphNode(nodeId) {
  const manual = graphManualPositions();
  if (!manual.has(nodeId)) return false;
  manual.delete(nodeId);
  $('unfreezeGraphNodes').disabled = manual.size === 0;
  scheduleGraphLayoutSave(100);
  return true;
}
function unfreezeAllGraphNodes() {
  const manual = graphManualPositions();
  if (!manual.size) return;
  manual.clear();
  const mode = $('graphMode').value;
  if ($('graphPhysics').checked) app.graphPhysicsPending[mode] = true;
  renderArtifactGraph();
  scheduleGraphLayoutSave(100);
}
function resetCurrentGraphLayout() {
  stopGraphPhysics();
  const mode = $('graphMode').value;
  const {nodes, edges} = graphModeData(mode);
  const live = app.graphLivePositions[mode] || (app.graphLivePositions[mode] = new Map());
  const manual = app.graphManualPositions[mode] || (app.graphManualPositions[mode] = new Map());
  live.clear(); manual.clear();
  const seed = graphFastLayout(nodes, edges, mode);
  for (const node of nodes) {
    const point = seed.get(node.id);
    if (point) live.set(node.id, {...point});
  }
  app.graphPhysicsPending[mode] = $('graphPhysics').checked && nodes.length > 1;
  renderArtifactGraph();
  scheduleGraphLayoutSave(0);
}
function stopGraphPhysics() {
  if (!app.graphSimulation) return;
  app.graphSimulation.stop();
  app.graphSimulation = null;
}
function graphPointerCoordinates(svg, event) {
  const point = svg.createSVGPoint();
  point.x = event.clientX; point.y = event.clientY;
  const matrix = svg.getScreenCTM();
  return matrix ? point.matrixTransform(matrix.inverse()) : point;
}
function graphLayout(nodes, edges) {
  const seed = graphFastLayout(nodes, edges, $('graphMode').value);
  const live = graphLivePositions();
  const manual = graphManualPositions();
  const positions = new Map();
  for (const node of nodes) {
    const source = manual.get(node.id) || live.get(node.id) || seed.get(node.id);
    if (source) positions.set(node.id, {x: source.x, y: source.y});
  }
  return {positions, seed};
}
function renderArtifactGraph() {
  stopGraphPhysics();
  const svg = $('artifactGraph');
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!app.graph) return;
  renderGraphControls();
  const {nodes: visibleNodes, edges} = graphData();
  const nodeById = new Map(visibleNodes.map(node => [node.id, node]));
  const {positions, seed} = graphLayout(visibleNodes, edges);
  const manualPositions = graphManualPositions();
  const livePositions = graphLivePositions();
  const initialWidth = Math.max(760, ...visibleNodes.map(node => {
    const pos = positions.get(node.id) || seed.get(node.id); const metrics = graphNodeMetrics(node);
    return pos ? pos.x + metrics.width + 35 : 0;
  }));
  const initialHeight = Math.max(500, ...visibleNodes.map(node => {
    const pos = positions.get(node.id) || seed.get(node.id); const metrics = graphNodeMetrics(node);
    return pos ? pos.y + metrics.height + 35 : 0;
  }));
  // Extra room lets the simulation breathe without constantly resizing the SVG.
  const width = initialWidth + 140;
  const height = initialHeight + 140;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);

  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
  marker.setAttribute('id', 'arrow'); marker.setAttribute('viewBox', '0 0 10 10'); marker.setAttribute('refX', '9'); marker.setAttribute('refY', '5'); marker.setAttribute('markerWidth', '6'); marker.setAttribute('markerHeight', '6'); marker.setAttribute('orient', 'auto-start-reverse');
  const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  arrow.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z'); arrow.setAttribute('fill', 'currentColor');
  marker.appendChild(arrow); defs.appendChild(marker); svg.appendChild(defs);

  const edgeRecords = [];
  const compassAnchor = (node, position, towardX, towardY) => {
    const metrics = graphNodeMetrics(node);
    const cx = position.x + metrics.width / 2;
    const cy = position.y + metrics.height / 2;
    const angle = Math.atan2(towardY - cy, towardX - cx);
    // SVG Y grows downward, so sectors proceed E, SE, S, SW, W, NW, N, NE.
    const sector = ((Math.round(angle / (Math.PI / 4)) % 8) + 8) % 8;
    const anchors = [
      [1, 0.5, 1, 0],
      [1, 1, Math.SQRT1_2, Math.SQRT1_2],
      [0.5, 1, 0, 1],
      [0, 1, -Math.SQRT1_2, Math.SQRT1_2],
      [0, 0.5, -1, 0],
      [0, 0, -Math.SQRT1_2, -Math.SQRT1_2],
      [0.5, 0, 0, -1],
      [1, 0, Math.SQRT1_2, -Math.SQRT1_2],
    ];
    const [rx, ry, nx, ny] = anchors[sector];
    return {x: position.x + rx * metrics.width, y: position.y + ry * metrics.height, nx, ny};
  };
  const updateEdgeGeometry = record => {
    const from = positions.get(record.edge.source); const to = positions.get(record.edge.target);
    if (!from || !to) return;
    const fromMetrics = graphNodeMetrics(record.fromNode); const toMetrics = graphNodeMetrics(record.toNode);
    const fromCenter = {x: from.x + fromMetrics.width / 2, y: from.y + fromMetrics.height / 2};
    const toCenter = {x: to.x + toMetrics.width / 2, y: to.y + toMetrics.height / 2};
    const start = compassAnchor(record.fromNode, from, toCenter.x, toCenter.y);
    const end = compassAnchor(record.toNode, to, fromCenter.x, fromCenter.y);
    const distance = Math.hypot(end.x - start.x, end.y - start.y);
    const bend = Math.max(32, Math.min(130, distance * 0.34));
    const c1x = start.x + start.nx * bend;
    const c1y = start.y + start.ny * bend;
    const c2x = end.x + end.nx * bend;
    const c2y = end.y + end.ny * bend;
    record.path.setAttribute('d', `M ${start.x} ${start.y} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${end.x} ${end.y}`);
    if (record.label) {
      // Cubic Bezier midpoint at t=.5, which keeps labels near the actual curve.
      const labelX = (start.x + 3 * c1x + 3 * c2x + end.x) / 8;
      const labelY = (start.y + 3 * c1y + 3 * c2y + end.y) / 8;
      record.label.setAttribute('x', String(labelX));
      record.label.setAttribute('y', String(labelY - 5));
    }
  };

  for (const edge of edges) {
    const from = positions.get(edge.source); const to = positions.get(edge.target);
    if (!from || !to) continue;
    const fromNode = nodeById.get(edge.source);
    const toNode = nodeById.get(edge.target);
    if (!fromNode || !toNode) continue;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const deletedEdge = Boolean(fromNode.deleted || toNode.deleted);
    line.setAttribute('class', `edge${deletedEdge ? ' deleted' : ''}`); line.setAttribute('marker-end', 'url(#arrow)');
    svg.appendChild(line);
    const edgeLabel = $('graphMode').value === 'lineage'
      ? truncate(String(edge.label || ''), 24)
      : provenanceEdgeLabel(edge, nodeById);
    let label = null;
    if (edgeLabel) {
      label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('text-anchor', 'middle'); label.setAttribute('class', 'edge-label'); label.textContent = edgeLabel;
      svg.appendChild(label);
    }
    const record = {edge, path: line, label, fromNode, toNode};
    edgeRecords.push(record);
    updateEdgeGeometry(record);
  }

  const groups = new Map();
  const connectedEdges = new Map(visibleNodes.map(node => [node.id, []]));
  for (const record of edgeRecords) {
    connectedEdges.get(record.edge.source)?.push(record);
    connectedEdges.get(record.edge.target)?.push(record);
  }
  const clampPosition = (node, next) => {
    const metrics = graphNodeMetrics(node);
    return {
      x: Math.max(18, Math.min(next.x, width - metrics.width - 18)),
      y: Math.max(18, Math.min(next.y, height - metrics.height - 18)),
    };
  };
  const updateNodeGeometry = node => {
    const group = groups.get(node.id); const position = positions.get(node.id);
    if (!group || !position) return;
    group.setAttribute('transform', `translate(${position.x},${position.y})`);
  };
  const updateAllGeometry = () => {
    for (const node of visibleNodes) updateNodeGeometry(node);
    for (const record of edgeRecords) updateEdgeGeometry(record);
  };
  const setNodePosition = (node, next, remember = false) => {
    const position = clampPosition(node, next);
    positions.set(node.id, position);
    livePositions.set(node.id, {...position});
    if (remember) {
      manualPositions.set(node.id, {...position});
      $('unfreezeGraphNodes').disabled = false;
    }
    updateNodeGeometry(node);
    for (const record of connectedEdges.get(node.id) || []) updateEdgeGeometry(record);
  };
  const mode = $('graphMode').value;
  const ensurePhysics = (strength = 0.8) => {
    if (!$('graphPhysics').checked || visibleNodes.length < 2) return;
    if (!app.graphSimulation) {
      app.graphSimulation = startGraphPhysics({
        nodes: visibleNodes,
        edges,
        positions,
        seed,
        manualPositions,
        livePositions,
        nodeById,
        width,
        height,
        updateAllGeometry,
        onSettled: () => {
          app.graphPhysicsPending[mode] = false;
          scheduleGraphLayoutSave(150);
        },
      });
    }
    app.graphSimulation.reheat(strength);
  };

  for (const node of visibleNodes) {
    const pos = positions.get(node.id);
    if (!pos) continue;
    const metrics = graphNodeMetrics(node);
    const selected = node.node_type === 'operation' ? node.id === app.selectedOperation : node.id === app.selectedArtifact;
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `node ${node.node_type}${node.deleted ? ' deleted' : ''}${selected ? ' selected' : ''}`);
    group.setAttribute('transform', `translate(${pos.x},${pos.y})`);
    groups.set(node.id, group);
    if (node.node_type === 'operation') {
      const shape = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const inset = 18; const mid = metrics.height / 2;
      shape.setAttribute('class', 'node-shape');
      shape.setAttribute('d', `M ${inset} 0 H ${metrics.width - inset} L ${metrics.width} ${mid} L ${metrics.width - inset} ${metrics.height} H ${inset} L 0 ${mid} Z`);
      group.appendChild(shape);
    } else {
      const shape = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const fold = 22;
      shape.setAttribute('class', 'node-shape');
      shape.setAttribute('d', `M 0 0 H ${metrics.width - fold} L ${metrics.width} ${fold} V ${metrics.height} H 0 Z`);
      const foldLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      foldLine.setAttribute('class', 'node-fold');
      foldLine.setAttribute('d', `M ${metrics.width - fold} 0 V ${fold} H ${metrics.width}`);
      group.append(shape, foldLine);
    }
    const textX = node.node_type === 'operation' ? 24 : 12;
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    title.setAttribute('x', String(textX)); title.setAttribute('y', node.node_type === 'operation' ? '20' : '22'); title.textContent = truncate(node.label, node.node_type === 'operation' ? 20 : 28);
    const sub = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    sub.setAttribute('x', String(textX)); sub.setAttribute('y', node.node_type === 'operation' ? '37' : '43'); sub.setAttribute('class', 'node-sub');
    sub.textContent = node.node_type === 'operation'
      ? (node.operation_type === node.label ? 'operation' : node.operation_type)
      : `${node.artifact_type} · ${node.size_display}${node.status === 'complete' ? '' : ` · ${node.status}`}${node.deleted ? ' · deleted' : ''}`;
    group.append(title, sub);

    let drag = null;
    group.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      event.preventDefault();
      const point = graphPointerCoordinates(svg, event);
      const current = positions.get(node.id);
      drag = {
        pointerId: event.pointerId,
        offsetX: point.x - current.x,
        offsetY: point.y - current.y,
        startClientX: event.clientX,
        startClientY: event.clientY,
        moved: false,
      };
      group.setPointerCapture(event.pointerId);
      group.classList.add('dragging');
    });
    group.addEventListener('pointermove', event => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      if (Math.hypot(event.clientX - drag.startClientX, event.clientY - drag.startClientY) > 3) drag.moved = true;
      if (!drag.moved) return;
      const point = graphPointerCoordinates(svg, event);
      setNodePosition(node, {x: point.x - drag.offsetX, y: point.y - drag.offsetY}, true);
      if ($('graphPhysics').checked) ensurePhysics(0.9);
    });
    const finishDrag = event => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const moved = drag.moved;
      drag = null;
      group.classList.remove('dragging');
      if (group.hasPointerCapture(event.pointerId)) group.releasePointerCapture(event.pointerId);
      if (!moved) {
        const now = Date.now();
        const last = app.graphLastClick;
        const doubleClick = last && last.mode === mode && last.nodeId === node.id && now - last.when < 360;
        if (doubleClick && releaseGraphNode(node.id)) {
          app.graphLastClick = null;
          if ($('graphPhysics').checked) ensurePhysics(0.9);
        } else {
          app.graphLastClick = {mode, nodeId: node.id, when: now};
          node.node_type === 'operation' ? selectOperation(node.id) : selectArtifact(node.id);
        }
      } else {
        scheduleGraphLayoutSave(100);
        if ($('graphPhysics').checked) ensurePhysics(0.7);
      }
    };
    group.addEventListener('pointerup', finishDrag);
    group.addEventListener('pointercancel', event => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      drag = null;
      group.classList.remove('dragging');
    });
    svg.appendChild(group);
  }

  updateAllGeometry();
  for (const [nodeId, position] of positions) livePositions.set(nodeId, {...position});
  if ($('graphPhysics').checked && app.graphPhysicsPending[mode] && visibleNodes.length > 1) ensurePhysics(0.9);
}
function graphNodeMetrics(node) {
  return node.node_type === 'operation'
    ? {width: 150, height: 46}
    : {width: 220, height: 58};
}
function startGraphPhysics({nodes, edges, positions, seed, manualPositions, livePositions, nodeById, width, height, updateAllGeometry, onSettled}) {
  const validEdges = edges.filter(edge => positions.has(edge.source) && positions.has(edge.target));
  const velocity = new Map(nodes.map(node => [node.id, {x: 0, y: 0}]));
  const forces = new Map(nodes.map(node => [node.id, {x: 0, y: 0}]));
  const metrics = new Map(nodes.map(node => [node.id, graphNodeMetrics(node)]));
  const cellWidth = 260;
  const cellHeight = 110;
  let frame = null;
  let alpha = 1;
  let stopped = false;
  let quietFrames = 0;

  const center = (node, position) => {
    const size = metrics.get(node.id);
    return {x: position.x + size.width / 2, y: position.y + size.height / 2};
  };
  const seedCenters = nodes.flatMap(node => {
    const target = seed.get(node.id);
    return target ? [center(node, target)] : [];
  });
  const gravityCenter = seedCenters.length ? {
    x: seedCenters.reduce((sum, point) => sum + point.x, 0) / seedCenters.length,
    y: seedCenters.reduce((sum, point) => sum + point.y, 0) / seedCenters.length,
  } : {x: width / 2, y: height / 2};
  const edgeSprings = validEdges.flatMap(edge => {
    const sourceNode = nodeById.get(edge.source); const targetNode = nodeById.get(edge.target);
    const seedSource = seed.get(edge.source); const seedTarget = seed.get(edge.target);
    if (!sourceNode || !targetNode || !seedSource || !seedTarget) return [];
    const seedSourceCenter = center(sourceNode, seedSource); const seedTargetCenter = center(targetNode, seedTarget);
    return [{...edge, restLength: Math.max(90, Math.abs(seedTargetCenter.x - seedSourceCenter.x))}];
  });
  const addForce = (id, x, y) => {
    const force = forces.get(id);
    force.x += x; force.y += y;
  };
  const schedule = () => {
    if (!stopped && frame === null) frame = requestAnimationFrame(tick);
  };
  const reheat = strength => {
    if (stopped) return;
    alpha = Math.max(alpha, strength);
    quietFrames = 0;
    schedule();
  };
  const stop = () => {
    stopped = true;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
  };

  const tick = () => {
    frame = null;
    if (stopped || !$('graphPhysics').checked) return;
    for (const force of forces.values()) { force.x = 0; force.y = 0; }

    // Preserve the layered DAG semantics. X is fairly strongly tethered to its
    // deterministic layer; Y is only weakly tethered so the graph can relax.
    for (const node of nodes) {
      if (manualPositions.has(node.id)) continue;
      const position = positions.get(node.id); const target = seed.get(node.id);
      if (!position || !target) continue;
      const c = center(node, position);
      addForce(node.id,
        (target.x - position.x) * 0.052 + (gravityCenter.x - c.x) * 0.0007,
        (target.y - position.y) * 0.0012 + (gravityCenter.y - c.y) * 0.0016);
    }

    // Real edge springs: the rest length is the horizontal separation implied by
    // the layered seed. Diagonal edges therefore shorten naturally without
    // collapsing graph depth, and overly compressed edges push back.
    for (const edge of edgeSprings) {
      const sourceNode = nodeById.get(edge.source); const targetNode = nodeById.get(edge.target);
      const sourcePos = positions.get(edge.source); const targetPos = positions.get(edge.target);
      if (!sourceNode || !targetNode || !sourcePos || !targetPos) continue;
      const sourceCenter = center(sourceNode, sourcePos); const targetCenter = center(targetNode, targetPos);
      let dx = targetCenter.x - sourceCenter.x; let dy = targetCenter.y - sourceCenter.y;
      let distance = Math.hypot(dx, dy);
      if (distance < 0.001) { dx = 0.001; distance = 0.001; }
      const extension = Math.max(-120, Math.min(220, distance - edge.restLength));
      const spring = extension * 0.016;
      const fx = dx / distance * spring;
      const fy = dy / distance * spring;
      if (!manualPositions.has(edge.source)) addForce(edge.source, fx, fy);
      if (!manualPositions.has(edge.target)) addForce(edge.target, -fx, -fy);
    }

    // Spatial hashing keeps collision/near-repulsion local instead of doing an
    // O(n^2) all-pairs pass on every animation frame.
    const buckets = new Map();
    for (let index = 0; index < nodes.length; index++) {
      const node = nodes[index]; const position = positions.get(node.id);
      if (!position) continue;
      const c = center(node, position);
      const key = `${Math.floor(c.x / cellWidth)},${Math.floor(c.y / cellHeight)}`;
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(index);
    }
    const visited = new Set();
    for (let index = 0; index < nodes.length; index++) {
      const a = nodes[index]; const aPos = positions.get(a.id);
      if (!aPos) continue;
      const aCenter = center(a, aPos); const aSize = metrics.get(a.id);
      const bx = Math.floor(aCenter.x / cellWidth); const by = Math.floor(aCenter.y / cellHeight);
      for (let dxCell = -1; dxCell <= 1; dxCell++) {
        for (let dyCell = -1; dyCell <= 1; dyCell++) {
          for (const otherIndex of buckets.get(`${bx + dxCell},${by + dyCell}`) || []) {
            if (otherIndex <= index) continue;
            const pairKey = `${index}:${otherIndex}`;
            if (visited.has(pairKey)) continue;
            visited.add(pairKey);
            const b = nodes[otherIndex]; const bPos = positions.get(b.id);
            if (!bPos) continue;
            const bCenter = center(b, bPos); const bSize = metrics.get(b.id);
            let dx = bCenter.x - aCenter.x; let dy = bCenter.y - aCenter.y;
            if (dx === 0 && dy === 0) dy = 0.01;
            const overlapX = (aSize.width + bSize.width) / 2 + 26 - Math.abs(dx);
            const overlapY = (aSize.height + bSize.height) / 2 + 22 - Math.abs(dy);
            if (overlapX > 0 && overlapY > 0) {
              const sameLayer = Math.abs((seed.get(a.id)?.x || 0) - (seed.get(b.id)?.x || 0)) < 1;
              if (sameLayer || overlapY <= overlapX) {
                const push = Math.min(16, overlapY * 0.16) * (dy >= 0 ? 1 : -1);
                if (!manualPositions.has(a.id)) addForce(a.id, 0, -push);
                if (!manualPositions.has(b.id)) addForce(b.id, 0, push);
              } else {
                const push = Math.min(14, overlapX * 0.12) * (dx >= 0 ? 1 : -1);
                if (!manualPositions.has(a.id)) addForce(a.id, -push, 0);
                if (!manualPositions.has(b.id)) addForce(b.id, push, 0);
              }
            } else {
              const distanceSquared = dx * dx + dy * dy;
              if (distanceSquared < 52000) {
                const distance = Math.sqrt(distanceSquared);
                const strength = Math.min(1.4, 60 / Math.max(45, distance));
                const ux = dx / distance; const uy = dy / distance;
                if (!manualPositions.has(a.id)) addForce(a.id, -ux * strength, -uy * strength);
                if (!manualPositions.has(b.id)) addForce(b.id, ux * strength, uy * strength);
              }
            }
          }
        }
      }
    }

    let motion = 0;
    for (const node of nodes) {
      const position = positions.get(node.id); const size = metrics.get(node.id);
      if (!position || !size) continue;
      if (manualPositions.has(node.id)) {
        const fixed = manualPositions.get(node.id);
        position.x = fixed.x; position.y = fixed.y;
        const v = velocity.get(node.id); v.x = 0; v.y = 0;
      } else {
        const force = forces.get(node.id); const v = velocity.get(node.id);
        v.x = (v.x + force.x * alpha) * 0.78;
        v.y = (v.y + force.y * alpha) * 0.78;
        const stepX = Math.max(-10, Math.min(10, v.x));
        const stepY = Math.max(-12, Math.min(12, v.y));
        position.x = Math.max(18, Math.min(position.x + stepX, width - size.width - 18));
        position.y = Math.max(18, Math.min(position.y + stepY, height - size.height - 18));
        motion += Math.abs(stepX) + Math.abs(stepY);
      }
      livePositions.set(node.id, {x: position.x, y: position.y});
    }
    updateAllGeometry();

    alpha *= 0.982;
    const averageMotion = motion / Math.max(1, nodes.length - manualPositions.size);
    quietFrames = averageMotion < 0.035 && alpha < 0.12 ? quietFrames + 1 : 0;
    if (quietFrames < 14 && alpha > 0.018) schedule();
    else if (typeof onSettled === 'function') onSettled();
  };

  schedule();
  return {stop, reheat};
}
function graphFastLayout(nodes, edges, mode = $('graphMode').value) {
  if (!nodes.length) return new Map();
  const ids = new Set(nodes.map(node => node.id));
  const depth = new Map(nodes.map(node => [node.id, 0]));
  for (let iteration = 0; iteration < nodes.length; iteration++) {
    let changed = false;
    for (const edge of edges) {
      if (!ids.has(edge.source) || !ids.has(edge.target)) continue;
      const candidate = (depth.get(edge.source) || 0) + 1;
      if (candidate > (depth.get(edge.target) || 0) && candidate <= nodes.length) {
        depth.set(edge.target, candidate); changed = true;
      }
    }
    if (!changed) break;
  }
  const levels = new Map();
  for (const node of nodes.slice().sort((a, b) => a.label.localeCompare(b.label))) {
    const level = depth.get(node.id) || 0;
    if (!levels.has(level)) levels.set(level, []);
    levels.get(level).push(node);
  }

  const incoming = new Map(nodes.map(node => [node.id, []]));
  const outgoing = new Map(nodes.map(node => [node.id, []]));
  for (const edge of edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target)) continue;
    outgoing.get(edge.source).push(edge.target);
    incoming.get(edge.target).push(edge.source);
  }
  const levelKeys = Array.from(levels.keys()).sort((a, b) => a - b);
  const rebuildOrder = () => {
    const order = new Map();
    for (const level of levelKeys) {
      levels.get(level).forEach((node, index) => order.set(node.id, index));
    }
    return order;
  };
  const sortLevelByNeighbors = (level, neighbors, order) => {
    const group = levels.get(level);
    const previousIndex = new Map(group.map((node, index) => [node.id, index]));
    const barycenter = node => {
      const positions = (neighbors.get(node.id) || [])
        .map(id => order.get(id))
        .filter(value => Number.isFinite(value));
      if (!positions.length) return null;
      return positions.reduce((total, value) => total + value, 0) / positions.length;
    };
    group.sort((a, b) => {
      const aCenter = barycenter(a); const bCenter = barycenter(b);
      if (aCenter === null && bCenter === null) return previousIndex.get(a.id) - previousIndex.get(b.id);
      if (aCenter === null) return 1;
      if (bCenter === null) return -1;
      if (aCenter !== bCenter) return aCenter - bCenter;
      return previousIndex.get(a.id) - previousIndex.get(b.id);
    });
  };

  // Barycentric sweeps get close to a good Sugiyama ordering, but can stop with
  // obviously removable crossings. A transpose pass then tests adjacent nodes
  // directly and keeps swaps that reduce crossings to neighbors on the same rank.
  const pairCrossings = (upperNode, lowerNode, neighbors, order) => {
    const upperNeighbors = neighbors.get(upperNode.id) || [];
    const lowerNeighbors = neighbors.get(lowerNode.id) || [];
    let crossings = 0;
    for (const upperId of upperNeighbors) {
      const upperOrder = order.get(upperId); const upperLevel = depth.get(upperId);
      if (!Number.isFinite(upperOrder)) continue;
      for (const lowerId of lowerNeighbors) {
        if (depth.get(lowerId) !== upperLevel) continue;
        const lowerOrder = order.get(lowerId);
        if (Number.isFinite(lowerOrder) && upperOrder > lowerOrder) crossings += 1;
      }
    }
    return crossings;
  };
  const transposeLevel = level => {
    const group = levels.get(level);
    if (!group || group.length < 2) return false;
    let changedAny = false;
    for (let pass = 0; pass < group.length * 2; pass++) {
      let changed = false;
      let order = rebuildOrder();
      for (let index = 0; index < group.length - 1; index++) {
        const upper = group[index]; const lower = group[index + 1];
        const current = pairCrossings(upper, lower, incoming, order)
          + pairCrossings(upper, lower, outgoing, order);
        const swapped = pairCrossings(lower, upper, incoming, order)
          + pairCrossings(lower, upper, outgoing, order);
        if (swapped < current) {
          group[index] = lower; group[index + 1] = upper;
          changed = true; changedAny = true;
          order = rebuildOrder();
        }
      }
      if (!changed) break;
    }
    return changedAny;
  };

  // Preserve the previous six-sweep barycentric result as the baseline.
  for (let sweep = 0; sweep < 6; sweep++) {
    let order = rebuildOrder();
    for (const level of levelKeys.slice(1)) {
      sortLevelByNeighbors(level, incoming, order);
      order = rebuildOrder();
    }
    order = rebuildOrder();
    for (const level of levelKeys.slice(0, -1).reverse()) {
      sortLevelByNeighbors(level, outgoing, order);
      order = rebuildOrder();
    }
  }
  // Then monotonically improve that ordering: every accepted adjacent swap
  // removes more crossings than it creates. Multiple rounds let improvements in
  // one rank expose new improvements in neighboring ranks.
  for (let round = 0; round < 4; round++) {
    let changed = false;
    for (const level of levelKeys) changed = transposeLevel(level) || changed;
    if (!changed) break;
  }

  const verticalGap = 34;
  const levelHeights = new Map();
  let maxLevelHeight = 0;
  for (const level of levelKeys) {
    const group = levels.get(level);
    const total = group.reduce((sum, node) => sum + graphNodeMetrics(node).height, 0)
      + Math.max(0, group.length - 1) * verticalGap;
    levelHeights.set(level, total);
    maxLevelHeight = Math.max(maxLevelHeight, total);
  }
  const positions = new Map();
  let x = 35;
  const horizontalGap = graphHorizontalGap(mode);
  for (const level of levelKeys) {
    const group = levels.get(level);
    const levelWidth = Math.max(...group.map(node => graphNodeMetrics(node).width));
    let y = 35 + (maxLevelHeight - levelHeights.get(level)) / 2;
    for (const node of group) {
      positions.set(node.id, {x, y});
      y += graphNodeMetrics(node).height + verticalGap;
    }
    x += levelWidth + horizontalGap;
  }
  return positions;
}

"""
