//! HNSW graph structure and operations.

use crate::error::{HnswError, HnswResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    /// Node ID (corresponds to vector index).
    pub id: usize,

    /// Maximum layer this node appears in.
    pub max_layer: usize,

    /// Neighbors at each layer. neighbors[layer] = list of neighbor IDs.
    pub neighbors: Vec<Vec<u32>>,
}

impl HnswNode {
    /// Create a new node.
    pub fn new(id: usize, max_layer: usize) -> Self {
        Self {
            id,
            max_layer,
            neighbors: vec![Vec::new(); max_layer + 1],
        }
    }

    /// Get neighbors at a specific layer.
    pub fn get_neighbors(&self, layer: usize) -> &[u32] {
        if layer <= self.max_layer {
            &self.neighbors[layer]
        } else {
            &[]
        }
    }

    /// Set neighbors at a specific layer.
    pub fn set_neighbors(&mut self, layer: usize, neighbors: Vec<u32>) {
        if layer <= self.max_layer {
            self.neighbors[layer] = neighbors;
        }
    }

    /// Add a neighbor at a specific layer.
    pub fn add_neighbor(&mut self, layer: usize, neighbor: u32) {
        if layer <= self.max_layer && !self.neighbors[layer].contains(&neighbor) {
            self.neighbors[layer].push(neighbor);
        }
    }

    /// Remove a neighbor at a specific layer.
    pub fn remove_neighbor(&mut self, layer: usize, neighbor: u32) {
        if layer <= self.max_layer {
            self.neighbors[layer].retain(|&n| n != neighbor);
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.id as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.max_layer as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.neighbors.len() as u32).to_le_bytes());

        for layer_neighbors in &self.neighbors {
            bytes.extend_from_slice(&(layer_neighbors.len() as u32).to_le_bytes());
            for &n in layer_neighbors {
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        }

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> HnswResult<(Self, usize)> {
        if data.len() < 12 {
            return Err(HnswError::InvalidData("Node data too short".into()));
        }

        let id = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let max_layer = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let num_layers = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;

        let mut offset = 12;
        let mut neighbors = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            if data.len() < offset + 4 {
                return Err(HnswError::InvalidData("Node data truncated".into()));
            }

            let num_neighbors = u32::from_le_bytes(
                data[offset..offset + 4].try_into().unwrap()
            ) as usize;
            offset += 4;

            let layer_neighbors: Vec<u32> = (0..num_neighbors)
                .map(|i| {
                    u32::from_le_bytes(
                        data[offset + i * 4..offset + (i + 1) * 4].try_into().unwrap()
                    )
                })
                .collect();
            offset += num_neighbors * 4;
            neighbors.push(layer_neighbors);
        }

        Ok((Self { id, max_layer, neighbors }, offset))
    }
}

/// The HNSW graph structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HnswGraph {
    /// All nodes in the graph.
    pub nodes: Vec<HnswNode>,
}

impl HnswGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Create a graph with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: HnswNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: usize) -> Option<&HnswNode> {
        self.nodes.get(id)
    }

    /// Get a mutable node by ID.
    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut HnswNode> {
        self.nodes.get_mut(id)
    }

    /// Get neighbors of a node at a specific layer.
    pub fn get_neighbors(&self, id: usize, layer: usize) -> &[u32] {
        self.nodes
            .get(id)
            .map(|n| n.get_neighbors(layer))
            .unwrap_or(&[])
    }

    /// Set neighbors for a node at a specific layer.
    pub fn set_neighbors(&mut self, id: usize, layer: usize, neighbors: Vec<u32>) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.set_neighbors(layer, neighbors);
        }
    }

    /// Add a bidirectional edge between two nodes.
    pub fn add_edge(&mut self, node1: usize, node2: usize, layer: usize) {
        if let Some(n1) = self.nodes.get_mut(node1) {
            n1.add_neighbor(layer, node2 as u32);
        }
        if let Some(n2) = self.nodes.get_mut(node2) {
            n2.add_neighbor(layer, node1 as u32);
        }
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());

        for node in &self.nodes {
            let node_bytes = node.to_bytes();
            bytes.extend_from_slice(&(node_bytes.len() as u32).to_le_bytes());
            bytes.extend(node_bytes);
        }

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> HnswResult<Self> {
        if data.len() < 4 {
            return Err(HnswError::InvalidData("Graph data too short".into()));
        }

        let num_nodes = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        let mut nodes = Vec::with_capacity(num_nodes);

        for _ in 0..num_nodes {
            if data.len() < offset + 4 {
                return Err(HnswError::InvalidData("Graph data truncated".into()));
            }

            let node_len = u32::from_le_bytes(
                data[offset..offset + 4].try_into().unwrap()
            ) as usize;
            offset += 4;

            let (node, _) = HnswNode::from_bytes(&data[offset..offset + node_len])?;
            nodes.push(node);
            offset += node_len;
        }

        Ok(Self { nodes })
    }

    /// Get graph statistics.
    pub fn stats(&self) -> GraphStats {
        if self.nodes.is_empty() {
            return GraphStats::default();
        }

        let max_layer = self.nodes.iter().map(|n| n.max_layer).max().unwrap_or(0);
        let mut total_edges = 0;
        let mut edges_per_layer = vec![0; max_layer + 1];

        for node in &self.nodes {
            for (layer, neighbors) in node.neighbors.iter().enumerate() {
                total_edges += neighbors.len();
                if layer < edges_per_layer.len() {
                    edges_per_layer[layer] += neighbors.len();
                }
            }
        }

        GraphStats {
            num_nodes: self.nodes.len(),
            max_layer,
            total_edges: total_edges / 2, // Each edge counted twice
            avg_edges_per_node: total_edges as f64 / self.nodes.len() as f64 / 2.0,
            edges_per_layer,
        }
    }
}

/// Graph statistics.
#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub max_layer: usize,
    pub total_edges: usize,
    pub avg_edges_per_node: f64,
    pub edges_per_layer: Vec<usize>,
}

/// Thread-safe wrapper for concurrent graph construction.
pub struct ConcurrentHnswGraph {
    nodes: Vec<RwLock<HnswNode>>,
}

impl ConcurrentHnswGraph {
    /// Create with pre-allocated nodes.
    pub fn new(num_nodes: usize, max_layers: &[usize]) -> Self {
        let nodes = (0..num_nodes)
            .map(|id| RwLock::new(HnswNode::new(id, max_layers[id])))
            .collect();
        Self { nodes }
    }

    /// Set neighbors for a node at a layer (thread-safe).
    pub fn set_neighbors(&self, id: usize, layer: usize, neighbors: Vec<u32>) {
        if let Some(node_lock) = self.nodes.get(id) {
            let mut node = node_lock.write();
            node.set_neighbors(layer, neighbors);
        }
    }

    /// Add a neighbor to a node (thread-safe).
    pub fn add_neighbor(&self, id: usize, layer: usize, neighbor: u32) {
        if let Some(node_lock) = self.nodes.get(id) {
            let mut node = node_lock.write();
            node.add_neighbor(layer, neighbor);
        }
    }

    /// Get neighbors (thread-safe read).
    pub fn get_neighbors(&self, id: usize, layer: usize) -> Vec<u32> {
        self.nodes
            .get(id)
            .map(|n| n.read().get_neighbors(layer).to_vec())
            .unwrap_or_default()
    }

    /// Convert to regular graph (consumes self).
    pub fn into_graph(self) -> HnswGraph {
        let nodes = self
            .nodes
            .into_iter()
            .map(|n| n.into_inner())
            .collect();
        HnswGraph { nodes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = HnswNode::new(42, 3);
        assert_eq!(node.id, 42);
        assert_eq!(node.max_layer, 3);
        assert_eq!(node.neighbors.len(), 4);
    }

    #[test]
    fn test_node_neighbors() {
        let mut node = HnswNode::new(0, 2);

        node.add_neighbor(0, 1);
        node.add_neighbor(0, 2);
        node.add_neighbor(1, 3);

        assert_eq!(node.get_neighbors(0), &[1, 2]);
        assert_eq!(node.get_neighbors(1), &[3]);
        assert!(node.get_neighbors(2).is_empty());
    }

    #[test]
    fn test_node_serialization() {
        let mut node = HnswNode::new(5, 2);
        node.add_neighbor(0, 1);
        node.add_neighbor(0, 2);
        node.add_neighbor(1, 3);

        let bytes = node.to_bytes();
        let (restored, _) = HnswNode::from_bytes(&bytes).unwrap();

        assert_eq!(node.id, restored.id);
        assert_eq!(node.max_layer, restored.max_layer);
        assert_eq!(node.neighbors, restored.neighbors);
    }

    #[test]
    fn test_graph_operations() {
        let mut graph = HnswGraph::new();

        let node0 = HnswNode::new(0, 1);
        let node1 = HnswNode::new(1, 1);
        let node2 = HnswNode::new(2, 0);

        graph.add_node(node0);
        graph.add_node(node1);
        graph.add_node(node2);

        graph.add_edge(0, 1, 0);
        graph.add_edge(0, 2, 0);
        graph.add_edge(1, 0, 1);

        assert_eq!(graph.get_neighbors(0, 0), &[1, 2]);
        assert_eq!(graph.get_neighbors(1, 0), &[0]);
        assert_eq!(graph.get_neighbors(0, 1), &[1]);
    }

    #[test]
    fn test_graph_serialization() {
        let mut graph = HnswGraph::new();

        let mut node0 = HnswNode::new(0, 1);
        node0.add_neighbor(0, 1);
        graph.add_node(node0);

        let mut node1 = HnswNode::new(1, 1);
        node1.add_neighbor(0, 0);
        graph.add_node(node1);

        let bytes = graph.to_bytes();
        let restored = HnswGraph::from_bytes(&bytes).unwrap();

        assert_eq!(graph.len(), restored.len());
        assert_eq!(graph.get_neighbors(0, 0), restored.get_neighbors(0, 0));
    }
}
