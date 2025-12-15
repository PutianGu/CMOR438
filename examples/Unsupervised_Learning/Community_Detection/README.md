# Community Detection

## Overview
Community detection identifies groups of nodes in a graph that are more strongly connected to each other than to the rest of the graph. In this project, the graph is provided as an adjacency matrix and the output is a community label for each node.

## Algorithm
This repository demonstrates a simple community detection approach suitable for teaching:
- Convert the adjacency matrix into a graph connectivity structure.
- Identify communities based on graph connectivity structure (e.g., connected components) or an iterative label-based method (depending on the implementation choice).
- Output an integer label for each node indicating its community membership.

## Key Parameters
Common parameters (depending on the implementation):
- `method`: which strategy is used (e.g., connected-components style vs. iterative labeling).
- `assume_undirected`: whether to treat the adjacency matrix as undirected.
- `max_iter`: iteration limit for iterative methods.

## Complexity
- Connectivity-based methods can be near-linear in the number of nodes and edges: `O(n + m)`.
- Iterative methods typically cost multiple passes over edges: roughly `O(max_iter * m)`.

## Strengths & Trade-offs
- Pros:
  - Works directly on graph structure (no feature vectors required).
  - Useful for social networks and relational datasets.
- Cons:
  - Different definitions of “community” lead to different algorithms and results.
  - Some methods can be sensitive to graph sparsity/noise and parameter choices.

## Data
- Dataset: Zachary’s Karate Club network (34 nodes).
- Input: adjacency matrix (undirected/unweighted in the classic version).
- Output: a community label for each node; visualization can color nodes by community.

