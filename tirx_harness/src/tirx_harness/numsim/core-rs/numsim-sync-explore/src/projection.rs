//! Splitting the fixed program into independently explored transition systems.

use crate::event::ResourceId;
use crate::program::Program;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProjectionMode {
    /// One transition system for the whole launch (the product space).
    Whole,
    /// Connected components of the warp<->resource bipartite graph. Sound
    /// without any causal annotation: components share no warp and no resource.
    Components,
    /// One transition system per resource, each command gated by the
    /// happens-before predecessors observed in the reference run (today's
    /// `FixedSyncProjectionKey` + `causal_predecessors`).
    PerResource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProjectionKey {
    Whole,
    Component(usize),
    Resource(ResourceId),
}

#[derive(Clone, Debug)]
pub struct ProjectionSpec {
    pub key: ProjectionKey,
    /// Global command ids in this projection.
    pub commands: Vec<usize>,
    /// Apply happens-before gates from the reference run.
    pub gated: bool,
}

pub fn project(program: &Program, mode: ProjectionMode) -> Vec<ProjectionSpec> {
    match mode {
        ProjectionMode::Whole => vec![program.whole_spec()],
        ProjectionMode::PerResource => {
            let mut by_resource = vec![Vec::new(); program.resources.len()];
            for (index, command) in program.commands.iter().enumerate() {
                by_resource[command.resource].push(index);
            }
            by_resource
                .into_iter()
                .enumerate()
                .filter(|(_, commands)| !commands.is_empty())
                .map(|(resource, commands)| ProjectionSpec {
                    key: ProjectionKey::Resource(program.resources[resource]),
                    commands,
                    gated: true,
                })
                .collect()
        }
        ProjectionMode::Components => {
            // Union-find over warps [0, W) and resources [W, W + R).
            let warps = program.warp_ids.len();
            let mut parent = (0..warps + program.resources.len()).collect::<Vec<_>>();
            fn find(parent: &mut [usize], mut node: usize) -> usize {
                while parent[node] != node {
                    parent[node] = parent[parent[node]];
                    node = parent[node];
                }
                node
            }
            for command in &program.commands {
                let a = find(&mut parent, command.warp);
                let b = find(&mut parent, warps + command.resource);
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
            let mut components = std::collections::BTreeMap::<usize, Vec<usize>>::new();
            for (index, command) in program.commands.iter().enumerate() {
                let root = find(&mut parent, command.warp);
                components.entry(root).or_default().push(index);
            }
            components
                .into_values()
                .enumerate()
                .map(|(index, commands)| ProjectionSpec {
                    key: ProjectionKey::Component(index),
                    commands,
                    gated: false,
                })
                .collect()
        }
    }
}
