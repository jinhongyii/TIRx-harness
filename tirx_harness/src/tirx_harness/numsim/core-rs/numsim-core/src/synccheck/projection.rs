//! Splitting the fixed program into independently explored transition systems.

use super::program::Program;
use crate::sync::ResourceId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProjectionMode {
    /// One transition system for the whole launch (the product space).
    Whole,
    /// Connected components of the warp<->resource graph; sound without clocks.
    Components,
    /// One transition system per resource group (resources an atomic command
    /// touches together stay together), gated by reference-run happens-before
    /// (today's `FixedSyncProjectionKey` + `causal_predecessors`).
    PerResource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProjectionKey {
    Whole,
    Component(usize),
    /// Anchor resource of a resource group.
    Resource(ResourceId),
}

#[derive(Clone, Debug)]
pub struct ProjectionSpec {
    pub key: ProjectionKey,
    pub commands: Vec<usize>,
    pub gated: bool,
    /// Number of resources in the group.
    pub resource_count: usize,
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (find(parent, a), find(parent, b));
    if a != b {
        parent[a.max(b)] = a.min(b);
    }
}

pub fn project(program: &Program, mode: ProjectionMode) -> Vec<ProjectionSpec> {
    let all = (0..program.commands.len()).collect::<Vec<_>>();
    if mode == ProjectionMode::Whole || program.commands.is_empty() {
        return vec![ProjectionSpec { key: ProjectionKey::Whole, commands: all, gated: false, resource_count: program.resources.len() }];
    }
    let warps = program.warp_ids.len();
    let n = program.resources.len();
    let mut parent = (0..warps + n).collect::<Vec<_>>();
    for c in 0..program.commands.len() {
        let rs = program.command_resources(c);
        for w in rs.windows(2) {
            union(&mut parent, warps + w[0], warps + w[1]);
        }
        if mode == ProjectionMode::Components {
            for &p in &program.commands[c].participants {
                if let Some(&r) = rs.first() {
                    union(&mut parent, p, warps + r);
                }
            }
        }
    }
    let mut groups = std::collections::BTreeMap::<usize, (Vec<usize>, usize)>::new();
    for r in 0..n {
        let root = find(&mut parent, warps + r);
        groups.entry(root).or_default().1 += 1;
    }
    for c in 0..program.commands.len() {
        let rs = program.command_resources(c);
        let Some(&r) = rs.first() else { continue };
        let root = find(&mut parent, warps + r);
        groups.entry(root).or_default().0.push(c);
    }
    groups
        .into_iter()
        .filter(|(_, (cmds, _))| !cmds.is_empty())
        .enumerate()
        .map(|(i, (root, (commands, resource_count)))| ProjectionSpec {
            key: match mode {
                ProjectionMode::Components => ProjectionKey::Component(i),
                _ => ProjectionKey::Resource(program.resources[root - warps]),
            },
            commands,
            gated: mode == ProjectionMode::PerResource,
            resource_count,
        })
        .collect()
}
