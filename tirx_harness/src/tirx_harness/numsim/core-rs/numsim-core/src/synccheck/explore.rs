//! Explicit-state DFS over a finite transition system.
//!
//! Port of `sync_partial_order.rs` (`explore_sync_states`,
//! `explore_sync_states_with_sleep_sets`) with the strong-diamond test moved
//! from the model into the explorer (it only uses `step` and `enabled`).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

pub trait TransitionSystem {
    type State: Clone + Eq + Hash;
    type Transition: Clone + Ord + Hash + Debug;
    type Error: Clone + Debug;
    type Deadlock: Clone + Debug;

    fn initial_state(&self) -> Self::State;
    /// Every enabled transition. An empty set in an incomplete state is a deadlock.
    fn enabled(&self, state: &Self::State) -> Vec<Self::Transition>;
    fn step(&self, state: &Self::State, transition: &Self::Transition)
        -> Result<Self::State, Self::Error>;
    fn is_complete(&self, state: &Self::State) -> bool;
    fn describe_deadlock(&self, state: &Self::State) -> Self::Deadlock;
    /// An enabled transition that can be moved to the front of every complete
    /// execution without losing errors, deadlocks or distinct terminal states.
    fn persistent_transition(
        &self,
        _state: &Self::State,
        _enabled: &[Self::Transition],
    ) -> Option<Self::Transition> {
        None
    }
    /// Proof obligation of the strong-diamond reduction beyond one step:
    /// no transition that can occur before `transition` (enabled now, or
    /// enabled later by any sequence of other transitions) conflicts with it
    /// or can disable it. The one-step diamond test alone misses a
    /// transition exposed only after two pruned siblings (review S8). The
    /// default declines, which keeps the search exhaustive.
    fn independent_of_future(&self, _state: &Self::State, _transition: &Self::Transition) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_states: usize,
    pub max_transitions: usize,
    /// Wall-clock deadline, checked every 256 states (review V2C-31).
    pub deadline: Option<std::time::Instant>,
}

impl Default for Limits {
    /// Today's `SyncStateSearchLimits::default()`; production derives them from
    /// `ResourceLimits.max_backtrack_nodes` / `max_loop_steps`.
    fn default() -> Self {
        Self {
            max_states: 1_000_000,
            max_transitions: 10_000_000,
            deadline: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub sleep_sets: bool,
    pub strong_diamonds: bool,
    pub persistent: bool,
    pub stop_on_first_failure: bool,
}

impl Options {
    pub const NONE: Self = Self {
        sleep_sets: false,
        strong_diamonds: false,
        persistent: false,
        stop_on_first_failure: true,
    };
    pub const ALL: Self = Self {
        sleep_sets: true,
        strong_diamonds: true,
        persistent: true,
        stop_on_first_failure: true,
    };
}

impl Default for Options {
    fn default() -> Self {
        Self::ALL
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Termination {
    Exhausted,
    FirstFailure,
    StateLimit(usize),
    TransitionLimit(usize),
    /// The wall-clock deadline passed.
    WallTime,
}

#[derive(Clone, Debug)]
pub enum Failure<T, E, D> {
    Error { transition: T, error: E, witness: Vec<T> },
    Deadlock { deadlock: D, witness: Vec<T> },
}

#[derive(Clone, Debug)]
pub struct SearchResult<T, E, D> {
    pub visited_states: usize,
    pub explored_transitions: usize,
    pub strong_diamond_pruned: usize,
    pub sleep_pruned: usize,
    pub complete_states: usize,
    /// Up to two distinct complete schedules (non-confluence witnesses).
    pub complete_witnesses: Vec<Vec<T>>,
    pub failures: Vec<Failure<T, E, D>>,
    pub termination: Termination,
}

impl<T, E, D> SearchResult<T, E, D> {
    pub fn confluent(&self) -> bool {
        self.complete_states == 1
    }

    pub fn proves_clean(&self) -> bool {
        self.termination == Termination::Exhausted && self.failures.is_empty() && self.confluent()
    }
}

struct Node<S, T> {
    state: Arc<S>,
    sleep: BTreeSet<T>,
    parent: Option<(usize, T)>,
}

fn witness<S, T: Clone>(nodes: &[Node<S, T>], mut id: usize, last: Option<&T>) -> Vec<T> {
    let mut path = Vec::new();
    while let Some((parent, transition)) = &nodes[id].parent {
        path.push(transition.clone());
        id = *parent;
    }
    path.reverse();
    path.extend(last.cloned());
    path
}

/// Keep a subset-minimal antichain of sleep sets per semantic state: a smaller
/// sleep set permits every continuation a larger one does.
fn register_sleep<S: Eq + Hash, T: Clone + Ord>(
    visited: &mut HashMap<Arc<S>, Vec<BTreeSet<T>>>,
    state: &Arc<S>,
    sleep: &BTreeSet<T>,
) -> bool {
    match visited.get_mut(state) {
        Some(contexts) => {
            if contexts.iter().any(|current| current.is_subset(sleep)) {
                return false;
            }
            contexts.retain(|current| !sleep.is_subset(current));
            contexts.push(sleep.clone());
            true
        }
        None => {
            visited.insert(Arc::clone(state), vec![sleep.clone()]);
            true
        }
    }
}

/// Explore every reachable state (modulo the selected reductions).
pub fn explore<M: TransitionSystem>(
    model: &M,
    limits: Limits,
    options: Options,
) -> SearchResult<M::Transition, M::Error, M::Deadlock> {
    let initial = Arc::new(model.initial_state());
    let mut visited: HashMap<Arc<M::State>, Vec<BTreeSet<M::Transition>>> = HashMap::new();
    visited.insert(Arc::clone(&initial), vec![BTreeSet::new()]);
    let mut nodes = vec![Node {
        state: initial,
        sleep: BTreeSet::new(),
        parent: None,
    }];
    let mut stack = vec![0usize];
    let mut result = SearchResult {
        visited_states: 0,
        explored_transitions: 0,
        strong_diamond_pruned: 0,
        sleep_pruned: 0,
        complete_states: 0,
        complete_witnesses: Vec::new(),
        failures: Vec::new(),
        termination: Termination::Exhausted,
    };
    let mut complete = HashSet::<Arc<M::State>>::new();

    let mut popped = 0u64;
    'search: while let Some(id) = stack.pop() {
        popped += 1;
        if popped % 256 == 0 && limits.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            result.termination = Termination::WallTime;
            break;
        }
        let state = Arc::clone(&nodes[id].state);
        if options.sleep_sets
            && !visited
                .get(&state)
                .is_some_and(|contexts| contexts.iter().any(|sleep| *sleep == nodes[id].sleep))
        {
            // A less restrictive context for this state superseded this one.
            continue;
        }
        if model.is_complete(&state) {
            if complete.insert(Arc::clone(&state)) && result.complete_witnesses.len() < 2 {
                result.complete_witnesses.push(witness(&nodes, id, None));
            }
            continue;
        }
        let mut enabled = model.enabled(&state);
        enabled.sort();
        enabled.dedup();
        if enabled.is_empty() {
            result.failures.push(Failure::Deadlock {
                deadlock: model.describe_deadlock(&state),
                witness: witness(&nodes, id, None),
            });
            if options.stop_on_first_failure {
                result.termination = Termination::FirstFailure;
                break;
            }
            continue;
        }

        // One-step successors, computed once per state.
        let successors = enabled
            .iter()
            .map(|transition| model.step(&state, transition).map(Arc::new))
            .collect::<Vec<_>>();

        let mut sleep = if options.sleep_sets {
            let mut inherited = nodes[id].sleep.clone();
            inherited.retain(|t| enabled.binary_search(t).is_ok());
            inherited
        } else {
            BTreeSet::new()
        };
        let active = enabled.iter().filter(|t| !sleep.contains(*t)).count();

        let persistent = (options.persistent && sleep.is_empty())
            .then(|| model.persistent_transition(&state, &enabled))
            .flatten()
            .and_then(|t| enabled.binary_search(&t).ok());
        let canonical = (options.strong_diamonds && persistent.is_none() && active > 1)
            .then(|| all_strong_diamonds(model, &enabled, &successors))
            .filter(|all| *all)
            .and_then(|_| {
                enabled
                    .iter()
                    .position(|t| !sleep.contains(t) && model.independent_of_future(&state, t))
            });
        if canonical.is_some() {
            result.strong_diamond_pruned += active - 1;
        }

        let mut children = Vec::new();
        for (index, transition) in enabled.iter().enumerate() {
            if persistent.is_some_and(|chosen| chosen != index)
                || canonical.is_some_and(|chosen| chosen != index)
            {
                continue;
            }
            if sleep.contains(transition) {
                result.sleep_pruned += 1;
                continue;
            }
            if result.explored_transitions >= limits.max_transitions {
                result.termination = Termination::TransitionLimit(limits.max_transitions);
                break 'search;
            }
            result.explored_transitions += 1;
            match &successors[index] {
                Ok(next) => {
                    let child_sleep = if options.sleep_sets {
                        let next_enabled = model.enabled(next).into_iter().collect::<BTreeSet<_>>();
                        sleep
                            .iter()
                            .filter(|sleeping| {
                                let Ok(sleeping_index) = enabled.binary_search(sleeping) else {
                                    return false;
                                };
                                let Ok(after_sleeping) = &successors[sleeping_index] else {
                                    return false;
                                };
                                next_enabled.contains(*sleeping)
                                    && commutes(model, transition, next, sleeping, after_sleeping)
                            })
                            .cloned()
                            .collect()
                    } else {
                        BTreeSet::new()
                    };
                    let is_new = !visited.contains_key(next);
                    if is_new && visited.len() >= limits.max_states {
                        result.termination = Termination::StateLimit(limits.max_states);
                        break 'search;
                    }
                    let fresh = if options.sleep_sets {
                        register_sleep(&mut visited, next, &child_sleep)
                    } else if is_new {
                        visited.insert(Arc::clone(next), vec![BTreeSet::new()]);
                        true
                    } else {
                        false
                    };
                    if fresh {
                        children.push(nodes.len());
                        nodes.push(Node {
                            state: Arc::clone(next),
                            sleep: child_sleep,
                            parent: Some((id, transition.clone())),
                        });
                    }
                }
                Err(error) => {
                    result.failures.push(Failure::Error {
                        transition: transition.clone(),
                        error: error.clone(),
                        witness: witness(&nodes, id, Some(transition)),
                    });
                    if options.stop_on_first_failure {
                        result.termination = Termination::FirstFailure;
                        break 'search;
                    }
                }
            }
            if options.sleep_sets && canonical.is_none() {
                sleep.insert(transition.clone());
            }
        }
        stack.extend(children.into_iter().rev());
    }
    result.visited_states = visited.len();
    result.complete_states = complete.len();
    result
}

/// `left` then `right` and `right` then `left` both succeed and reach the same state.
fn commutes<M: TransitionSystem>(
    model: &M,
    left: &M::Transition,
    after_left: &M::State,
    right: &M::Transition,
    after_right: &M::State,
) -> bool {
    if left == right {
        return false;
    }
    match (model.step(after_left, right), model.step(after_right, left)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// Every pair of enabled transitions is a *strong* diamond: both orders reach
/// the same state and neither first step changes the other enabled
/// transitions (no newly exposed or disabled successor). Together with
/// [`TransitionSystem::independent_of_future`] for the chosen transition this
/// makes `{canonical}` a persistent set; the one-step test alone (today's
/// `sync_fixed_unified.rs:5382-5423`) is not sufficient.
fn all_strong_diamonds<M: TransitionSystem>(
    model: &M,
    enabled: &[M::Transition],
    successors: &[Result<Arc<M::State>, M::Error>],
) -> bool {
    let mut after_enabled = Vec::with_capacity(enabled.len());
    for (index, successor) in successors.iter().enumerate() {
        let Ok(next) = successor else {
            return false;
        };
        let mut expected = enabled.to_vec();
        expected.remove(index);
        let mut actual = model.enabled(next);
        actual.sort();
        actual.dedup();
        if actual != expected {
            return false;
        }
        after_enabled.push(next);
    }
    for left in 0..enabled.len() {
        for right in left + 1..enabled.len() {
            if !commutes(
                model,
                &enabled[left],
                after_enabled[left],
                &enabled[right],
                after_enabled[right],
            ) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` actors each perform one step on a shared value; `overwrite` makes the
    /// steps non-commuting (last writer wins), so terminal states differ.
    struct Actors {
        n: usize,
        overwrite: bool,
    }

    impl TransitionSystem for Actors {
        type State = (Vec<bool>, i64);
        type Transition = usize;
        type Error = ();
        type Deadlock = ();

        fn initial_state(&self) -> Self::State {
            (vec![false; self.n], 0)
        }
        fn enabled(&self, state: &Self::State) -> Vec<usize> {
            (0..self.n).filter(|&i| !state.0[i]).collect()
        }
        fn step(&self, state: &Self::State, actor: &usize) -> Result<Self::State, ()> {
            let mut next = state.clone();
            next.0[*actor] = true;
            next.1 = if self.overwrite { *actor as i64 } else { next.1 + 1 };
            Ok(next)
        }
        fn is_complete(&self, state: &Self::State) -> bool {
            state.0.iter().all(|done| *done)
        }
        fn describe_deadlock(&self, _: &Self::State) {}
        // Increments commute and never disable each other.
        fn independent_of_future(&self, _: &Self::State, _: &usize) -> bool {
            !self.overwrite
        }
    }

    #[test]
    fn commuting_actors_collapse_with_strong_diamonds() {
        let model = Actors { n: 10, overwrite: false };
        let plain = explore(&model, Limits::default(), Options::NONE);
        let sleep = explore(&model, Limits::default(), Options { sleep_sets: true, ..Options::NONE });
        let all = explore(&model, Limits::default(), Options::ALL);
        assert!(plain.proves_clean() && sleep.proves_clean() && all.proves_clean());
        // 2^10 subsets of finished actors.
        assert_eq!(plain.visited_states, 1024);
        assert_eq!(plain.explored_transitions, 10 * 512);
        // Sleep sets keep the states but explore each one once.
        assert_eq!(sleep.visited_states, 1024);
        assert_eq!(sleep.explored_transitions, 1023);
        // Strong diamonds walk one canonical chain.
        assert_eq!(all.visited_states, 11);
        assert_eq!(all.explored_transitions, 10);
    }

    #[test]
    fn non_commuting_actors_are_non_confluent_under_every_reduction() {
        let model = Actors { n: 3, overwrite: true };
        for options in [Options::NONE, Options { sleep_sets: true, ..Options::NONE }, Options::ALL] {
            let result = explore(&model, Limits::default(), options);
            assert_eq!(result.termination, Termination::Exhausted);
            assert_eq!(result.complete_states, 3, "{options:?}");
            assert!(!result.proves_clean());
            assert_eq!(result.complete_witnesses.len(), 2);
        }
    }

    #[test]
    fn limits_stop_the_search() {
        let model = Actors { n: 12, overwrite: false };
        let result = explore(&model, Limits { max_states: 100, max_transitions: usize::MAX, deadline: None }, Options::NONE);
        assert_eq!(result.termination, Termination::StateLimit(100));
        let result = explore(&model, Limits { max_states: usize::MAX, max_transitions: 50, deadline: None }, Options::NONE);
        assert_eq!(result.termination, Termination::TransitionLimit(50));
    }

    /// Review S8: `t`, `s1`, `s2` are pairwise one-step strong diamonds, but
    /// `u` becomes enabled only after both `s1` and `s2` and fails if `t` has
    /// not run. Choosing `t` as the canonical transition on the one-step test
    /// alone hides the failure; the independence hook must veto it.
    struct TwoStep {
        claims_independence: bool,
    }

    impl TransitionSystem for TwoStep {
        // (t, s1, s2, u) done flags.
        type State = [bool; 4];
        type Transition = u8;
        type Error = &'static str;
        type Deadlock = ();

        fn initial_state(&self) -> Self::State {
            [false; 4]
        }
        fn enabled(&self, s: &Self::State) -> Vec<u8> {
            let mut out = (0..3).filter(|&i| !s[i as usize]).collect::<Vec<_>>();
            if s[1] && s[2] && !s[3] {
                out.push(3);
            }
            out
        }
        fn step(&self, s: &Self::State, t: &u8) -> Result<Self::State, &'static str> {
            if *t == 3 && !s[0] {
                return Err("u ran before t");
            }
            let mut n = *s;
            n[*t as usize] = true;
            Ok(n)
        }
        fn is_complete(&self, s: &Self::State) -> bool {
            s.iter().all(|d| *d)
        }
        fn describe_deadlock(&self, _: &Self::State) {}
        fn independent_of_future(&self, _: &Self::State, _: &u8) -> bool {
            self.claims_independence
        }
    }

    #[test]
    fn one_step_diamonds_need_the_independence_proof() {
        let unsound = explore(&TwoStep { claims_independence: true }, Limits::default(), Options::ALL);
        assert!(unsound.failures.is_empty(), "the one-step test alone misses the failure");
        let sound = explore(&TwoStep { claims_independence: false }, Limits::default(), Options::ALL);
        assert!(!sound.failures.is_empty());
    }
}
