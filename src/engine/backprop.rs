use super::Value;
use std::collections::HashSet;
use std::hash::Hash;
use termtree::Tree;

impl Value {
    /// Performs backpropagation to compute gradients for all Values in the graph.
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();

        self.topological_sort(&mut topo, &mut visited);
        topo.reverse();

        // ∂z/∂z = 1
        self.borrow_mut().grad = 1.0;

        // Backpropagation through the topology
        topo.iter().for_each(|v| {
            if let Some(backprop) = v.borrow()._backward {
                backprop(&v.borrow());
            }
        });
    }

    /// Topological sort for order.
    fn topological_sort(
        &self,
        topo: &mut Vec<Value>,
        visited: &mut HashSet<Value>,
    ) {
        if visited.insert(self.clone()) {
            self.borrow()._prev.iter().for_each(|child| {
                child.topological_sort(topo, visited);
            });

            topo.push(self.clone());
        }
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow()._uuid.hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        self.borrow()._uuid == other.borrow()._uuid
    }
}

impl Eq for Value {}

// --- Extras ---

impl Value {
    /// Returns tree with final output as root and inputs as leaves.
    pub fn tree(&self) -> Tree<Value> {
        let mut root = Tree::new(self.clone());
        if self.borrow()._op.is_some() {
            self.borrow()._prev.iter().for_each(|p| {
                root.push(p.tree());
            })
        }
        root
    }
}
