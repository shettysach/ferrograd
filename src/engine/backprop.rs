use crate::engine::value::Value;
use std::collections::HashSet;
use termtree::Tree;

#[allow(clippy::mutable_key_type)]
impl Value {
    /// Performs backpropagation to compute gradients for all Values in the graph.
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();

        self.topological_sort(&mut topo, &mut visited);
        topo.reverse();

        // ∂z/∂z = 1
        self.borrow_mut().grad = 1.0;

        // Backpropagation through the computation graph.
        topo.iter().for_each(|v| {
            if let Some(backprop) = v.borrow().backward {
                backprop(&v.borrow());
            }
        });
    }

    // Topological sort for order.
    fn topological_sort(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            self.borrow().prev.iter().for_each(|child| {
                child.topological_sort(topo, visited);
            });

            topo.push(self.clone());
        }
    }
}

// --- Extras ---

impl Value {
    /// Returns tree with final output as root and inputs as leaves.
    pub fn tree(&self) -> Tree<Value> {
        let mut root = Tree::new(self.clone());
        let node = self.borrow();

        if node.op.is_some() {
            node.prev.iter().for_each(|p| {
                root.push(p.tree());
            })
        }
        root
    }
}
