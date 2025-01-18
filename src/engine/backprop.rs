use crate::engine::value::{Prev, Value};
use std::collections::HashSet;
use termtree::Tree;

#[allow(clippy::mutable_key_type)]
impl Value {
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();

        self.topological_sort(&mut topo, &mut visited);
        topo.reverse();

        // ∂z/∂z = 1
        self.borrow_mut().grad = 1.0;

        // Backpropagation through the computation graph.
        for v in topo.iter() {
            if let Some(backprop) = v.borrow().backward {
                backprop(&v.borrow());
            }
        }
    }

    fn topological_sort(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            match &self.borrow().prev {
                Prev::Binary(a, b) => {
                    a.topological_sort(topo, visited);
                    b.topological_sort(topo, visited);
                }
                Prev::Unary(a) => {
                    a.topological_sort(topo, visited);
                }
                Prev::Init => {}
            };
            topo.push(self.clone());
        }
    }

    /// Returns tree with final output as root and inputs as leaves.
    pub fn tree(&self) -> Tree<Value> {
        let mut root = Tree::new(self.clone());
        let node = self.borrow();

        match &node.prev {
            Prev::Binary(a, b) => {
                root.push(a.tree());
                root.push(b.tree());
            }
            Prev::Unary(a) => {
                root.push(a.tree());
            }
            Prev::Init => {}
        }

        root
    }
}
