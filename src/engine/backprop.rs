use super::Value;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

impl Value {
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();

        self.topological_sort(&mut topo, &mut visited);
        topo.reverse();

        // dz/dz = 1
        self.borrow_mut().grad = 1.0;

        // Backpropagate
        for v in topo {
            if let Some(backprop) = v.borrow().backward {
                backprop(&v.borrow());
            }
        }
    }

    fn topological_sort(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            self.borrow().prev.iter().for_each(|child| {
                child.topological_sort(topo, visited);
            });

            topo.push(self.clone());
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().uuid.hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().uuid == other.borrow().uuid
    }
}

impl Eq for Value {}
