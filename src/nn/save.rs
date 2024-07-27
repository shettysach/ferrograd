use crate::nn::mlp::MultiLayerPerceptron;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::{fs::File, io::Result, path::Path};

// Thanks to https://gist.github.com/rust-play/e710e311a2ad808b5a8789d5e4457426

impl MultiLayerPerceptron {
    /// Save model weights at the given filepath.
    pub fn save(&self, filepath: &str) -> Result<()> {
        let mut file = File::create(filepath)?;
        let model_weights = self
            .parameters()
            .into_iter()
            .map(|value| value.borrow().data);

        for weight in model_weights {
            file.write_f64::<BigEndian>(weight)?;
        }

        Ok(())
    }

    /// Read model weights from the given filepath and load state.
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::open(path)?;
        let buf_len: usize = file.metadata()?.len() as usize / 8; // 8 bytes for one f64
        let mut buf: Vec<f64> = vec![0.0; buf_len];
        file.read_f64_into::<BigEndian>(&mut buf)?;

        let params = self.parameters();
        assert!(
            buf_len == self.parameters().len(),
            "Mismatching number of parameters"
        );

        params
            .into_iter()
            .zip(buf)
            .for_each(|(value, weight)| value.borrow_mut().data = weight);

        Ok(())
    }
}
