use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;

fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embbedings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embbedings, hidden_size))
}

// -----------------------------------------------------------------------------

fn linear(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Linear> {
    todo!()
}

fn main() {
    println!("Hello, world!");
}
