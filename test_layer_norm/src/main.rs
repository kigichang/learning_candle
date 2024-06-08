fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use candle_core::{Device, Tensor};
    use candle_nn::{LayerNorm, Module};

    #[test]
    fn test_layer_norm() -> Result<()> {
        let w = Tensor::new(&[1.0_f32, 1., 1.], &Device::Cpu)?;
        let b = Tensor::new(&[0.0_f32, 0., 0.], &Device::Cpu)?;

        let layer = LayerNorm::new(w, b, 1e-5);

        let x = Tensor::new(
            &[[[1.0_f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]],
            &Device::Cpu,
        )?;

        let y = layer.forward(&x)?;
        println!("y: {:?}", y.to_vec3::<f32>()?);

        Ok(())
    }
}
