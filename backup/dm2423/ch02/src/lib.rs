use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Module, Sequential, VarBuilder};
pub(crate) struct Unet {
    first_block_down: Sequential,
}

struct Gelu {}

impl Module for Gelu {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.gelu()
    }
}

impl Unet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let first_block_down = {
            let mut cfg = candle_nn::conv::Conv2dConfig::default();
            cfg.padding = 1;

            let seq = candle_nn::seq();
            let seq = seq
                .add(candle_nn::conv2d(1, 32, 3, cfg, vb.clone())?)
                .add(Gelu {});
            seq
        };
        Ok(Self { first_block_down })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unet() {
        let map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(
            &map,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        );

        let _unet = Unet::new(vb).unwrap();

        println!("all vars: {:?}", map.all_vars());

        map.save("unet.safetensors").unwrap();
    }
}
