use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;

fn compute_output(w: &Tensor, x: &Tensor) -> Result<f32> {
    Ok(x.mul(w)? // ai * bi
        .sum_all()? // sum(ai * bi)
        .sign()? // sign(sum(ai * bi))
        .to_scalar::<f32>()?) // 取值
}

fn show_learning(w: &Tensor) {
    println!("w: {:?}", w.to_vec1::<f32>().unwrap());
}

fn main() -> Result<()> {
    let mut rng = thread_rng();
    let device = Device::Cpu;

    const LEARNING_RATE: f32 = 0.1;
    let mut index_list = [0, 1, 2, 3];
    let x_tran = [
        [1.0_f32, -1.0, -1.0],
        [1.0_f32, -1.0, 1.0],
        [1.0_f32, 1.0, -1.0],
        [1.0_f32, 1.0, 1.0],
    ];

    let y_train = [1.0, 1.0, 1.0, -1.0];

    let mut w = Tensor::new(&[0.2_f32, -0.6, 0.25], &device)?;

    show_learning(&w);

    let mut all_correct = false;
    while !all_correct {
        all_correct = true;
        index_list.shuffle(&mut rng);
        for i in index_list.iter() {
            let x = Tensor::new(&x_tran[*i], &Device::Cpu)?;
            let y = y_train[*i];

            let p_out = compute_output(&w, &x)?;
            if y != p_out {
                let rate = Tensor::full(y * LEARNING_RATE, 3, &device)?;
                let rate = x.mul(&rate)?;
                w = w.add(&rate)?;
                all_correct = false;
                show_learning(&w);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow;
    use candle_core::Tensor;

    #[test]
    fn test_matrix() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let w = Tensor::new(&[0.9_f32, -0.6, -0.5, 0.2, 0.6, 0.6], &device)?.reshape((2, 3))?;
        let x = Tensor::new(
            &[
                1.0_f32, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
            ],
            &device,
        )?
        .reshape((3, 4))?;

        let result = w.matmul(&x)?;
        println!("{:?}", result.to_vec2::<f32>()?);
        Ok(())
    }
}
