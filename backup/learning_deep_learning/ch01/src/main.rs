use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() -> Result<()> {
    let mut rng = thread_rng();

    const LEARNING_RATE: f32 = 0.1;
    let mut index_list = [0, 1, 2, 3];

    let x_tran = [
        [1.0_f32, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ];

    let y_train = [1.0, 1.0, 1.0, -1.0];

    let mut w = [0.2_f32, -0.6, 0.25];

    show_learning(&w);

    let mut all_correct = false;
    while !all_correct {
        all_correct = true;
        index_list.shuffle(&mut rng);

        for i in index_list.iter() {
            let x = &x_tran[*i];
            let y = y_train[*i];

            let p_out = compute_output(x, &w)?;
            if y != p_out {
                for j in 0..w.len() {
                    w[j] += y * LEARNING_RATE * x[j];
                }
                all_correct = false;
                show_learning(&w);
            }
        }
    }
    Ok(())
}

pub fn tensor_compute_output(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok(a.mul(b)? // ai * bi
        .sum_all()? // sum(ai * bi)
        .sign()? // sign(sum(ai * bi))
        .to_scalar::<f32>()?) // 取值
}

pub fn compute_output(a: &[f32], b: &[f32]) -> Result<f32> {
    let device = Device::Cpu;
    let a = Tensor::new(a, &device)?;
    let b = Tensor::new(b, &device)?;
    tensor_compute_output(&a, &b)
}

pub fn show_learning(w: &[f32]) {
    println!("{:?}", w);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_output() -> Result<()> {
        assert_eq!(compute_output(&[0.9, -0.6, -0.5], &[1.0, -1.0, -1.0])?, 1.0);
        assert_eq!(compute_output(&[0.9, -0.6, -0.5], &[1.0, -1.0, 1.0])?, 1.0);
        assert_eq!(compute_output(&[0.9, -0.6, -0.5], &[1.0, 1.0, -1.0])?, 1.0);
        assert_eq!(compute_output(&[0.9, -0.6, -0.5], &[1.0, 1.0, 1.0])?, -1.0);

        Ok(())
    }

    #[test]
    fn test_rand_slice() {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let mut y = [1, 2, 3, 4, 5];
        println!("{:?}", y);
        y.shuffle(&mut rng);
        println!("{:?}", y);
    }

    #[test]
    fn test_tensor_full() {
        let device = Device::Cpu;
        let a = Tensor::full(1.0_f32, 3, &device).unwrap();
        println!("{:?}", a.to_vec1::<f32>().unwrap());
    }
}
