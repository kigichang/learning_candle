#[cfg(test)]
mod tests {
    use anyhow::Result;
    use candle_core::{Device, Tensor, D};

    #[test]
    fn test_tensor() -> Result<()> {
        let data = [[1.0_f32], [2.0], [3.0]]; // 3x1 martix
        let tensor = Tensor::new(&data, &Device::Cpu)?;
        println!("{:?}", tensor.to_vec2::<f32>()?);

        let nested_data = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]; // 3x3 martix
        let nested_tensor = Tensor::new(&nested_data, &Device::Cpu)?;
        println!("{:?}", nested_tensor.to_vec2::<f32>()?);

        let a = nested_tensor.matmul(&tensor)?;
        println!("{:?}", a.to_vec2::<f32>()?);
        println!("{:?}", a.sign()?.to_vec2::<f32>()?);

        //let a = tensor.matmul(&nested_tensor)?; // Error: shape mismatch in matmul, lhs: [3, 1], rhs: [3, 3]

        Ok(())
    }

    #[test]
    fn test_like_full() -> Result<()> {
        let data = [1_u32, 2, 3]; // vector
        let tensor = Tensor::new(&data, &Device::Cpu)?;
        println!("{:?}", tensor.to_vec1::<u32>()?); // vector is 1-dem martix

        let zero_tensor = tensor.zeros_like()?; // zero tensor
        println!("{:?}", zero_tensor.to_vec1::<u32>()?);

        let ones_tensor = tensor.ones_like()?; // ones tensor
        println!("{:?}", ones_tensor.to_vec1::<u32>()?);

        let tensor = Tensor::full(2.0_f32, 3, &Device::Cpu)?; // vector with 3 elements, each element is 2.0
        println!("{:?}", tensor.to_vec1::<f32>()?);

        let tensor = Tensor::full(2.0_f32, (2, 3), &Device::Cpu)?; // 2x3 martix, each element is 2.0
        println!("{:?}", tensor.to_vec2::<f32>()?);

        Ok(())
    }

    #[test]
    fn test_randn() -> Result<()> {
        let a = Tensor::randn(0_f32, 1., (2, 3), &Device::Cpu)?; // random 2x3 martix with normal distribution
        println!("{:?}", a.to_vec2::<f32>()?); // show 2-dem martix

        let b = Tensor::randn(0_f32, 1., (3, 3), &Device::Cpu)?; // random 3x3 martix with normal distribution
        println!("{:?}", b.to_vec2::<f32>()?); // show 2-dem martix

        let c = a.matmul(&b)?; // matrix multiplication, result 2x3 martix
        println!("matmul: {:?}", c.to_vec2::<f32>()?); // show 2-dem martix

        Ok(())
    }

    #[test]
    fn test_shape_dim() -> Result<()> {
        let a = Tensor::randn(0.0_f32, 0.5_f32, (2, 3, 4), &Device::Cpu)?; // random 2x3x4 tensor with normal distribution
        println!("{:?}", a.to_vec3::<f32>()?);

        println!("shape: {:?}", a.shape());
        println!("dims: {:?}", a.dims());
        println!("dim3: {:?}", a.dims3()?);

        println!("dim with D::Minus1: {:?}", a.dim(D::Minus1)?);
        println!("dim with D::Minus2: {:?}", a.dim(D::Minus2)?);

        let b = a.reshape((3, 8))?; // reshape to 3x8 tensor
        println!("{:?}", b.to_vec2::<f32>()?);

        Ok(())
    }

    #[test]
    fn test_broadcast_mul() -> Result<()> {
        let x = Tensor::new(
            &[1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &Device::Cpu,
        )?
        .reshape((4, 3))?;

        let w = Tensor::new(&[0.2_f32, -0.6, 0.25], &Device::Cpu)?;
        let ans = Tensor::new(&[1.0_f32, 1., 1., 1.], &Device::Cpu)?;

        let y = x.broadcast_mul(&w)?; // broadcast multiplication
        println!("broadcast_mul: {:?}", y.to_vec2::<f32>()?);

        let y = y.sum_keepdim(D::Minus1)?;
        println!("sum_keepdim: {:?}", y.to_vec2::<f32>()?);

        let y1 = y.sum(1)?;
        println!("sum(1): {:?}", y1.to_vec1::<f32>()?);

        let y1 = y.sum(D::Minus1)?;
        println!("sum(D::Minus1): {:?}", y1.to_vec1::<f32>()?);

        let sign = y.sum(D::Minus1)?.sign()?;
        println!("sign: {:?}", sign.to_vec1::<f32>()?);
        println!("eq: {:?}", sign.eq(&ans)?.to_vec1::<u8>()?);
        println!("eq_sum: {}", sign.eq(&ans)?.sum_all()?.to_scalar::<u8>()?);

        let y_all = y.sum_all()?;
        println!("sum_all; {:?}", y_all.to_vec0::<f32>()?);

        Ok(())
    }
}
