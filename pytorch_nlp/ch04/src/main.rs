fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests_4_1 {

    use candle_core::{DType, Device, IndexOp, Tensor, D};
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    fn init_tracing() {
        let (chrome_layer, _) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
    }

    #[test]
    fn test_tensor_basic() {
        init_tracing();
        let t = Tensor::new(&[[1_i64, 2, 3], [4, 5, 6]], &Device::Cpu).unwrap();
        println!(
            "{:?}, {:?}, {:?}",
            t.to_vec2::<i64>().unwrap(),
            t.shape(),
            t.dtype()
        );
    }

    #[test]
    fn test_rand_ones_zeros() {
        init_tracing();
        let rand_t = Tensor::rand(0.0_f32, 1.0_f32, (3, 3), &Device::Cpu).unwrap();
        let ones_t = Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap();
        let zeros_t = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();

        println!("{:?}", rand_t.to_vec2::<f32>().unwrap());
        println!("{:?}", ones_t.to_vec2::<f32>().unwrap());
        println!("{:?}", zeros_t.to_vec2::<f32>().unwrap());

        let arange_t = Tensor::arange(1.0_f32, 3.0_f32, &Device::Cpu).unwrap();
        println!("{:?}", arange_t.to_vec1::<f32>().unwrap());

        let arange_t = Tensor::arange_step(1.0_f32, 3.0_f32, 0.5_f32, &Device::Cpu).unwrap();
        println!("{:?}", arange_t.to_vec1::<f32>().unwrap());

        let eye_t = Tensor::eye(3, DType::F32, &Device::Cpu).unwrap(); // 3x3 identity matrix
        println!("{:?}", eye_t.to_vec2::<f32>().unwrap());
    }

    #[test]
    fn test_cat() {
        init_tracing();

        let t1 = Tensor::new(&[1.0_f32, 2., 3.], &Device::Cpu).unwrap();
        let t2 = Tensor::new(&[4.0_f32, 5., 6.], &Device::Cpu).unwrap();

        let t3 = Tensor::cat(&[&t1, &t2], 0).unwrap();
        println!(
            "vec concat: {:?}, {:?}",
            t3.to_vec1::<f32>().unwrap(),
            t3.shape()
        );

        let t1 = Tensor::new(&[[1.0_f32, 2., 3.], [1., 2., 3.]], &Device::Cpu).unwrap();
        let t2 = Tensor::new(&[[4.0_f32, 5., 6.], [4., 5., 6.]], &Device::Cpu).unwrap();

        let t3 = Tensor::cat(&[&t1, &t2], 0).unwrap();
        println!(
            "2x2 cat dim0: {:?}, {:?}",
            t3.to_vec2::<f32>().unwrap(),
            t3.shape()
        );

        let t4 = Tensor::cat(&[&t1, &t2], 1).unwrap();
        println!(
            "2x2 cat dim1: {:?}, {:?}",
            t4.to_vec2::<f32>().unwrap(),
            t4.shape()
        );

        let t3 = Tensor::cat(&[&t1, &t2], D::Minus2).unwrap();
        println!(
            "2x2 cat (D:Minus2): {:?}, {:?}",
            t3.to_vec2::<f32>().unwrap(),
            t3.shape()
        );

        let t4 = Tensor::cat(&[&t1, &t2], D::Minus1).unwrap();
        println!(
            "2x2 cat (D:Minus1): {:?}, {:?}",
            t4.to_vec2::<f32>().unwrap(),
            t4.shape()
        );
    }

    #[test]
    fn test_stack() {
        init_tracing();

        let t1 = Tensor::new(&[[1.0_f32, 2., 3.], [1., 2., 3.]], &Device::Cpu).unwrap();
        let t2 = Tensor::new(&[[4.0_f32, 5., 6.], [4., 5., 6.]], &Device::Cpu).unwrap();

        let t3 = Tensor::stack(&[&t1, &t2], 0).unwrap();
        println!(
            "stack dim0\n{:#?}\nshape:{:?}",
            t3.to_vec3::<f32>().unwrap(),
            t3.shape()
        );

        let t4 = Tensor::stack(&[&t1, &t2], 1).unwrap();
        println!(
            "stack dim1\n{:#?}\nshape:{:?}",
            t4.to_vec3::<f32>().unwrap(),
            t4.shape()
        );

        let t5 = Tensor::stack(&[&t1, &t2], 2).unwrap();
        println!(
            "stack dim2\n{:#?}\nshape:{:?}",
            t5.to_vec3::<f32>().unwrap(),
            t5.shape()
        );
    }

    #[test]
    fn test_chunk() {
        let t = Tensor::new(&[1.0_f32, 2., 3., 4., 5.], &Device::Cpu).unwrap();

        let chunks = t.chunk(1, 0).unwrap();
        println!("chunks(size=1, dim=0): {:?}, {:?}", chunks, chunks.len());

        let chunks = t.chunk(2, 0).unwrap();
        println!("chunks(size=2, dim=0): {:?}, {:?}", chunks, chunks.len());

        let chunks = t.chunk(3, 0).unwrap();
        println!("chunks(size=3, dim=0): {:?}, {:?}", chunks, chunks.len());

        let chunks = t.chunk(4, 0).unwrap();
        println!("chunks(size=4, dim=0): {:?}, {:?}", chunks, chunks.len());

        let chunks = t.chunk(5, 0).unwrap();
        println!("chunks(size=5, dim=0): {:?}, {:?}", chunks, chunks.len());

        let t = Tensor::new(&[[1.0_f32, 2., 3.], [4., 5., 6.]], &Device::Cpu).unwrap();
        let chunks = t.chunk(1, 0).unwrap();
        println!("chunks(size=1, dim=0): {:?}, {:?}", chunks, chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("{i}={:?}", chunk.to_vec2::<f32>().unwrap());
        }

        let chunks = t.chunk(2, 0).unwrap();
        println!("chunks(size=2, dim=0): {:?}, {:?}", chunks, chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("{i}={:?}", chunk.to_vec2::<f32>().unwrap());
        }

        let chunks = t.chunk(3, 0).unwrap();
        println!("chunks(size=3, dim=0): {:?}, {:?}", chunks, chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            println!("{i}={:?}", chunk.to_vec2::<f32>().unwrap());
        }

        let chunks = t.chunk(1, 1).unwrap();
        println!("chunks(size=1, dim=1): {:?}, {:?}", chunks, chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("{i}={:?}", chunk.to_vec2::<f32>().unwrap());
        }

        let chunks = t.chunk(2, 1).unwrap();
        println!("chunks(size=2, dim=1): {:?}, {:?}", chunks, chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("{i}={:?}", chunk.to_vec2::<f32>().unwrap());
        }

        let chunks = t.chunk(3, 1).unwrap();
        println!("chunks(size=3, dim=1): {:?}, {:?}", chunks, chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("{i}={:?}", chunk.to_vec2::<f32>().unwrap());
        }

        let chunks = t.chunk(4, 1).unwrap();
        println!("chunks(size=4, dim=1): {:?}, {:?}", chunks, chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("{i}={:?}", chunk.to_vec2::<f32>().unwrap());
        }
    }

    #[test]
    fn test_split() {
        todo!("split test")
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::new(&[1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9.], &Device::Cpu).unwrap();
        let t1 = t.reshape((3, 3)).unwrap();
        println!("{:?}", t1.to_vec2::<f32>().unwrap());

        //let t1 = t.reshape((3, -1)).unwrap(); // error
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::new(&[[1.0_f32, 2., 3.], [4., 5., 6.]], &Device::Cpu).unwrap();
        let t1 = t.transpose(0, 1).unwrap();
        println!("transpose (0, 1): {:?}", t1.to_vec2::<f32>().unwrap());

        let vec: Vec<f32> = Vec::from_iter(1..=24)
            .into_iter()
            .map(|x| x as f32)
            .collect();

        let t = Tensor::new(vec.as_slice(), &Device::Cpu)
            .unwrap()
            .reshape((2, 3, 4))
            .unwrap();
        println!("original: {:?}", t.to_vec3::<f32>().unwrap());

        let t1 = t.transpose(0, 1).unwrap();
        println!("transpose(0, 1): {:?}", t1.to_vec3::<f32>().unwrap());

        let t2 = t.transpose(0, 2).unwrap();
        println!("transpose(0, 2): {:?}", t2.to_vec3::<f32>().unwrap());

        let t3 = t.transpose(1, 2).unwrap();
        println!("transpose(1, 2): {:?}", t3.to_vec3::<f32>().unwrap());
    }

    #[test]
    fn test_squeeze() {
        let t = Tensor::from_iter(1..=6_i64, &Device::Cpu)
            .unwrap()
            .reshape((2, 3))
            .unwrap();

        println!("orginal: {:?}", t.to_vec2::<i64>().unwrap());

        let t2 = t.unsqueeze(0).unwrap();
        println!(
            "unsqueeze(0): {:?}, {:?}",
            t2.to_vec3::<i64>().unwrap(),
            t2.shape()
        );
        let t20 = t.squeeze(0).unwrap();
        println!(
            "\tsqueeze(0): {:?}, {:?}",
            t20.to_vec2::<i64>().unwrap(),
            t20.shape()
        );

        let t3 = t.unsqueeze(1).unwrap();
        println!(
            "unsqueeze(1): {:?}, {:?}",
            t3.to_vec3::<i64>().unwrap(),
            t3.shape()
        );

        let t31 = t3.squeeze(1).unwrap();
        println!(
            "\tsqueeze(1): {:?}, {:?}",
            t31.to_vec2::<i64>().unwrap(),
            t31.shape()
        );

        let t30 = t3.squeeze(0).unwrap();
        println!(
            "\tsqueeze(0): {:?}, {:?}",
            t30.to_vec3::<i64>().unwrap(),
            t30.shape()
        );

        let t4 = t.unsqueeze(2).unwrap();
        println!(
            "unsqueeze(2): {:?}, {:?}",
            t4.to_vec3::<i64>().unwrap(),
            t4.shape()
        );

        let t42 = t4.squeeze(D::Minus1).unwrap();
        println!(
            "\tsqueeze(D::Minus1): {:?}, {:?}",
            t42.to_vec2::<i64>().unwrap(),
            t42.shape()
        );

        let t42 = t4.squeeze(2).unwrap();
        println!(
            "\tsqueeze(2): {:?}, {:?}",
            t42.to_vec2::<i64>().unwrap(),
            t42.shape()
        );

        let t41 = t4.squeeze(1).unwrap();
        println!(
            "\tsqueeze(1): {:?}, {:?}",
            t41.to_vec3::<i64>().unwrap(),
            t41.shape()
        );
    }

    #[test]
    fn test_expand() {
        let x = Tensor::new(&[[[0.5_f32, 0.1, 0.3]], [[0.8, 0.2, 0.1]]], &Device::Cpu).unwrap();
        println!("x:{:?}, {:?}", x.to_vec3::<f32>().unwrap(), x.shape());

        let y = x.expand((2, 8, 3)).unwrap();
        println!("y:{:?}, {:?}", y.to_vec3::<f32>().unwrap(), y.shape());
    }

    #[test]
    fn test_repeat() {
        let x = Tensor::new(&[[[0.5_f32, 0.1, 0.3]], [[0.8, 0.2, 0.1]]], &Device::Cpu).unwrap();
        println!("x:{:?}, {:?}", x.to_vec3::<f32>().unwrap(), x.shape());

        let y = x.repeat((2, 2, 2)).unwrap();
        println!("y:{:?}, {:?}", y.to_vec3::<f32>().unwrap(), y.shape());

        let y = x.repeat((1, 2, 3)).unwrap();
        println!("y:{:?}, {:?}", y.to_vec3::<f32>().unwrap(), y.shape());
    }

    #[test]
    fn test_index() {
        let t = Tensor::from_iter(1..=6_i64, &Device::Cpu)
            .unwrap()
            .reshape((2, 3))
            .unwrap();

        println!("{:?}", t.to_vec2::<i64>().unwrap());

        let t1 = t.i(1).unwrap();
        println!("{:?}", t1.to_vec1::<i64>().unwrap());

        let t2 = t.i((1, 2)).unwrap();
        println!("{:?}", t2.to_scalar::<i64>().unwrap());

        let t = Tensor::from_iter(1..=9_i64, &Device::Cpu)
            .unwrap()
            .reshape((3, 3))
            .unwrap();

        let t1 = t.i((.., 1)).unwrap();
        println!("{:?}", t1.to_vec1::<i64>().unwrap());
    }
}

#[cfg(test)]
mod tests_4_2 {

    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_add() {
        let t1 = Tensor::arange(1_i64, 4_i64, &Device::Cpu).unwrap();
        println!("t1: {:?}", t1.to_vec1::<i64>().unwrap());
        let t2 = Tensor::arange(4_i64, 7_i64, &Device::Cpu).unwrap();
        println!("t2: {:?}", t2.to_vec1::<i64>().unwrap());

        let t3 = t1.add(&t2).unwrap();
        println!("t3: {:?}", t3.to_vec1::<i64>().unwrap());

        let t3 = (&t1 + &t2).unwrap(); // 要用 `&` 否則會被 move
        println!("t3: {:?}", t3.to_vec1::<i64>().unwrap());

        let t3 = t1.broadcast_add(&t2).unwrap();
        println!("t3: {:?}", t3.to_vec1::<i64>().unwrap());

        let one = Tensor::ones(1, DType::I64, &Device::Cpu).unwrap();
        println!("one: {:?}", one.to_vec1::<i64>().unwrap());

        let t3 = t1.broadcast_add(&one).unwrap();
        println!("t3: {:?}", t3.to_vec1::<i64>().unwrap());

        // let t3 = (&t1 + &one).unwrap(); // error
        // let t3 = t1.add(&one).unwrap(); // error

        let t1 = Tensor::from_iter(1..=3_i64, &Device::Cpu).unwrap();
        let t2 = Tensor::from_iter(1..=9_i64, &Device::Cpu)
            .unwrap()
            .reshape((3, 3))
            .unwrap();

        println!("t1: {:?}", t1.to_vec1::<i64>().unwrap());
        println!("t2: {:?}", t2.to_vec2::<i64>().unwrap());

        let t3 = t1.broadcast_add(&t2).unwrap();
        println!("t3: {:?}", t3.to_vec2::<i64>().unwrap());

        let t3 = t2.broadcast_add(&t1).unwrap();
        println!("t3: {:?}", t3.to_vec2::<i64>().unwrap());

        // let t1 = Tensor::from_iter(1..=3_i64, &Device::Cpu).unwrap();
        // let t2 = Tensor::from_iter(1..=9_i64, &Device::Cpu).unwrap();
        // let t3 = t1.broadcast_add(&t2).unwrap(); // error
    }
}
