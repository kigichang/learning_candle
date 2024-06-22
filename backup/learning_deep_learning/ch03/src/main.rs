use std::result;

use anyhow::{Error as E, Result};
use candle_core::{utils, DType, Device, IndexOp, D};
use candle_datasets::vision::mnist;
use candle_nn::{encoding::one_hot, linear, loss, ops, Module, Optimizer, VarBuilder, VarMap, SGD};
use clap::Parser;

const IMAGE_DIM: usize = 28 * 28;
const LABLES: usize = 10;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    local_mnist: Option<String>,

    #[arg(long, default_value_t = 200)]
    epochs: usize,

    #[arg(long)]
    save: Option<String>,

    #[arg(long)]
    load: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = if utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else if utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };

    println!("device: {:?}", device);

    let m = if let Some(directory) = args.local_mnist {
        mnist::load_dir(directory)?
    } else {
        mnist::load()?
    };

    // let a = m.train_images.i(0)?;
    // println!("a: {:?}", a.to_vec1::<f32>()?);

    let train_labels = one_hot::<u32>(m.train_labels.clone(), 10, 1, 0)?;
    let test_labels = one_hot::<u32>(m.test_labels.clone(), 10, 1, 0)?;
    println!("train_images: {:?}", m.train_images.shape());
    println!("train_labels: {:?}", train_labels.shape());

    println!("test_images: {:?}", m.test_images.shape());
    println!("test_labels: {:?}", test_labels.shape());

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let train_images = m.train_images.to_device(&device)?;
    let train_labels = train_labels.to_device(&device)?;

    let test_images = m.test_images.to_device(&device)?;
    let test_labels = test_labels.to_device(&device)?;

    let learning_rate = 0.01_f64;
    let mut sgd = SGD::new(varmap.all_vars(), learning_rate)?;

    // let w1 = vb.get_with_hints(
    //     (IMAGE_DIM, 25),
    //     "weight",
    //     candle_nn::init::Init::Uniform { lo: -0.1, up: 0.1 },
    // )?;
    // let b1 = vb.get_with_hints(25, "bias", candle_nn::init::ZERO)?;

    // let ln1 = linear::Linear::new(w1, Some(b1));

    let ln1 = linear::linear(IMAGE_DIM, 25, vb.pp("ln1"))?;
    let ln2 = linear::linear(25, LABLES, vb.pp("ln2"))?;

    if let Some(load) = &args.load {
        println!("loading weights from {load}");
        varmap.load(load)?
    }

    for epoch in 1..=args.epochs {
        let result = ln1.forward(&train_images)?;
        let result = result.tanh()?;

        let result = ln2.forward(&result)?;
        let result = candle_nn::ops::sigmoid(&result)?;
        println!("{epoch}: 1 result: {:?}", result.i(0)?.to_vec1::<f32>());
        let result = result.argmax(D::Minus1)?;
        println!("{epoch}: 2 result: {:?}", result.i(0)?.to_scalar::<u32>());

        let loss = loss::mse(&result, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = ln1.forward(&test_images)?.tanh()?;
        let test_logits = candle_nn::ops::sigmoid(&ln2.forward(&test_logits)?)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        println!("sum_ok: {:?}", sum_ok);

        // let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        // println!(
        //     "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
        //     loss.to_scalar::<f32>()?,
        //     100. * test_accuracy
        // );
    }

    if let Some(save) = &args.save {
        println!("saving trained weights in {save}");
        varmap.save(save)?
    }

    Ok(())
}
