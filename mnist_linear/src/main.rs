use anyhow::Result;
use candle_core::{DType, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use clap::Parser;

const IMAGE_DIM: usize = 28 * 28;
const LABLES: usize = 10;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    local_mnist: Option<String>,

    #[arg(long, default_value_t = 1.0)]
    learning_rate: f64,

    #[arg(long, default_value_t = 200)]
    epochs: usize,

    #[arg(long)]
    load: Option<String>,

    #[arg(long)]
    save: Option<String>,
}

#[derive(Debug)]
struct TrainingArgs {
    learning_rate: f64,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
}

struct LinearModel {
    linear: Linear,
}

impl LinearModel {
    fn new(vb: &VarBuilder) -> Result<Self> {
        let linear = linear_z(IMAGE_DIM, LABLES, vb)?;
        Ok(Self { linear })
    }
}

impl Module for LinearModel {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.linear.forward(x)
    }
}

fn linear_z(in_dim: usize, out_dim: usize, vb: &VarBuilder) -> Result<Linear> {
    let w = vb.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
    let b = vb.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(w, Some(b)))
}

fn training_loop(m: candle_datasets::vision::Dataset, args: &TrainingArgs) -> Result<()> {
    let device = candle_core::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&device)?;
    let train_images = m.train_images.to_device(&device)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = LinearModel::new(&vb)?;

    if let Some(load) = &args.load {
        println!("loading model from {load}...");
        varmap.load(load)?;
    }

    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;
    let test_images = m.test_images.to_device(&device)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&device)?;

    for epoch in 1..=args.epochs {
        let logits = model.forward(&train_images)?;
        let log_sum = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sum, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / (test_labels.dims1()? as f32);

        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }

    if let Some(save) = &args.save {
        println!("saving model to {save}...");
        varmap.save(save)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let m = if let Some(directory) = args.local_mnist {
        candle_datasets::vision::mnist::load_dir(directory)?
    } else {
        candle_datasets::vision::mnist::load()?
    };

    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    let training_args = TrainingArgs {
        learning_rate: args.learning_rate,
        load: args.load,
        save: args.save,
        epochs: args.epochs,
    };

    training_loop(m, &training_args)
}
