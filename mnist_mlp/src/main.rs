use anyhow::Result;
use candle_core::{utils, DType, Device, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};
use clap::Parser;

const IMAGE_DIM: usize = 28 * 28;
const LABLES: usize = 10;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 0.05)]
    learning_rate: f64,

    #[arg(long, default_value_t = 200)]
    epochs: usize,

    #[arg(long)]
    save: Option<String>,

    #[arg(long)]
    load: Option<String>,

    #[arg(long)]
    local_mnist: Option<String>,
}

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
    save: Option<String>,
    load: Option<String>,
}

struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Mlp {
    fn new(vb: &VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(IMAGE_DIM, 100, vb.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, LABLES, vb.pp("ln2"))?;

        Ok(Self { ln1, ln2 })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.ln1.forward(x)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

fn training_loop(m: candle_datasets::vision::Dataset, args: TrainingArgs) -> Result<()> {
    let device = if utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else if utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };

    println!("Device: {:?}", device);

    let train_images = m.train_images.to_device(&device)?;

    let train_labels = m
        .train_labels
        .to_dtype(candle_core::DType::U32)?
        .to_device(&device)?;

    let test_images = m.test_images.to_device(&device)?;
    let test_labels = m
        .test_labels
        .to_dtype(candle_core::DType::U32)?
        .to_device(&device)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = Mlp::new(&vb)?;

    if let Some(load) = &args.load {
        println!("loading weights from {load}");
        varmap.load(load)?
    }

    let mut sgd = SGD::new(varmap.all_vars(), args.learning_rate)?; // use Optimizer

    for epoch in 1..=args.epochs {
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    if let Some(save) = &args.save {
        println!("saving trained weights in {save}");
        varmap.save(save)?
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

    println!(
        "train-images-test: {:?} {:?}",
        m.train_images.dim(0),
        m.train_images.dim(1)
    );

    let training_args = TrainingArgs {
        learning_rate: args.learning_rate,
        epochs: args.epochs,
        save: args.save,
        load: args.load,
    };

    training_loop(m, training_args)
}
