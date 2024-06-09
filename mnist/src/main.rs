use anyhow::Result;
use candle_core::Device;
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    local_mnist: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("args: {:?}", args);
    let device = Device::cuda_if_available(0)?;
    println!("device: {:?}", device);

    let m = if let Some(directory) = args.local_mnist {
        candle_datasets::vision::mnist::load_dir(directory)?
    } else {
        candle_datasets::vision::mnist::load()?
    };

    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    Ok(())
}
