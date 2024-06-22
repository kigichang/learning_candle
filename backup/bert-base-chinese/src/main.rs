use std::{default, fs};

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::Sampling;
use candle_transformers::models::bert::{self, BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

fn main() -> Result<()> {
    let device = Device::Cpu;

    let default_model = "google-bert/bert-base-chinese".to_string();
    let default_revision = "main".to_string();

    let repo = Repo::with_revision(default_model, RepoType::Model, default_revision);

    // let (config_filename, tokenizer_filename, weights_filename) = {
    //     let api = Api::new()?;
    //     let api = api.repo(repo);
    //     let config = api.get("config.json")?;
    //     let tokenizer = api.get("tokenizer.json")?;
    //     let weights = api.get("model.safetensors")?;
    //     (config, tokenizer, weights)
    // };

    let config_filename = "config.json";
    let tokenizer_filename = "tokenizer.json";
    let weights_filename = "pytorch_model.bin.zip";

    println!("{config_filename:?} {tokenizer_filename:?} {weights_filename:?}");

    let config = fs::read_to_string(config_filename)?;
    let mut config: Config = serde_json::from_str(&config)?;
    let tokenizers = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = VarBuilder::from_pth(&weights_filename, DTYPE, &device)?;
    let model = BertModel::load(vb, &config).unwrap();

    // println!("model 1");
    // let model = BertModel::load(vb, &config)?;
    // println!("model 2");

    // let mask_id = tokenizers.token_to_id("<mask>").unwrap();
    // println!("<mask>: {}", mask_id);
    // let test_sentence = "Tom has fully <mask> <mask> <mask> illness.".to_string();

    // println!("tokenizers");
    // let input_seq = tokenizers
    //     .encode(test_sentence.clone(), true)
    //     .map_err(E::msg)?;

    // println!("input_seq: {:?}", input_seq);
    // println!("token_ids: {:?}", input_seq.get_ids());
    // let masks = input_seq
    //     .get_ids()
    //     .iter()
    //     .enumerate()
    //     .filter_map(|(i, v)| if *v == mask_id { Some(i) } else { None })
    //     .collect::<Vec<usize>>();

    // println!("masked_position: {:?}", masks);

    // let input_ids = Tensor::new(input_seq.get_ids(), &device)?;
    // let input_ids = Tensor::stack(&[&input_ids], 0)?;
    // println!("input_ids.shape: {:?}", input_ids.shape());
    // let token_type_ids = input_ids.zeros_like()?;
    // let token_logits = model.forward(&input_ids, &token_type_ids)?;

    // println!("token_logits: {:?}", token_logits.shape());

    // let mask_token_index = input_seq.get_attention_mask()[1];
    // println!("mask_token_index: {}", mask_token_index);

    // let input_ids = Tensor::new(input_seq.get_ids(), &device)?;
    // let input_ids = Tensor::stack(&[&input_ids], 0)?;
    // let token_type_ids = input_ids.zeros_like()?;

    // let token_logits = model.forward(&input_ids, &token_type_ids)?;
    // let masked_token_logits = token_logits.i((0, mask_token_index as usize, ..))?;
    // println!("masked_token_logits: {:?}", masked_token_logits);

    // let mut gen = candle_transformers::generation::LogitsProcessor::from_sampling(
    //     1000,
    //     Sampling::TopK {
    //         k: 5,
    //         temperature: 1.0,
    //     },
    // );

    // println!("{:?}", gen.sample(&masked_token_logits));
    // println!("{:?}", gen.sample(&masked_token_logits));
    // println!("{:?}", gen.sample(&masked_token_logits));
    // println!("{:?}", gen.sample(&masked_token_logits));
    // println!("{:?}", gen.sample(&masked_token_logits));
    // println!("{:?}", gen.sample(&masked_token_logits));

    // println!("tokens {:?}", tokens);

    // let token_ids = Tensor::new(tokens.get_ids(), &device)?;
    // let token_ids = Tensor::stack(&[&token_ids], 0)?;
    // let token_type_ids = token_ids.zeros_like()?;
    // let output = model.forward(&token_ids, &token_type_ids)?;

    // // let input_ids = &[[0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]];
    // // let input_ids = Tensor::new(input_ids, &model.device)?;
    // // let token_ids = input_ids.zeros_like().unwrap();
    // // let output = model.forward(&input_ids, &token_ids)?;
    // // let expected_shape = [1, 11, 768];
    // // assert_eq!(output.shape().dims(), &expected_shape);

    // println!("output: {:?}", output.shape());

    // let output = output.to_vec3::<f32>()?;

    // println!("[0][0]: {:?}", output[0][0]);

    // // let output_0_0_0 = tokenizers.decode(&output[0][0], false).map_err(E::msg)?;
    // // println!("{:?}", output_0_0_0);
    Ok(())
}
