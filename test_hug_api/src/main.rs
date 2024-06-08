use anyhow::{anyhow, Result};
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};

fn main() -> Result<()> {
    let default_model = "roberta-base".to_string();
    let default_revision = "main".to_string();
    let (model_id, revision) = (default_model, default_revision);
    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let offline = false;

    let (config_filename, _tokenizer_filename, _weights_filename) = if offline {
        let cache = Cache::default().repo(repo);
        (
            cache
                .get("config.json")
                .ok_or(anyhow!("Missing config file in cache"))?,
            cache
                .get("tokenizer.json")
                .ok_or(anyhow!("Missing tokenizer file in cache"))?,
            cache
                .get("model.safetensors")
                .ok_or(anyhow!("Missing weights file in cache"))?,
        )
    } else {
        let api = Api::new()?;
        let api = api.repo(repo);
        (
            api.get("config.json")?,
            api.get("tokenizer.json")?,
            api.get("model.safetensors")?,
        )
    };

    println!("config_filename: {}", config_filename.display());
    Ok(())
}
