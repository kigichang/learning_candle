fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use candle_core::utils;
    #[test]
    fn test_device() {
        println!("cuda:{}", utils::cuda_is_available());
        println!("num threads: {}", utils::get_num_threads());
        println!("has_accelerate: {}", utils::has_accelerate());
        println!("has_mkl: {}", utils::has_mkl());
        println!("metal_is_available: {}", utils::metal_is_available());
        println!("with avx: {}", utils::with_avx());
        println!("with f16c: {}", utils::with_f16c());
        println!("with neon: {}", utils::with_neon());
        println!("with simd128: {}", utils::with_simd128());
    }
}
