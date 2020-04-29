use std::env;

fn main() -> failure::Fallible<()> {
    let text = env::args().skip(1).next().unwrap();

    let model = tch::CModule::load("torchscript_cpu_model.pt")?;
    let backend = nnsplit::TchRsBackend::new(model, tch::Device::Cpu, 32);

    let splitter = nnsplit::NNSplit::new(&backend as &dyn nnsplit::Backend)?;

    let input: Vec<&str> = vec![&text]; // input can consist of multiple texts to allow parallelization
    println!("{:#?}", splitter.split(input));

    Ok(())
}
