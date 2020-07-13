use nnsplit::{NNSplit, NNSplitOptions};
use std::time::Instant;

fn main() {
    let data: Vec<String> =
        serde_json::from_str(include_str!("../../benchmarks/sample.json")).unwrap();
    let data: Vec<&str> = data.iter().map(|x| x.as_str()).collect();

    for batch_size in &[256, 1024] {
        for device in &[tch::Device::Cpu, tch::Device::Cuda(0)] {
            let splitter = NNSplit::load(
                "de",
                *device,
                NNSplitOptions {
                    batch_size: *batch_size,
                    ..NNSplitOptions::default()
                },
            )
            .unwrap();

            println!("{} {:#?}", batch_size, device);
            let now = Instant::now();

            splitter.split(&data);
            println!("Time: {}", now.elapsed().as_millis());
        }
    }
}
