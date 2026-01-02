use activations::SIGMOID;
use network::Network;

pub mod activations;
pub mod matrix;
pub mod network;

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut net = Network::new(vec![2, 3, 1], 0.1, SIGMOID);

    net.train(inputs.clone(), targets.clone(), 5000);

    for x in inputs {
        let out = net.feed_forward(x.clone());
		println!("{:?} -> {:?}", x, out);

    }
}
