use super::{activations::Activation, matrix::Matrix};

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>,
}

impl<'a> Network<'a> {
    pub fn new(layers: Vec<usize>, lr: f64, activation: Activation<'a>) -> Self {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Self {
            layers,
            weights,
            biases,
            data: vec![],
            learning_rate: lr,
            activation,
        }
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.weights.len() {
            current = self.weights[i]
                .multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);
            self.data.push(current.clone());
        }

        current.transpose().data[0].clone()
    }

    pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        let mut errors =
            Matrix::from(vec![targets]).transpose()
                .subtract(&Matrix::from(vec![outputs]).transpose());

        for i in (0..self.weights.len()).rev() {
            let activated = self.data[i + 1].clone();
            let gradients = activated
                .map(self.activation.derivative)
                .dot_multiply(&errors)
				.map(&|x| x * self.learning_rate);


            let weight_deltas =
                gradients.multiply(&self.data[i].transpose());

            let old_weights = self.weights[i].clone();

            self.weights[i] = self.weights[i].add(&weight_deltas);
            self.biases[i] = self.biases[i].add(&gradients);

            errors = old_weights.transpose().multiply(&errors);
        }
    }

    pub fn train(
        &mut self,
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        epochs: usize,
    ) {
        for epoch in 1..=epochs {
            for i in 0..inputs.len() {
                let out = self.feed_forward(inputs[i].clone());
                self.back_propagate(out, targets[i].clone());
            }

            if epoch % 1000 == 0 || epoch == epochs {
                println!("Epoch {} / {}", epoch, epochs);
            }
        }
    }
}
