use std::f64::consts::E;

#[derive(Clone)]
pub struct Activation<'a> {
    // NOTE: derivative expects ACTIVATED value, not pre-activation
    pub function: &'a dyn Fn(f64) -> f64,
    pub derivative: &'a dyn Fn(f64) -> f64,
}

pub const IDENTITY: Activation = Activation {
    function: &|x| x,
    derivative: &|_| 1.0,
};

pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + E.powf(-x)),
    derivative: &|y| y * (1.0 - y), // y = sigmoid(x)
};

pub const TANH: Activation = Activation {
    function: &|x| x.tanh(),
    derivative: &|y| 1.0 - y * y, // y = tanh(x)
};

pub const RELU: Activation = Activation {
    function: &|x| x.max(0.0),
    derivative: &|y| if y > 0.0 { 1.0 } else { 0.0 },
};
