use rand;
use rand::Rng;

use simplenn::function::activation::*;
use simplenn::function::optimizer::*;
use simplenn::function::loss::*;
use simplenn::NN;
use simplenn::NNModel;
use simplenn::persistence::*;

struct Intelligence {
	nna:NN<Adam,Mse>,
	nnb:NN<Adam,Mse>,
}
impl Intelligence {
	pub fn new (savepath:String) -> Intelligence {
		let mut rnd = rand::XorShiftRng::new_unseeded();
		let mut units:Vec<(usize,Box<ActivateF>)> = Vec::new();
		units.push((4304,Box::new(FReLU::new())));
		units.push((4304,Box::new(FReLU::new())));
		units.push((1,Box::new(FTanh::new())));

		let mut model:NNModel = NNModel::with_bias_and_unit_initializer(2152,
										units,
										TextFileInputReader::new(format!("{}/nn.a.txt",savepath).as_str()).unwrap(),
										0f64,move || {
											let i = rnd.next_u32();
											if i % 2 == 0{
												rnd.next_f64()
											} else {
												-rnd.next_f64()
											}
										}).unwrap();
		let mut nnb = NN::new(model,|s| Adam::new(s),Mse::new());

		let mut rnd = rand::XorShiftRng::new_unseeded();
		let mut units:Vec<(usize,Box<ActivateF>)> = Vec::new();
		units.push((4304,Box::new(FReLU::new())));
		units.push((4304,Box::new(FReLU::new())));
		units.push((1,Box::new(FTanh::new())));

		let mut model:NNModel = NNModel::with_bias_and_unit_initializer(2152,
										units,
										TextFileInputReader::new(format!("{}/nn.a.txt",savepath).as_str()).unwrap(),
										0f64,move || {
											let i = rnd.next_u32();
											if i % 2 == 0{
												rnd.next_f64()
											} else {
												-rnd.next_f64()
											}
										}).unwrap();
		let mut nna = NN::new(model,|s| Adam::new(s),Mse::new());

		Intelligence {
			nna:nna,
			nnb:nnb,
		}
	}
}