use rand;
use rand::Rng;

use simplenn::function::activation::*;
use simplenn::function::optimizer::*;
use simplenn::function::loss::*;
use simplenn::NN;
use simplenn::NNModel;
use simplenn::NNUnits;
use simplenn::persistence::*;

struct Intelligence {
	nna:NN<Adam,Mse>,
	nnb:NN<Adam,Mse>,
	nnsavepath:String,
}
impl Intelligence {
	pub fn new (savepath:String) -> Intelligence {
		let mut rnd = rand::XorShiftRng::new_unseeded();

		let mut model:NNModel = NNModel::with_bias_and_unit_initializer(
										NNUnits::new(2120,
											(4240,Box::new(FReLU::new())),
											(4240,Box::new(FReLU::new())))
											.add((1,Box::new(FTanh::new()))),
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

		let mut rnd = rand::XorShiftRng::new_unseeded();

		let mut model:NNModel = NNModel::with_bias_and_unit_initializer(
										NNUnits::new(2120,
											(4240,Box::new(FReLU::new())),
											(4240,Box::new(FReLU::new())))
											.add((1,Box::new(FTanh::new()))),
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

		Intelligence {
			nna:nna,
			nnb:nnb,
			nnsavepath:savepath,
		}
	}
}