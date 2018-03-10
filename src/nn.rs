use rand;
use rand::Rng;

use simplenn::function::activation::*;
use simplenn::function::optimizer::*;
use simplenn::function::loss::*;
use simplenn::NN;
use simplenn::NNModel;
use simplenn::NNUnits;
use simplenn::persistence::*;

use usiagent::shogi::*;

pub struct Intelligence {
	nna:NN<SGD,CrossEntropy>,
	nnb:NN<SGD,CrossEntropy>,
	nnsavepath:String,
}
impl Intelligence {
	pub fn new (savepath:String) -> Intelligence {
		let mut rnd = rand::XorShiftRng::new_unseeded();

		let model:NNModel = NNModel::with_bias_and_unit_initializer(
										NNUnits::new(2282,
											(4564,Box::new(FReLU::new())),
											(4564,Box::new(FReLU::new())))
											.add((1,Box::new(FSigmoid::new()))),
										TextFileInputReader::new(format!("{}/nn.a.txt",savepath).as_str()).unwrap(),
										0f64,move || {
											let i = rnd.next_u32();
											if i % 2 == 0{
												rnd.next_f64()
											} else {
												-rnd.next_f64()
											}
										}).unwrap();
		let nna = NN::new(model,|_| SGD::new(0.5),CrossEntropy::new());

		let mut rnd = rand::XorShiftRng::new_unseeded();

		let model:NNModel = NNModel::with_bias_and_unit_initializer(
										NNUnits::new(2282,
											(4564,Box::new(FReLU::new())),
											(4564,Box::new(FReLU::new())))
											.add((1,Box::new(FSigmoid::new()))),
										TextFileInputReader::new(format!("{}/nn.a.txt",savepath).as_str()).unwrap(),
										0f64,move || {
											let i = rnd.next_u32();
											if i % 2 == 0{
												rnd.next_f64()
											} else {
												-rnd.next_f64()
											}
										}).unwrap();
		let nnb = NN::new(model,|_| SGD::new(0.5),CrossEntropy::new());

		Intelligence {
			nna:nna,
			nnb:nnb,
			nnsavepath:savepath,
		}
	}

	pub fn evalute(&mut self,t:Teban,b:&Banmen,mc:&MochigomaCollections) -> i32 {
		0
	}
}