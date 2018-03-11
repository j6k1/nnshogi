use rand;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Mutex;
use std::i32;

use simplenn::function::activation::*;
use simplenn::function::optimizer::*;
use simplenn::function::loss::*;
use simplenn::NN;
use simplenn::NNModel;
use simplenn::NNUnits;
use simplenn::persistence::*;
use simplenn::error::InvalidStateError;

use usiagent::shogi::*;
use usiagent::event::EventQueue;
use usiagent::event::UserEvent;
use usiagent::event::UserEventKind;
use usiagent::error::EventDispatchError;
use usiagent::event::GameEndState;

use error::*;

pub struct Intelligence {
	nna:NN<SGD,CrossEntropy>,
	nnb:NN<SGD,CrossEntropy>,
	nnsavepath:String,
	quited:bool,
}
impl Intelligence {
	pub fn new (savepath:String) -> Intelligence {
		let mut rnd = rand::XorShiftRng::new_unseeded();

		let model:NNModel = NNModel::with_bias_and_unit_initializer(
										NNUnits::new(2344,
											(2344,Box::new(FReLU::new())),
											(2344,Box::new(FReLU::new())))
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
										NNUnits::new(2344,
											(2344,Box::new(FReLU::new())),
											(2344,Box::new(FReLU::new())))
											.add((1,Box::new(FSigmoid::new()))),
										TextFileInputReader::new(format!("{}/nn.b.txt",savepath).as_str()).unwrap(),
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
			quited:false,
		}
	}

	pub fn evalute(&mut self,t:Teban,b:&Banmen,mc:&MochigomaCollections)
		-> Result<i32,InvalidStateError> {
		let mut input:Vec<f64> = Vec::new();
		input.extend_from_slice(&self.make_input(t,b,mc));

		let nnaanswer = self.nna.solve(&input)?;
		let nnbanswer = self.nnb.solve(&input)?;

		let mut rnd = rand::XorShiftRng::new_unseeded();

		let a = rnd.next_f64();
		let b = 1f64 - a;

		let nnaanswer = nnaanswer[0];
		let nnbanswer = nnbanswer[0];

		let answer = nnaanswer * a + nnbanswer * b;

		Ok(((answer - 0.5) * i32::MAX as f64) as i32)
	}

	pub fn learning<'a>(&mut self,teban:Teban,
		history:Vec<(Banmen,MochigomaCollections)>,s:&GameEndState,
		event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
		-> Result<(),CommonError> {

		let mut t = teban;

		for h in history.iter().rev() {
			match self.handle_events(&*event_queue) {
				Err(_) => {
					return Err(CommonError::Fail(
							String::from("An error occurred while processing the event.")));
				},
				_ => (),
			}

			if self.quited {
				break;
			}

			let mut input:Vec<f64> = Vec::new();
			input.extend_from_slice(&self.make_input(t,&h.0,&h.1));

			let promise_a = self.nna.promise_of_learn(&input)?;
			let promise_b = self.nnb.promise_of_learn(&input)?;

			let mut rnd = rand::XorShiftRng::new_unseeded();

			let a = rnd.next_f64();
			let b = 1f64 - a;

			let nnaanswer = promise_a.r[0];
			let nnbanswer = promise_b.r[0];

			let mut answer:Vec<f64> = Vec::new();

			match s {
				&GameEndState::Win => {
					answer.push(nnaanswer * a + nnbanswer * b);
				},
				_ =>  {
					answer.push(0f64);
				}
			}

			self.nna.latter_part_of_learning(&answer,promise_a)?;
			self.nnb.latter_part_of_learning(&answer,promise_b)?;

			t = t.opposite();
		}

		self.save()
	}

	fn save(&mut self) -> Result<(),CommonError>{
		self.nna.save(
			PersistenceWithTextFile::new(
				&format!("{}/nn.a.txt",self.nnsavepath.as_str()))?)?;
		self.nnb.save(
			PersistenceWithTextFile::new(
				&format!("{}/nn.b.txt",self.nnsavepath.as_str()))?)?;
		Ok(())
	}

	fn handle_events<'a>(&mut self,event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
		-> Result<(), EventDispatchError<'a,EventQueue<UserEvent,UserEventKind>,UserEvent,CommonError>>
		{
		self.dispatch_events(event_queue)?;

		Ok(())
	}

	fn dispatch_events<'a>(&mut self, event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)-> Result<(), EventDispatchError<'a,EventQueue<UserEvent,UserEventKind>,UserEvent,CommonError>>
		{
		let events = {
			event_queue.lock()?.drain_events()
		};

		for e in &events {
			match e {
				&UserEvent::Quit => {
					self.quited = true
				},
				_ => (),
			};
		}

		Ok(())
	}

	pub fn make_input(&self,t:Teban,b:&Banmen,mc:&MochigomaCollections) -> [f64; 2344] {
		let mut inputs:[f64; 2344] = [0f64; 2344];

		match b {
			&Banmen(ref kinds) => {
				for y in 0..9 {
					for x in 0..9 {
						let kind = kinds[y][x];
						let (x,y) = match t {
							Teban::Sente => (x,y),
							Teban::Gote => (8 - x, 8 - y),
						};

						match kind {
							KomaKind::SFu => {
								inputs[y * 9 + x] = 1f64;
							},
							KomaKind::SKyou => {
								inputs[y * 9 + x + 81] = 1f64;
							},
							KomaKind::SKei => {
								inputs[y * 9 + x + 81 * 2] = 1f64;
							},
							KomaKind::SGin => {
								inputs[y * 9 + x + 81 * 3] = 1f64;
							},
							KomaKind::SKin => {
								inputs[y * 9 + x + 81 * 4] = 1f64;
							},
							KomaKind::SKaku => {
								inputs[y * 9 + x + 81 * 5] = 1f64;
							},
							KomaKind::SHisha => {
								inputs[y * 9 + x + 81 * 6] = 1f64;
							},
							KomaKind::SOu => {
								inputs[y * 9 + x + 81 * 7] = 1f64;
							},
							KomaKind::SFuN => {
								inputs[y * 9 + x + 81 * 8] = 1f64;
							},
							KomaKind::SKyouN => {
								inputs[y * 9 + x + 81 * 9] = 1f64;
							},
							KomaKind::SKeiN => {
								inputs[y * 9 + x + 81 * 10] = 1f64;
							},
							KomaKind::SGinN => {
								inputs[y * 9 + x + 81 * 11] = 1f64;
							},
							KomaKind::SKakuN => {
								inputs[y * 9 + x + 81 * 12] = 1f64;
							},
							KomaKind::SHishaN => {
								inputs[y * 9 + x + 81 * 13] = 1f64;
							},
							KomaKind::GFu => {
								inputs[y * 9 + x + 81 * 14] = 1f64;
							},
							KomaKind::GKyou => {
								inputs[y * 9 + x + 81 * 15] = 1f64;
							},
							KomaKind::GKei => {
								inputs[y * 9 + x + 81 * 16] = 1f64;
							},
							KomaKind::GGin => {
								inputs[y * 9 + x + 81 * 17] = 1f64;
							},
							KomaKind::GKin => {
								inputs[y * 9 + x + 81 * 18] = 1f64;
							},
							KomaKind::GKaku => {
								inputs[y * 9 + x + 81 * 19] = 1f64;
							},
							KomaKind::GHisha => {
								inputs[y * 9 + x + 81 * 20] = 1f64;
							},
							KomaKind::GOu => {
								inputs[y * 9 + x + 81 * 21] = 1f64;
							},
							KomaKind::GFuN => {
								inputs[y * 9 + x + 81 * 22] = 1f64;
							},
							KomaKind::GKyouN => {
								inputs[y * 9 + x + 81 * 23] = 1f64;
							},
							KomaKind::GKeiN => {
								inputs[y * 9 + x + 81 * 24] = 1f64;
							},
							KomaKind::GGinN => {
								inputs[y * 9 + x + 81 * 25] = 1f64;
							},
							KomaKind::GKakuN => {
								inputs[y * 9 + x + 81 * 26] = 1f64;
							},
							KomaKind::GHishaN => {
								inputs[y * 9 + x + 81 * 27] = 1f64;
							},
							KomaKind::Blank => (),
						}
					}
				}
			}
		}

		let ms = HashMap::new();
		let mg = HashMap::new();
		let (ms,mg) = match mc {
			&MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
			&MochigomaCollections::Empty => (&ms,&mg),
		};

		let (ms,mg) = match t {
			Teban::Sente => (ms,mg),
			Teban::Gote => (mg,ms),
		};

		for k in &MOCHIGOMA_KINDS {
			match ms.get(&k) {
				Some(c) => {
					let offset = match k {
						&MochigomaKind::Fu => 81 * 28,
						&MochigomaKind::Kyou => 81 * 28 + 18,
						&MochigomaKind::Kei => 81 * 28 + 18 + 4,
						&MochigomaKind::Gin => 81 * 28 + 18 + 8,
						&MochigomaKind::Kin => 81 * 28 + 18 + 12,
						&MochigomaKind::Kaku => 81 * 28 + 18 + 16,
						&MochigomaKind::Hisha => 81 * 28 + 18 + 18,
					};

					let offset = offset as usize;

					for i in 0..(*c as usize) {
						inputs[offset + i] = 1f64;
					}
				},
				None => (),
			}
			match mg.get(&k) {
				Some(c) => {
					let offset = match k {
						&MochigomaKind::Fu => 81 * 28 + 18 + 20,
						&MochigomaKind::Kyou => 81 * 28 + 18 + 20 + 18,
						&MochigomaKind::Kei => 81 * 28 + 18 + 20 + 18 + 4,
						&MochigomaKind::Gin => 81 * 28 + 18 + 20 + 18 + 8,
						&MochigomaKind::Kin => 81 * 28 + 18 + 20 + 18 + 12,
						&MochigomaKind::Kaku => 81 * 28 + 18 + 20 + 18 + 16,
						&MochigomaKind::Hisha => 81 * 28 + 18 + 20 + 18 + 18,
					};

					let offset = offset as usize;

					for i in 0..(*c as usize) {
						inputs[offset + i] = 1f64;
					}
				},
				None => (),
			}
		}
		inputs
	}
}