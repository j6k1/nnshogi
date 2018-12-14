use rand;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Mutex;
use std::i32;
use std::fs;

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
	nna:NN<Adam,Mse>,
	nnb:NN<Adam,Mse>,
	nna_filename:String,
	nnb_filename:String,
	nnsavedir:String,
	learning_mode:bool,
	quited:bool,
}
impl Intelligence {
	pub fn new (savedir:String,nna_filename:String,nnb_filename:String,learning_mode:bool) -> Intelligence {
		let mut rnd = rand::XorShiftRng::new_unseeded();

		let model:NNModel = NNModel::with_unit_initializer(
										NNUnits::new(2344,
											(100,Box::new(FReLU::new())),
											(100,Box::new(FReLU::new())))
											.add((1,Box::new(FSigmoid::new()))),
										BinFileInputReader::new(
											format!("{}/{}",savedir,nna_filename).as_str()).unwrap(),
										move || {
											let i = rnd.next_u32();
											if i % 2 == 0{
												rnd.next_f64()
											} else {
												-rnd.next_f64()
											}
										}).unwrap();
		let nna = NN::new(model,|s| Adam::new(s),Mse::new());

		let mut rnd = rand::XorShiftRng::new_unseeded();

		let model:NNModel = NNModel::with_unit_initializer(
										NNUnits::new(2344,
											(100,Box::new(FReLU::new())),
											(100,Box::new(FReLU::new())))
											.add((1,Box::new(FSigmoid::new()))),
										BinFileInputReader::new(
											format!("{}/{}",savedir,nnb_filename).as_str()).unwrap(),
										move || {
											let i = rnd.next_u32();
											if i % 2 == 0{
												rnd.next_f64()
											} else {
												-rnd.next_f64()
											}
										}).unwrap();
		let nnb = NN::new(model,|s| Adam::new(s),Mse::new());

		Intelligence {
			nna:nna,
			nnb:nnb,
			nna_filename:nna_filename,
			nnb_filename:nnb_filename,
			nnsavedir:savedir,
			learning_mode:learning_mode,
			quited:false,
		}
	}

	pub fn evalute(&mut self,t:Teban,b:&Banmen,mc:&MochigomaCollections)
		-> Result<f64,InvalidStateError> {
		let mut input:Vec<f64> = Vec::new();
		input.extend_from_slice(&self.make_input(t,b,mc));

		let nnaanswera = self.nna.solve(&input)?;
		let nnbanswerb = self.nnb.solve(&input)?;

		let (a,b) = if self.learning_mode {
			let mut rnd = rand::XorShiftRng::new_unseeded();

			let a = rnd.next_f64();
			let b = 1f64 - a;

			(a,b)
		} else {
			(0.5f64,0.5f64)
		};

		let nnaanswera = nnaanswera[0];
		let nnbanswerb = nnbanswerb[0];

		let answer = nnaanswera * a + nnbanswerb * b;

		Ok(answer * i32::MAX as f64)
	}

	pub fn learning<'a>(&mut self,enable_shake_shake:bool,
		teban:Teban,last_teban:Teban,
		history:Vec<(Banmen,MochigomaCollections,u64,u64)>,s:&GameEndState,
		event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
		-> Result<(),CommonError> {

		let mut t = last_teban;

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

			let mut rnd = rand::XorShiftRng::new_unseeded();
			let a = rnd.next_f64();
			let b = 1f64 - a;

			match s {
				&GameEndState::Win if t == teban && enable_shake_shake => {
					self.nna.learn(&input,&(0..1).map(|_| a)
													.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| b)
													.collect::<Vec<f64>>())?;
				},
				&GameEndState::Win if t == teban => {
					self.nna.learn(&input,&(0..1).map(|_| 1f64)
													.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| 1f64)
													.collect::<Vec<f64>>())?;
				},
				&GameEndState::Win => {
					self.nna.learn(&input,&(0..1).map(|_| 0f64)
													.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| 0f64)
													.collect::<Vec<f64>>())?;
				},
				_ if t == teban => {
					self.nna.learn(&input,&(0..1).map(|_| 0f64)
													.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| 0f64)
													.collect::<Vec<f64>>())?;
				},
				_ if enable_shake_shake => {
					self.nna.learn(&input,&(0..1).map(|_| a)
												.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| b)
													.collect::<Vec<f64>>())?;
				},
				_  => {
					self.nna.learn(&input,&(0..1).map(|_| 1f64)
												.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| 1f64)
													.collect::<Vec<f64>>())?;
				}
			}

			t = t.opposite();
		}

		self.save()
	}

	fn save(&mut self) -> Result<(),CommonError>{
		self.nna.save(
			PersistenceWithBinFile::new(
				&format!("{}/{}.tmp",self.nnsavedir,self.nna_filename))?)?;
		self.nnb.save(
			PersistenceWithBinFile::new(
				&format!("{}/{}.tmp",self.nnsavedir,self.nnb_filename))?)?;
		fs::rename(&format!("{}/{}.tmp", self.nnsavedir,self.nna_filename),
						&format!("{}/{}", self.nnsavedir,self.nna_filename))?;
		fs::rename(&format!("{}/{}.tmp", self.nnsavedir,self.nnb_filename),
						&format!("{}/{}", self.nnsavedir,self.nnb_filename))?;
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
						let (x,y) = match t {
							Teban::Sente => (x,y),
							Teban::Gote => (8 - x, 8 - y),
						};
						let kind = kinds[y][x];

						match kind {
							KomaKind::SFu if t == Teban::Sente => {
								inputs[y * 9 + x] = 1f64;
							},
							KomaKind::GFu if t == Teban::Gote => {
								inputs[y * 9 + x] = 1f64;
							},
							KomaKind::SKyou if t == Teban::Sente => {
								inputs[y * 9 + x + 81] = 1f64;
							},
							KomaKind::GKyou if t == Teban::Gote => {
								inputs[y * 9 + x + 81] = 1f64;
							},
							KomaKind::SKei if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 2] = 1f64;
							},
							KomaKind::GKei if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 2] = 1f64;
							},
							KomaKind::SGin if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 3] = 1f64;
							},
							KomaKind::GGin if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 3] = 1f64;
							},
							KomaKind::SKin if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 4] = 1f64;
							},
							KomaKind::GKin if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 4] = 1f64;
							},
							KomaKind::SKaku if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 5] = 1f64;
							},
							KomaKind::GKaku if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 5] = 1f64;
							},
							KomaKind::SHisha if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 6] = 1f64;
							},
							KomaKind::GHisha if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 6] = 1f64;
							},
							KomaKind::SOu if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 7] = 1f64;
							},
							KomaKind::GOu if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 7] = 1f64;
							},
							KomaKind::SFuN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 8] = 1f64;
							},
							KomaKind::GFuN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 8] = 1f64;
							},
							KomaKind::SKyouN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 9] = 1f64;
							},
							KomaKind::GKyouN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 9] = 1f64;
							},
							KomaKind::SKeiN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 10] = 1f64;
							},
							KomaKind::GKeiN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 10] = 1f64;
							},
							KomaKind::SGinN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 11] = 1f64;
							},
							KomaKind::GGinN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 11] = 1f64;
							},
							KomaKind::SKakuN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 12] = 1f64;
							},
							KomaKind::GKakuN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 12] = 1f64;
							},
							KomaKind::SHishaN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 13] = 1f64;
							},
							KomaKind::GHishaN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 13] = 1f64;
							},
							KomaKind::GFu if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 14] = 1f64;
							},
							KomaKind::SFu if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 14] = 1f64;
							},
							KomaKind::GKyou if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 15] = 1f64;
							},
							KomaKind::SKyou if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 15] = 1f64;
							},
							KomaKind::GKei if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 16] = 1f64;
							},
							KomaKind::SKei if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 16] = 1f64;
							},
							KomaKind::GGin if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 17] = 1f64;
							},
							KomaKind::SGin if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 17] = 1f64;
							},
							KomaKind::GKin if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 18] = 1f64;
							},
							KomaKind::SKin if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 18] = 1f64;
							},
							KomaKind::GKaku if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 19] = 1f64;
							},
							KomaKind::SKaku if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 19] = 1f64;
							},
							KomaKind::GHisha if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 20] = 1f64;
							},
							KomaKind::SHisha if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 20] = 1f64;
							},
							KomaKind::GOu if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 21] = 1f64;
							},
							KomaKind::SOu if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 21] = 1f64;
							},
							KomaKind::GFuN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 22] = 1f64;
							},
							KomaKind::SFuN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 22] = 1f64;
							},
							KomaKind::GKyouN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 23] = 1f64;
							},
							KomaKind::SKyouN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 23] = 1f64;
							},
							KomaKind::GKeiN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 24] = 1f64;
							},
							KomaKind::SKeiN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 24] = 1f64;
							},
							KomaKind::GGinN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 25] = 1f64;
							},
							KomaKind::SGinN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 25] = 1f64;
							},
							KomaKind::GKakuN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 26] = 1f64;
							},
							KomaKind::SKakuN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 26] = 1f64;
							},
							KomaKind::GHishaN if t == Teban::Sente => {
								inputs[y * 9 + x + 81 * 27] = 1f64;
							},
							KomaKind::SHishaN if t == Teban::Gote => {
								inputs[y * 9 + x + 81 * 27] = 1f64;
							},
							_ => (),
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