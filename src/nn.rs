use rand;
use rand::Rng;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use rand::distributions::Distribution;
use statrs::distribution::Normal;
use std;
use std::collections::HashMap;
use std::sync::Mutex;
use std::fs;

use simplenn::function::activation::*;
use simplenn::function::optimizer::*;
use simplenn::function::loss::*;
use simplenn::NN;
use simplenn::NNModel;
use simplenn::NNUnits;
use simplenn::SnapShot;
use simplenn::persistence::*;
use simplenn::error::InvalidStateError;

use usiagent::shogi::*;
use usiagent::event::EventQueue;
use usiagent::event::UserEvent;
use usiagent::event::UserEventKind;
use usiagent::error::EventDispatchError;
use usiagent::event::GameEndState;
use usiagent::TryFrom;

use error::*;

pub struct Intelligence {
	nna:NN<SGD,CrossEntropy>,
	nnb:NN<SGD,CrossEntropy>,
	nna_filename:String,
	nnb_filename:String,
	nnsavedir:String,
	enable_shake_shake:bool,
	quited:bool,
}
pub const F64_FRACTION_MAX:u64 = std::u64::MAX >> 12;
impl Intelligence {
	pub fn new (savedir:String,nna_filename:String,nnb_filename:String,enable_shake_shake:bool) -> Intelligence {
		let mut rnd = rand::thread_rng();
		let mut rnd = XorShiftRng::from_seed(rnd.gen());
		let n = Normal::new(0.0, 1.0).unwrap();

		let model:NNModel = NNModel::with_unit_initializer(
										NNUnits::new(2344,
											(100,Box::new(FReLU::new())),
											(100,Box::new(FReLU::new())))
											.add((1,Box::new(FSigmoid::new()))),
										BinFileInputReader::new(
											format!("{}/{}",savedir,nna_filename).as_str()).unwrap(),
										move || {
											n.sample(&mut rnd) * 0.025
										}).unwrap();
		let nna = NN::new(model,|_| SGD::new(0.01),CrossEntropy::new());

		let mut rnd = rand::thread_rng();
		let mut rnd = XorShiftRng::from_seed(rnd.gen());
		let n = Normal::new(0.0, 1.0).unwrap();

		let model:NNModel = NNModel::with_unit_initializer(
										NNUnits::new(2344,
											(100,Box::new(FReLU::new())),
											(100,Box::new(FReLU::new())))
											.add((1,Box::new(FSigmoid::new()))),
										BinFileInputReader::new(
											format!("{}/{}",savedir,nnb_filename).as_str()).unwrap(),
										move || {
											n.sample(&mut rnd) * 0.025
										}).unwrap();
		let nnb = NN::new(model,|_| SGD::new(0.01),CrossEntropy::new());

		Intelligence {
			nna:nna,
			nnb:nnb,
			nna_filename:nna_filename,
			nnb_filename:nnb_filename,
			nnsavedir:savedir,
			enable_shake_shake:enable_shake_shake,
			quited:false,
		}
	}

	pub fn make_snapshot(&self,t:Teban,b:&Banmen,mc:&MochigomaCollections)
		-> Result<(SnapShot,SnapShot),InvalidStateError> {
		let input = self.make_input(t,b,mc);

		let ssa = self.nna.solve_shapshot(&input)?;
		let ssb = self.nnb.solve_shapshot(&input)?;

		Ok((ssa,ssb))
	}

	pub fn evalute(&self,t:Teban,b:&Banmen,mc:&MochigomaCollections)
		-> Result<i64,InvalidStateError> {
		let input = self.make_input(t,b,mc);

		let nnaanswera = self.nna.solve(&input)?;
		let nnbanswerb = self.nnb.solve(&input)?;

		let (a,b) = if self.enable_shake_shake {
			let mut rnd = rand::thread_rng();
			let mut rnd = XorShiftRng::from_seed(rnd.gen());

			let a = rnd.gen();
			let b = 1f64 - a;

			(a,b)
		} else {
			(0.5f64,0.5f64)
		};

		let nnaanswera = nnaanswera[0];
		let nnbanswerb = nnbanswerb[0];

		let answer = nnaanswera * a + nnbanswerb * b - 0.5;

		Ok((answer * F64_FRACTION_MAX as f64) as i64)
	}

	pub fn evalute_by_diff(&self,snapshot:&(SnapShot,SnapShot),is_opposite:bool,t:Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move)
		-> Result<(i64,(SnapShot,SnapShot)),CommonError> {
		let input = self.make_diff_input(is_opposite,t,b,mc,m)?;

		let ssa = self.nna.solve_diff(&input,&snapshot.0)?;
		let ssb = self.nnb.solve_diff(&input,&snapshot.1)?;

		let (a,b) = if self.enable_shake_shake {
			let mut rnd = rand::thread_rng();
			let mut rnd = XorShiftRng::from_seed(rnd.gen());

			let a = rnd.gen();
			let b = 1f64 - a;

			(a,b)
		} else {
			(0.5f64,0.5f64)
		};

		let nnaanswera = ssa.r[0];
		let nnbanswerb = ssb.r[0];

		let answer = nnaanswera * a + nnbanswerb * b - 0.5;

		Ok(((answer * F64_FRACTION_MAX as f64) as i64,(ssa,ssb)))
	}

	pub fn evalute_by_snapshot(&self,snapshot:&(SnapShot,SnapShot)) -> i64 {
		let ssa = &snapshot.0;
		let ssb = &snapshot.1;

		let (a,b) = if self.enable_shake_shake {
			let mut rnd = rand::thread_rng();
			let mut rnd = XorShiftRng::from_seed(rnd.gen());

			let a = rnd.gen();
			let b = 1f64 - a;

			(a,b)
		} else {
			(0.5f64,0.5f64)
		};

		let nnaanswera = ssa.r[0];
		let nnbanswerb = ssb.r[0];

		let answer = nnaanswera * a + nnbanswerb * b - 0.5;

		(answer * F64_FRACTION_MAX as f64) as i64
	}


	pub fn learning_by_training_data<'a,D>(&mut self,
						last_teban:Teban,
						history:Vec<(Banmen,MochigomaCollections,u64,u64)>, s:&GameEndState,
						mut training_data_generator:D,
						event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
						-> Result<(),CommonError> where D: FnMut(&GameEndState,Teban) -> Option<(f64,f64)> {
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

			if let Some((a,b)) = training_data_generator(s,t) {
				self.nna.learn(&input, &(0..1).map(|_| a)
					.collect::<Vec<f64>>())?;
				self.nnb.learn(&input, &(0..1).map(|_| b)
					.collect::<Vec<f64>>())?;
			}

			t = t.opposite();
		}

		self.save()
	}

	/*
	pub fn learning<'a>(&mut self,enable_shake_shake:bool,
		teban:Teban,last_teban:Teban,
		history:Vec<(Banmen,MochigomaCollections,u64,u64)>,s:&GameEndState,
		event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
		-> Result<(),CommonError> {

		const BASE_RATE:f64 = 0.96;

		let mut rate = 1.0f64;

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

			let mut rnd = rand::thread_rng();
			let a:f64 = rnd.gen();
			let b = 1f64 - a;

			match s {
				&GameEndState::Win if t == teban && enable_shake_shake => {
					self.nna.learn(&input,&(0..1).map(|_| a / 2.0f64 + a / 2.0f64 * rate)
													.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| b / 2.0f64 + b / 2.0f64 * rate)
													.collect::<Vec<f64>>())?;
				},
				&GameEndState::Win if t == teban => {
					self.nna.learn(&input,&(0..1).map(|_| 0.5f64 + 0.5f64 * rate)
													.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| 0.5f64 + 0.5f64 * rate)
													.collect::<Vec<f64>>())?;
				},
				&GameEndState::Win => {
					self.nna.learn(&input,&(0..1).map(|_| 0.5f64 - 0.5f64 * rate)
													.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| 0.5f64 - 0.5f64 * rate)
													.collect::<Vec<f64>>())?;
				},
				_ if t == teban => {
					self.nna.learn(&input,&(0..1).map(|_| 0.5f64 - 0.5f64 * rate)
													.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| 0.5f64 - 0.5f64 * rate)
													.collect::<Vec<f64>>())?;
				},
				_ if enable_shake_shake => {
					self.nna.learn(&input,&(0..1).map(|_| a / 2.0f64 + a / 2.0f64 * rate)
												.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| b / 2.0f64 + b / 2.0f64 * rate)
													.collect::<Vec<f64>>())?;
				},
				_  => {
					self.nna.learn(&input,&(0..1).map(|_| 0.5f64 + 0.5f64 * rate)
												.collect::<Vec<f64>>())?;
					self.nnb.learn(&input,&(0..1).map(|_| 0.5f64 + 0.5f64 * rate)
													.collect::<Vec<f64>>())?;
				}
			}

			t = t.opposite();

			rate = rate * BASE_RATE;
		}

		self.save()
	}
	*/
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
						let kind = kinds[y][x];

						if kind != KomaKind::Blank {
							let index = self.input_index_of_banmen(t,kind,x as u32,y as u32).unwrap();

							inputs[index] = 1f64;
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

	pub fn make_diff_input(&self,is_opposite:bool,t:Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move) -> Result<Vec<(usize,f64)>,CommonError> {
		let mut d = Vec::new();

		match m {
			&Move::To(KomaSrcPosition(sx,sy),KomaDstToPosition(dx,dy,n)) => {
				match b {
					&Banmen(ref kinds) => {
						let (sx,sy) = (9-sx,sy-1);
						let (dx,dy) = (9-dx,dy-1);

						let sk = kinds[sy as usize][sx as usize];

						d.push((self.input_index_of_banmen(t,sk,sx,sy)?,-1f64));

						let dk = kinds[dy as usize][dx as usize];

						if dk != KomaKind::Blank {
							d.push((self.input_index_of_banmen(t,dk,dx,dy)?,-1f64));
						}

						if dk != KomaKind::Blank && dk != KomaKind::SOu && dk != KomaKind::GOu {
							d.push((self.input_index_with_of_mochigoma_get(is_opposite,t,MochigomaKind::try_from(dk)?,mc)?,1f64));
						}

						if n {
							d.push((self.input_index_of_banmen(t,sk.to_nari(),dx,dy)?,1f64));
						} else {
							d.push((self.input_index_of_banmen(t,sk,dx,dy)?,1f64));
						}
					}
				}
			},
			&Move::Put(kind,KomaDstPutPosition(dx,dy))  => {
				let (dx,dy) = (9-dx,dy-1);
				d.push((self.input_index_with_of_mochigoma_put(is_opposite,t,kind,mc)?,-1f64));
				d.push((self.input_index_of_banmen(t,KomaKind::from((t,kind)),dx,dy)?,1f64));
			}
		}

		Ok(d)
	}

	#[inline]
	fn input_index_of_banmen(&self,teban:Teban,kind:KomaKind,x:u32,y:u32) -> Result<usize,CommonError> {
		let index = match teban {
			Teban::Sente => {
				match kind {
					KomaKind::SFu => {
						y * 9 + x
					},
					KomaKind::SKyou => {
						y * 9 + x + 81
					},
					KomaKind::SKei => {
						y * 9 + x + 81 * 2
					},
					KomaKind::SGin => {
						y * 9 + x + 81 * 3
					},
					KomaKind::SKin => {
						y * 9 + x + 81 * 4
					},
					KomaKind::SKaku => {
						y * 9 + x + 81 * 5
					},
					KomaKind::SHisha => {
						y * 9 + x + 81 * 6
					},
					KomaKind::SOu => {
						y * 9 + x + 81 * 7
					},
					KomaKind::SFuN => {
						y * 9 + x + 81 * 8
					},
					KomaKind::SKyouN => {
						y * 9 + x + 81 * 9
					},
					KomaKind::SKeiN => {
						y * 9 + x + 81 * 10
					},
					KomaKind::SGinN => {
						y * 9 + x + 81 * 11
					},
					KomaKind::SKakuN => {
						y * 9 + x + 81 * 12
					},
					KomaKind::SHishaN => {
						y * 9 + x + 81 * 13
					},
					KomaKind::GFu => {
						y * 9 + x + 81 * 14
					},
					KomaKind::GKyou => {
						y * 9 + x + 81 * 15
					},
					KomaKind::GKei => {
						y * 9 + x + 81 * 16
					},
					KomaKind::GGin => {
						y * 9 + x + 81 * 17
					},
					KomaKind::GKin => {
						y * 9 + x + 81 * 18
					},
					KomaKind::GKaku => {
						y * 9 + x + 81 * 19
					},
					KomaKind::GHisha => {
						y * 9 + x + 81 * 20
					},
					KomaKind::GOu => {
						y * 9 + x + 81 * 21
					},
					KomaKind::GFuN => {
						y * 9 + x + 81 * 22
					},
					KomaKind::GKyouN => {
						y * 9 + x + 81 * 23
					},
					KomaKind::GKeiN => {
						y * 9 + x + 81 * 24
					},
					KomaKind::GGinN => {
						y * 9 + x + 81 * 25
					},
					KomaKind::GKakuN => {
						y * 9 + x + 81 * 26
					},
					KomaKind::GHishaN => {
						y * 9 + x + 81 * 27
					},
					_ => {
						return Err(CommonError::Fail(
									String::from(
										"Calculation of index of difference input data of neural network failed. (KomaKind is 'Blank')"
								)));
					},
				}
			},
			Teban::Gote => {
				let (x,y) = (8-x,8-y);

				match kind {
					KomaKind::GFu => {
						y * 9 + x
					},
					KomaKind::GKyou => {
						y * 9 + x + 81
					},
					KomaKind::GKei => {
						y * 9 + x + 81 * 2
					},
					KomaKind::GGin => {
						y * 9 + x + 81 * 3
					},
					KomaKind::GKin => {
						y * 9 + x + 81 * 4
					},
					KomaKind::GKaku => {
						y * 9 + x + 81 * 5
					},
					KomaKind::GHisha => {
						y * 9 + x + 81 * 6
					},
					KomaKind::GOu => {
						y * 9 + x + 81 * 7
					},
					KomaKind::GFuN => {
						y * 9 + x + 81 * 8
					},
					KomaKind::GKyouN => {
						y * 9 + x + 81 * 9
					},
					KomaKind::GKeiN => {
						y * 9 + x + 81 * 10
					},
					KomaKind::GGinN => {
						y * 9 + x + 81 * 11
					},
					KomaKind::GKakuN => {
						y * 9 + x + 81 * 12
					},
					KomaKind::GHishaN => {
						y * 9 + x + 81 * 13
					},
					KomaKind::SFu => {
						y * 9 + x + 81 * 14
					},
					KomaKind::SKyou => {
						y * 9 + x + 81 * 15
					},
					KomaKind::SKei => {
						y * 9 + x + 81 * 16
					},
					KomaKind::SGin => {
						y * 9 + x + 81 * 17
					},
					KomaKind::SKin => {
						y * 9 + x + 81 * 18
					},
					KomaKind::SKaku => {
						y * 9 + x + 81 * 19
					},
					KomaKind::SHisha => {
						y * 9 + x + 81 * 20
					},
					KomaKind::SOu => {
						y * 9 + x + 81 * 21
					},
					KomaKind::SFuN => {
						y * 9 + x + 81 * 22
					},
					KomaKind::SKyouN => {
						y * 9 + x + 81 * 23
					},
					KomaKind::SKeiN => {
						y * 9 + x + 81 * 24
					},
					KomaKind::SGinN => {
						y * 9 + x + 81 * 25
					},
					KomaKind::SKakuN => {
						y * 9 + x + 81 * 26
					},
					KomaKind::SHishaN => {
						y * 9 + x + 81 * 27
					},
					_ => {
						return Err(CommonError::Fail(
									String::from(
										"Calculation of index of difference input data of neural network failed. (KomaKind is 'Blank')"
								)));
					}
				}
			}
		};

		Ok(index as usize)
	}

	#[inline]
	fn input_index_with_of_mochigoma_get(&self,is_opposite:bool,teban:Teban,kind:MochigomaKind,mc:&MochigomaCollections) -> Result<usize,CommonError> {
		let ms = HashMap::new();
		let mg = HashMap::new();

		let (ms,mg) = match mc {
			&MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
			&MochigomaCollections::Empty => (&ms,&mg),
		};

		let mc = match teban {
			Teban::Sente if is_opposite => mg,
			Teban::Sente => ms,
			Teban::Gote if is_opposite => ms,
			Teban::Gote => mg,
		};

		let offset = if is_opposite {
			match kind {
				MochigomaKind::Fu => 81 * 28 + 18 + 20,
				MochigomaKind::Kyou => 81 * 28 + 18 + 20 + 18,
				MochigomaKind::Kei => 81 * 28 + 18 + 20 + 18 + 4,
				MochigomaKind::Gin => 81 * 28 + 18 + 20 + 18 + 8,
				MochigomaKind::Kin => 81 * 28 + 18 + 20 + 18 + 12,
				MochigomaKind::Kaku => 81 * 28 + 18 + 20 + 18 + 16,
				MochigomaKind::Hisha => 81 * 28 + 18 + 20 + 18 + 18,
			}
		} else {
			match kind {
				MochigomaKind::Fu => 81 * 28,
				MochigomaKind::Kyou => 81 * 28 + 18,
				MochigomaKind::Kei => 81 * 28 + 18 + 4,
				MochigomaKind::Gin => 81 * 28 + 18 + 8,
				MochigomaKind::Kin => 81 * 28 + 18 + 12,
				MochigomaKind::Kaku => 81 * 28 + 18 + 16,
				MochigomaKind::Hisha => 81 * 28 + 18 + 18,
			}
		};

		match mc.get(&kind) {
			Some(c) => {
				let offset = offset as usize;

				Ok(offset + *c as usize)
			},
			_ => {
				Ok(offset as usize)
			}
		}
	}

	#[inline]
	fn input_index_with_of_mochigoma_put(&self,is_opposite:bool,teban:Teban,kind:MochigomaKind,mc:&MochigomaCollections) -> Result<usize,CommonError> {
		let ms = HashMap::new();
		let mg = HashMap::new();

		let (ms,mg) = match mc {
			&MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
			&MochigomaCollections::Empty => (&ms,&mg),
		};

		let mc = match teban {
			Teban::Sente if is_opposite => mg,
			Teban::Sente => ms,
			Teban::Gote if is_opposite => ms,
			Teban::Gote => mg,
		};

		let offset = if is_opposite {
			match kind {
				MochigomaKind::Fu => 81 * 28 + 18 + 20,
				MochigomaKind::Kyou => 81 * 28 + 18 + 20 + 18,
				MochigomaKind::Kei => 81 * 28 + 18 + 20 + 18 + 4,
				MochigomaKind::Gin => 81 * 28 + 18 + 20 + 18 + 8,
				MochigomaKind::Kin => 81 * 28 + 18 + 20 + 18 + 12,
				MochigomaKind::Kaku => 81 * 28 + 18 + 20 + 18 + 16,
				MochigomaKind::Hisha => 81 * 28 + 18 + 20 + 18 + 18,
			}
		} else {
			match kind {
				MochigomaKind::Fu => 81 * 28,
				MochigomaKind::Kyou => 81 * 28 + 18,
				MochigomaKind::Kei => 81 * 28 + 18 + 4,
				MochigomaKind::Gin => 81 * 28 + 18 + 8,
				MochigomaKind::Kin => 81 * 28 + 18 + 12,
				MochigomaKind::Kaku => 81 * 28 + 18 + 16,
				MochigomaKind::Hisha => 81 * 28 + 18 + 18,
			}
		};

		match mc.get(&kind) {
			Some(c) if *c > 0 => {
				let offset = offset as usize;

				Ok(offset + (*c as usize - 1))
			},
			_ => {
				Err(CommonError::Fail(
					String::from(
						"Calculation of index of difference input data of neural network failed. (The number of holding pieces is 0)"
				)))
			}
		}
	}
}