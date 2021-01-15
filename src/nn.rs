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
use simplenn::Metrics;
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
	bias_shake_shake:bool,
	quited:bool,
}
pub const F64_FRACTION_MAX:u64 = std::u64::MAX >> 12;

const BANMEN_SIZE:usize = 81;
const KOMA_COUNT:usize = 40;

const SELF_TEBAN_INDEX:usize = 0;
const OPPONENT_TEBAN_INDEX:usize = SELF_TEBAN_INDEX + (KOMA_COUNT - 1);

const OU_INDEX:usize = OPPONENT_TEBAN_INDEX + (KOMA_COUNT - 1);
const FU_INDEX:usize = OU_INDEX + BANMEN_SIZE * (KOMA_COUNT - 1);
const KYOU_INDEX:usize = FU_INDEX + BANMEN_SIZE;
const KEI_INDEX:usize = KYOU_INDEX + BANMEN_SIZE;
const GIN_INDEX:usize = KEI_INDEX + BANMEN_SIZE;
const KIN_INDEX:usize = GIN_INDEX + BANMEN_SIZE;
const KAKU_INDEX:usize = KIN_INDEX + BANMEN_SIZE;
const HISHA_INDEX:usize = KAKU_INDEX + BANMEN_SIZE;
const NARIFU_INDEX:usize = HISHA_INDEX + BANMEN_SIZE;
const NARIKYOU_INDEX:usize = NARIFU_INDEX + BANMEN_SIZE;
const NARIKEI_INDEX:usize = NARIKYOU_INDEX + BANMEN_SIZE;
const NARIGIN_INDEX:usize = NARIKEI_INDEX + BANMEN_SIZE;
const NARIKAKU_INDEX:usize = NARIGIN_INDEX + BANMEN_SIZE;
const NARIHISHA_INDEX:usize = NARIKAKU_INDEX + BANMEN_SIZE;
const OPPONENT_FU_INDEX:usize = NARIHISHA_INDEX + BANMEN_SIZE;
const OPPONENT_KYOU_INDEX:usize = OPPONENT_FU_INDEX + BANMEN_SIZE;
const OPPONENT_KEI_INDEX:usize = OPPONENT_KYOU_INDEX + BANMEN_SIZE;
const OPPONENT_GIN_INDEX:usize = OPPONENT_KEI_INDEX + BANMEN_SIZE;
const OPPONENT_KIN_INDEX:usize = OPPONENT_GIN_INDEX + BANMEN_SIZE;
const OPPONENT_KAKU_INDEX:usize = OPPONENT_KIN_INDEX + BANMEN_SIZE;
const OPPONENT_HISHA_INDEX:usize = OPPONENT_KAKU_INDEX + BANMEN_SIZE;
const OPPONENT_OU_INDEX:usize = OPPONENT_HISHA_INDEX + BANMEN_SIZE;
const OPPONENT_NARIFU_INDEX:usize = OPPONENT_OU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKYOU_INDEX:usize = OPPONENT_NARIFU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKEI_INDEX:usize = OPPONENT_NARIKYOU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIGIN_INDEX:usize = OPPONENT_NARIKEI_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKAKU_INDEX:usize = OPPONENT_NARIGIN_INDEX + BANMEN_SIZE;
const OPPONENT_NARIHISHA_INDEX:usize = OPPONENT_NARIKAKU_INDEX + BANMEN_SIZE;

const MOCHIGOMA_FU_INDEX:usize = OPPONENT_NARIHISHA_INDEX + BANMEN_SIZE;
const MOCHIGOMA_KYOU_INDEX:usize = MOCHIGOMA_FU_INDEX + 19;
const MOCHIGOMA_KEI_INDEX:usize = MOCHIGOMA_KYOU_INDEX + 5;
const MOCHIGOMA_GIN_INDEX:usize = MOCHIGOMA_KEI_INDEX + 5;
const MOCHIGOMA_KIN_INDEX:usize = MOCHIGOMA_GIN_INDEX + 5;
const MOCHIGOMA_KAKU_INDEX:usize = MOCHIGOMA_KIN_INDEX + 5;
const MOCHIGOMA_HISHA_INDEX:usize = MOCHIGOMA_KAKU_INDEX + 3;
const OPPONENT_MOCHIGOMA_FU_INDEX:usize = MOCHIGOMA_HISHA_INDEX + 3;
const OPPONENT_MOCHIGOMA_KYOU_INDEX:usize = OPPONENT_MOCHIGOMA_FU_INDEX + 19;
const OPPONENT_MOCHIGOMA_KEI_INDEX:usize = OPPONENT_MOCHIGOMA_KYOU_INDEX + 5;
const OPPONENT_MOCHIGOMA_GIN_INDEX:usize = OPPONENT_MOCHIGOMA_KEI_INDEX + 5;
const OPPONENT_MOCHIGOMA_KIN_INDEX:usize = OPPONENT_MOCHIGOMA_GIN_INDEX + 5;
const OPPONENT_MOCHIGOMA_KAKU_INDEX:usize = OPPONENT_MOCHIGOMA_KIN_INDEX + 5;
const OPPONENT_MOCHIGOMA_HISHA_INDEX:usize = OPPONENT_MOCHIGOMA_KAKU_INDEX + 3;

const SELF_INDEX_MAP:[usize; 7] = [
	MOCHIGOMA_FU_INDEX,
	MOCHIGOMA_KYOU_INDEX,
	MOCHIGOMA_KEI_INDEX,
	MOCHIGOMA_GIN_INDEX,
	MOCHIGOMA_KIN_INDEX,
	MOCHIGOMA_KAKU_INDEX,
	MOCHIGOMA_HISHA_INDEX
];

const OPPONENT_INDEX_MAP:[usize; 7] = [
	OPPONENT_MOCHIGOMA_FU_INDEX,
	OPPONENT_MOCHIGOMA_KYOU_INDEX,
	OPPONENT_MOCHIGOMA_KEI_INDEX,
	OPPONENT_MOCHIGOMA_GIN_INDEX,
	OPPONENT_MOCHIGOMA_KIN_INDEX,
	OPPONENT_MOCHIGOMA_KAKU_INDEX,
	OPPONENT_MOCHIGOMA_HISHA_INDEX
];

impl Intelligence {
	pub fn new (savedir:String,nna_filename:String,nnb_filename:String,enable_shake_shake:bool) -> Intelligence {
		let mut rnd = rand::thread_rng();
		let mut rnd = XorShiftRng::from_seed(rnd.gen());
		let n = Normal::new(0.0, 1.0).unwrap();

		let model:NNModel = NNModel::with_unit_initializer(
										NNUnits::new(5514,
											(256,Box::new(FReLU::new())),
											(32,Box::new(FReLU::new())))
											.add((32,Box::new(FReLU::new())))
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
										NNUnits::new(5514,
												 (256,Box::new(FReLU::new())),
												 (32,Box::new(FReLU::new())))
											.add((32,Box::new(FReLU::new())))
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
			bias_shake_shake:enable_shake_shake,
			quited:false,
		}
	}

	pub fn make_snapshot(&self,is_self:bool,t:Teban,b:&Banmen,mc:&MochigomaCollections)
		-> Result<(SnapShot,SnapShot),InvalidStateError> {
		let input = Intelligence::make_input(is_self,t,b,mc);

		let ssa = self.nna.solve_shapshot(&input)?;
		let ssb = self.nnb.solve_shapshot(&input)?;

		Ok((ssa,ssb))
	}

	pub fn evalute(&self,is_self:bool,t:Teban,b:&Banmen,mc:&MochigomaCollections)
		-> Result<i64,InvalidStateError> {
		let input = Intelligence::make_input(is_self,t,b,mc);

		let nnaanswera = self.nna.solve(&input)?;
		let nnbanswerb = self.nnb.solve(&input)?;

		let (a,b) = if self.bias_shake_shake {
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

	pub fn evalute_by_diff(&self, snapshot:&(SnapShot,SnapShot), is_self:bool, t:Teban, b:&Banmen, mc:&MochigomaCollections, m:&Move)
						   -> Result<(i64,(SnapShot,SnapShot)),CommonError> {
		let input = Intelligence::make_diff_input(is_self, t, b, mc, m)?;

		let ssa = self.nna.solve_diff(&input,&snapshot.0)?;
		let ssb = self.nnb.solve_diff(&input,&snapshot.1)?;

		let (a,b) = if self.bias_shake_shake {
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

		let (a,b) = if self.bias_shake_shake {
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
						history:Vec<(Banmen,MochigomaCollections,u64,u64)>,
						s:&GameEndState,
						training_data_generator:&D,
						a:f64,b:f64,
						_:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
						-> Result<(Metrics,Metrics,Metrics,Metrics),CommonError>
		where D: Fn(&GameEndState,Teban,f64) -> f64 {

		let mut teban = last_teban;

		let msa = self.nna.learn_batch(history.iter().rev().map(move |(banmen,mc,_,_)| {
			let input = Intelligence::make_input(teban == last_teban,teban, banmen, mc);

			let t = training_data_generator(s,teban,a);

			teban = teban.opposite();

			(input.to_vec(),(0..1).map(|_| t).collect())
		}))?;

		let mut teban = last_teban.opposite();

		let moa = self.nna.learn_batch(history.iter().rev().map(move |(banmen,mc,_,_)| {
			let input = Intelligence::make_input(teban == last_teban,teban, banmen, mc);

			let t = training_data_generator(s,teban,a);

			teban = teban.opposite();

			(input.to_vec(),(0..1).map(|_| t).collect())
		}))?;

		let mut teban = last_teban;

		let msb = self.nnb.learn_batch(history.iter().rev().map(move |(banmen,mc,_,_)| {
			let input = Intelligence::make_input(teban == last_teban,teban, banmen, mc);

			let t = training_data_generator(s,teban,b);

			teban = teban.opposite();

			(input.to_vec(),(0..1).map(|_| t).collect())
		}))?;

		let mut teban = last_teban.opposite();

		let mob = self.nnb.learn_batch(history.iter().rev().map(move |(banmen,mc,_,_)| {
			let input = Intelligence::make_input(teban == last_teban,teban, banmen, mc);

			let t = training_data_generator(s,teban,b);

			teban = teban.opposite();

			(input.to_vec(),(0..1).map(|_| t).collect())
		}))?;

		self.save()?;

		Ok((msa,moa,msb,mob))
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

	#[allow(dead_code)]
	fn handle_events<'a>(&mut self,event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
		-> Result<(), EventDispatchError<'a,EventQueue<UserEvent,UserEventKind>,UserEvent,CommonError>>
		{
		self.dispatch_events(event_queue)?;

		Ok(())
	}

	#[allow(dead_code)]
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

	pub fn make_input(is_self:bool,t:Teban,b:&Banmen,mc:&MochigomaCollections) -> [f64; 5514] {
		let mut inputs:[f64; 5514] = [0f64; 5514];

		let index = if is_self {
			SELF_TEBAN_INDEX
		} else {
			OPPONENT_TEBAN_INDEX
		};

		for i in 0..(KOMA_COUNT-1) {
			inputs[index + i] = 1f64;
		}

		match b {
			&Banmen(ref kinds) => {
				for y in 0..9 {
					for x in 0..9 {
						let kind = kinds[y][x];

						if t == Teban::Sente && kind == KomaKind::SOu || t == Teban::Gote && kind == KomaKind::GOu {
							let index = Intelligence::input_index_of_banmen(t,kind,x as u32,y as u32).unwrap();

							for i in 0..(KOMA_COUNT-1) {
								inputs[index + i] = 1f64;
							}
						} else if kind != KomaKind::Blank {
							let index = Intelligence::input_index_of_banmen(t,kind,x as u32,y as u32).unwrap();

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

		for &k in &MOCHIGOMA_KINDS {
			match ms.get(&k).unwrap_or(&0) {
				&c => {
					let offset = SELF_INDEX_MAP[k as usize];

					let offset = offset as usize;

					inputs[offset + c as usize] = 1f64;
				}
			}
			match mg.get(&k).unwrap_or(&0) {
				&c => {
					let offset = OPPONENT_INDEX_MAP[k as usize];

					let offset = offset as usize;

					inputs[offset + c as usize] = 1f64;
				}
			}
		}
		inputs
	}

	pub fn make_diff_input(is_self:bool, t:Teban, b:&Banmen, mc:&MochigomaCollections, m:&Move) -> Result<Vec<(usize, f64)>,CommonError> {
		let mut d = Vec::new();

		let (addi,subi) = if is_self {
			(SELF_TEBAN_INDEX,OPPONENT_TEBAN_INDEX)
		} else {
			(OPPONENT_TEBAN_INDEX,SELF_TEBAN_INDEX)
		};

		for i in 0..(KOMA_COUNT-1) {
			d.push((subi + i, -1f64));
			d.push((addi + i,1f64));
		}

		match m {
			&Move::To(KomaSrcPosition(sx,sy),KomaDstToPosition(dx,dy,n)) => {
				match b {
					&Banmen(ref kinds) => {
						let (sx,sy) = (9-sx,sy-1);
						let (dx,dy) = (9-dx,dy-1);

						let sk = kinds[sy as usize][sx as usize];

						if t == Teban::Sente && sk == KomaKind::SOu || t == Teban::Gote && sk == KomaKind::GOu {
							let si = Intelligence::input_index_of_banmen(t, sk, sx, sy)?;
							let di = Intelligence::input_index_of_banmen(t,sk,dx,dy)?;

							for i in 0..(KOMA_COUNT-1) {
								d.push((si + i, -1f64));
								d.push((di + i,1f64));
							}
						} else {
							d.push((Intelligence::input_index_of_banmen(t, sk, sx, sy)?, -1f64));

							if n {
								d.push((Intelligence::input_index_of_banmen(t,sk.to_nari(),dx,dy)?,1f64));
							} else {
								d.push((Intelligence::input_index_of_banmen(t,sk,dx,dy)?,1f64));
							}
						}

						let dk = kinds[dy as usize][dx as usize];

						if dk != KomaKind::Blank {
							d.push((Intelligence::input_index_of_banmen(t,dk,dx,dy)?,-1f64));
						}

						if dk != KomaKind::Blank && dk != KomaKind::SOu && dk != KomaKind::GOu {
							let offset = Intelligence::input_index_with_of_mochigoma_get(is_self, t, MochigomaKind::try_from(dk)?, mc)?;

							d.push((offset, 1f64));
						}
					}
				}
			},
			&Move::Put(kind,KomaDstPutPosition(dx,dy))  => {
				let (dx,dy) = (9-dx,dy-1);
				let offset = Intelligence::input_index_with_of_mochigoma_get(is_self, t, kind, mc)?;

				if offset < 1 {
					return Err(CommonError::Fail(
						String::from(
							"Calculation of index of difference input data of neural network failed. (The number of holding pieces is 0)"
						)))
				} else {
					d.push((offset, -1f64));
					d.push((offset - 1, 1f64));

					d.push((Intelligence::input_index_of_banmen(t, KomaKind::from((t, kind)), dx, dy)?, 1f64));
				}
			}
		}

		Ok(d)
	}

	#[inline]
	fn input_index_of_banmen(teban:Teban,kind:KomaKind,x:u32,y:u32) -> Result<usize,CommonError> {
		const SENTE_INDEX_MAP:[usize; 28] = [
			FU_INDEX,
			KYOU_INDEX,
			KEI_INDEX,
			GIN_INDEX,
			KIN_INDEX,
			KAKU_INDEX,
			HISHA_INDEX,
			OU_INDEX,
			NARIFU_INDEX,
			NARIKYOU_INDEX,
			NARIKEI_INDEX,
			NARIGIN_INDEX,
			NARIKAKU_INDEX,
			NARIHISHA_INDEX,
			OPPONENT_FU_INDEX,
			OPPONENT_KYOU_INDEX,
			OPPONENT_KEI_INDEX,
			OPPONENT_GIN_INDEX,
			OPPONENT_KIN_INDEX,
			OPPONENT_KAKU_INDEX,
			OPPONENT_HISHA_INDEX,
			OPPONENT_OU_INDEX,
			OPPONENT_NARIFU_INDEX,
			OPPONENT_NARIKYOU_INDEX,
			OPPONENT_NARIKEI_INDEX,
			OPPONENT_NARIGIN_INDEX,
			OPPONENT_NARIKAKU_INDEX,
			OPPONENT_NARIHISHA_INDEX
		];

		const GOTE_INDEX_MAP:[usize; 28] = [
			OPPONENT_FU_INDEX,
			OPPONENT_KYOU_INDEX,
			OPPONENT_KEI_INDEX,
			OPPONENT_GIN_INDEX,
			OPPONENT_KIN_INDEX,
			OPPONENT_KAKU_INDEX,
			OPPONENT_HISHA_INDEX,
			OPPONENT_OU_INDEX,
			OPPONENT_NARIFU_INDEX,
			OPPONENT_NARIKYOU_INDEX,
			OPPONENT_NARIKEI_INDEX,
			OPPONENT_NARIGIN_INDEX,
			OPPONENT_NARIKAKU_INDEX,
			OPPONENT_NARIHISHA_INDEX,
			FU_INDEX,
			KYOU_INDEX,
			KEI_INDEX,
			GIN_INDEX,
			KIN_INDEX,
			KAKU_INDEX,
			HISHA_INDEX,
			OU_INDEX,
			NARIFU_INDEX,
			NARIKYOU_INDEX,
			NARIKEI_INDEX,
			NARIGIN_INDEX,
			NARIKAKU_INDEX,
			NARIHISHA_INDEX
		];

		let index = match teban {
			Teban::Sente | Teban::Gote if kind == KomaKind::Blank => {
				return Err(CommonError::Fail(
					String::from(
						"Calculation of index of difference input data of neural network failed. (KomaKind is 'Blank')"
					)));
			},
			Teban::Sente => {
				SENTE_INDEX_MAP[kind as usize] + y as usize * 9 + x as usize
			},
			Teban::Gote => {
				let (x,y) = (8-x,8-y);

				GOTE_INDEX_MAP[kind as usize] + y as usize * 9 + x as usize
			}
		};

		Ok(index as usize)
	}

	#[inline]
	fn input_index_with_of_mochigoma_get(is_self:bool, teban:Teban, kind:MochigomaKind, mc:&MochigomaCollections)
										 -> Result<usize,CommonError> {

		let ms = HashMap::new();
		let mg = HashMap::new();

		let (ms,mg) = match mc {
			&MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
			&MochigomaCollections::Empty => (&ms,&mg),
		};

		let mc = match teban {
			Teban::Sente if is_self => ms,
			Teban::Sente => mg,
			Teban::Gote if is_self => mg,
			Teban::Gote => ms,
		};

		let offset = if is_self {
			SELF_INDEX_MAP[kind as usize]
		} else {
			OPPONENT_INDEX_MAP[kind as usize]
		};

		match mc.get(&kind).unwrap_or(&0) {
			&c => {
				let offset = offset as usize;

				Ok(offset + c as usize)
			}
		}
	}
}