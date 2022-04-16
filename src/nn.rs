use std;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::fs;
use std::ops::DerefMut;
use std::rc::Rc;
use nncombinator::activation::{ReLu, Sigmoid};
use nncombinator::arr::{Arr, DiffArr};
use nncombinator::device::DeviceCpu;
use nncombinator::layer::{ActivationLayer, AddLayer, AddLayerTrain, AskDiffInput, BatchTrain, DiffInput, DiffLinearLayer, ForwardAll, ForwardDiff, InputLayer, LinearLayer, LinearOutputLayer};
use nncombinator::{Cons, Stack};
use nncombinator::persistence::TextFilePersistence;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;

use simplenn::function::activation::*;
use simplenn::function::optimizer::*;
use simplenn::function::loss::*;
use simplenn::{Metrics, Quantization, UnitsConverter};
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
use packedsfen::yaneuraou;
use packedsfen::hcpe;
use packedsfen::yaneuraou::reader::PackedSfenReader;
use packedsfen::traits::Reader;
use simplenn::types::{FxS16, One};
use packedsfen::hcpe::reader::HcpeReader;
use packedsfen::hcpe::haffman_code::GameResult;

pub struct Intelligence<NN> where NN: ForwardDiff<f32> {
	nna:NN,
	nnb:NN,
	nna_filename:String,
	nnb_filename:String,
	nnsavedir:String,
	packed_sfen_reader:PackedSfenReader,
	hcpe_reader:HcpeReader,
	bias_shake_shake:bool,
	quited:bool,
}

const BANMEN_SIZE:usize = 81;
const KOMA_COUNT:usize = 40;

const SELF_TEBAN_INDEX:usize = 0;
const OPPONENT_TEBAN_INDEX:usize = SELF_TEBAN_INDEX + 1;

const OU_INDEX:usize = OPPONENT_TEBAN_INDEX + 1;
const FU_INDEX:usize = OU_INDEX + BANMEN_SIZE;
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
pub struct IntelligenceCreator;
impl IntelligenceCreator {
	pub fn new(savedir:String,nna_filename:String,nnb_filename:String,enable_shake_shake:bool)
		-> Trainer<impl ForwardDiff<f32> + AskDiffInput<f32>> {

		let mut rnd = prelude::thread_rng();
		let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

		let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
		let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
		let n3 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

		let device = DeviceCpu::new();

		let net:InputLayer<f32,DiffInput<DiffArr<f32,2517>,f32,2517,256>,_> = InputLayer::new();

		let rnd = rnd_base.clone();

		let mut nna = net.add_layer(|l| {
			let rnd = rnd.clone();
			DiffLinearLayer::<_,_,_,_,2517,256>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,256,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,Sigmoid::new(&device),&device)
		}).add_layer_train(|l| {
			LinearOutputLayer::new(l,&device)
		});

		let mut rnd = prelude::thread_rng();
		let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

		let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
		let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
		let n3 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

		let device = DeviceCpu::new();

		let net:InputLayer<f32,DiffInput<DiffArr<f32,2517>,f32,2517,256>,_> = InputLayer::new();

		let rnd = rnd_base.clone();

		let mut nnb = net.add_layer(|l| {
			let rnd = rnd.clone();
			DiffLinearLayer::<_,_,_,_,2517,256>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,256,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,Sigmoid::new(&device),&device)
		}).add_layer_train(|l| {
			LinearOutputLayer::new(l,&device)
		});

		Trainer {
			nna:nna,
			nnb:nnb,
			nna_filename:nna_filename,
			nnb_filename:nnb_filename,
			nnsavedir:savedir,
			packed_sfen_reader:PackedSfenReader::new(),
			hcpe_reader:HcpeReader::new(),
			bias_shake_shake:enable_shake_shake,
			quited:false,
		}
	}
}
impl<NN> Intelligence<NN> where NN: ForwardDiff<f32> + AskDiffInput<f32> {
	pub fn make_input(is_self:bool,t:Teban,b:&Banmen,mc:&MochigomaCollections) -> Arr<f32,2517> {
		let mut inputs = Arr::new();

		let index = if is_self {
			SELF_TEBAN_INDEX
		} else {
			OPPONENT_TEBAN_INDEX
		};

		inputs[index] = 1f32;

		match b {
			&Banmen(ref kinds) => {
				for y in 0..9 {
					for x in 0..9 {
						let kind = kinds[y][x];

						if kind != KomaKind::Blank {
							let index = Intelligence::input_index_of_banmen(t,kind,x as u32,y as u32).unwrap();

							inputs[index] = 1f32;
						}
					}
				}
			}
		}

		let ms = Mochigoma::new();
		let mg = Mochigoma::new();
		let (ms,mg) = match mc {
			&MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
			&MochigomaCollections::Empty => (&ms,&mg),
		};

		let (ms,mg) = match t {
			Teban::Sente => (ms,mg),
			Teban::Gote => (mg,ms),
		};

		for &k in &MOCHIGOMA_KINDS {
			let c = ms.get(k);

			for i in 0..c {
				let offset = SELF_INDEX_MAP[k as usize];

				let offset = offset as usize;

				inputs[offset + i as usize] = 1f32;
			}

			let c = mg.get(k);

			for i in 0..c {
				let offset = OPPONENT_INDEX_MAP[k as usize];

				let offset = offset as usize;

				inputs[offset + i as usize] = 1f32;
			}
		}
		inputs
	}

	pub fn make_diff_input(is_self:bool, t:Teban, b:&Banmen, mc:&MochigomaCollections, m:&Move) -> Result<Vec<(usize, f32)>,CommonError> {
		let mut d = Vec::new();

		let (addi,subi) = if is_self {
			(SELF_TEBAN_INDEX,OPPONENT_TEBAN_INDEX)
		} else {
			(OPPONENT_TEBAN_INDEX,SELF_TEBAN_INDEX)
		};

		d.push((subi + i,1.));
		d.push((addi + i,1.));

		match m {
			&Move::To(KomaSrcPosition(sx,sy),KomaDstToPosition(dx,dy,n)) => {
				match b {
					&Banmen(ref kinds) => {
						let (sx,sy) = (9-sx,sy-1);
						let (dx,dy) = (9-dx,dy-1);

						let sk = kinds[sy as usize][sx as usize];

						d.push((Intelligence::input_index_of_banmen(t, sk, sx, sy)?, -1.));

						if n {
							d.push((Intelligence::input_index_of_banmen(t,sk.to_nari(),dx,dy)?,1.));
						} else {
							d.push((Intelligence::input_index_of_banmen(t,sk,dx,dy)?,1.));
						}

						let dk = kinds[dy as usize][dx as usize];

						if dk != KomaKind::Blank {
							d.push((Intelligence::input_index_of_banmen(t,dk,dx,dy)?,-1.));
						}

						if dk != KomaKind::Blank && dk != KomaKind::SOu && dk != KomaKind::GOu {
							let offset = Intelligence::input_index_with_of_mochigoma_get(is_self, t, MochigomaKind::try_from(dk)?, mc)?;

							d.push((offset+1, 1.));
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
					d.push((offset, -1.));

					d.push((Intelligence::input_index_of_banmen(t, KomaKind::from((t, kind)), dx, dy)?, 1.));
				}
			}
		}

		Ok(d)
	}

	pub fn make_snapshot(&self,is_self:bool,t:Teban,b:&Banmen,mc:&MochigomaCollections)
		-> (impl Stack,impl Stack) {

		let sa = self.nna.forward_diff(DiffInput::NotDiff(Self::make_input(
			is_self,t,b,mc
		)));

		let sb = self.nnb.forward_diff(DiffInput::NotDiff(Self::make_input(
			is_self,t,b,mc
		)));

		(sa,sb)
	}

	pub fn evalute(&self,is_self:bool,t:Teban,b:&Banmen,mc:&MochigomaCollections)
		-> i32 {
		let input = Self::make_input(is_self,t,b,mc);

		let nnaanswera = self.nna.forward_all(input.clone());
		let nnaanswerb = self.nnb.forward_all(input.clone());

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

		let answer = nnaanswera * a.into() + nnbanswerb * b.into() - 0.5.into();

		(answer * (1 << 29) as f32) as i32
	}

	pub fn evalute_by_diff<SA,SB>(&self, snapshot:&(Cons<SA,Arr<f32,1>>,Cons<SB,Arr<f32,1>>), is_self:bool, t:Teban, b:&Banmen, mc:&MochigomaCollections, m:&Move)
		-> Result<(i32,(Cons<SA,f32>,Cons<SB,f32>)),CommonError> where SA: Stack, SB: Stack {
		let (sa,sb) = snapshot;

		let input = Intelligence::make_diff_input(is_self, t, b, mc, m)?;
		let o = self.nna.ask_diff_input(sa);

		let sa = self.nna.forward_diff(DiffInput::Diff(input.clone(),o));

		let o = self.nnb.ask_diff_input(sb);

		let sb = self.nna.forward_diff(DiffInput::Diff(input.clone(),o));

		let (a,b) = if self.bias_shake_shake {
			let mut rnd = rand::thread_rng();
			let mut rnd = XorShiftRng::from_seed(rnd.gen());

			let a = rnd.gen();
			let b = 1f64 - a;

			(a,b)
		} else {
			(0.5f64,0.5f64)
		};

		let answer = nnaanswera * a + nnbanswerb * b - 0.5;

		Ok(((answer * (1 << 29) as f32) as i32,(sa,sb)))
	}

	pub fn evalute_by_snapshot<SA,SB>(&self,snapshot:&(Cons<SA,Arr<f32,1>>,Cons<SB,Arr<f32,1>>)) -> i32 where SA: Stack, SB: Stack {
		match snapshot {
			&(sa,sb) => {
				let (a,b) = if self.bias_shake_shake {
					let mut rnd = rand::thread_rng();
					let mut rnd = XorShiftRng::from_seed(rnd.gen());

					let a = rnd.gen();
					let b = 1f64 - a;

					(a,b)
				} else {
					(0.5f64,0.5f64)
				};

				let nnaanswera = sa.1;
				let nnbanswerb = sb.1;

				let answer = nnaanswera * a+ nnbanswerb * b - 0.5;

				(answer * (1 << 29) as f32) as i32
			}
		}
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

		let ms = Mochigoma::new();
		let mg = Mochigoma::new();

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

		let c = mc.get(kind);
		let offset = offset as usize;

		Ok(offset + c as usize)
	}
}
pub struct Trainer<NN> {
	nna:NN,
	nnb:NN,
	nna_filename:String,
	nnb_filename:String,
	nnsavedir:String,
	packed_sfen_reader:PackedSfenReader,
	hcpe_reader:HcpeReader,
	bias_shake_shake:bool,
	quited:bool,
}
pub struct TrainerCreator;

impl TrainerCreator {
	pub fn new(savedir:String,nna_filename:String,nnb_filename:String,enable_shake_shake:bool)
		-> Trainer<impl BatchTrain<f32> + ForwardAll> {
		let mut rnd = prelude::thread_rng();
		let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

		let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
		let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
		let n3 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

		let device = DeviceCpu::new();

		let net:InputLayer<f32,Arr<f32,2517>,_> = InputLayer::new();

		let rnd = rnd_base.clone();

		let mut nna = net.add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,2517,256>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,256,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,Sigmoid::new(&device),&device)
		}).add_layer_train(|l| {
			LinearOutputLayer::new(l,&device)
		});

		let mut rnd = prelude::thread_rng();
		let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

		let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
		let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
		let n3 = Normal::<f32>::new(0.0, 1f32/100f32.sqrt()).unwrap();

		let device = DeviceCpu::new();

		let net:InputLayer<f32,Arr<f32,2517>,_> = InputLayer::new();

		let rnd = rnd_base.clone();

		let mut nnb = net.add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,2517,256>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,256,100>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,_,100,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,Sigmoid::new(&device),&device)
		}).add_layer_train(|l| {
			LinearOutputLayer::new(l,&device)
		});

		Trainer {
			nna:nna,
			nnb:nnb,
			nna_filename:nna_filename,
			nnb_filename:nnb_filename,
			nnsavedir:savedir,
			packed_sfen_reader:PackedSfenReader::new(),
			hcpe_reader:HcpeReader::new(),
			bias_shake_shake:enable_shake_shake,
			quited:false,
		}
	}
}
impl<NN> Trainer<NN> {

	pub fn learning_by_training_data<'a,D>(&mut self,
										   last_teban:Teban,
										   history:Vec<(Banmen,MochigomaCollections,u64,u64)>,
										   s:&GameEndState,
										   learn_max_threads:usize,
										   training_data_generator:&D,
										   a:f64,b:f64,
										   _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
										   -> Result<(Metrics,Metrics,Metrics,Metrics),CommonError>
		where D: Fn(&GameEndState,Teban,f64) -> f64 {

		let mut teban = last_teban;

		let msa = self.nna.learn_batch_parallel(learn_max_threads,history.iter().rev().map(move |(banmen,mc,_,_)| {
			let input = Intelligence::make_input(true,teban, banmen, mc);

			let t = training_data_generator(s,teban,a);

			teban = teban.opposite();

			(input.to_vec(),(0..1).map(|_| t).collect())
		}))?;

		let mut teban = last_teban.opposite();

		let moa = self.nna.learn_batch_parallel(learn_max_threads,history.iter().rev().map(move |(banmen,mc,_,_)| {
			let input = Intelligence::make_input(false,teban, banmen, mc);

			let t = training_data_generator(s,teban,a);

			teban = teban.opposite();

			(input.to_vec(),(0..1).map(|_| t).collect())
		}))?;

		let mut teban = last_teban;

		let msb = self.nnb.learn_batch_parallel(learn_max_threads,history.iter().rev().map(move |(banmen,mc,_,_)| {
			let input = Intelligence::make_input(true,teban, banmen, mc);

			let t = training_data_generator(s,teban,b);

			teban = teban.opposite();

			(input.to_vec(),(0..1).map(|_| t).collect())
		}))?;

		let mut teban = last_teban.opposite();

		let mob = self.nnb.learn_batch_parallel(learn_max_threads,history.iter().rev().map(move |(banmen,mc,_,_)| {
			let input = Intelligence::make_input(false,teban, banmen, mc);

			let t = training_data_generator(s,teban,b);

			teban = teban.opposite();

			(input.to_vec(),(0..1).map(|_| t).collect())
		}))?;

		self.save()?;

		Ok((msa,moa,msb,mob))
	}

	pub fn learning_by_packed_sfens<'a,D>(&mut self,
										  packed_sfens:Vec<Vec<u8>>,
										  learn_max_threads:usize,
										  training_data_generator:&D,
										  a:f64,b:f64,
										  _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
										  -> Result<(Metrics,Metrics,Metrics,Metrics),CommonError>
		where D: Fn(&GameEndState,f64) -> f64 {

		let mut sfens_with_extended = Vec::with_capacity(packed_sfens.len());

		for entry in packed_sfens.into_iter() {
			let ((teban,banmen,mc),yaneuraou::haffman_code::ExtendFields {
				value: _,
				best_move: _,
				end_ply: _,
				game_result
			}) = self.packed_sfen_reader.read_sfen_with_extended(entry).map_err(|e| {
				CommonError::Fail(format!("{}",e))
			})?;

			sfens_with_extended.push((teban,banmen,mc,game_result));
		}

		let msa = self.nna.learn_batch_parallel(learn_max_threads,
												sfens_with_extended.iter()
													.map(|(teban,banmen,mc,es)| {
														let teban = *teban;

														let input = Intelligence::make_input(true,teban, banmen, mc);

														let t = training_data_generator(&es,a);

														(input.to_vec(),(0..1).map(|_| t).collect())
													}))?;


		let moa = self.nna.learn_batch_parallel(learn_max_threads,
												sfens_with_extended.iter()
													.map(|(teban,banmen,mc,es)| {
														// 非手番側であるため、手番と勝敗を反転
														let teban = teban.opposite();
														let es = match es {
															&GameEndState::Win => GameEndState::Lose,
															&GameEndState::Lose => GameEndState::Win,
															&GameEndState::Draw => GameEndState::Draw
														};
														let input = Intelligence::make_input(false,teban, banmen, mc);

														let t = training_data_generator(&es,a);

														(input.to_vec(),(0..1).map(|_| t).collect())
													}))?;


		let msb = self.nnb.learn_batch_parallel(learn_max_threads,
												sfens_with_extended.iter()
													.map(|(teban,banmen,mc,es)| {
														let teban = *teban;

														let input = Intelligence::make_input(true,teban, banmen, mc);

														let t = training_data_generator(&es,b);

														(input.to_vec(),(0..1).map(|_| t).collect())
													}))?;


		let mob = self.nnb.learn_batch_parallel(learn_max_threads,
												sfens_with_extended.iter()
													.map(|(teban,banmen,mc,es)| {
														// 非手番側であるため、手番と勝敗を反転
														let teban = teban.opposite();
														let es = match es {
															&GameEndState::Win => GameEndState::Lose,
															&GameEndState::Lose => GameEndState::Win,
															&GameEndState::Draw => GameEndState::Draw
														};
														let input = Intelligence::make_input(false,teban, banmen, mc);

														let t = training_data_generator(&es,b);

														(input.to_vec(),(0..1).map(|_| t).collect())
													}))?;

		self.save()?;

		Ok((msa,moa,msb,mob))
	}

	pub fn learning_by_hcpe<'a,D>(&mut self,
								  hcpes:Vec<Vec<u8>>,
								  learn_max_threads:usize,
								  training_data_generator:&D,
								  a:f64,b:f64,
								  _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
								  -> Result<(Metrics,Metrics,Metrics,Metrics),CommonError>
		where D: Fn(&GameEndState,f64) -> f64 {

		let mut sfens_with_extended = Vec::with_capacity(hcpes.len());

		for entry in hcpes.into_iter() {
			let ((teban,banmen,mc),hcpe::haffman_code::ExtendFields {
				eval: _,
				best_move: _,
				game_result
			}) = self.hcpe_reader.read_sfen_with_extended(entry).map_err(|e| {
				CommonError::Fail(format!("{}",e))
			})?;

			sfens_with_extended.push((teban, banmen, mc, game_result));
		}

		let msa = self.nna.learn_batch_parallel(learn_max_threads,
												sfens_with_extended.iter()
													.map(|(teban,banmen,mc,es)| {
														let teban = *teban;

														let input = Intelligence::make_input(true,teban, banmen, mc);

														let es = match (es,teban) {
															(&GameResult::Draw,_) => GameEndState::Draw,
															(&GameResult::SenteWin,Teban::Sente) |
															(&GameResult::GoteWin,Teban::Gote) => {
																GameEndState::Win
															},
															(&GameResult::SenteWin,Teban::Gote) |
															(&GameResult::GoteWin,Teban::Sente) => {
																GameEndState::Lose
															}
														};

														let t = training_data_generator(&es,a);

														(input.to_vec(),(0..1).map(|_| t).collect())
													}))?;


		let moa = self.nna.learn_batch_parallel(learn_max_threads,
												sfens_with_extended.iter()
													.map(|(teban,banmen,mc,es)| {
														// 非手番側であるため、手番と勝敗を反転
														let teban = teban.opposite();

														let es = match (es,teban) {
															(&GameResult::Draw,_) => GameEndState::Draw,
															(&GameResult::SenteWin,Teban::Sente) |
															(&GameResult::GoteWin,Teban::Gote) => {
																GameEndState::Lose
															},
															(&GameResult::SenteWin,Teban::Gote) |
															(&GameResult::GoteWin,Teban::Sente) => {
																GameEndState::Win
															}
														};

														let input = Intelligence::make_input(false,teban, banmen, mc);

														let t = training_data_generator(&es,a);

														(input.to_vec(),(0..1).map(|_| t).collect())
													}))?;


		let msb = self.nnb.learn_batch_parallel(learn_max_threads,
												sfens_with_extended.iter()
													.map(|(teban,banmen,mc,es)| {
														let teban = *teban;

														let input = Intelligence::make_input(true,teban, banmen, mc);

														let es = match (es,teban) {
															(&GameResult::Draw,_) => GameEndState::Draw,
															(&GameResult::SenteWin,Teban::Sente) |
															(&GameResult::GoteWin,Teban::Gote) => {
																GameEndState::Win
															},
															(&GameResult::SenteWin,Teban::Gote) |
															(&GameResult::GoteWin,Teban::Sente) => {
																GameEndState::Lose
															}
														};

														let t = training_data_generator(&es,b);

														(input.to_vec(),(0..1).map(|_| t).collect())
													}))?;


		let mob = self.nnb.learn_batch_parallel(learn_max_threads,
												sfens_with_extended.iter()
													.map(|(teban,banmen,mc,es)| {
														// 非手番側であるため、手番と勝敗を反転
														let teban = teban.opposite();

														let es = match (es,teban) {
															(&GameResult::Draw,_) => GameEndState::Draw,
															(&GameResult::SenteWin,Teban::Sente) |
															(&GameResult::GoteWin,Teban::Gote) => {
																GameEndState::Lose
															},
															(&GameResult::SenteWin,Teban::Gote) |
															(&GameResult::GoteWin,Teban::Sente) => {
																GameEndState::Win
															}
														};

														let input = Intelligence::make_input(false,teban, banmen, mc);

														let t = training_data_generator(&es,b);

														(input.to_vec(),(0..1).map(|_| t).collect())
													}))?;

		self.save()?;

		Ok((msa,moa,msb,mob))
	}

	fn save(&mut self) -> Result<(),CommonError>{
		self.nna.save(
			TextFilePersistence::new(
				&format!("{}/{}.tmp",self.nnsavedir,self.nna_filename))?)?;
		self.nnb.save(
			TextFilePersistence::new(
				&format!("{}/{}.tmp",self.nnsavedir,self.nnb_filename))?)?;
		fs::rename(&format!("{}/{}.tmp", self.nnsavedir,self.nna_filename),
				   &format!("{}/{}", self.nnsavedir,self.nna_filename))?;
		fs::rename(&format!("{}/{}.tmp", self.nnsavedir,self.nnb_filename),
				   &format!("{}/{}", self.nnsavedir,self.nnb_filename))?;
		Ok(())
	}
}