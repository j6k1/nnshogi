use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::fs;
use std::ops::DerefMut;
use std::path::Path;
use std::rc::Rc;
use std::convert::TryFrom;

use nncombinator::activation::{ReLu, Tanh};
use nncombinator::arr::{Arr, DiffArr, VecArr};
use nncombinator::cuda::mem::{Alloctype, MemoryPool};
use nncombinator::device::{DeviceCpu, DeviceGpu};
use nncombinator::layer::{ActivationLayer, AddLayer, AddLayerTrain, AskDiffInput, BatchForwardBase, BatchTrain, DiffInput, DiffLinearLayer, ForwardAll, ForwardDiff, InputLayer, LinearLayer, LinearOutputLayer, PreTrain, TryAddLayer};
use nncombinator::lossfunction::Mse;
use nncombinator::optimizer::{Adam};
use nncombinator::persistence::{BinFilePersistence, Linear, Persistence, SaveToFile};
use nncombinator::Stack;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;

use usiagent::shogi::*;
use usiagent::event::EventQueue;
use usiagent::event::UserEvent;
use usiagent::event::UserEventKind;
use usiagent::error::EventDispatchError;
use usiagent::event::GameEndState;

use error::*;
use packedsfen::yaneuraou;
use packedsfen::hcpe;
use packedsfen::yaneuraou::reader::PackedSfenReader;
use packedsfen::traits::Reader;
use packedsfen::hcpe::reader::HcpeReader;
use packedsfen::hcpe::haffman_code::GameResult;

pub struct Intelligence<NN>
	where NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
			  PreTrain<f32> + ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> {
	nna:NN,
	nnb:NN,
	quited:bool,
}

const BANMEN_SIZE:usize = 81;

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
const SCALE:f32 = 1.;

pub struct IntelligenceCreator;
impl IntelligenceCreator {
	pub fn create(savedir:String,nna_filename:String,nnb_filename:String)
		-> Result<Intelligence<impl ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
							 PreTrain<f32> + ForwardDiff<f32> +
							 AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static>,ApplicationError> {

		let mut rnd = prelude::thread_rng();
		let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

		let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
		let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
		let n3 = Normal::<f32>::new(0.0, 1f32/32f32.sqrt()).unwrap();

		let device = DeviceCpu::new()?;

		let net:InputLayer<f32,DiffInput<DiffArr<f32,2517>,f32,2517,256>,_> = InputLayer::new();

		let rnd = rnd_base.clone();

		let mut nna = net.add_layer(|l| {
			let rnd = rnd.clone();
			DiffLinearLayer::<_,_,_,DeviceCpu<f32>,_,2517,256>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,DeviceCpu<f32>,_,256,32>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,DeviceCpu<f32>,_,32,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,Tanh::new(&device),&device)
		}).add_layer_train(|l| {
			LinearOutputLayer::new(l,&device)
		});

		let mut rnd = prelude::thread_rng();
		let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

		let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
		let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
		let n3 = Normal::<f32>::new(0.0, 1f32/32f32.sqrt()).unwrap();

		let device = DeviceCpu::new()?;

		let net:InputLayer<f32,DiffInput<DiffArr<f32,2517>,f32,2517,256>,_> = InputLayer::new();

		let rnd = rnd_base.clone();

		let mut nnb = net.add_layer(|l| {
			let rnd = rnd.clone();
			DiffLinearLayer::<_,_,_,DeviceCpu<f32>,_,2517,256>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,DeviceCpu<f32>,_,256,32>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).add_layer(|l| {
			let rnd = rnd.clone();
			LinearLayer::<_,_,_,DeviceCpu<f32>,_,32,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
		}).add_layer(|l| {
			ActivationLayer::new(l,Tanh::new(&device),&device)
		}).add_layer_train(|l| {
			LinearOutputLayer::new(l,&device)
		});

		if Path::new(&format!("{}/{}",savedir,nna_filename)).exists() {
			let mut pa = BinFilePersistence::new(
				&format!("{}/{}", savedir, nna_filename))?;

			nna.load(&mut pa)?;
		}

		if Path::new(&format!("{}/{}",savedir,nnb_filename)).exists() {
			let mut pb = BinFilePersistence::new(
				&format!("{}/{}", savedir, nna_filename))?;

			nnb.load(&mut pb)?;
		}

		Ok(Intelligence::new(nna,nnb))
	}
}
impl<NN> Intelligence<NN>
	where NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
			  PreTrain<f32> + ForwardDiff<f32> +
			  AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static {
	pub fn new(nna:NN,nnb:NN) -> Intelligence<NN> {
		Intelligence {
			nna:nna,
			nnb:nnb,
			quited:false,
		}
	}

	pub fn make_snapshot(&self, is_self:bool, t:Teban, b:&Banmen, mc:&MochigomaCollections) -> Result<(<NN as PreTrain<f32>>::OutStack, <NN as PreTrain<f32>>::OutStack), CommonError>
	{

		let sa = self.nna.forward_diff(DiffInput::NotDiff(InputCreator::make_input(
			is_self,t,b,mc
		) * SCALE))?;

		let sb = self.nnb.forward_diff(DiffInput::NotDiff(InputCreator::make_input(
			is_self,t,b,mc
		) * SCALE))?;

		Ok((sa,sb))
	}

	pub fn evalute(&self, is_self:bool, t:Teban, b:&Banmen, mc:&MochigomaCollections) -> Result<i32, CommonError>
	{
		let input = InputCreator::make_input(is_self,t,b,mc);

		let nnaanswera = self.nna.forward_all(DiffInput::NotDiff(input.clone() * SCALE))?;
		let nnbanswerb = self.nnb.forward_all(DiffInput::NotDiff(input.clone() * SCALE))?;

		let answer = nnaanswera[0] + nnbanswerb[0];

		Ok((answer * (1 << 29) as f32) as i32)
	}

	pub fn evalute_by_diff(&self, snapshot:&(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack), is_self:bool, t:Teban, b:&Banmen, mc:&MochigomaCollections, m:&Move)
		-> Result<(i32,(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)),CommonError> {
		let (sa,sb) = snapshot;

		let input = InputCreator::make_diff_input(is_self, t, b, mc, m)?;
		let o = self.nna.ask_diff_input(sa);

		let sa = self.nna.forward_diff(DiffInput::Diff(input.clone() * SCALE,o))?;

		let o = self.nnb.ask_diff_input(sb);

		let sb = self.nna.forward_diff(DiffInput::Diff(input.clone() * SCALE,o))?;

		let answer = sa.map(|ans| ans[0].clone()) + sb.map(|ans| ans[0].clone());

		Ok(((answer * (1 << 29) as f32) as i32,(sa,sb)))
	}

	pub fn evalute_by_snapshot(&self,snapshot:&(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)) -> i32 {
		match snapshot {
			(sa,sb) => {
				let nnaanswera = sa.map(|ans| ans[0].clone());
				let nnbanswerb = sb.map(|ans| ans[0].clone());

				let answer = nnaanswera + nnbanswerb;

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
}
pub struct Trainer<NN>
	where NN: BatchTrain<f32,DeviceGpu<f32>> + ForwardAll + Persistence<f32,BinFilePersistence<f32>,Linear> {

	nna:NN,
	nnb:NN,
	optimizer:Adam<f32>,
	nna_filename:String,
	nnb_filename:String,
	nnsavedir:String,
	packed_sfen_reader:PackedSfenReader,
	hcpe_reader:HcpeReader,
	bias_shake_shake:bool,
}
pub struct TrainerCreator;

impl TrainerCreator {
	pub fn create(savedir:String, nna_filename:String, nnb_filename:String, enable_shake_shake:bool)
		-> Result<Trainer<impl ForwardAll<Input=Arr<f32,2517>,Output=Arr<f32,1>> +
						BatchForwardBase<BatchInput=VecArr<f32,Arr<f32,2517>>,BatchOutput=VecArr<f32,Arr<f32,1>>> +
						BatchTrain<f32,DeviceGpu<f32>> + Persistence<f32,BinFilePersistence<f32>,Linear>>,ApplicationError> {

		let mut rnd = prelude::thread_rng();
		let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

		let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
		let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
		let n3 = Normal::<f32>::new(0.0, 1f32/32f32.sqrt()).unwrap();

		let memory_pool = Arc::new(Mutex::new(MemoryPool::with_size(1024 * 1024 * 1024 *  4,Alloctype::Device)?));

		let device = DeviceGpu::new(&memory_pool)?;

		let net:InputLayer<f32,Arr<f32,2517>,_> = InputLayer::new();

		let rnd = rnd_base.clone();

		let mut nna = net.try_add_layer(|l| {
			let rnd = rnd.clone();
			Ok(LinearLayer::<_,_,_,DeviceGpu<f32>,_,2517,256>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)?)
		})?.add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).try_add_layer(|l| {
			let rnd = rnd.clone();
			Ok(LinearLayer::<_,_,_,DeviceGpu<f32>,_,256,32>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)?)
		})?.add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).try_add_layer(|l| {
			let rnd = rnd.clone();
			Ok(LinearLayer::<_,_,_,DeviceGpu<f32>,_,32,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)?)
		})?.add_layer(|l| {
			ActivationLayer::new(l,Tanh::new(&device),&device)
		}).add_layer_train(|l| {
			LinearOutputLayer::new(l,&device)
		});

		let mut rnd = prelude::thread_rng();
		let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

		let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
		let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
		let n3 = Normal::<f32>::new(0.0, 1f32/32f32.sqrt()).unwrap();

		let device = DeviceGpu::new(&memory_pool)?;

		let net:InputLayer<f32,Arr<f32,2517>,_> = InputLayer::new();

		let rnd = rnd_base.clone();

		let mut nnb = net.try_add_layer(|l| {
			let rnd = rnd.clone();
			Ok(LinearLayer::<_,_,_,DeviceGpu<f32>,_,2517,256>::new(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)?)
		})?.add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).try_add_layer(|l| {
			let rnd = rnd.clone();
			Ok(LinearLayer::<_,_,_,DeviceGpu<f32>,_,256,32>::new(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)?)
		})?.add_layer(|l| {
			ActivationLayer::new(l,ReLu::new(&device),&device)
		}).try_add_layer(|l| {
			let rnd = rnd.clone();
			Ok(LinearLayer::<_,_,_,DeviceGpu<f32>,_,32,1>::new(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)?)
		})?.add_layer(|l| {
			ActivationLayer::new(l,Tanh::new(&device),&device)
		}).add_layer_train(|l| {
			LinearOutputLayer::new(l,&device)
		});

		if Path::new(&format!("{}/{}",savedir,nna_filename)).exists() {
			let mut pa = BinFilePersistence::new(
				&format!("{}/{}", savedir, nna_filename))?;

			nna.load(&mut pa)?;
		}

		if Path::new(&format!("{}/{}",savedir,nnb_filename)).exists() {
			let mut pb = BinFilePersistence::new(
				&format!("{}/{}", savedir, nna_filename))?;

			nnb.load(&mut pb)?;
		}

		Ok(Trainer {
			nna:nna,
			nnb:nnb,
			optimizer:Adam::new(),
			nna_filename:nna_filename,
			nnb_filename:nnb_filename,
			nnsavedir:savedir,
			packed_sfen_reader:PackedSfenReader::new(),
			hcpe_reader:HcpeReader::new(),
			bias_shake_shake:enable_shake_shake,
		})
	}
}
impl<NN> Trainer<NN>
	where NN: ForwardAll<Input=Arr<f32,2517>,Output=Arr<f32,1>> +
			  BatchForwardBase<BatchInput=VecArr<f32,Arr<f32,2517>>,BatchOutput=VecArr<f32,Arr<f32,1>>> +
			  BatchTrain<f32,DeviceGpu<f32>> + Persistence<f32,BinFilePersistence<f32>,Linear> {
	pub fn calc_alpha_beta(bias_shake_shake:bool) -> (f32,f32) {
		if bias_shake_shake {
			let mut rnd = rand::thread_rng();
			let mut rnd = XorShiftRng::from_seed(rnd.gen());

			let a = rnd.gen();
			let b = 1f32 - a ;

			(a,b)
		} else {
			(0.5f32,0.5f32)
		}
	}

	pub fn learning_by_training_csa<'a>(&mut self,
										last_teban:Teban,
										history:Vec<(Banmen,MochigomaCollections,u64,u64)>,
										s:&GameEndState,
										_:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
										-> Result<(f32,f32),CommonError> {

		let lossf = Mse::new();

		let mut teban = last_teban;
		let bias_shake_shake = self.bias_shake_shake;
		
		let batch = history.iter().rev().map(move |(banmen,mc,_,_)| {
			let (a, b) = Self::calc_alpha_beta(bias_shake_shake);

			let input = InputCreator::make_input(true, teban, banmen, mc);

			let t = match s {
				GameEndState::Win if teban == last_teban => {
					1f32
				}
				GameEndState::Win => {
					-1f32
				},
				GameEndState::Lose if teban == last_teban => {
					-1f32
				},
				GameEndState::Lose => {
					1f32
				},
				_ => 0f32
			};

			teban = teban.opposite();

			(t,input,a,b)
		}).fold(((Vec::new(),Vec::new()),(Vec::new(),Vec::new())), | mut acc, (t,input,a,b) | {
			let mut ans = Arr::<f32, 1>::new();
			ans[0] = t * a;

			(acc.0).0.push(ans);
			(acc.0).1.push(input.clone() * SCALE);

			let mut ans = Arr::<f32, 1>::new();
			ans[0] = t * b;

			(acc.1).0.push(ans);
			(acc.1).1.push(input * SCALE);

			acc
		});

		let msa = self.nna.batch_train((batch.0).0.into(),(batch.0).1.into(),&mut self.optimizer,&lossf)?;
		let msb = self.nna.batch_train((batch.1).0.into(),(batch.1).1.into(),&mut self.optimizer,&lossf)?;

		self.save()?;

		Ok((msa,msb))
	}

	pub fn test_by_csa(&mut self,
						   teban:Teban,
						   kyokumen:&(Banmen,MochigomaCollections,u64,u64))
						   -> Result<f32,ApplicationError> {
		let (banmen,mc,_,_) = kyokumen;

		let input = InputCreator::make_input(true, teban, &banmen, &mc);

		let ra = self.nna.forward_all(input.clone() * SCALE)?;
		let rb = self.nnb.forward_all(input * SCALE)?;

		Ok(ra[0] + rb[0])
	}

	pub fn learning_by_packed_sfens<'a>(&mut self,
										  packed_sfens:Vec<Vec<u8>>,
										  _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
		-> Result<(f32,f32),CommonError> {

		let lossf = Mse::new();
		let bias_shake_shake = self.bias_shake_shake;

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

		let batch = sfens_with_extended.iter()
			.map(|(teban,banmen,mc,es)| {
				let (a,b) = Self::calc_alpha_beta(bias_shake_shake);

				let teban = *teban;

				let input = InputCreator::make_input(true, teban, banmen, mc);

				let t = match es {
					GameEndState::Win => {
						1f32
					}
					GameEndState::Lose => {
						-1f32
					},
					_ => 0f32
				};

				(t,input,a,b)
			}).fold(((Vec::new(),Vec::new()),(Vec::new(),Vec::new())), | mut acc, (t,input,a,b) | {
				let mut ans = Arr::<f32, 1>::new();
				ans[0] = t * a;

				(acc.0).0.push(ans);
				(acc.0).1.push(input.clone() * SCALE);

				let mut ans = Arr::<f32, 1>::new();
				ans[0] = t * b;

				(acc.1).0.push(ans);
				(acc.1).1.push(input * SCALE);

				acc
			});

		let msa = self.nna.batch_train((batch.0).0.into(),(batch.0).1.into(),&mut self.optimizer,&lossf)?;
		let msb = self.nnb.batch_train((batch.1).0.into(),(batch.1).1.into(),&mut self.optimizer,&lossf)?;

		Ok((msa,msb))
	}

	pub fn test_by_packed_sfens(&mut self,
										packed_sfen:Vec<u8>)
										-> Result<(GameEndState,f32),ApplicationError> {
		let ((teban,banmen,mc),yaneuraou::haffman_code::ExtendFields {
			value: _,
			best_move: _,
			end_ply: _,
			game_result
		}) = self.packed_sfen_reader.read_sfen_with_extended(packed_sfen).map_err(|e| {
			ApplicationError::LearningError(format!("{}",e))
		})?;

		let input = InputCreator::make_input(true, teban, &banmen, &mc);

		let ra = self.nna.forward_all(input.clone() * SCALE)?;
		let rb = self.nnb.forward_all(input * SCALE)?;

		Ok((game_result,ra[0] + rb[0]))
	}

	pub fn learning_by_hcpe<'a>(&mut self,
								  hcpes:Vec<Vec<u8>>,
								  _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
								  -> Result<(f32,f32),CommonError> {

		let lossf = Mse::new();
		let bias_shake_shake = self.bias_shake_shake;

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

		let batch = sfens_with_extended.iter()
			.map(|(teban,banmen,mc,es)| {
				let (a,b) = Self::calc_alpha_beta(bias_shake_shake);

				let teban = *teban;

				let input = InputCreator::make_input(true,teban, banmen, mc);

				let es = match (es,teban) {
					(GameResult::Draw,_) => GameEndState::Draw,
					(GameResult::SenteWin,Teban::Sente) |
					(GameResult::GoteWin,Teban::Gote) => {
						GameEndState::Win
					},
					(GameResult::SenteWin,Teban::Gote) |
					(GameResult::GoteWin,Teban::Sente) => {
						GameEndState::Lose
					}
				};

				let t = match es {
					GameEndState::Win => {
						1f32
					}
					GameEndState::Lose => {
						-1f32
					},
					_ => 0f32
				};

				(t,input,a,b)
			}).fold(((Vec::new(),Vec::new()),(Vec::new(),Vec::new())),| mut acc, (t,input,a,b) | {
				let mut ans = Arr::<f32,1>::new();
				ans[0] = t * a;

				(acc.0).0.push(ans);
				(acc.0).1.push(input.clone() * SCALE);

				let mut ans = Arr::<f32,1>::new();
				ans[0] = t * b;

				(acc.1).0.push(ans);
				(acc.1).1.push(input * SCALE);

				acc
			});

		let msa = self.nna.batch_train((batch.0).0.into(),(batch.0).1.into(),&mut self.optimizer,&lossf)?;
		let msb = self.nnb.batch_train((batch.1).0.into(),(batch.1).1.into(),&mut self.optimizer,&lossf)?;

		Ok((msa,msb))
	}

	pub fn test_by_hcpe(&mut self,
						hcpe:Vec<u8>)
						-> Result<(GameEndState,f32),ApplicationError> {
		let ((teban,banmen,mc),hcpe::haffman_code::ExtendFields {
			eval: _,
			best_move: _,
			game_result
		}) = self.hcpe_reader.read_sfen_with_extended(hcpe).map_err(|e| {
			ApplicationError::LearningError(format!("{}",e))
		})?;

		let input = InputCreator::make_input(true, teban, &banmen, &mc);

		let ra = self.nna.forward_all(input.clone() * SCALE)?;
		let rb = self.nnb.forward_all(input * SCALE)?;

		let s = match game_result {
			GameResult::SenteWin if teban == Teban::Sente => {
				GameEndState::Win
			},
			GameResult::SenteWin => {
				GameEndState::Lose
			},
			GameResult::GoteWin if teban == Teban::Gote => {
				GameEndState::Win
			},
			GameResult::GoteWin => {
				GameEndState::Lose
			},
			_ => GameEndState::Draw
		};

		Ok((s,ra[0] + rb[0]))
	}

	pub fn save(&mut self) -> Result<(),ApplicationError> {
		let mut pa = BinFilePersistence::new(
			&format!("{}/{}.tmp",self.nnsavedir,self.nna_filename))?;
		let mut pb = BinFilePersistence::new(
			&format!("{}/{}.tmp",self.nnsavedir,self.nnb_filename))?;

		self.nna.save(&mut pa)?;
		self.nnb.save(&mut pb)?;

		pa.save(&format!("{}/{}.tmp",self.nnsavedir,self.nna_filename))?;
		pb.save(&format!("{}/{}.tmp",self.nnsavedir,self.nnb_filename))?;

		fs::rename(&format!("{}/{}.tmp", self.nnsavedir,self.nna_filename),
				   &format!("{}/{}", self.nnsavedir,self.nna_filename))?;
		fs::rename(&format!("{}/{}.tmp", self.nnsavedir,self.nnb_filename),
				   &format!("{}/{}", self.nnsavedir,self.nnb_filename))?;
		Ok(())
	}
}
pub struct InputCreator;

impl InputCreator {
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
							let index = InputCreator::input_index_of_banmen(t,kind,x as u32,y as u32).unwrap();

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

	pub fn make_diff_input(is_self:bool, t:Teban, b:&Banmen, mc:&MochigomaCollections, m:&Move) -> Result<DiffArr<f32,2517>,CommonError> {
		let mut d = DiffArr::new();

		let (addi,subi) = if is_self {
			(SELF_TEBAN_INDEX,OPPONENT_TEBAN_INDEX)
		} else {
			(OPPONENT_TEBAN_INDEX,SELF_TEBAN_INDEX)
		};

		d.push(subi,-1.)?;
		d.push(addi,1.)?;

		match m {
			&Move::To(KomaSrcPosition(sx,sy),KomaDstToPosition(dx,dy,n)) => {
				match b {
					&Banmen(ref kinds) => {
						let (sx,sy) = (9-sx,sy-1);
						let (dx,dy) = (9-dx,dy-1);

						let sk = kinds[sy as usize][sx as usize];

						d.push(InputCreator::input_index_of_banmen(t, sk, sx, sy)?, -1.)?;

						if n {
							d.push(InputCreator::input_index_of_banmen(t,sk.to_nari(),dx,dy)?,1.)?;
						} else {
							d.push(InputCreator::input_index_of_banmen(t,sk,dx,dy)?,1.)?;
						}

						let dk = kinds[dy as usize][dx as usize];

						if dk != KomaKind::Blank {
							d.push(InputCreator::input_index_of_banmen(t,dk,dx,dy)?,-1.)?;
						}

						if dk != KomaKind::Blank && dk != KomaKind::SOu && dk != KomaKind::GOu {
							let offset = InputCreator::input_index_with_of_mochigoma_get(t, MochigomaKind::try_from(dk)?, mc)?;

							d.push(offset+1, 1.)?;
						}
					}
				}
			},
			&Move::Put(kind,KomaDstPutPosition(dx,dy))  => {
				let (dx,dy) = (9-dx,dy-1);
				let offset = InputCreator::input_index_with_of_mochigoma_get(t, kind, mc)?;

				if offset < 1 {
					return Err(CommonError::Fail(
						String::from(
							"Calculation of index of difference input data of neural network failed. (The number of holding pieces is 0)"
						)))
				} else {
					d.push(offset, -1.)?;

					d.push(InputCreator::input_index_of_banmen(t, KomaKind::from((t, kind)), dx, dy)?, 1.)?;
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
	fn input_index_with_of_mochigoma_get(teban:Teban, kind:MochigomaKind, mc:&MochigomaCollections)
										 -> Result<usize,CommonError> {

		let ms = Mochigoma::new();
		let mg = Mochigoma::new();

		let (ms,mg) = match mc {
			&MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
			&MochigomaCollections::Empty => (&ms,&mg),
		};

		let mc = match teban {
			Teban::Sente => ms,
			Teban::Gote => mg,
		};

		let offset = if teban == Teban::Sente {
			SELF_INDEX_MAP[kind as usize]
		} else {
			OPPONENT_INDEX_MAP[kind as usize]
		};

		let c = mc.get(kind);
		let offset = offset as usize;

		Ok(offset + c as usize)
	}
}