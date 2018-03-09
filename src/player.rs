use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::fmt;
use rand;
use rand::Rng;
use error::*;
use std::num::Wrapping;

use usiagent::player::*;
use usiagent::command::*;
use usiagent::event::*;
use usiagent::shogi::*;
use usiagent::OnErrorHandler;
use usiagent::Logger;
use usiagent::error::PlayerError;

const KOMA_KIND_MAX:usize = KomaKind::Blank as usize;
const MOCHIGOMA_KIND_MAX:usize = MochigomaKind::Hisha as usize;
const MOCHIGOMA_MAX:usize = 18;
const SUJI_MAX:usize = 9;
const DAN_MAX:usize = 9;

pub struct NNShogiPlayer {
	stop:bool,
	kyokumen_hash_seeds:[[u64; KOMA_KIND_MAX + 1]; SUJI_MAX * DAN_MAX],
	mochigoma_hash_seeds:[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX],
}
impl fmt::Debug for NNShogiPlayer {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "NNShogiPlayer")
	}
}
impl NNShogiPlayer {
	pub fn new() -> NNShogiPlayer {
		let mut rnd = rand::XorShiftRng::new_unseeded();

		let mut kyokumen_hash_seeds:[[u64; KOMA_KIND_MAX + 1]; SUJI_MAX * DAN_MAX] = [[0; KOMA_KIND_MAX + 1]; SUJI_MAX * DAN_MAX];
		let mut mochigoma_hash_seeds:[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX] = [[0; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX];

		for i in 0..(KOMA_KIND_MAX + 1) {
			for j in 0..(SUJI_MAX * DAN_MAX) {
				kyokumen_hash_seeds[i][j] = rnd.next_u64();
			}
		}

		for i in 0..MOCHIGOMA_MAX {
			for j in 0..(MOCHIGOMA_KIND_MAX + 1) {
				mochigoma_hash_seeds[i][j] = rnd.next_u64();
			}
		}

		NNShogiPlayer {
			stop:false,
			kyokumen_hash_seeds:kyokumen_hash_seeds,
			mochigoma_hash_seeds:mochigoma_hash_seeds,
		}
	}

	fn calc_hash<AF,PF>(&self,h:u64,b:&Banmen,mc:&HashMap<MochigomaKind,u32>,m:&Move,add:AF,pull:PF)
		-> u64 where AF: Fn(u64,u64) -> u64, PF: Fn(u64,u64) -> u64 {
		match b {
			&Banmen(ref kinds) => {
				match m {
					&Move::To(ref ms, KomaDstToPosition(dx, dy, mn)) => {
						let sx = 9 - ms.0 as usize;
						let sy = ms.1 as usize;
						let dx = 9 - dx as usize;
						let dy = dy as usize;

						let mut hash = h;
						let k = kinds[sy][sx] as usize;

						hash =  pull(hash,self.kyokumen_hash_seeds[k][sx * 9 + sy]);
						hash = add(hash,self.kyokumen_hash_seeds[KomaKind::Blank as usize][sx * 9 + sy]);

						let dk = kinds[dy][dx] as usize;

						hash =  pull(hash,self.kyokumen_hash_seeds[dk][dx * 9 + dy]);
						hash = add(hash,self.kyokumen_hash_seeds[k][dx * 9 + dy]);
						hash
					},
					&Move::Put(ref mk, ref md) => {
						let mut hash = h;
						let c = match mc.get(&mk) {
							Some(c) => (*c - 1) as usize,
							None => {
								return hash;
							}
						};
						let k = *mk as usize;

						hash = pull(hash,self.mochigoma_hash_seeds[c][k]);

						let dx = md.0 as usize;
						let dy = md.1 as usize;

						hash = add(hash,self.kyokumen_hash_seeds[k + 1usize][dx * 9 + dy]);

						hash
					}
				}
			}
		}
	}

	fn calc_main_hash(&self,h:u64,b:&Banmen,mc:&HashMap<MochigomaKind,u32>,m:&Move) -> u64 {
		self.calc_hash(h,b,mc,m,|h,v| h ^ v, |h,v| h ^ v)
	}

	fn calc_sub_hash(&self,h:u64,b:&Banmen,mc:&HashMap<MochigomaKind,u32>,m:&Move) -> u64 {
		self.calc_hash(h,b,mc,m,|h,v| {
			let h = Wrapping(h);
			let v = Wrapping(v);
			(h + v).0
		}, |h,v| {
			let h = Wrapping(h);
			let v = Wrapping(v);
			(h - v).0
		})
	}
}
impl USIPlayer<CommonError> for NNShogiPlayer {
	const ID: &'static str = "nnshogi";
	const AUTHOR: &'static str = "jinpu";
	fn get_option_kinds(&mut self) -> Result<HashMap<String,SysEventOptionKind>,CommonError> {
		Ok(HashMap::new())
	}
	fn get_options(&mut self) -> Result<HashMap<String,UsiOptType>,CommonError> {
		Ok(HashMap::new())
	}
	fn take_ready(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
	fn set_option(&mut self,name:String,value:SysEventOption) -> Result<(),CommonError> {
		Ok(())
	}
	fn newgame(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
	fn set_position(&mut self,teban:Teban,ban:Banmen,
					ms:HashMap<MochigomaKind,u32>,mg:HashMap<MochigomaKind,u32>,n:u32,m:Vec<Move>)
		-> Result<(),CommonError> {
		Ok(())
	}
	fn think<L>(&mut self,limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
			info_sender:&USIInfoSender,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
			-> Result<BestMove,CommonError> where L: Logger {
		Ok(BestMove::Win)
	}
	fn think_mate<L>(&mut self,limit:&UsiGoMateTimeLimit,event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
			info_sender:&USIInfoSender,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
			-> Result<CheckMate,CommonError> where L: Logger {
		Ok(CheckMate::NotiImplemented)
	}
	fn on_stop(&mut self,e:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		self.stop = true;
		Ok(())
	}
	fn gameover(&mut self,s:&GameEndState,event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>) -> Result<(),CommonError> {
		Ok(())
	}
	fn on_quit(&mut self,e:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		Ok(())
	}

	fn quit(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
}
