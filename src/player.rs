use std;
use std::collections::HashMap;
use std::collections::BTreeMap;
use std::fmt;
use rand;
use rand::Rng;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use error::*;
use std::num::Wrapping;
use std::time::{Instant,Duration};
use std::ops::Neg;
use std::ops::Add;
use std::ops::Sub;
use std::sync::atomic;
use std::sync::atomic::AtomicBool;

use usiagent::player::*;
use usiagent::event::*;
use usiagent::command::*;
use usiagent::shogi::*;
use usiagent::rule::*;
use usiagent::hash::*;
use usiagent::OnErrorHandler;
use usiagent::logger::*;
use usiagent::error::PlayerError;
use usiagent::error::UsiProtocolError;
use usiagent::error::EventHandlerError;
use usiagent::TryFrom;
use simplenn::SnapShot;

use nn::Intelligence;
use solver::*;

const KOMA_KIND_MAX:usize = KomaKind::Blank as usize;
const MOCHIGOMA_KIND_MAX:usize = MochigomaKind::Hisha as usize;
const MOCHIGOMA_MAX:usize = 18;
const SUJI_MAX:usize = 9;
const DAN_MAX:usize = 9;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Evaluation {
	Result(Score,Option<Move>),
	Timeout(Option<Score>,Option<Move>),
	Error,
}
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
enum Score {
	NEGINFINITE,
	Value(i64),
	INFINITE,
}
impl Neg for Score {
	type Output = Score;

	fn neg(self) -> Score {
		match self {
			Score::Value(v) => Score::Value(-v),
			Score::INFINITE => Score::NEGINFINITE,
			Score::NEGINFINITE => Score::INFINITE,
		}
	}
}
impl Add<i64> for Score {
	type Output = Self;

	fn add(self, other:i64) -> Self::Output {
		match self {
			Score::Value(v) => Score::Value(v + other),
			Score::INFINITE => Score::INFINITE,
			Score::NEGINFINITE => Score::NEGINFINITE,
		}
	}
}
impl Sub<i64> for Score {
	type Output = Self;

	fn sub(self, other:i64) -> Self::Output {
		match self {
			Score::Value(v) => Score::Value(v - other),
			Score::INFINITE => Score::INFINITE,
			Score::NEGINFINITE => Score::NEGINFINITE,
		}
	}
}
const BASE_DEPTH:u32 = 2;
const MAX_DEPTH:u32 = 6;
const TIMELIMIT_MARGIN:u64 = 50;
const NETWORK_DELAY:u32 = 1100;
const DEFALUT_DISPLAY_EVALUTE_SCORE:bool = false;
const DEFAULT_ADJUST_DEPTH:bool = true;
const MAX_THREADS:u32 = 1;
const MAX_PLY:u32 = 200;
const MAX_PLY_TIMELIMIT:u64 = 0;
const TURN_COUNT:u32 = 50;
const MIN_TURN_COUNT:u32 = 5;

type Strategy<L,S> = fn (&Arc<Search>,
						&mut Environment<L,S>,
						&mut UserEventDispatcher<Search,CommonError,L>,
						&mut UserEventDispatcher<Solver<CommonError>,CommonError,L>,
						&Arc<(SnapShot,SnapShot)>,&Arc<(SnapShot,SnapShot)>,
						Teban,&Arc<State>,Score,Score,
						&Arc<MochigomaCollections>,
						&KyokumenMap<u64,u32>,
						&mut Option<KyokumenMap<u64,bool>>,
						&mut Option<KyokumenMap<u64,bool>>,
						&KyokumenMap<u64,()>,
						u64,u64,
						u32,u32,u32,u64,
						&Vec<(u32,LegalMove)>,bool) -> Evaluation;
pub struct Environment<L,S> where L: Logger, S: InfoSender {
	solver:Solver<CommonError>,
	event_queue:Arc<Mutex<UserEventQueue>>,
	evalutor:Arc<Intelligence>,
	info_sender:S,
	on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
	limit:Option<Instant>,
	current_limit:Option<Instant>,
	stop:Arc<AtomicBool>,
	quited:Arc<AtomicBool>,
	kyokumen_score_map:KyokumenMap<u64,i64>
}
impl<L,S> Clone for Environment<L,S> where L: Logger, S: InfoSender {
	fn clone(&self) -> Self {
		Environment {
			solver:Solver::new(),
			event_queue:self.event_queue.clone(),
			evalutor:self.evalutor.clone(),
			info_sender:self.info_sender.clone(),
			on_error_handler:self.on_error_handler.clone(),
			limit:self.limit.clone(),
			current_limit:self.current_limit.clone(),
			stop:self.stop.clone(),
			quited:self.quited.clone(),
			kyokumen_score_map:self.kyokumen_score_map.clone()
		}
	}
}
impl<L,S> Environment<L,S> where L: Logger, S: InfoSender {
	pub fn new(event_queue:Arc<Mutex<UserEventQueue>>,
			   evalutor:Arc<Intelligence>,
			   info_sender:S,
			   on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
			   limit:Option<Instant>,
			   current_limit:Option<Instant>) -> Environment<L,S> {
		let stop = Arc::new(AtomicBool::new(false));
		let quited = Arc::new(AtomicBool::new(false));

		Environment {
			solver:Solver::new(),
			event_queue:event_queue,
			evalutor:evalutor,
			info_sender:info_sender,
			on_error_handler:on_error_handler,
			limit:limit,
			current_limit:current_limit,
			stop:stop,
			quited:quited,
			kyokumen_score_map:KyokumenMap::new()
		}
	}
}
pub struct KyokumenHash {
	kyokumen_hash_seeds:[[u64; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1],
	mochigoma_hash_seeds:[[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2],
}
impl KyokumenHash {
	pub fn new() -> KyokumenHash {
		let mut rnd = rand::thread_rng();
		let mut rnd = XorShiftRng::from_seed(rnd.gen());

		let mut kyokumen_hash_seeds:[[u64; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1] = [[0; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1];
		let mut mochigoma_hash_seeds:[[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2] = [[[0; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2];

		for i in 0..(KOMA_KIND_MAX + 1) {
			for j in 0..(SUJI_MAX * DAN_MAX) {
				kyokumen_hash_seeds[i][j] = rnd.gen();
			}
		}

		for i in 0..MOCHIGOMA_MAX {
			for j in 0..(MOCHIGOMA_KIND_MAX + 1) {
				mochigoma_hash_seeds[0][i][j] = rnd.gen();
				mochigoma_hash_seeds[1][i][j] = rnd.gen();
			}
		}

		KyokumenHash {
			kyokumen_hash_seeds:kyokumen_hash_seeds,
			mochigoma_hash_seeds:mochigoma_hash_seeds,
		}
	}

	fn calc_hash<AF,PF>(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,
						m:&Move,obtained:&Option<MochigomaKind>,add:AF,pull:PF)
						-> u64 where AF: Fn(u64,u64) -> u64, PF: Fn(u64,u64) -> u64 {
		match b {
			&Banmen(ref kinds) => {
				match m {
					&Move::To(KomaSrcPosition(sx,sy), KomaDstToPosition(dx, dy, n)) => {
						let sx = (9 - sx) as usize;
						let sy = (sy - 1) as usize;
						let dx = (9 - dx) as usize;
						let dy = dy as usize - 1;

						let mut hash = h;
						let k = kinds[sy][sx];

						hash =  pull(hash,self.kyokumen_hash_seeds[k as usize][sy * 8 + sx]);
						hash = add(hash,self.kyokumen_hash_seeds[KomaKind::Blank as usize][sy * 8 + sx]);

						let dk = kinds[dy][dx] as usize;

						hash =  pull(hash,self.kyokumen_hash_seeds[dk][dy * 8 + dx]);

						let k = if n {
							match k {
								KomaKind::SFu => KomaKind::SFuN,
								KomaKind::SKyou => KomaKind::SKyouN,
								KomaKind::SKei => KomaKind::SKeiN,
								KomaKind::SGin => KomaKind::SGinN,
								KomaKind::SKaku => KomaKind::SKakuN,
								KomaKind::SHisha => KomaKind::SHishaN,
								KomaKind::GFu => KomaKind::GFuN,
								KomaKind::GKyou => KomaKind::GKyouN,
								KomaKind::GKei => KomaKind::GKeiN,
								KomaKind::GGin => KomaKind::GGinN,
								KomaKind::GKaku => KomaKind::GKakuN,
								KomaKind::GHisha => KomaKind::GHishaN,
								k => k,
							}
						} else {
							k
						} as usize;

						hash = add(hash,self.kyokumen_hash_seeds[k][dy * 8 + dx]);

						hash = match obtained  {
							&None => hash,
							&Some(ref obtained) => {
								let c =  match t {
									&Teban::Sente => {
										match mc {
											&MochigomaCollections::Pair(ref mc,_) => {
												match mc.get(obtained) {
													Some(c) => *c as usize,
													None => 0,
												}
											},
											&MochigomaCollections::Empty => 0,
										}
									},
									&Teban::Gote => {
										match mc {
											&MochigomaCollections::Pair(_,ref mc) => {
												match mc.get(obtained) {
													Some(c) => *c as usize,
													None => 0,
												}
											},
											&MochigomaCollections::Empty => 0,
										}
									}
								};

								let k = *obtained as usize;

								match t {
									&Teban::Sente => {
										hash = add(hash,self.mochigoma_hash_seeds[0][c][k]);
									},
									&Teban::Gote => {
										hash = add(hash,self.mochigoma_hash_seeds[1][c][k]);
									}
								}
								hash
							}
						};

						hash
					},
					&Move::Put(ref mk, ref md) => {
						let mut hash = h;

						let c = match t {
							&Teban::Sente => {
								match mc {
									&MochigomaCollections::Pair(ref mc,_) => {
										match mc.get(&mk) {
											None | Some(&0) => {
												return hash;
											}
											Some(c) => *c as usize,
										}
									},
									&MochigomaCollections::Empty => {
										return hash;
									}
								}
							},
							&Teban::Gote => {
								match mc {
									&MochigomaCollections::Pair(_,ref mc) => {
										match mc.get(&mk) {
											None | Some(&0) => {
												return hash;
											}
											Some(c) => *c as usize,
										}
									},
									&MochigomaCollections::Empty => {
										return hash;
									}
								}
							}
						};

						let k = *mk as usize;

						match t {
							&Teban::Sente => {
								hash = pull(hash,self.mochigoma_hash_seeds[0][c-1][k]);
							},
							&Teban::Gote => {
								hash = pull(hash,self.mochigoma_hash_seeds[1][c-1][k]);
							}
						}

						let dx = 9 - md.0 as usize;
						let dy = md.1 as usize - 1;
						let dk = kinds[dy][dx] as usize;

						hash = pull(hash,self.kyokumen_hash_seeds[dk as usize][dy * 8 + dx]);

						let k = KomaKind::from((*t,*mk)) as usize;

						hash = add(hash,self.kyokumen_hash_seeds[k as usize][dy * 8 + dx]);
						hash
					}
				}
			}
		}
	}

	pub fn calc_main_hash(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
		self.calc_hash(h,t,b,mc,m,obtained,|h,v| h ^ v, |h,v| h ^ v)
	}

	pub fn calc_sub_hash(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
		self.calc_hash(h,t,b,mc,m,obtained,|h,v| {
			let h = Wrapping(h);
			let v = Wrapping(v);
			(h + v).0
		}, |h,v| {
			let h = Wrapping(h);
			let v = Wrapping(v);
			(h - v).0
		})
	}

	fn calc_initial_hash(&self,b:&Banmen,
						 ms:&HashMap<MochigomaKind,u32>,mg:&HashMap<MochigomaKind,u32>) -> (u64,u64) {
		let mut mhash:u64 = 0;
		let mut shash:Wrapping<u64> = Wrapping(0u64);

		match b {
			&Banmen(ref kinds) => {
				for y in 0..9 {
					for x in 0..9 {
						let k = kinds[y][x] as usize;
						mhash = mhash ^ self.kyokumen_hash_seeds[k][y * 8 + x];
						shash = shash + Wrapping(self.kyokumen_hash_seeds[k][y * 8 + x]);
					}
				}
			}
		}
		for k in &MOCHIGOMA_KINDS {
			match ms.get(&k) {
				Some(c) => {
					for i in 0..(*c as usize) {
						mhash = mhash ^ self.mochigoma_hash_seeds[0][i][*k as usize];
						shash = shash + Wrapping(self.mochigoma_hash_seeds[0][i][*k as usize]);
					}
				},
				None => (),
			}
			match mg.get(&k) {
				Some(c) => {
					for i in 0..(*c as usize) {
						mhash = mhash ^ self.mochigoma_hash_seeds[1][i][*k as usize];
						shash = shash + Wrapping(self.mochigoma_hash_seeds[1][i][*k as usize]);
					}
				},
				None => (),
			}
		}

		(mhash,shash.0)
	}
}
pub struct Search {
	kyokumenhash:KyokumenHash,
	base_depth:u32,
	max_depth:u32,
	max_threads:u32,
	max_ply:Option<u32>,
	max_ply_mate:Option<u32>,
	max_ply_timelimit:Option<Duration>,
	network_delay:u32,
	turn_count:u32,
	min_turn_count:u32,
	display_evalute_score:bool,
	adjust_depth:bool,
}
impl Search {
	pub fn new() -> Search {
		let max_ply_timelimit = if MAX_PLY_TIMELIMIT >  0 {
			Some(Duration::from_millis(MAX_PLY_TIMELIMIT))
		} else {
			None
		};

		Search {
			kyokumenhash:KyokumenHash::new(),
			base_depth:BASE_DEPTH,
			max_depth:MAX_DEPTH,
			max_threads:MAX_THREADS,
			max_ply:Some(MAX_PLY),
			max_ply_mate:None,
			max_ply_timelimit:max_ply_timelimit,
			network_delay:NETWORK_DELAY,
			turn_count:TURN_COUNT,
			min_turn_count:MIN_TURN_COUNT,
			display_evalute_score:DEFALUT_DISPLAY_EVALUTE_SCORE,
			adjust_depth:DEFAULT_ADJUST_DEPTH,
		}
	}

	fn create_event_dispatcher<T,L>(&self,on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,stop:&Arc<AtomicBool>,quited:&Arc<AtomicBool>)
		-> UserEventDispatcher<T,CommonError,L> where L: Logger {

		let mut event_dispatcher = USIEventDispatcher::new(&on_error_handler.clone());

		{
			let stop = stop.clone();

			event_dispatcher.add_handler(UserEventKind::Stop, move |_,e| {
				match e {
					&UserEvent::Stop => {
						stop.store(true,atomic::Ordering::Release);
						Ok(())
					},
					e => Err(EventHandlerError::InvalidState(e.event_kind())),
				}
			});
		}

		{
			let stop = stop.clone();
			let quited = quited.clone();

			event_dispatcher.add_handler(UserEventKind::Quit, move |_,e| {
				match e {
					&UserEvent::Quit => {
						quited.store(true,atomic::Ordering::Release);
						stop.store(true,atomic::Ordering::Release);
						Ok(())
					},
					e => Err(EventHandlerError::InvalidState(e.event_kind())),
				}
			});
		}

		event_dispatcher
	}

	fn timelimit_reached(&self,limit:&Option<Instant>) -> bool {
		let network_delay = self.network_delay;
		limit.map_or(false,|l| {
			l < Instant::now() || l - Instant::now() <= Duration::from_millis(network_delay as u64 + TIMELIMIT_MARGIN)
		})
	}

	fn send_message<L,S>(&self, info_sender:&mut S,
			on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, message:&str)
		where L: Logger, S: InfoSender,
			Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
		commands.push(UsiInfoSubCommand::Str(String::from(message)));

		match info_sender.send(commands) {
			Ok(_) => (),
			Err(ref e) => {
				let _ = on_error_handler.lock().map(|h| h.call(e));
			}
		}
	}

	fn send_seldepth<L,S>(&self, info_sender:&mut S,
			on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, depth:u32, seldepth:u32)
		where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
		commands.push(UsiInfoSubCommand::Depth(depth));
		commands.push(UsiInfoSubCommand::SelDepth(seldepth));


		match info_sender.send(commands) {
			Ok(_) => (),
			Err(ref e) => {
				let _ = on_error_handler.lock().map(|h| h.call(e));
			}
		}
	}
	/*
	fn send_depth<L>(&self, info_sender:&USIInfoSender,
			on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, depth:u32)
		where L: Logger {
		let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
		commands.push(UsiInfoSubCommand::Depth(depth));

		match info_sender.send(commands) {
			Ok(_) => (),
			Err(ref e) => {
				let _ = on_error_handler.lock().map(|h| h.call(e));
			}
		}
	}
	*/

	fn make_snapshot(&self,is_self:bool,evalutor:&Arc<Intelligence>,teban:Teban,state:&State,mc:&MochigomaCollections)
		-> Result<(SnapShot,SnapShot),CommonError> {

		let r = evalutor.make_snapshot(is_self,teban,state.get_banmen(),mc)?;

		Ok(r)
	}
	/*
	fn evalute<L,S>(&self,evalutor:&Arc<Intelligence>,teban:Teban,state:&State,mc:&MochigomaCollections,m:&Option<Move>,
					info_sender:&mut S,on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>)
		-> Evaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let s = match evalutor.evalute(teban,state.get_banmen(),mc) {
			Ok(s) => s,
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
				return Evaluation::Error;
			}
		};

		if self.display_evalute_score {
			self.send_message(info_sender, on_error_handler, &format!("evalute score = {}",s));
		}

		Evaluation::Result(Score::Value(s),m.clone())
	}
	*/

	fn evalute_by_diff<L,S>(&self,evalutor:&Arc<Intelligence>,
								self_snapshot:&Arc<(SnapShot,SnapShot)>,
								opponent_snapshot:&Arc<(SnapShot,SnapShot)>,
								teban:Teban,state:&Option<&Arc<State>>,
								mc:&Option<&Arc<MochigomaCollections>>,m:&Option<Move>,
					info_sender:&mut S,on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<(Evaluation,(SnapShot,SnapShot),(SnapShot,SnapShot)),CommonError> where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let state = match state {
			&Some(ref state) => state,
			&None => {
				self.send_message(info_sender, on_error_handler, "prev_state is none!");
				return Err(CommonError::Fail(String::from("prev_state is none!")));
			}
		};

		let mc = match mc {
			&Some(ref mc) => mc,
			&None => {
				self.send_message(info_sender, on_error_handler, "prev_mc is none!");
				return Err(CommonError::Fail(String::from("prev_mc is none!")));
			}
		};

		let m = match m {
			Some(ref m) => {
				m
			},
			None => {
				self.send_message(info_sender, on_error_handler, "m is none!");
				return Err(CommonError::Fail(String::from("m is none!")));
			}
		};

		let (ss,self_snapshot) = evalutor.evalute_by_diff(&self_snapshot,true,teban,state.get_banmen(),mc,m)?;
		let (os,opponent_snapshot) = evalutor.evalute_by_diff(&opponent_snapshot,false,teban.opposite(),state.get_banmen(),mc,m)?;
		let s = ss - os;

		if self.display_evalute_score {
			let teban_str = match teban {
				Teban::Sente => "sente",
				Teban::Gote =>  "gote"
			};
			self.send_message(info_sender, on_error_handler, &format!("evalute score =  {0: >17} ({1})",s,teban_str));
		}

		Ok((Evaluation::Result(Score::Value(s),Some(m.clone())),self_snapshot,opponent_snapshot))
	}

	fn evalute_by_snapshot(&self,evalutor:&Arc<Intelligence>,
						   self_snapshot:&Arc<(SnapShot,SnapShot)>,
						   opponent_snapshot:&Arc<(SnapShot,SnapShot)>)
		-> Score {

		let ss = evalutor.evalute_by_snapshot(self_snapshot);
		let os = evalutor.evalute_by_snapshot(opponent_snapshot);

		Score::Value(ss - os)
	}

	fn negascout<L,S>(self:&Arc<Self>,
								env:&mut Environment<L,S>,
					  			event_dispatcher:&mut UserEventDispatcher<Search,CommonError,L>,
					  			solver_event_dispatcher:&mut UserEventDispatcher<Solver<CommonError>,CommonError,L>,
								self_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								opponent_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								teban:Teban,state:&Arc<State>,
								alpha:Score,beta:Score,
								m:Option<Move>,mc:&Arc<MochigomaCollections>,
								prev_state:&Option<Arc<State>>,
								prev_mc:&Option<Arc<MochigomaCollections>>,
								obtained:Option<ObtainKind>,
								current_kyokumen_map:&KyokumenMap<u64,u32>,
								self_already_oute_map:&mut Option<KyokumenMap<u64,bool>>,
								opponent_already_oute_map:&mut Option<KyokumenMap<u64,bool>>,
								oute_kyokumen_map:&KyokumenMap<u64,()>,
								mhash:u64,shash:u64,
								depth:u32,current_depth:u32,base_depth:u32,
								node_count:u64,
								strategy:Strategy<L,S>,
	) -> Evaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		if current_depth > base_depth {
			self.send_seldepth(&mut env.info_sender, &env.on_error_handler, base_depth, current_depth);
		}

		if let Some(ObtainKind::Ou) = obtained {
			return Evaluation::Result(Score::NEGINFINITE, None);
		}

		if self.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
			self.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
			return Evaluation::Timeout(None, None);
		}

		if depth == 0 || current_depth == self.max_depth {
			if let Some(&s) = env.kyokumen_score_map.get(teban, &mhash, &shash) {
				return Evaluation::Result(Score::Value(s), None);
			}
		}

		let (current_opponent_nn_ss,current_self_nn_ss) = if prev_state.is_some() {
			let (opponent_nn_snapshot, self_nn_snapshot) = match self.evalute_by_diff(&env.evalutor,
																					  &opponent_nn_snapshot,
																					  &self_nn_snapshot,
																					  teban.opposite(),
																					  &prev_state.as_ref(), &prev_mc.as_ref(),
																					  &m, &mut env.info_sender, &env.on_error_handler) {
				Ok((_, oss, sss)) => {
					(Arc::new(oss), Arc::new(sss))
				},
				Err(ref e) => {
					let _ = env.on_error_handler.lock().map(|h| h.call(e));
					return Evaluation::Error;
				}
			};
			(Some(opponent_nn_snapshot), Some(self_nn_snapshot))
		} else {
			(None, None)
		};

		let self_nn_snapshot = match current_self_nn_ss {
			Some(ref ss) => ss,
			None => self_nn_snapshot,
		};

		let opponent_nn_snapshot = match current_opponent_nn_ss {
			Some(ref ss) => ss,
			None => opponent_nn_snapshot,
		};

		{
			let s = self.evalute_by_snapshot(&env.evalutor, opponent_nn_snapshot, self_nn_snapshot);

			if let Score::Value(s) = s {
				env.kyokumen_score_map.insert(teban, mhash, shash, s);
			}

			if depth == 0 || current_depth == self.max_depth {
				return Evaluation::Result(-s, None);
			}
		}

		let _ = event_dispatcher.dispatch_events(self,&*env.event_queue);

		if self.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
			self.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
			return Evaluation::Timeout(None,None);
		}

		let (mvs,responded_oute) = if Rule::is_mate(teban.opposite(),&*state) {
			if depth == 0 || current_depth == self.max_depth {
				if self.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
					self.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
					return Evaluation::Timeout(None,None);
				}
			}

			let mvs = Rule::respond_oute_only_moves_all(teban, &*state, &*mc);

			if mvs.len() == 0 {
				return Evaluation::Result(Score::NEGINFINITE,None);
			} else if depth == 0 || current_depth == self.max_depth {
				return Evaluation::Result(-self.evalute_by_snapshot(&env.evalutor,opponent_nn_snapshot,self_nn_snapshot),None);
			} else {
				(mvs,true)
			}
		} else {
			let network_delay = self.network_delay;
			let limit = env.limit.clone();
			let checkmate_limit = self.max_ply_timelimit.map(|l| Instant::now() + l);

			let mut check_timelimit = move || {
				limit.map_or(false,|l| {
					let now = Instant::now();
						l < now ||
						l - now <= Duration::from_millis(network_delay as u64 + TIMELIMIT_MARGIN) ||
						checkmate_limit.map(|l| l < now).unwrap_or(false)
				})
			};
			{
				let this = self.clone();

				let mut info_sender = env.info_sender.clone();
				let on_error_handler = env.on_error_handler.clone();

				let mut on_searchstart = |depth,_| {
					this.send_seldepth(&mut info_sender, &on_error_handler, base_depth, current_depth + depth);
				};

				match env.solver.checkmate(false,teban, state, mc,
											self.max_ply,
											None,
											&mut oute_kyokumen_map.clone(),
											self_already_oute_map,
											&mut current_kyokumen_map.clone(),
											&*this.clone(),
											mhash, shash,
											&mut check_timelimit,
											&env.stop,
											&mut on_searchstart,
											&env.event_queue,
											solver_event_dispatcher) {
					MaybeMate::MateMoves(_,ref mvs) if mvs.len() > 0 => {
						return Evaluation::Result(Score::INFINITE,Some(mvs[0].to_move()));
					},
					MaybeMate::MateMoves(_,_) => {
						return Evaluation::Result(Score::INFINITE,None);
					},
					_ => ()
				}
			}

			if self.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
				self.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
				return Evaluation::Timeout(None,None);
			}

			let mvs:Vec<LegalMove> = Rule::legal_moves_all(teban, &*state, &*mc);

			(mvs,false)
		};

		if mvs.len() == 0 {
			return Evaluation::Result(Score::NEGINFINITE,None);
		} else if self.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
			self.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
			return Evaluation::Timeout(None,Some(mvs[0].to_move()));
		} else if mvs.len() == 1 {
			let r = match self.evalute_by_diff(&env.evalutor,
											   &self_nn_snapshot,
											   &opponent_nn_snapshot,
											   teban,
											   &Some(&state),
											   &Some(&mc),
											   &Some(mvs[0].to_move()),
											   &mut env.info_sender, &env.on_error_handler) {
				Ok((r,_,_)) => {
					r
				},
				Err(ref e) => {
					let _ = env.on_error_handler.lock().map(|h| h.call(e));
					return Evaluation::Error;
				}
			};

			return r
		}

		let _ = event_dispatcher.dispatch_events(self,&*env.event_queue);

		if self.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
			self.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
			return Evaluation::Timeout(None,Some(mvs[0].to_move()));
		}

		let mut mvs = mvs.into_iter().map(|m| {
			let ps = Rule::apply_move_to_partial_state_none_check(&*state,teban,&*mc,m.to_applied_move());

			let (x,y,kind) = match m {
				LegalMove::To(ref mv) => {
					let banmen = state.get_banmen();
					let (sx,sy) = mv.src().square_to_point();
					let (x,y) = mv.dst().square_to_point();
					let kind = banmen.0[sy as usize][sx as usize];

					let kind = if mv.is_nari() {
						kind.to_nari()
					} else {
						kind
					};

					(x,y,kind)
				},
				LegalMove::Put(ref mv) => {
					let (x,y) = mv.dst().square_to_point();
					let kind = mv.kind();

					(x,y,KomaKind::from((teban,kind)))
				}
			};
			if Rule::is_mate_with_partial_state_and_point_and_kind(teban,&ps,x,y,kind) ||
			   Rule::is_mate_with_partial_state_repeat_move_kinds(teban,&ps) {
				(10,m)
			} else {
				match m {
					LegalMove::To(ref mv) if mv.obtained().is_some() => {
						(5,m)
					},
					_ => (1,m),
				}
			}
		}).collect::<Vec<(u32,LegalMove)>>();

		mvs.sort_by(|a,b| b.0.cmp(&a.0));

		strategy(self,
					env,
					event_dispatcher,
					solver_event_dispatcher,
					&self_nn_snapshot,
					&opponent_nn_snapshot,
					teban,state,
					alpha,beta,mc,
					current_kyokumen_map,
					self_already_oute_map,
					opponent_already_oute_map,
					oute_kyokumen_map,
					mhash,shash,
					depth,
					current_depth,base_depth,
					node_count,
					&mvs,
					responded_oute)
	}

	fn startup_strategy(&self,teban:Teban,state:&State,mc:&MochigomaCollections,
						m:&LegalMove,mhash:u64,shash:u64,
						priority:u32,
						oute_kyokumen_map:&KyokumenMap<u64,()>,
						current_kyokumen_map:&KyokumenMap<u64,u32>,
						depth:u32,_:bool)
		-> Option<(u32,Option<ObtainKind>,u64,u64,KyokumenMap<u64,()>,KyokumenMap<u64,u32>,bool)> {

		let obtained = match m {
			LegalMove::To(ref m) => m.obtained(),
			_ => None,
		};

		let mut oute_kyokumen_map = oute_kyokumen_map.clone();
		let mut current_kyokumen_map = current_kyokumen_map.clone();

		let (mhash,shash) = {
			let o = obtained.and_then(|o| MochigomaKind::try_from(o).ok());

			let mhash = self.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
			let shash = self.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

			if priority == 10 {
				match oute_kyokumen_map.get(teban,&mhash,&shash) {
					Some(_) => {
						return None;
					},
					None => {
						oute_kyokumen_map.insert(teban,mhash,shash,());
					},
				}
			}

			(mhash,shash)
		};

		if priority < 10 {
			oute_kyokumen_map.clear(teban);
		}

		let depth = match priority {
			5 => depth + 1,
			_ => depth,
		};

		let is_sennichite = match current_kyokumen_map.get(teban,&mhash,&shash).unwrap_or(&0) {
			&c if c >= 3 => {
				return None;
			},
			&c if c > 0 => {
				current_kyokumen_map.insert(teban,mhash,shash,c+1);

				true
			},
			_ => false,
		};

		Some((depth,obtained,mhash,shash,oute_kyokumen_map,current_kyokumen_map,is_sennichite))
	}

	fn single_search<L,S>(search:&Arc<Search>,
								env:&mut Environment<L,S>,
						  		event_dispatcher:&mut UserEventDispatcher<Search,CommonError,L>,
						  		solver_event_dispatcher:&mut UserEventDispatcher<Solver<CommonError>,CommonError,L>,
								self_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								opponent_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								teban:Teban,state:&Arc<State>,
								mut alpha:Score,beta:Score,
								mc:&Arc<MochigomaCollections>,
								current_kyokumen_map:&KyokumenMap<u64,u32>,
								self_already_oute_map:&mut Option<KyokumenMap<u64,bool>>,
								opponent_already_oute_map:&mut Option<KyokumenMap<u64,bool>>,
								oute_kyokumen_map:&KyokumenMap<u64,()>,
								mhash:u64,shash:u64,
								depth:u32,current_depth:u32,base_depth:u32,
								node_count:u64,
								mvs:&Vec<(u32,LegalMove)>,
								responded_oute:bool)
		-> Evaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let mut scoreval = Score::NEGINFINITE;
		let mut best_move:Option<Move> = None;

		let mut processed_nodes:u32 = 0;
		let start_time = Instant::now();

		for (priority,m) in mvs {
			processed_nodes += 1;
			let nodes = node_count * mvs.len() as u64 - processed_nodes as u64;

			match search.startup_strategy(teban,state,mc,m,
											mhash,shash,
										 	*priority,
											oute_kyokumen_map,
											current_kyokumen_map,
											depth,responded_oute) {
				Some(r) => {
					let (depth,obtained,mhash,shash,
						 oute_kyokumen_map,
						 current_kyokumen_map,
						 is_sennichite) = r;

					let m = m.to_applied_move();
					let prev_state = Some(state.clone());
					let prev_mc = Some(mc.clone());

					let next = Rule::apply_move_none_check(&state,teban,mc,m);

					match next {
						(state,mc,_) => {
							if is_sennichite {
								let s = if Rule::is_mate(teban.opposite(),&state) {
									Score::NEGINFINITE
								} else {
									Score::Value(0)
								};

								if s > scoreval {
									scoreval = s;
									best_move = Some(m.to_move());
									if alpha < scoreval {
										alpha = scoreval;
									}
									if scoreval >= beta {
										return Evaluation::Result(scoreval,best_move);
									}
								}

								continue;
							}

							let repeat = match alpha {
								Score::NEGINFINITE | Score::INFINITE => 1,
								Score::Value(_) => 2,
							};

							let state = Arc::new(state);
							let mc = Arc::new(mc);

							for i in 0..repeat {
								let b = match (i,repeat) {
									(0,2) => alpha + 1,
									(1,2) | (0,1) => beta,
									_ => {
										return Evaluation::Error;
									}
								};

								match search.negascout(
									env,
									event_dispatcher,
									solver_event_dispatcher,
									opponent_nn_snapshot,
									self_nn_snapshot,
									teban.opposite(),&state,
									-b,-alpha,Some(m.to_move()),&mc,
									&prev_state,&prev_mc,
									obtained,&current_kyokumen_map,
									opponent_already_oute_map,
									self_already_oute_map,
									&oute_kyokumen_map,
									mhash,shash,
									depth-1,
									current_depth+1,base_depth,
									nodes,
									Search::single_search) {

									Evaluation::Timeout(s,_) => {
										if let Some(s) = s {
											if -s > scoreval {
												scoreval = -s;
												best_move = Some(m.to_move());
											}
										}

										return match best_move {
											Some(best_move) => Evaluation::Timeout(Some(scoreval),Some(best_move)),
											None => Evaluation::Timeout(Some(scoreval),Some(m.to_move())),
										};
									},
									Evaluation::Result(s,_) => {
										if -s > scoreval {
											scoreval = -s;
											best_move = Some(m.to_move());
											if alpha < scoreval {
												alpha = scoreval;
											}
											if scoreval >= beta {
												return Evaluation::Result(scoreval,best_move);
											}
										}
										if -s < alpha {
											break;
										}
									},
									Evaluation::Error => {
										return Evaluation::Error
									}
								}
							}
						}
					}

					let _ = event_dispatcher.dispatch_events(search,&*env.event_queue);

					if search.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
						search.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
						return match best_move {
							Some(best_move) => Evaluation::Timeout(Some(scoreval),Some(best_move)),
							None => Evaluation::Timeout(Some(scoreval),Some(m.to_move())),
						};
					} else if (current_depth > 1 && search.adjust_depth && nodes <= std::u32::MAX as u64 &&
						env.current_limit.map(|l| Instant::now() + (Instant::now() - start_time) / processed_nodes * nodes as u32 > l).unwrap_or(false)
					) || env.current_limit.map(|l| Instant::now() >= l).unwrap_or(false) {
						return Evaluation::Result(scoreval,best_move);
					}
				},
				None => (),
			}
		}

		Evaluation::Result(scoreval,best_move)
	}

	fn parallel_search<L,S>(search:&Arc<Search>,
								env:&mut Environment<L,S>,
								event_dispatcher:&mut UserEventDispatcher<Search,CommonError,L>,
								_:&mut UserEventDispatcher<Solver<CommonError>,CommonError,L>,
								self_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								opponent_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								teban:Teban,state:&Arc<State>,
								mut alpha:Score,beta:Score,
								mc:&Arc<MochigomaCollections>,
								current_kyokumen_map:&KyokumenMap<u64,u32>,
								self_already_oute_map:&mut Option<KyokumenMap<u64,bool>>,
								opponent_already_oute_map:&mut Option<KyokumenMap<u64,bool>>,
								oute_kyokumen_map:&KyokumenMap<u64,()>,
								mhash:u64,shash:u64,
								depth:u32,current_depth:u32,base_depth:u32,
								node_count:u64,
								mvs:&Vec<(u32,LegalMove)>,
								responded_oute:bool)
		-> Evaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let mut scoreval = Score::NEGINFINITE;
		let mut best_move:Option<Move> = None;

		let (sender,receiver):(_,Receiver<(Evaluation,AppliedMove)>) = mpsc::channel();
		let mut threads = search.max_threads;

		let mvs_count = mvs.len() as u64;

		let mut it = mvs.into_iter();
		let mut processed_nodes:u32 = 0;
		let start_time = Instant::now();

		loop {
			if threads == 0 {
				let r = match receiver.recv() {
					Ok(r) => r,
					Err(ref e) => {
						let _ = env.on_error_handler.lock().map(|h| h.call(e));
						let _ = search.termination(receiver, threads, &env.stop, scoreval, best_move);
						return Evaluation::Error;
					}
				};
				threads += 1;
				processed_nodes += 1;
				let nodes = node_count * mvs_count - processed_nodes as u64;

				match r {
					(Evaluation::Timeout(s,_),m) => {
						if let Some(s) = s {
							if -s > scoreval {
								scoreval = -s;
								best_move = Some(m.to_move());
							}
						}

						match best_move {
							Some(best_move) => {
								let r = search.termination(receiver, threads, &env.stop, scoreval, Some(best_move));
								let (scoreval,best_move) = r;
								return Evaluation::Timeout(scoreval,best_move);
							},
							None => {
								let r = search.termination(receiver, threads, &env.stop, scoreval, best_move);
								let (scoreval,best_move) = r;
								return Evaluation::Timeout(scoreval,best_move.or(Some(m.to_move())));
							},
						};
					},
					(Evaluation::Result(s,_),m) => {
						if -s > scoreval {
							scoreval = -s;
							best_move = Some(m.to_move());
							if alpha < scoreval {
								alpha = scoreval;
							}
							if scoreval >= beta {
								let (scoreval,best_move) = search.termination(receiver, threads, &env.stop, scoreval, best_move);
								return Evaluation::Result(scoreval.unwrap_or(Score::NEGINFINITE),best_move);
							}
						}

						if (current_depth > 1 && search.adjust_depth && nodes <= std::u32::MAX as u64 &&
							env.current_limit.map(|l| Instant::now() + (Instant::now() - start_time) / processed_nodes * nodes as u32 > l).unwrap_or(false)
						) || env.current_limit.map(|l| Instant::now() >= l).unwrap_or(false) {
							let (scoreval, best_move) = search.termination(receiver, threads, &env.stop, scoreval, best_move);
							return Evaluation::Result(scoreval.unwrap_or(Score::NEGINFINITE),best_move);
						}
					},
					(Evaluation::Error,_) => {
						let _ = search.termination(receiver, threads, &env.stop, scoreval, best_move);
						return Evaluation::Error;
					}
				}
			} else if let Some((priority,m)) = it.next() {
				match search.startup_strategy(teban,state,mc,m,
												mhash,shash,
											 	*priority,
												oute_kyokumen_map,
												current_kyokumen_map,
												depth,responded_oute) {
					Some(r) => {
						let (depth,obtained,mhash,shash,
							 oute_kyokumen_map,
							 current_kyokumen_map,
							 is_sennichite) = r;

						let m = m.to_applied_move();
						let prev_state = Some(state.clone());
						let prev_mc = Some(mc.clone());

						let next = Rule::apply_move_none_check(&state,teban,mc,m);

						match next {
							(state,mc,_) => {
								if is_sennichite {
									let s = if Rule::is_mate(teban.opposite(),&state) {
										Score::NEGINFINITE
									} else {
										Score::Value(0)
									};
									if s > scoreval {
										scoreval = s;
										best_move = Some(m.to_move());
										if alpha < scoreval {
											alpha = scoreval;
										}
										if scoreval >= beta {
											let (scoreval, best_move) = search.termination(receiver, threads, &env.stop, scoreval, best_move);
											return Evaluation::Result(scoreval.unwrap_or(Score::NEGINFINITE),best_move);
										}
									}
									continue;
								}

								let search = search.clone();
								let opponent_nn_snapshot = opponent_nn_snapshot.clone();
								let self_nn_snapshot = self_nn_snapshot.clone();
								let state = Arc::new(state);
								let mc = Arc::new(mc);
								let mut self_already_oute_map = self_already_oute_map.clone();
								let mut opponent_already_oute_map = opponent_already_oute_map.clone();
								let mut env = env.clone();

								let sender = sender.clone();

								let b = thread::Builder::new();
								let _ = b.stack_size(1024 * 1024 * 200).spawn(move || {
									let search = search.clone();

									let mut r = Evaluation::Result(Score::NEGINFINITE,None);

									let repeat = match alpha {
										Score::NEGINFINITE | Score::INFINITE => 1,
										Score::Value(_) => 2,
									};

									let mut a = alpha;

									for i in 0..repeat {
										let b = match (i,repeat) {
											(0,2) => alpha + 1,
											(1,2) | (0,1) => beta,
											_ => {
												let _ = sender.send((Evaluation::Error,m));
												return;
											}
										};

										let mut event_dispatcher = search.create_event_dispatcher(&env.on_error_handler,&env.stop,&env.quited);
										let mut solver_event_dispatcher = search.create_event_dispatcher(&env.on_error_handler,&env.stop,&env.quited);

										r = search.negascout(
											&mut env,
											&mut event_dispatcher,
											&mut solver_event_dispatcher,
											&opponent_nn_snapshot,
											&self_nn_snapshot,
											teban.opposite(),&state,
											-b,-a,Some(m.to_move()),&mc,
											&prev_state,&prev_mc,
											obtained,&current_kyokumen_map,
											&mut opponent_already_oute_map,
											&mut self_already_oute_map,
											&oute_kyokumen_map,
											mhash,shash,
											depth-1,current_depth+1,base_depth,
											node_count,
											Search::single_search);

										match r {
											Evaluation::Result(s,_) => {
												if -s <= alpha || -s  >= beta {
													break;
												} else {
													a = -s;
												}
											},
											_ => {
												break;
											}
										}
									}
									let _ = sender.send((r,m));
								});

								threads -= 1;
							}
						}

						let _ = event_dispatcher.dispatch_events(search,&*env.event_queue);

						if search.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
							let (scoreval, best_move) = search.termination(receiver, threads, &env.stop, scoreval, best_move);

							return match best_move {
								Some(best_move) => Evaluation::Timeout(scoreval,Some(best_move)),
								None => Evaluation::Timeout(scoreval,best_move.or(Some(m.to_move()))),
							};
						}
					},
					None => (),
				}
			} else {
				break;
			}
		}

		while threads < search.max_threads {
			match receiver.recv() {
				Ok(r) => {
					threads += 1;

					match r {
						(Evaluation::Timeout(s,_),m) => {
							if let Some(s) = s {
								if -s > scoreval {
									scoreval = -s;
									best_move = Some(m.to_move());
								}
							}

							match best_move {
								Some(best_move) => {
									let (scoreval, best_move) = search.termination(receiver, threads, &env.stop, scoreval, Some(best_move));
									return Evaluation::Timeout(scoreval,best_move);
								},
								None => {
									let (scoreval, best_move) = search.termination(receiver, threads, &env.stop, scoreval, best_move);
									return Evaluation::Timeout(scoreval,best_move.or(Some(m.to_move())));
								},
							};
						},
						(Evaluation::Result(s,_),m) => {
							if -s > scoreval {
								scoreval = -s;
								best_move = Some(m.to_move());
								if alpha < scoreval {
									alpha = scoreval;
								}
								if scoreval >= beta {
									let (scoreval, best_move) = search.termination(receiver, threads, &env.stop, scoreval, best_move);
									return Evaluation::Result(scoreval.unwrap_or(Score::NEGINFINITE),best_move);
								}
							}

							let nodes = node_count * mvs_count - processed_nodes as u64;

							if (current_depth > 1 && search.adjust_depth && nodes <= std::u32::MAX as u64 &&
								env.current_limit.map(|l| Instant::now() + (Instant::now() - start_time) / processed_nodes * nodes as u32 > l).unwrap_or(false)
							) || env.current_limit.map(|l| Instant::now() >= l).unwrap_or(false) {
								let (scoreval, best_move) = search.termination(receiver, threads, &env.stop, scoreval, best_move);
								return Evaluation::Result(scoreval.unwrap_or(Score::NEGINFINITE),best_move);
							}
						},
						(Evaluation::Error,_) => {
							let _ = search.termination(receiver, threads, &env.stop, scoreval, best_move);
							return Evaluation::Error;
						}
					}
				},
				Err(ref e) => {
					threads += 1;
					let _ = env.on_error_handler.lock().map(|h| h.call(e));
					let _ = search.termination(receiver, threads, &env.stop, scoreval, best_move);
					return Evaluation::Error;
				}
			};
		}

		Evaluation::Result(scoreval,best_move)
	}

	fn termination(&self,r:Receiver<(Evaluation,AppliedMove)>,
				   threads:u32,stop:&Arc<AtomicBool>,
				   score:Score,best_move:Option<Move>) -> (Option<Score>,Option<Move>) {
		stop.store(true,atomic::Ordering::Release);

		let mut score = score;
		let mut best_move = best_move;

		for _ in threads..self.max_threads {
			if let Ok((r,m)) = r.recv() {
				match r {
					Evaluation::Result(s, _) => {
						if -s > score {
							score = -s;
							best_move = Some(m.to_move());
						}
					},
					Evaluation::Timeout(Some(s), _) => {
						if -s > score {
							score = -s;
							best_move = Some(m.to_move());
						}
					},
					_ => ()
				}
			}
		}

		if best_move.is_some() {
			(Some(score), best_move)
		} else {
			(None, best_move)
		}
	}

	#[inline]
	pub fn calc_main_hash(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
		self.kyokumenhash.calc_main_hash(h,t,b,mc,m,obtained)
	}

	#[inline]
	pub fn calc_sub_hash(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
		self.kyokumenhash.calc_sub_hash(h,t,b,mc,m,obtained)
	}

	#[inline]
	fn calc_initial_hash(&self,b:&Banmen,
		ms:&HashMap<MochigomaKind,u32>,mg:&HashMap<MochigomaKind,u32>) -> (u64,u64) {
		self.kyokumenhash.calc_initial_hash(b,ms,mg)
	}
}
pub struct NNShogiPlayer {
	search:Arc<Search>,
	kyokumen:Option<Kyokumen>,
	mhash:u64,
	shash:u64,
	oute_kyokumen_map:KyokumenMap<u64,()>,
	kyokumen_map:KyokumenMap<u64,u32>,
	remaining_turns:u32,
	nna_filename:String,
	nnb_filename:String,
	bias_shake_shake:bool,
	learn_max_threads: usize,
	evalutor:Option<Arc<Intelligence>>,
	pub history:Vec<(Banmen,MochigomaCollections,u64,u64)>,
	count_of_move_started:u32,
	moved:bool,
}
impl fmt::Debug for NNShogiPlayer {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "NNShogiPlayer")
	}
}
impl NNShogiPlayer {
	pub fn new(nna_filename:String, nnb_filename:String, bias_shake_shake:bool,learn_max_threads:usize) -> NNShogiPlayer {
		NNShogiPlayer {
			search:Arc::new(Search::new()),
			kyokumen:None,
			mhash:0,
			shash:0,
			oute_kyokumen_map:KyokumenMap::new(),
			kyokumen_map:KyokumenMap::new(),
			remaining_turns:TURN_COUNT,
			nna_filename:nna_filename,
			nnb_filename:nnb_filename,
			bias_shake_shake,
			learn_max_threads:learn_max_threads,
			evalutor:None,
			history:Vec::new(),
			count_of_move_started:0,
			moved:false,
		}
	}
}
impl USIPlayer<CommonError> for NNShogiPlayer {
	const ID: &'static str = "nnshogi";
	const AUTHOR: &'static str = "jinpu";
	fn get_option_kinds(&mut self) -> Result<BTreeMap<String,SysEventOptionKind>,CommonError> {
		let mut kinds:BTreeMap<String,SysEventOptionKind> = BTreeMap::new();
		kinds.insert(String::from("USI_Hash"),SysEventOptionKind::Num);
		kinds.insert(String::from("USI_Ponder"),SysEventOptionKind::Bool);
		kinds.insert(String::from("MaxDepth"),SysEventOptionKind::Num);
		kinds.insert(String::from("MAX_PLY"),SysEventOptionKind::Num);
		kinds.insert(String::from("MAX_PLY_MATE"),SysEventOptionKind::Num);
		kinds.insert(String::from("MAX_PLY_TIMELIMIT"),SysEventOptionKind::Num);
		kinds.insert(String::from("TURN_COUNT"),SysEventOptionKind::Num);
		kinds.insert(String::from("MIN_TURN_COUNT"),SysEventOptionKind::Num);
		kinds.insert(String::from("Threads"),SysEventOptionKind::Num);
		kinds.insert(String::from("BaseDepth"),SysEventOptionKind::Num);
		kinds.insert(String::from("NetworkDelay"),SysEventOptionKind::Num);
		kinds.insert(String::from("DispEvaluteScore"),SysEventOptionKind::Bool);
		kinds.insert(String::from("AdjustDepth"),SysEventOptionKind::Bool);

		Ok(kinds)
	}
	fn get_options(&mut self) -> Result<BTreeMap<String,UsiOptType>,CommonError> {
		let mut options:BTreeMap<String,UsiOptType> = BTreeMap::new();
		options.insert(String::from("BaseDepth"),UsiOptType::Spin(1,100,Some(BASE_DEPTH as i64)));
		options.insert(String::from("MaxDepth"),UsiOptType::Spin(1,100,Some(MAX_DEPTH as i64)));
		options.insert(String::from("MAX_PLY"),UsiOptType::Spin(0,1000,Some(MAX_PLY as i64)));
		options.insert(String::from("MAX_PLY_MATE"),UsiOptType::Spin(0,10000,Some(0)));
		options.insert(String::from("MAX_PLY_TIMELIMIT"),UsiOptType::Spin(0,300000,Some(MAX_PLY_TIMELIMIT as i64)));
		options.insert(String::from("TURN_COUNT"),UsiOptType::Spin(0,1000,Some(TURN_COUNT as i64)));
		options.insert(String::from("MIN_TURN_COUNT"),UsiOptType::Spin(0,1000,Some(MIN_TURN_COUNT as i64)));
		options.insert(String::from("Threads"),UsiOptType::Spin(1,100,Some(MAX_THREADS as i64)));
		options.insert(String::from("NetworkDelay"),UsiOptType::Spin(0,10000,Some(NETWORK_DELAY as i64)));
		options.insert(String::from("DispEvaluteScore"),UsiOptType::Check(Some(DEFALUT_DISPLAY_EVALUTE_SCORE)));
		options.insert(String::from("AdjustDepth"),UsiOptType::Check(Some(DEFAULT_ADJUST_DEPTH)));

		Ok(options)
	}
	fn take_ready(&mut self) -> Result<(),CommonError> {
		match self.evalutor {
			Some(_) => (),
			None => {
				self.evalutor = Some(Arc::new(Intelligence::new(
										String::from("data"),
										self.nna_filename.clone(),
										self.nnb_filename.clone(),self.bias_shake_shake)));
			}
		}
		Ok(())
	}
	fn set_option(&mut self,name:String,value:SysEventOption) -> Result<(),CommonError> {
		match Arc::get_mut(&mut self.search) {
			Some(search) => {
				match &*name {
					"MaxDepth" => {
						search.max_depth = match value {
							SysEventOption::Num(depth) => {
								depth as u32
							},
							_ => MAX_DEPTH,
						};
					},
					"BaseDepth" => {
						search.base_depth = match value {
							SysEventOption::Num(depth) => {
								depth as u32
							},
							_ => BASE_DEPTH,
						};
					},
					"Threads" => {
						search.max_threads = match value {
							SysEventOption::Num(max_threads) => {
								max_threads as u32
							},
							_ => MAX_THREADS,
						};
					},
					"NetworkDelay" => {
						search.network_delay = match value {
							SysEventOption::Num(n) => {
								n as u32
							},
							_ => NETWORK_DELAY,
						}
					},
					"DispEvaluteScore" => {
						search.display_evalute_score =  match value {
							SysEventOption::Bool(b) => {
								b
							},
							_ => DEFALUT_DISPLAY_EVALUTE_SCORE,
						}
					},
					"AdjustDepth" => {
						search.adjust_depth =  match value {
							SysEventOption::Bool(b) => {
								b
							},
							_ => DEFAULT_ADJUST_DEPTH,
						}
					},
					"MAX_PLY" => {
						search.max_ply = match value {
							SysEventOption::Num(0) => {
								None
							},
							SysEventOption::Num(depth) => {
								Some(depth as u32)
							},
							_ => Some(MAX_PLY),
						};
					},
					"MAX_PLY_MATE" => {
						search.max_ply_mate = match value {
							SysEventOption::Num(0) => {
								None
							},
							SysEventOption::Num(depth) => {
								Some(depth as u32)
							},
							_ => None,
						};
					},
					"MAX_PLY_TIMELIMIT" => {
						search.max_ply_timelimit = match value {
							SysEventOption::Num(0) => {
								None
							},
							SysEventOption::Num(limit) => {
								Some(Duration::from_millis(limit as u64))
							},
							_ if MAX_PLY_TIMELIMIT > 0 => Some(Duration::from_millis(MAX_PLY_TIMELIMIT)),
							_ => None,
						};
					},
					"TURN_COUNT" => {
						search.turn_count = match value {
							SysEventOption::Num(c) => {
								c as u32
							},
							_ => TURN_COUNT,
						};
					},
					"MIN_TURN_COUNT" => {
						search.min_turn_count = match value {
							SysEventOption::Num(c) => {
								c as u32
							},
							_ => MIN_TURN_COUNT,
						};
					},
					_ => (),
				}
				Ok(())
			},
			None => {
				Err(CommonError::Fail(String::from(
					"Could not get a mutable reference of searcher."
				)))
			}
		}
	}
	fn newgame(&mut self) -> Result<(),CommonError> {
		self.kyokumen = None;
		self.history.clear();
		self.count_of_move_started = 0;
		self.remaining_turns = self.search.turn_count;
		Ok(())
	}
	fn set_position(&mut self,teban:Teban,banmen:Banmen,
					ms:HashMap<MochigomaKind,u32>,mg:HashMap<MochigomaKind,u32>,_:u32,m:Vec<Move>)
		-> Result<(),CommonError> {
		self.history.clear();
		self.kyokumen_map = KyokumenMap::new();

		let kyokumen_map:KyokumenMap<u64,u32> = KyokumenMap::new();
		let (mhash,shash) = self.search.calc_initial_hash(&banmen,&ms,&mg);

		let teban = teban;
		let state = State::new(banmen);

		let mc = MochigomaCollections::new(ms,mg);


		let history:Vec<(Banmen,MochigomaCollections,u64,u64)> = Vec::new();

		let (t,state,mc,r) = self.apply_moves(state,teban,
												mc,&m.into_iter()
													.map(|m| m.to_applied_move())
													.collect::<Vec<AppliedMove>>(),
												(mhash,shash,kyokumen_map,history),
												|s,t,banmen,mc,m,o,r| {
			let (prev_mhash,prev_shash,mut kyokumen_map,mut history) = r;

			let (mhash,shash) = match m {
				&Some(ref m) => {
					let mhash = s.search.calc_main_hash(prev_mhash,&t,&banmen,&mc,&m.to_move(),&o);
					let shash = s.search.calc_sub_hash(prev_shash,&t,&banmen,&mc,&m.to_move(),&o);

					match kyokumen_map.get(t,&mhash,&shash).unwrap_or(&0) {
						&c => {
							kyokumen_map.insert(t,mhash,shash,c+1);
						}
					};
					(mhash,shash)
				},
				&None => {
					(prev_mhash,prev_shash)
				}
			};

			history.push((banmen.clone(),mc.clone(),prev_mhash,prev_shash));
			(mhash,shash,kyokumen_map,history)
		});

		let (mhash,shash,kyokumen_map,history) = r;

		let mut oute_kyokumen_map:KyokumenMap<u64,()> = KyokumenMap::new();
		let mut current_teban = t.opposite();

		let mut current_cont = true;
		let mut opponent_cont = true;

		for h in history.iter().rev() {
			match &h {
				&(ref banmen,_, mhash,shash) => {
					if current_cont && Rule::is_mate(current_teban,&State::new(banmen.clone())) {
						oute_kyokumen_map.insert(current_teban,*mhash,*shash,());
					} else if !opponent_cont {
						break;
					} else {
						current_cont = false;
					}
				}
			}

			std::mem::swap(&mut current_cont, &mut opponent_cont);

			current_teban = current_teban.opposite();
		}

		self.kyokumen = Some(Kyokumen {
			state:state,
			mc:mc,
			teban:t
		});
		self.mhash = mhash;
		self.shash = shash;
		self.oute_kyokumen_map = oute_kyokumen_map;
		self.kyokumen_map = kyokumen_map;
		self.history = history;
		self.count_of_move_started += 1;
		self.moved = false;
		Ok(())
	}
	fn think<L,S>(&mut self,think_start_time:Instant,
			limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<UserEventQueue>>,
			info_sender:S,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<BestMove,CommonError>
		where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let (teban,state,mc) = self.kyokumen.as_ref().map(|k| (k.teban,&k.state,&k.mc)).ok_or(
			UsiProtocolError::InvalidState(
						String::from("Position information is not initialized."))
		)?;

		let limit = limit.to_instant(teban,think_start_time);
		let current_limit = limit.map(|l| think_start_time + (l  - think_start_time) / self.remaining_turns);

		let (mhash,shash) = (self.mhash.clone(), self.shash.clone());
		let kyokumen_map = self.kyokumen_map.clone();
		let oute_kyokumen_map = self.oute_kyokumen_map.clone();
		let base_depth = self.search.base_depth;

		match self.evalutor {
			Some(ref evalutor) => {
				let self_nn_snapshot = self.search.make_snapshot(false,evalutor,teban,state,mc)?;
				let opponent_nn_snapshot = self.search.make_snapshot(true,evalutor,teban.opposite(),state,mc)?;

				let prev_state:Option<Arc<State>> = None;
				let prev_mc:Option<Arc<MochigomaCollections>> = None;

				let strategy = if self.search.max_threads > 1 {
					Search::parallel_search
				} else {
					Search::single_search
				};

				let mut env = Environment::new(
													event_queue,
													evalutor.clone(),
													info_sender.clone(),
													on_error_handler.clone(),
													limit,current_limit);

				let mut event_dispatcher = self.search.create_event_dispatcher(&on_error_handler,&env.stop,&env.quited);
				let mut solver_event_dispatcher = self.search.create_event_dispatcher(&on_error_handler,&env.stop,&env.quited);

				let result = match self.search.negascout(
							&mut env,
							&mut event_dispatcher,
							&mut solver_event_dispatcher,
							&Arc::new(self_nn_snapshot),&Arc::new(opponent_nn_snapshot),
							teban,&Arc::new(state.clone()), Score::NEGINFINITE,
							Score::INFINITE, None,&Arc::new(mc.clone()),
							&prev_state,
							&prev_mc,
							None, &kyokumen_map,
							&mut Some(KyokumenMap::new()),
							&mut Some(KyokumenMap::new()),
							&oute_kyokumen_map,
							mhash,shash,
							base_depth, 1, base_depth,
							1,
							strategy) {
					Evaluation::Result(_,None) => {
						BestMove::Resign
					},
					Evaluation::Result(Score::NEGINFINITE,_) => {
						BestMove::Resign
					},
					Evaluation::Result(_,Some(m)) => {
						BestMove::Move(m,None)
					},
					Evaluation::Timeout(Some(Score::NEGINFINITE),_) => {
						BestMove::Resign
					}
					Evaluation::Timeout(_,Some(m)) => {
						BestMove::Move(m,None)
					}
					Evaluation::Timeout(_,None) if env.quited.load(atomic::Ordering::Acquire) => {
						BestMove::Abort
					},
					Evaluation::Timeout(_,None) => {
						BestMove::Resign
					},
					Evaluation::Error => {
						BestMove::Resign
					}
				};

				if self.remaining_turns > self.search.min_turn_count {
					self.remaining_turns -= 1;
				}

				if let BestMove::Move(m,_) = result {
					let h = match self.history.last() {
						Some(&(ref banmen,ref mc,mhash,shash)) => {
							let (next,nmc,o) = Rule::apply_move_none_check(&State::new(banmen.clone()),teban,mc,m.to_applied_move());
							self.moved = true;
							let mhash = self.search.calc_main_hash(mhash,&teban,banmen,mc,&m,&o);
							let shash = self.search.calc_sub_hash(shash,&teban,banmen,mc,&m,&o);
							(next.get_banmen().clone(),nmc.clone(),mhash,shash)
						},
						None => {
							return Err(CommonError::Fail(String::from("The history of banmen has not been set yet.")));
						}
					};
					self.history.push(h);
				}

				Ok(result)
			},
			None =>  {
				Err(CommonError::Fail(format!("evalutor is not initialized!")))
			}
		}
	}
	fn think_ponder<L,S>(&mut self,_:&UsiGoTimeLimit,_:Arc<Mutex<UserEventQueue>>,
			_:S,_:Arc<Mutex<OnErrorHandler<L>>>)
			-> Result<BestMove,CommonError> where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		unimplemented!();
	}
	fn think_mate<L,S>(&mut self,limit:&UsiGoMateTimeLimit,event_queue:Arc<Mutex<UserEventQueue>>,
			info_sender:S,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<CheckMate,CommonError>
		where L: Logger, S: InfoSender,
			Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let (teban,state,mc) = self.kyokumen.as_ref().map(|k| (k.teban,&k.state,&k.mc)).ok_or(
			UsiProtocolError::InvalidState(
						String::from("Position information is not initialized."))
		)?;

		let (mhash,shash) = (self.mhash.clone(), self.shash.clone());

		let limit = limit.to_instant(Instant::now());

		let search = Search::new();

		let mut info_sender = info_sender.clone();

		let mut on_searchstart = |depth,_| {
			search.send_seldepth(&mut info_sender, &on_error_handler, search.base_depth, 0 + depth);
		};

		let stop = Arc::new(AtomicBool::new(false));
		let quited = Arc::new(AtomicBool::new(false));

		let mut event_dispatcher = self.search.create_event_dispatcher(&on_error_handler, &stop, &quited);

		let mut solver = Solver::new();

		let network_delay = search.network_delay;

		let mut check_timelimit = move || {
			limit.map_or(false,|l| {
				let now = Instant::now();
					l < now ||
					l - now <= Duration::from_millis(network_delay as u64 + TIMELIMIT_MARGIN)
			})
		};

		match solver.checkmate(false,teban, state, mc,
									self.search.max_ply_mate.clone(),
									None,
									&mut KyokumenMap::new(),
									&mut Some(KyokumenMap::new()),
									&mut KyokumenMap::new(),
									&search,
									mhash, shash,
									&mut check_timelimit,
									&stop,
									&mut on_searchstart,
									&event_queue,
									&mut event_dispatcher) {
			MaybeMate::MateMoves(_,ref mvs) => {
				Ok(CheckMate::Moves(mvs.into_iter().map(|m| m.to_move()).collect::<Vec<Move>>()))
			},
			MaybeMate::Nomate => {
				Ok(CheckMate::Nomate)
			},
			MaybeMate::Continuation => {
				Err(CommonError::Fail(String::from("logic error.")))
			},
			_ => {
				Ok(CheckMate::Timeout)
			}
		}
	}
	fn on_stop(&mut self,_:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		Ok(())
	}
	fn gameover<L>(&mut self,s:&GameEndState,
		event_queue:Arc<Mutex<UserEventQueue>>,
		_:Arc<Mutex<OnErrorHandler<L>>>) -> Result<(),CommonError> where L: Logger, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let teban = self.kyokumen.as_ref().map(|k| k.teban).ok_or(
			UsiProtocolError::InvalidState(
						String::from("Information of 'teban' is not set."))
		)?;

		if self.count_of_move_started > 0 {
			let teban = if self.moved {
				teban.opposite()
			} else {
				teban
			};

			match self.evalutor {
				Some(ref mut evalutor) => {
					match Arc::get_mut(evalutor) {
						Some(evalutor)  => {
							let (a,b) = if self.bias_shake_shake {
								let mut rnd = rand::thread_rng();

								let a: f64 = rnd.gen();
								let b: f64 = 1f64 - a;

								(a,b)
							} else {
								(1f64,1f64)
							};

							evalutor.learning_by_training_data(teban,
															   self.history.clone(),
															   s,
															   self.learn_max_threads,
															   &move |s,t, ab| {
									match s {
										&GameEndState::Win if t == teban => {
											ab
										}
										&GameEndState::Win => {
											0f64
										},
										&GameEndState::Lose if t == teban => {
											0f64
										},
										&GameEndState::Lose => {
											ab
										},
										_ => 0.5f64
									}
								}, a,b,&*event_queue)?;
						},
						None => {
							return Err(CommonError::Fail(String::from(
								"Could not get a mutable reference of evaluator."
							)));
						}
					}
				},
				None => (),
			}
			self.history = Vec::new();
		}
		Ok(())
	}

	fn on_ponderhit(&mut self,_:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		Ok(())
	}

	fn on_quit(&mut self,_:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		Ok(())
	}

	fn quit(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
}
