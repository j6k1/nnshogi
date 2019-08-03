use std;
use std::collections::HashMap;
use std::fmt;
use rand;
use rand::Rng;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use error::*;
use std::num::Wrapping;
use std::time::{Instant,Duration};
use std::cmp::Ordering;
use std::ops::Neg;
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

const KOMA_KIND_MAX:usize = KomaKind::Blank as usize;
const MOCHIGOMA_KIND_MAX:usize = MochigomaKind::Hisha as usize;
const MOCHIGOMA_MAX:usize = 18;
const SUJI_MAX:usize = 9;
const DAN_MAX:usize = 9;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Evaluation {
	Result(Score,Option<Move>),
	Timeout(Option<Move>),
	Error,
}
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum OuteEvaluation {
	Result(i32),
	Timeout,
	Error,
}
#[derive(Clone, Copy, PartialEq, Debug)]
enum Score {
	NEGINFINITE,
	Value(i64),
	INFINITE,
}
impl Neg for Score {
	type Output = Score;

	fn neg(self) -> Score {
		match self {
			Score::INFINITE => Score::NEGINFINITE,
			Score::NEGINFINITE => Score::INFINITE,
			Score::Value(v) => Score::Value(-v),
		}
	}
}
impl PartialOrd for Score {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering>	{
		Some(match *self {
			Score::INFINITE if *other == Score::INFINITE => {
				Ordering::Equal
			},
			Score::INFINITE => Ordering::Greater,
			Score::NEGINFINITE if *other == Score::NEGINFINITE => {
				Ordering::Equal
			},
			Score::NEGINFINITE => Ordering::Less,
			Score::Value(l) => {
				match *other {
					Score::Value(r) => l.partial_cmp(&r)?,
					Score::INFINITE => Ordering::Less,
					Score::NEGINFINITE => Ordering::Greater,
				}
			},
		})
	}
}
const BASE_DEPTH:u32 = 2;
const MAX_DEPTH:u32 = 6;
const TIMELIMIT_MARGIN:u64 = 50;
const NETWORK_DELAY:u32 = 1100;
const DEFALUT_DISPLAY_EVALUTE_SCORE:bool = false;
const MAX_THREADS:u32 = 1;

type Strategy<L,S> = fn (&Arc<Search>,
						&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
						&mut USIEventDispatcher<UserEventKind,
										UserEvent,Search,L,CommonError>,
						&Arc<Intelligence>,&mut S,
						&Arc<Mutex<OnErrorHandler<L>>>,
						&Arc<(SnapShot,SnapShot)>,&Arc<(SnapShot,SnapShot)>,
						Teban,&Arc<State>,Score,Score,
						&Arc<MochigomaCollections>,
						Option<ObtainKind>,
						&KyokumenMap<u64,u32>,
						&Arc<RwLock<KyokumenMap<u64,bool>>>,
						&KyokumenMap<u64,()>,
						u64,u64,Option<Instant>,
						u32,u32,u32,
						&Arc<AtomicBool>,&Arc<AtomicBool>,
						&Vec<(u32,LegalMove)>,bool) -> Evaluation;

struct Search {
	kyokumen_hash_seeds:[[u64; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1],
	mochigoma_hash_seeds:[[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2],
	base_depth:u32,
	max_depth:u32,
	max_threads:u32,
	network_delay:u32,
	display_evalute_score:bool,
}
impl Search {
	pub fn new() -> Search {
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

		Search {
			kyokumen_hash_seeds:kyokumen_hash_seeds,
			mochigoma_hash_seeds:mochigoma_hash_seeds,
			base_depth:BASE_DEPTH,
			max_depth:MAX_DEPTH,
			max_threads:MAX_THREADS,
			network_delay:NETWORK_DELAY,
			display_evalute_score:DEFALUT_DISPLAY_EVALUTE_SCORE,
		}
	}

	fn create_event_dispatcher<L>(&self,on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,stop:&Arc<AtomicBool>,quited:&Arc<AtomicBool>)
		-> USIEventDispatcher<UserEventKind,UserEvent,Search,L,CommonError> where L: Logger {

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

			event_dispatcher.add_handler(UserEventKind::Stop, move |_,e| {
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
				on_error_handler.lock().map(|h| h.call(e)).is_err();
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
				on_error_handler.lock().map(|h| h.call(e)).is_err();
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
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}
	}
	*/

	fn make_snapshot(&self,evalutor:&Arc<Intelligence>,teban:Teban,state:&State,mc:&MochigomaCollections)
		-> Result<(SnapShot,SnapShot),CommonError> {

		let r = evalutor.make_snapshot(teban,state.get_banmen(),mc)?;

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

	fn evalute_by_diff<L,S>(&self,evalutor:&Arc<Intelligence>,snapshot:&Arc<(SnapShot,SnapShot)>,
								is_opposite:bool,teban:Teban,state:&Option<&Arc<State>>,
								mc:&Option<&Arc<MochigomaCollections>>,m:&Option<Move>,
					info_sender:&mut S,on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<(Evaluation,(SnapShot,SnapShot)),CommonError> where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
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

		let (s,snapshot) = evalutor.evalute_by_diff(&snapshot,is_opposite,teban,state.get_banmen(),mc,m)?;

		if self.display_evalute_score {
			self.send_message(info_sender, on_error_handler, &format!("evalute score = {}",s));
		}

		Ok((Evaluation::Result(Score::Value(s),Some(m.clone())),snapshot))
	}

	fn evalute_by_snapshot(&self,evalutor:&Arc<Intelligence>,snapshot:&Arc<(SnapShot,SnapShot)>)
		-> Evaluation {

		let s = evalutor.evalute_by_snapshot(snapshot);

		Evaluation::Result(Score::Value(s),None)
	}

	fn alphabeta<L,S>(&self,
			this:&Arc<Search>,
			event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
								event_dispatcher:&mut USIEventDispatcher<UserEventKind,
													UserEvent,Search,L,CommonError>,
								evalutor:&Arc<Intelligence>,
								info_sender:&mut S,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								self_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								opponent_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								teban:Teban,state:&Arc<State>,
								alpha:Score,beta:Score,
								m:Option<Move>,mc:&Arc<MochigomaCollections>,
								prev_state:&Option<Arc<State>>,
								prev_mc:&Option<Arc<MochigomaCollections>>,
								obtained:Option<ObtainKind>,
								current_kyokumen_map:&KyokumenMap<u64,u32>,
								already_oute_map:&Arc<RwLock<KyokumenMap<u64,bool>>>,
								oute_kyokumen_map:&KyokumenMap<u64,()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,
								depth:u32,current_depth:u32,base_depth:u32,
								stop:&Arc<AtomicBool>,
								quited:&Arc<AtomicBool>,
								strategy:Strategy<L,S>
	) -> Evaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		if current_depth > base_depth {
			self.send_seldepth(info_sender, on_error_handler, base_depth, current_depth);
		}

		if let Some(ObtainKind::Ou) = obtained {
			return Evaluation::Result(Score::NEGINFINITE,None);
		}

		let _ = event_dispatcher.dispatch_events(self,&*event_queue);

		if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return Evaluation::Timeout(None);
		}

		let (mvs,responded_oute) = if Rule::is_mate(teban.opposite(),&*state) {
			if depth == 0 || current_depth == self.max_depth {
				if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(None);
				}
			}

			let mvs = Rule::respond_oute_only_moves_all(teban, &*state, &*mc);

			if mvs.len() == 0 {
				return Evaluation::Result(Score::NEGINFINITE,None);
			} else if depth == 0 || current_depth == self.max_depth {
				return self.evalute_by_snapshot(evalutor,self_nn_snapshot);
			} else {
				(mvs,true)
			}
		} else {
			let oute_mvs = Rule::oute_only_moves_all(teban,&*state,&*mc);

			if oute_mvs.len() > 0 {
				if let LegalMove::To(ref m) = oute_mvs[0] {
					if m.obtained() == Some(ObtainKind::Ou) {
						return Evaluation::Result(Score::INFINITE,Some(oute_mvs[0].to_move()));
					}
				}
			}

			let _ = event_dispatcher.dispatch_events(self,&*event_queue);

			if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
				self.send_message(info_sender, on_error_handler, "think timeout!");
				return Evaluation::Timeout(None)
			}

			if depth == 0 || current_depth == self.max_depth {
				if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(None);
				} else {
					return self.evalute_by_snapshot(evalutor,self_nn_snapshot);
				}
			}

			for m in &oute_mvs {
				let o = match m {
					LegalMove::To(ref m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
					_ => None,
				};

				let mhash = self.calc_main_hash(mhash,&teban,state.get_banmen(),&*mc,&m.to_move(),&o);
				let shash = self.calc_sub_hash(shash,&teban,state.get_banmen(),&*mc,&m.to_move(),&o);

				let mut oute_kyokumen_map = oute_kyokumen_map.clone();

				if let Some(_) =  oute_kyokumen_map.get(teban,&mhash,&shash) {
					continue;
				} else {
					oute_kyokumen_map.insert(teban,mhash,shash,());
				}

				match already_oute_map.write() {
					Ok(mut already_oute_map) => {
						if let None = already_oute_map.get(teban,&mhash,&shash) {
							already_oute_map.insert(teban,mhash,shash,false);
						}
					},
					Err(ref e) => {
						let _ = on_error_handler.lock().map(|h| h.call(e));
						return Evaluation::Error;
					}
				}

				let mut current_kyokumen_map = current_kyokumen_map.clone();

				match current_kyokumen_map.get(teban,&mhash,&shash).unwrap_or(&0) {
					&c if c >= 3 => {
						continue;
					},
					&c => {
						current_kyokumen_map.insert(teban,mhash,shash,c+1);
					}
				}

				let next = Rule::apply_move_none_check(&*state,teban,&*mc,m.to_applied_move());

				match next {
					(ref next,ref mc,_) if !Rule::is_mate(teban.opposite(),next) => {
						let is_put_fu = match m {
							LegalMove::Put(ref m) if m.kind() == MochigomaKind::Fu => true,
							_ => false,
						};

						let _ = event_dispatcher.dispatch_events(self,&*event_queue);

						if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
							self.send_message(info_sender, on_error_handler, "think timeout!");
							return Evaluation::Timeout(Some(m.to_move()));
						}

						match self.respond_oute_only(event_queue,
															event_dispatcher,
															info_sender,
															on_error_handler,
															teban.opposite(),next,mc,
															&current_kyokumen_map,
															already_oute_map,
															&oute_kyokumen_map,
															mhash,shash,limit,
															current_depth+1,
															base_depth,stop,
															) {
							OuteEvaluation::Result(d) if d >= 0 &&
								!(is_put_fu && d - current_depth as i32 == 2) => {
								return Evaluation::Result(Score::INFINITE,Some(m.to_move()));
							},
							OuteEvaluation::Timeout => {
								return Evaluation::Timeout(Some(m.to_move()));
							},
							_ => (),
						}
					},
					_ => (),
				}

				let _ = event_dispatcher.dispatch_events(self,&*event_queue);

				if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(Some(m.to_move()));
				}
			}

			if oute_mvs.len() == 0 {
				let _ = event_dispatcher.dispatch_events(self,&*event_queue);

				if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(None);
				}
			} else {
				if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(Some(oute_mvs[0].to_move()));
				}
			}

			let mvs:Vec<LegalMove> = Rule::legal_moves_all(teban, &*state, &*mc);

			(mvs,false)
		};

		let (current_self_nn_ss,current_opponent_nn_ss) = if prev_state.is_some() {
			let self_nn_snapshot = 	match self.evalute_by_diff(evalutor,
																&self_nn_snapshot,
																true,
																teban,
																&prev_state.as_ref(),&prev_mc.as_ref(),
																&m,info_sender,on_error_handler) {
				Ok((_,ss)) => Arc::new(ss),
				Err(ref e) => {
					on_error_handler.lock().map(|h| h.call(e)).is_err();
					return Evaluation::Error;
				}
			};

			let opponent_nn_snapshot = match self.evalute_by_diff(evalutor,
																	&opponent_nn_snapshot,
																	false,
																	teban.opposite(),
																	&prev_state.as_ref(),&prev_mc.as_ref(),
																	&m,info_sender,on_error_handler) {
				Ok((_,ss)) => Arc::new(ss),
				Err(ref e) => {
					on_error_handler.lock().map(|h| h.call(e)).is_err();
					return Evaluation::Error;
				}
			};
			(Some(self_nn_snapshot),Some(opponent_nn_snapshot))
		} else {
			(None,None)
		};

		let self_nn_snapshot = match current_self_nn_ss {
			Some(ref ss) => ss,
			None => self_nn_snapshot,
		};

		let opponent_nn_snapshot = match current_opponent_nn_ss {
			Some(ref ss) => ss,
			None => opponent_nn_snapshot,
		};

		if mvs.len() == 0 {
			return Evaluation::Result(Score::NEGINFINITE,None);
		} else if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return Evaluation::Timeout(Some(mvs[0].to_move()));
		} else if mvs.len() == 1 {
			let r = match self.evalute_by_diff(evalutor,&self_nn_snapshot,false,teban,&Some(&state), &Some(&mc), &Some(mvs[0].to_move()), info_sender, on_error_handler) {
				Ok((r,_)) => r,
				Err(ref e) => {
					on_error_handler.lock().map(|h| h.call(e)).is_err();
					return Evaluation::Error;
				}
			};

			return r;
		}

		let _ = event_dispatcher.dispatch_events(self,&*event_queue);

		if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return Evaluation::Timeout(Some(mvs[0].to_move()));
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

		strategy(this,event_queue,
					event_dispatcher,
					evalutor,
					info_sender,
					on_error_handler,
					&self_nn_snapshot,
					&opponent_nn_snapshot,
					teban,state,
					alpha,beta,mc,
					obtained,current_kyokumen_map,
					already_oute_map,
					oute_kyokumen_map,
					mhash,shash,limit,depth,
					current_depth,base_depth,
					stop,quited,
					&mvs,
					responded_oute)
	}

	fn startup_strategy(&self,teban:Teban,state:&State,mc:&MochigomaCollections,
						m:&LegalMove,mhash:u64,shash:u64,
						priority:u32,
						oute_kyokumen_map:&KyokumenMap<u64,()>,
						current_kyokumen_map:&KyokumenMap<u64,u32>,
						depth:u32,responded_oute:bool)
		-> Option<(u32,u64,u64,KyokumenMap<u64,()>,KyokumenMap<u64,u32>,bool)> {

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

		let depth = match priority {
			10 | 5 => depth + 1,
			_ if responded_oute => depth + 1,
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

		Some((depth,mhash,shash,oute_kyokumen_map,current_kyokumen_map,is_sennichite))
	}

	fn single_search<L,S>(search:&Arc<Search>,
								event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
								event_dispatcher:&mut USIEventDispatcher<UserEventKind,
													UserEvent,Search,L,CommonError>,
								evalutor:&Arc<Intelligence>,
								info_sender:&mut S,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								self_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								opponent_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								teban:Teban,state:&Arc<State>,
								mut alpha:Score,beta:Score,
								mc:&Arc<MochigomaCollections>,
								obtained:Option<ObtainKind>,
								current_kyokumen_map:&KyokumenMap<u64,u32>,
								already_oute_map:&Arc<RwLock<KyokumenMap<u64,bool>>>,
								oute_kyokumen_map:&KyokumenMap<u64,()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,
								depth:u32,current_depth:u32,base_depth:u32,
								stop:&Arc<AtomicBool>,
								quited:&Arc<AtomicBool>,
								mvs:&Vec<(u32,LegalMove)>,
								responded_oute:bool)
		-> Evaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let mut scoreval = Score::NEGINFINITE;
		let mut best_move:Option<Move> = None;

		for (priority,m) in mvs {
			match search.startup_strategy(teban,state,mc,m,
											mhash,shash,
										 	*priority,
											oute_kyokumen_map,
											current_kyokumen_map,
											depth,responded_oute) {
				Some(r) => {
					let (depth,mut mhash,mut shash,
						 mut oute_kyokumen_map,
						 mut current_kyokumen_map,
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

							match search.alphabeta(
								search,
								event_queue,
								event_dispatcher,
								evalutor,
								info_sender,
								on_error_handler,
								opponent_nn_snapshot,
								self_nn_snapshot,
								teban.opposite(),&Arc::new(state),
								-beta,-alpha,Some(m.to_move()),&Arc::new(mc),
								&prev_state,&prev_mc,
								obtained,&current_kyokumen_map,
								already_oute_map,
								&oute_kyokumen_map,
								mhash,shash,limit,depth-1,
								current_depth+1,base_depth,
								stop,quited,Search::single_search) {

								Evaluation::Timeout(_) => {
									return match best_move {
										Some(best_move) => Evaluation::Timeout(Some(best_move)),
										None => Evaluation::Timeout(Some(m.to_move())),
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
								},
								Evaluation::Error => {
									return Evaluation::Error
								}
							}
						}
					}

					let _ = event_dispatcher.dispatch_events(search,&*event_queue);

					if search.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
						search.send_message(info_sender, on_error_handler, "think timeout!");
						return match best_move {
							Some(best_move) => Evaluation::Timeout(Some(best_move)),
							None => Evaluation::Timeout(Some(m.to_move())),
						};
					}
				},
				None => (),
			}
		}

		Evaluation::Result(scoreval,best_move)
	}

	fn parallel_search<L,S>(search:&Arc<Search>,
								event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
								event_dispatcher:&mut USIEventDispatcher<UserEventKind,
													UserEvent,Search,L,CommonError>,
								evalutor:&Arc<Intelligence>,
								info_sender:&mut S,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								self_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								opponent_nn_snapshot:&Arc<(SnapShot,SnapShot)>,
								teban:Teban,state:&Arc<State>,
								mut alpha:Score,beta:Score,
								mc:&Arc<MochigomaCollections>,
								obtained:Option<ObtainKind>,
								current_kyokumen_map:&KyokumenMap<u64,u32>,
								already_oute_map:&Arc<RwLock<KyokumenMap<u64,bool>>>,
								oute_kyokumen_map:&KyokumenMap<u64,()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,
								depth:u32,current_depth:u32,base_depth:u32,
								stop:&Arc<AtomicBool>,
								quited:&Arc<AtomicBool>,
								mvs:&Vec<(u32,LegalMove)>,
								responded_oute:bool)
		-> Evaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let mut scoreval = Score::NEGINFINITE;
		let mut best_move:Option<Move> = None;

		let (sender,receiver):(_,Receiver<(Evaluation,AppliedMove)>) = mpsc::channel();
		let mut threads = search.max_threads;

		for (priority,m) in mvs {
			match search.startup_strategy(teban,state,mc,m,
											mhash,shash,
										 	*priority,
											oute_kyokumen_map,
											current_kyokumen_map,
											depth,responded_oute) {
				Some(r) => {
					let (depth,mut mhash,mut shash,
						 mut oute_kyokumen_map,
						 mut current_kyokumen_map,
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
										search.termination(receiver, threads, stop);
										return Evaluation::Result(scoreval,best_move);
									}
								}

								continue;
							}

							if threads == 0 {
								let r = match receiver.recv() {
									Ok(r) => r,
									Err(ref e) => {
										on_error_handler.lock().map(|h| h.call(e)).is_err();
										search.termination(receiver, threads, stop);
										return Evaluation::Error;
									}
								};

								threads += 1;

								match r {
									(Evaluation::Timeout(_),m) => {
										match best_move {
											Some(best_move) => {
												search.termination(receiver, threads, stop);
												return Evaluation::Timeout(Some(best_move));
											},
											None => {
												search.termination(receiver, threads, stop);
												return Evaluation::Timeout(Some(m.to_move()));
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
												search.termination(receiver, threads, stop);
												return Evaluation::Result(scoreval,best_move);
											}
										}
									},
									(Evaluation::Error,_) => {
										search.termination(receiver, threads, stop);
										return Evaluation::Error;
									}
								}
							} else {
								let search = search.clone();
								let event_queue = event_queue.clone();
								let evalutor = evalutor.clone();
								let mut info_sender = info_sender.clone();
								let on_error_handler = on_error_handler.clone();
								let opponent_nn_snapshot = opponent_nn_snapshot.clone();
								let self_nn_snapshot = self_nn_snapshot.clone();
								let state = Arc::new(state);
								let mc = Arc::new(mc);
								let already_oute_map = already_oute_map.clone();
								let limit = limit.clone();
								let stop = stop.clone();
								let quited = quited.clone();

								let sender = sender.clone();

								let _ = thread::spawn(move || {
									let mut event_dispatcher = search.create_event_dispatcher(&on_error_handler, &stop, &quited);

									let search = search.clone();

									let r = {
										let evalutor = evalutor;

										search.alphabeta(
											&search,
											&event_queue,
											&mut event_dispatcher,
											&evalutor,
											&mut info_sender,
											&on_error_handler,
											&opponent_nn_snapshot,
											&self_nn_snapshot,
											teban.opposite(),&state,
											-beta,-alpha,Some(m.to_move()),&mc,
											&prev_state,&prev_mc,
											obtained,&current_kyokumen_map,
											&already_oute_map,
											&oute_kyokumen_map,
											mhash,shash,limit,depth-1,
											current_depth+1,base_depth,
											&stop,&quited,Search::single_search)
									};

									let _ = sender.send((r,m));
								});

								threads -= 1;
							}
						}
					}

					let _ = event_dispatcher.dispatch_events(search,&*event_queue);

					if search.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
						search.send_message(info_sender, on_error_handler, "think timeout!");
						search.termination(receiver, threads, stop);

						return match best_move {
							Some(best_move) => Evaluation::Timeout(Some(best_move)),
							None => Evaluation::Timeout(Some(m.to_move())),
						};
					}
				},
				None => (),
			}
		}

		while threads < search.max_threads {
			match receiver.recv() {
				Ok(r) => {
					threads += 1;

					match r {
						(Evaluation::Timeout(_),m) => {
							match best_move {
								Some(best_move) => {
									search.termination(receiver, threads, stop);
									return Evaluation::Timeout(Some(best_move));
								},
								None => {
									search.termination(receiver, threads, stop);
									return Evaluation::Timeout(Some(m.to_move()));
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
									search.termination(receiver, threads, stop);
									return Evaluation::Result(scoreval,best_move);
								}
							}
						},
						(Evaluation::Error,_) => {
							search.termination(receiver, threads, stop);
							return Evaluation::Error;
						}
					}
				},
				Err(ref e) => {
					threads += 1;
					on_error_handler.lock().map(|h| h.call(e)).is_err();
					search.termination(receiver, threads, stop);
					return Evaluation::Error;
				}
			};
		}

		Evaluation::Result(scoreval,best_move)
	}

	fn termination(&self,r:Receiver<(Evaluation,AppliedMove)>,threads:u32,stop:&Arc<AtomicBool>) {
		stop.store(true,atomic::Ordering::Release);

		for _ in threads..self.max_threads {
			let _ = r.recv();
		}
	}

	fn respond_oute_only<L,S>(&self,
								event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
								event_dispatcher:&mut USIEventDispatcher<UserEventKind,
													UserEvent,Search,L,CommonError>,
								info_sender:&mut S,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								teban:Teban,state:&State,
								mc:&MochigomaCollections,
								current_kyokumen_map:&KyokumenMap<u64,u32>,
								already_oute_map:&Arc<RwLock<KyokumenMap<u64,bool>>>,
								oute_kyokumen_map:&KyokumenMap<u64,()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,
								current_depth:u32,
								base_depth:u32,stop:&Arc<AtomicBool>)
		-> OuteEvaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let mvs = Rule::respond_oute_only_moves_all(teban,&state, mc);

		//self.send_seldepth(info_sender, on_error_handler, base_depth, current_depth);

		let _ = event_dispatcher.dispatch_events(self,&*event_queue);

		if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return OuteEvaluation::Timeout;
		}

		if mvs.len() == 0 {
			return OuteEvaluation::Result(current_depth as i32)
		} else {
			let mut maxdepth = -1;

			for m in &mvs {
				let o = match m {
					&LegalMove::To(ref m) => {
						m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
					},
					_ => None,
				};
				let mhash = self.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = self.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				let mut current_kyokumen_map = current_kyokumen_map.clone();

				match current_kyokumen_map.get(teban,&mhash,&shash).unwrap_or(&0) {
					&c if c >= 3 => {
						continue;
					},
					&c => {
						current_kyokumen_map.insert(teban,mhash,shash,c+1);
					}
				}

				let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

				match next {
					(ref next,ref mc,_) => {
						let oute_kyokumen_map = {
							let (x,y,kind) = match m {
								LegalMove::To(ref mv) => {
									let (dx,dy) = mv.dst().square_to_point();
									let kind = next.get_banmen().0[dy as usize][dx as usize];

									(dx,dy,kind)
								},
								LegalMove::Put(ref mv) => {
									let kind = KomaKind::from((teban,mv.kind()));
									let (dx,dy) = mv.dst().square_to_point();

									(dx,dy,kind)
								}
							};

							let ps = next.get_part();

							if Rule::is_mate_with_partial_state_and_point_and_kind(teban,ps,x,y,kind) ||
							   Rule::is_mate_with_partial_state_repeat_move_kinds(teban,ps) {

								let mut oute_kyokumen_map = oute_kyokumen_map.clone();

								match oute_kyokumen_map.get(teban,&mhash,&shash) {
									Some(_) => {
										continue;
									},
									None => {
										oute_kyokumen_map.insert(teban,mhash,shash,());
									},
								}
							}

							oute_kyokumen_map
						};

						match self.oute_only(event_queue,
												event_dispatcher,
												info_sender,
												on_error_handler,
												teban.opposite(),next,mc,
												&current_kyokumen_map,
												already_oute_map,
												oute_kyokumen_map,
												mhash,shash,limit,
												current_depth+1,base_depth,stop) {
							OuteEvaluation::Result(-1) => {
								return OuteEvaluation::Result(-1);
							},
							OuteEvaluation::Result(d) => {
								if d > maxdepth {
									maxdepth = d;
								}
							},
							OuteEvaluation::Timeout => {
								return OuteEvaluation::Timeout;
							},
							OuteEvaluation::Error => {
								return OuteEvaluation::Error;
							}
						}
					}
				}

				let _ = event_dispatcher.dispatch_events(self,&*event_queue);

				if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return OuteEvaluation::Timeout;
				}
			}

			OuteEvaluation::Result(maxdepth)
		}
	}

	fn oute_only<L,S>(&self,
						event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
						event_dispatcher:&mut USIEventDispatcher<UserEventKind,
													UserEvent,Search,L,CommonError>,
						info_sender:&mut S,
						on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
						teban:Teban,state:&State,
						mc:&MochigomaCollections,
						current_kyokumen_map:&KyokumenMap<u64,u32>,
						already_oute_map:&Arc<RwLock<KyokumenMap<u64,bool>>>,
						oute_kyokumen_map:&KyokumenMap<u64,()>,
						mhash:u64,shash:u64,
						limit:Option<Instant>,
						current_depth:u32,
						base_depth:u32,
						stop:&Arc<AtomicBool>)
		-> OuteEvaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let mvs = Rule::oute_only_moves_all(teban, &state, mc);

		//self.send_seldepth(info_sender, on_error_handler, base_depth, current_depth);

		let _ = event_dispatcher.dispatch_events(self,&*event_queue);

		if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return OuteEvaluation::Timeout;
		}

		if mvs.len() == 0 {
			OuteEvaluation::Result(-1)
		} else {
			for m in &mvs {
				match m {
					&LegalMove::To(ref m) => {
						if let Some(ObtainKind::Ou) = m.obtained() {
							return OuteEvaluation::Result(current_depth as i32);
						}
					},
					_ => ()
				}

				let is_put_fu = match m {
					&LegalMove::Put(ref m) if m.kind() == MochigomaKind::Fu => true,
					_ => false,
				};

				let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

				let o = match m {
					&LegalMove::To(ref m) => {
						m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
					},
					_ => None,
				};

				let mhash = self.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = self.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				let completed = match already_oute_map.read() {
					Ok(already_oute_map) => {
						already_oute_map.get(teban,&mhash,&shash).map(|&b| b).unwrap_or(false)
					},
					Err(ref e) => {
						let _ = on_error_handler.lock().map(|h| h.call(e));
						return OuteEvaluation::Error;
					}
				};

				if completed {
					return OuteEvaluation::Result(-1);
				} else {
					match already_oute_map.write() {
						Ok(mut already_oute_map) => {
							if let None = already_oute_map.get(teban,&mhash,&shash) {
								already_oute_map.insert(teban,mhash,shash,false);
							}
						},
						Err(ref e) => {
							let _ = on_error_handler.lock().map(|h| h.call(e));
							return OuteEvaluation::Error;
						}
					}
				}

				let mut current_kyokumen_map = current_kyokumen_map.clone();

				match current_kyokumen_map.get(teban,&mhash,&shash).unwrap_or(&0) {
					&c if c >= 3 => {
						continue;
					},
					&c => {
						current_kyokumen_map.insert(teban,mhash,shash,c+1);
					}
				}

				let mut oute_kyokumen_map = oute_kyokumen_map.clone();

				match oute_kyokumen_map.get(teban,&mhash,&shash) {
					Some(()) => {
						continue;
					},
					None => {
						oute_kyokumen_map.insert(teban,mhash,shash,());
					}
				}

				match next {
					(ref next,ref mc,_) => {
						match self.respond_oute_only(event_queue,
														event_dispatcher,
														info_sender,
														on_error_handler,
														teban.opposite(),next,mc,
														&current_kyokumen_map,
														already_oute_map,
														&oute_kyokumen_map,
														mhash,shash,limit,
														current_depth+1,base_depth,stop) {
							OuteEvaluation::Result(-1) => {
								return OuteEvaluation::Result(-1);
							}
							OuteEvaluation::Result(d) if d >= 0 &&
														!(is_put_fu && d - current_depth as i32 == 2)=> {
								return OuteEvaluation::Result(d);
							},
							OuteEvaluation::Timeout => {
								return OuteEvaluation::Timeout;
							},
							_ => (),
						}
					}
				}

				match already_oute_map.write() {
					Ok(mut already_oute_map) => {
						already_oute_map.insert(teban,mhash,shash,true);
					},
					Err(ref e) => {
						let _ = on_error_handler.lock().map(|h| h.call(e));
						return OuteEvaluation::Error;
					}
				}

				let _ = event_dispatcher.dispatch_events(self,&*event_queue);

				if self.timelimit_reached(&limit) || stop.load(atomic::Ordering::Acquire) {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return OuteEvaluation::Timeout;
				}
			}
			OuteEvaluation::Result(-1)
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

	fn calc_main_hash(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
		self.calc_hash(h,t,b,mc,m,obtained,|h,v| h ^ v, |h,v| h ^ v)
	}

	fn calc_sub_hash(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
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
pub struct NNShogiPlayer {
	search:Arc<Search>,
	kyokumen:Option<Arc<Kyokumen>>,
	mhash:u64,
	shash:u64,
	oute_kyokumen_map:KyokumenMap<u64,()>,
	kyokumen_map:KyokumenMap<u64,u32>,
	nna_filename:String,
	nnb_filename:String,
	learning_mode:bool,
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
	pub fn new(nna_filename:String,nnb_filename:String,learning_mode:bool) -> NNShogiPlayer {
		NNShogiPlayer {
			search:Arc::new(Search::new()),
			kyokumen:None,
			mhash:0,
			shash:0,
			oute_kyokumen_map:KyokumenMap::new(),
			kyokumen_map:KyokumenMap::new(),
			nna_filename:nna_filename,
			nnb_filename:nnb_filename,
			learning_mode:learning_mode,
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
	fn get_option_kinds(&mut self) -> Result<HashMap<String,SysEventOptionKind>,CommonError> {
		let mut kinds:HashMap<String,SysEventOptionKind> = HashMap::new();
		kinds.insert(String::from("USI_Hash"),SysEventOptionKind::Num);
		kinds.insert(String::from("USI_Ponder"),SysEventOptionKind::Bool);
		kinds.insert(String::from("MaxDepth"),SysEventOptionKind::Num);
		kinds.insert(String::from("Threads"),SysEventOptionKind::Num);
		kinds.insert(String::from("BaseDepth"),SysEventOptionKind::Num);
		kinds.insert(String::from("NetworkDelay"),SysEventOptionKind::Num);
		kinds.insert(String::from("DispEvaluteScore"),SysEventOptionKind::Bool);

		Ok(kinds)
	}
	fn get_options(&mut self) -> Result<HashMap<String,UsiOptType>,CommonError> {
		let mut options:HashMap<String,UsiOptType> = HashMap::new();
		options.insert(String::from("BaseDepth"),UsiOptType::Spin(1,100,Some(BASE_DEPTH as i64)));
		options.insert(String::from("MaxDepth"),UsiOptType::Spin(1,100,Some(MAX_DEPTH as i64)));
		options.insert(String::from("Threads"),UsiOptType::Spin(1,100,Some(MAX_THREADS as i64)));
		options.insert(String::from("NetworkDelay"),UsiOptType::Spin(0,10000,Some(NETWORK_DELAY as i64)));
		options.insert(String::from("DispEvaluteScore"),UsiOptType::Check(Some(DEFALUT_DISPLAY_EVALUTE_SCORE)));
		Ok(options)
	}
	fn take_ready(&mut self) -> Result<(),CommonError> {
		match self.evalutor {
			Some(_) => (),
			None => {
				self.evalutor = Some(Arc::new(Intelligence::new(
										String::from("data"),
										self.nna_filename.clone(),
										self.nnb_filename.clone(),self.learning_mode)));
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
					_ => (),
				}
				Ok(())
			},
			None => {
				Err(CommonError::Fail(String::from(
					"Could not get a mutable reference of evaluator."
				)))
			}
		}
	}
	fn newgame(&mut self) -> Result<(),CommonError> {
		self.kyokumen = None;
		self.history.clear();
		self.count_of_move_started = 0;
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

		let (t,state,mc,r) = self.apply_moves(teban,state,
												mc,m.into_iter()
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

		self.kyokumen = Some(Arc::new(Kyokumen {
			state:state,
			mc:mc,
			teban:t
		}));
		self.mhash = mhash;
		self.shash = shash;
		self.oute_kyokumen_map = oute_kyokumen_map;
		self.kyokumen_map = kyokumen_map;
		self.history = history;
		self.count_of_move_started += 1;
		self.moved = false;
		Ok(())
	}
	fn think<L,S>(&mut self,limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
			info_sender:S,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<BestMove,CommonError>
		where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let kyokumen = self.kyokumen.as_ref().map(|k| k.clone()).ok_or(
			UsiProtocolError::InvalidState(
						String::from("Position information is not initialized."))
		)?;

		let teban = kyokumen.teban;
		let state = &kyokumen.state;
		let mc = &kyokumen.mc;

		let limit = limit.to_instant(teban);
		let (mhash,shash) = (self.mhash.clone(), self.shash.clone());
		let kyokumen_map = self.kyokumen_map.clone();
		let oute_kyokumen_map = self.oute_kyokumen_map.clone();
		let base_depth = self.search.base_depth;

		let mut info_sender = info_sender;

		match self.evalutor {
			Some(ref evalutor) => {
				let self_nn_snapshot = self.search.make_snapshot(evalutor,teban,state,mc)?;
				let opponent_nn_snapshot = self.search.make_snapshot(evalutor,teban.opposite(),state,mc)?;

				let prev_state:Option<Arc<State>> = None;
				let prev_mc:Option<Arc<MochigomaCollections>> = None;

				let stop = Arc::new(AtomicBool::new(false));
				let quited = Arc::new(AtomicBool::new(false));

				let mut event_dispatcher = self.search.create_event_dispatcher(&on_error_handler, &stop, &quited);

				let strategy = if self.search.max_threads > 1 {
					Search::parallel_search
				} else {
					Search::single_search
				};

				let result = match self.search.alphabeta(&self.search.clone(),
							&event_queue,
							&mut event_dispatcher,
							evalutor,
							&mut info_sender, &on_error_handler,
							&Arc::new(self_nn_snapshot),&Arc::new(opponent_nn_snapshot),
							teban,&Arc::new(state.clone()), Score::NEGINFINITE,
							Score::INFINITE, None,&Arc::new(mc.clone()),
							&prev_state,
							&prev_mc,
							None, &kyokumen_map,
							&Arc::new(RwLock::new(KyokumenMap::new())),
							&oute_kyokumen_map,
							mhash,shash,
							limit, base_depth, 0, base_depth,
							&stop,
							&quited,
							strategy) {
					Evaluation::Result(_,Some(m)) => {
						BestMove::Move(m,None)
					},
					Evaluation::Result(_,None) => {
						BestMove::Resign
					},
					Evaluation::Timeout(Some(m)) => {
						BestMove::Move(m,None)
					}
					Evaluation::Timeout(None) if quited.load(atomic::Ordering::Acquire) => {
						BestMove::Abort
					},
					Evaluation::Timeout(None) => {
						BestMove::Resign
					},
					Evaluation::Error => {
						self.search.send_message(&mut info_sender, &on_error_handler.clone(), "error!");
						BestMove::Resign
					}
				};

				if let BestMove::Move(m,_) = result {
					let h = match self.history.last() {
						Some(&(ref banmen,ref mc,mhash,shash)) => {
							let (next,nmc,o) = Rule::apply_move_none_check(&State::new(banmen.clone()),teban,mc,m.to_applied_move());
							self.moved = true;
							let mut mhash = self.search.calc_main_hash(mhash,&teban,banmen,mc,&m,&o);
							let mut shash = self.search.calc_sub_hash(shash,&teban,banmen,mc,&m,&o);
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
	fn think_mate<L,S>(&mut self,_:&UsiGoMateTimeLimit,_:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
			_:S,_:Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<CheckMate,CommonError>
		where L: Logger, S: InfoSender,
			Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		Ok(CheckMate::NotiImplemented)
	}
	fn on_stop(&mut self,_:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		Ok(())
	}
	fn gameover<L>(&mut self,s:&GameEndState,
		event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
		_:Arc<Mutex<OnErrorHandler<L>>>) -> Result<(),CommonError> where L: Logger, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let kyokumen = self.kyokumen.as_ref().map(|k| k.clone()).ok_or(
			UsiProtocolError::InvalidState(
						String::from("Information of 'teban' is not set."))
		)?;

		let teban = kyokumen.teban;

		if self.count_of_move_started > 0 {
			let (teban,last_teban) = if self.moved {
				(teban,teban.opposite())
			} else {
				(teban,teban)
			};

			match self.evalutor {
				Some(ref mut evalutor) => {
					match Arc::get_mut(evalutor) {
						Some(evalutor)  => {
							evalutor.learning(true,teban,last_teban,self.history.clone(),s,&*event_queue)?;
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

	fn on_quit(&mut self,_:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		Ok(())
	}

	fn quit(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
}
