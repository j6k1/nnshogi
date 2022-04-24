use std;
use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::thread;
use std::sync::{Arc, mpsc};
use std::sync::Mutex;
use error::*;
use std::time::{Instant,Duration};
use std::ops::Neg;
use std::ops::Add;
use std::ops::Sub;
use std::sync::atomic;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::atomic::AtomicU64;

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

use nn::{Intelligence};
use solver::*;
use std::sync::mpsc::Receiver;
use nncombinator::arr::{Arr, DiffArr};
use nncombinator::layer::{AskDiffInput, DiffInput, ForwardAll, ForwardDiff, PreTrain};
use usiagent::output::USIOutputWriter;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Evaluation {
	Result(Score,Option<AppliedMove>),
	Timeout(Option<Score>,Option<AppliedMove>),
	Error,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Score {
	NEGINFINITE,
	Value(i32),
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
impl Add<i32> for Score {
	type Output = Self;

	fn add(self, other:i32) -> Self::Output {
		match self {
			Score::Value(v) => Score::Value(v + other),
			Score::INFINITE => Score::INFINITE,
			Score::NEGINFINITE => Score::NEGINFINITE,
		}
	}
}
impl Sub<i32> for Score {
	type Output = Self;

	fn sub(self, other:i32) -> Self::Output {
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

type Strategy<L,S,NN,ST> = fn (&Arc<Search<NN>>,
						&mut Environment<L,S,NN>,
						&mut UserEventDispatcher<Search<NN>,CommonError,L>,
						&mut UserEventDispatcher<Solver<CommonError,NN>,CommonError,L>,
						&Arc<(ST,ST)>,&Arc<(ST,ST)>,
						Teban,&Arc<State>,
						&Vec<AppliedMove>,
						Score,Score,
						&Arc<MochigomaCollections>,
						&KyokumenMap<u64,u32>,
						&mut Option<KyokumenMap<u64,bool>>,
						&mut Option<KyokumenMap<u64,bool>>,
						&KyokumenMap<u64,()>,
						u64,u64,
						u32,u32,u32,u64,
						&Vec<(u32,LegalMove)>,bool) -> Evaluation;
pub struct Environment<L,S,NN> where L: Logger, S: InfoSender,
										NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
											PreTrain<f32> +	ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static {
	solver:Solver<CommonError,NN>,
	event_queue:Arc<Mutex<UserEventQueue>>,
	evalutor:Arc<Intelligence<NN>>,
	info_sender:S,
	on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
	limit:Option<Instant>,
	current_limit:Option<Instant>,
	stop:Arc<AtomicBool>,
	quited:Arc<AtomicBool>,
	kyokumen_score_map:KyokumenMap<u64,(Score,u32)>,
	nodes:Arc<AtomicU64>,
	think_start_time:Instant
}
impl<L,S,NN> Clone for Environment<L,S,NN>
	where L: Logger,
		  S: InfoSender,
	      NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
		  	  PreTrain<f32> + ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static {
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
			kyokumen_score_map:self.kyokumen_score_map.clone(),
			nodes:self.nodes.clone(),
			think_start_time:self.think_start_time.clone()
		}
	}
}
impl<L,S,NN> Environment<L,S,NN> where L: Logger, S: InfoSender,
									   NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
									   	   PreTrain<f32> + ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static {
	pub fn new(event_queue:Arc<Mutex<UserEventQueue>>,
			   evalutor:Arc<Intelligence<NN>>,
			   info_sender:S,
			   on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
			   think_start_time:Instant,
			   limit:Option<Instant>,
			   current_limit:Option<Instant>) -> Environment<L,S,NN> {
		let stop = Arc::new(AtomicBool::new(false));
		let quited = Arc::new(AtomicBool::new(false));

		Environment {
			solver:Solver::new(),
			event_queue:event_queue,
			evalutor:evalutor,
			info_sender:info_sender,
			on_error_handler:on_error_handler,
			think_start_time:think_start_time,
			limit:limit,
			current_limit:current_limit,
			stop:stop,
			quited:quited,
			kyokumen_score_map:KyokumenMap::new(),
			nodes:Arc::new(AtomicU64::new(0))
		}
	}
}
pub struct Search<NN>
	where NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
			  PreTrain<f32> + ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static {
	kyokumenhash:KyokumenHash<u64>,
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
	nn_type:PhantomData<NN>
}
impl<NN> Search<NN>
	where NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
			  PreTrain<f32> + ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static {
	pub fn new() -> Search<NN> {
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
			nn_type:PhantomData::<NN>
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

	fn timeout_expected<S,L>(&self,search:&Arc<Search<NN>>,env:&mut Environment<L,S,NN>,start_time:Instant,
			current_depth:u32,nodes:u64,processed_nodes:u32
		) -> bool where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		const RATE:u64 = 8;

		if current_depth <= 1 {
			false
		} else {
			let nodes = nodes / RATE.pow(current_depth);

			(nodes > u32::MAX as u64) || (current_depth > 1 && search.adjust_depth &&
				env.current_limit.map(|l| {
					env.think_start_time + ((Instant::now() - start_time) / processed_nodes) * nodes as u32 > l
				}).unwrap_or(false)
			) || env.current_limit.map(|l| Instant::now() >= l).unwrap_or(false)
		}
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

	fn send_info<L,S>(&self, env:&mut Environment<L,S,NN>,
					  depth:u32, seldepth:u32, pv:&Vec<AppliedMove>)
		where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

		if depth < seldepth {
			commands.push(UsiInfoSubCommand::Depth(depth));
			commands.push(UsiInfoSubCommand::SelDepth(seldepth));
		}

		commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
		commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
		commands.push(UsiInfoSubCommand::Time((Instant::now() - env.think_start_time).as_millis() as u64));

		match env.info_sender.send(commands) {
			Ok(_) => (),
			Err(ref e) => {
				let _ = env.on_error_handler.lock().map(|h| h.call(e));
			}
		}
	}

	fn send_score<L,S>(&self, info_sender:&mut S,
				  on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
				  teban:Teban,
				  s:Score)
		where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		if self.display_evalute_score {
			let teban_str = match teban {
				Teban::Sente => "sente",
				Teban::Gote =>  "gote"
			};
			match &s {
				Score::INFINITE => {
					self.send_message(info_sender, on_error_handler, &format!("evalute score = inifinite. ({0})",teban_str));
				},
				Score::NEGINFINITE => {
					self.send_message(info_sender, on_error_handler, &format!("evalute score = neginifinite. ({0})",teban_str));
				},
				Score::Value(s) => {
					self.send_message(info_sender, on_error_handler, &format!("evalute score =  {0: >17} ({1})",s,teban_str));
				}
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

	fn make_snapshot(&self,is_self:bool,evalutor:&Arc<Intelligence<NN>>,teban:Teban,state:&State,mc:&MochigomaCollections)
		-> Result<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack),CommonError> {

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

	fn evalute_by_diff<L,S>(&self,evalutor:&Arc<Intelligence<NN>>,
								self_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
								opponent_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
								teban:Teban,state:&Option<&Arc<State>>,
								mc:&Option<&Arc<MochigomaCollections>>,m:Option<AppliedMove>,
					info_sender:&mut S,on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<(Evaluation,(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack),
				   			  (<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)),CommonError>
		where L: Logger, S: InfoSender,
			  Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

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
			Some(m) => {
				m
			},
			None => {
				self.send_message(info_sender, on_error_handler, "m is none!");
				return Err(CommonError::Fail(String::from("m is none!")));
			}
		};

		let (s,self_snapshot,opponent_snapshot) = {
			let m = m.to_move();
			let (ss, self_snapshot) = evalutor.evalute_by_diff(&self_snapshot, false, teban, state.get_banmen(), mc, &m)?;
			let (_, opponent_snapshot) = evalutor.evalute_by_diff(&opponent_snapshot, true, teban.opposite(), state.get_banmen(), mc, &m)?;
			(ss,self_snapshot,opponent_snapshot)
		};

		Ok((Evaluation::Result(Score::Value(s),Some(m)),self_snapshot,opponent_snapshot))
	}

	fn evalute_by_self_diff<L,S>(&self,evalutor:&Arc<Intelligence<NN>>,
							is_self:bool,
							self_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
							teban:Teban,state:&Option<&Arc<State>>,
							mc:&Option<&Arc<MochigomaCollections>>,m:Option<AppliedMove>,
							info_sender:&mut S,on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>)
							-> Result<(Evaluation,(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)),CommonError>
		where L: Logger, S: InfoSender,
			  Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

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
			Some(m) => {
				m
			},
			None => {
				self.send_message(info_sender, on_error_handler, "m is none!");
				return Err(CommonError::Fail(String::from("m is none!")));
			}
		};

		let (s,self_snapshot) = evalutor.evalute_by_diff(&self_snapshot, is_self, teban, state.get_banmen(), mc, &m.to_move())?;

		Ok((Evaluation::Result(Score::Value(s),Some(m)),self_snapshot))
	}

	#[allow(unused)]
	fn evalute_score_by_diff<L,S>(&self,evalutor:&Arc<Intelligence<NN>>,
							is_self:bool,
							self_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
							teban:Teban,state:&Option<&Arc<State>>,
							mc:&Option<&Arc<MochigomaCollections>>,m:Option<AppliedMove>,
							info_sender:&mut S,on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<Score,CommonError> where L: Logger, S: InfoSender,
										   Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
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
			Some(m) => {
				m
			},
			None => {
				self.send_message(info_sender, on_error_handler, "m is none!");
				return Err(CommonError::Fail(String::from("m is none!")));
			}
		};

		let s = {
			let m = m.to_move();
			let (ss, _) = evalutor.evalute_by_diff(&self_snapshot, is_self, teban, state.get_banmen(), mc, &m)?;
			ss
		};

		Ok(Score::Value(s))
	}

	#[allow(dead_code)]
	fn evalute_by_snapshot(&self,evalutor:&Arc<Intelligence<NN>>,
						   self_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>)
		-> Score {

		let ss = evalutor.evalute_by_snapshot(self_snapshot);

		Score::Value(ss)
	}

	fn negascout<L,S>(self:&Arc<Self>,
								env:&mut Environment<L,S,NN>,
					  			event_dispatcher:&mut UserEventDispatcher<Search<NN>,CommonError,L>,
					  			solver_event_dispatcher:&mut UserEventDispatcher<Solver<CommonError,NN>,CommonError,L>,
								self_nn_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
								opponent_nn_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
								teban:Teban,state:&Arc<State>,
								alpha:Score,beta:Score,
								m:Option<AppliedMove>,mc:&Arc<MochigomaCollections>,
								pv:&Vec<AppliedMove>,
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
								strategy:Strategy<L,S,NN,<NN as PreTrain<f32>>::OutStack>,
	) -> Evaluation where L: Logger, S: InfoSender,
						  Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		if let None = env.kyokumen_score_map.get(teban,&mhash,&shash) {
			env.nodes.fetch_add(1,atomic::Ordering::Release);
		}

		if let Some(ObtainKind::Ou) = obtained {
			return Evaluation::Result(Score::NEGINFINITE, None);
		}

		if self.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
			self.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
			return Evaluation::Timeout(None, None);
		}

		if let Some(&(s,d)) = env.kyokumen_score_map.get(teban, &mhash, &shash) {
			match s {
				Score::INFINITE => {
					self.send_message(&mut env.info_sender, &env.on_error_handler, "score corresponding to the hash was found in the map. value is infinite.");
					return Evaluation::Result(s, None);
				},
				Score::NEGINFINITE => {
					self.send_message(&mut env.info_sender, &env.on_error_handler, "score corresponding to the hash was found in the map. value is neginfinite.");
					return Evaluation::Result(s, None);
				},
				Score::Value(s) if d >= depth => {
					self.send_message(&mut env.info_sender, &env.on_error_handler, &format!("score corresponding to the hash was found in the map. value is {}.",s));
					return Evaluation::Result(Score::Value(s), None);
				},
				_ => ()
			}
		}

		if (depth == 0 || current_depth > self.max_depth) && !Rule::is_mate(teban.opposite(),&*state) {
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
					return Evaluation::Result(Score::INFINITE,Some(mvs[0].to_applied_move()));
				},
				MaybeMate::MateMoves(_,_) => {
					return Evaluation::Result(Score::INFINITE,None);
				},
				_ => ()
			}

			let r = self.evalute_score_by_diff(&env.evalutor,
											   false,
											   &self_nn_snapshot,
											   teban,
											   &prev_state.as_ref(), &prev_mc.as_ref(),
											   m, &mut env.info_sender, &env.on_error_handler);

			match r {
				Ok(s) => {
					return Evaluation::Result(s, None);
				},
				Err(ref e) => {
					let _ = env.on_error_handler.lock().map(|h| h.call(e));
					return Evaluation::Error;
				}
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
				let r = self.evalute_score_by_diff(&env.evalutor,false,
										   &self_nn_snapshot,
										   teban,
										   &prev_state.as_ref(), &prev_mc.as_ref(),
										   m, &mut env.info_sender, &env.on_error_handler);

				match r {
					Ok(s) => {
						return Evaluation::Result(s, None);
					},
					Err(ref e) => {
						let _ = env.on_error_handler.lock().map(|h| h.call(e));
						return Evaluation::Error;
					}
				}
			} else {
				(mvs,true)
			}
		} else {
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
			return Evaluation::Timeout(None,Some(mvs[0].to_applied_move()));
		} else if mvs.len() == 1 {
			let self_nn_snapshot = match self.evalute_by_self_diff(&env.evalutor,
									   false,
									   &self_nn_snapshot,
									   teban,
									   &prev_state.as_ref(), &prev_mc.as_ref(),
									   m, &mut env.info_sender, &env.on_error_handler) {
				Ok((_,s)) => {
					s
				},
				Err(ref e) => {
					let _ = env.on_error_handler.lock().map(|h| h.call(e));
					return Evaluation::Error;
				}
			};

			match self.evalute_score_by_diff(&env.evalutor,
											   true,
											   &Arc::new(self_nn_snapshot),
											   teban,
											   &Some(&state),
											   &Some(&mc),
											   Some(mvs[0].to_applied_move()),
											   &mut env.info_sender, &env.on_error_handler) {
				Ok(s) => {
					return Evaluation::Result(s, Some(mvs[0].to_applied_move()));
				},
				Err(ref e) => {
					let _ = env.on_error_handler.lock().map(|h| h.call(e));
					return Evaluation::Error;
				}
			};
		}

		let (current_self_nn_ss,current_opponent_nn_ss) = if prev_state.is_some() {
			let (self_nn_snapshot,opponent_nn_snapshot) = match self.evalute_by_diff(&env.evalutor,
																					  &self_nn_snapshot,
																					  &opponent_nn_snapshot,
																					  teban,
																					  &prev_state.as_ref(), &prev_mc.as_ref(),
																					  m, &mut env.info_sender, &env.on_error_handler) {
				Ok((_, sss, oss)) => {
					(Arc::new(sss), Arc::new(oss))
				},
				Err(ref e) => {
					let _ = env.on_error_handler.lock().map(|h| h.call(e));
					return Evaluation::Error;
				}
			};
			(Some(self_nn_snapshot), Some(opponent_nn_snapshot))
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

		let _ = event_dispatcher.dispatch_events(self,&*env.event_queue);

		if self.timelimit_reached(&env.limit) || env.stop.load(atomic::Ordering::Acquire) {
			self.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
			return Evaluation::Timeout(None,Some(mvs[0].to_applied_move()));
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
					teban,state,pv,
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
						m:LegalMove,mhash:u64,shash:u64,
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

			let mhash = self.calc_main_hash(mhash,teban,state.get_banmen(),mc,m.to_applied_move(),&o);
			let shash = self.calc_sub_hash(shash,teban,state.get_banmen(),mc,m.to_applied_move(),&o);

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
			5 | 10 => depth + 1,
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

	fn single_search<L,S>(search:&Arc<Search<NN>>,
								env:&mut Environment<L,S,NN>,
						  		event_dispatcher:&mut UserEventDispatcher<Search<NN>,CommonError,L>,
						  		solver_event_dispatcher:&mut UserEventDispatcher<Solver<CommonError,NN>,CommonError,L>,
								self_nn_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
								opponent_nn_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
								teban:Teban,state:&Arc<State>,pv:&Vec<AppliedMove>,
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
		-> Evaluation where L: Logger, S: InfoSender,
							Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let mut scoreval = Score::NEGINFINITE;
		let mut best_move:Option<AppliedMove> = None;

		let mut processed_nodes:u32 = 0;
		let start_time = Instant::now();

		for &(priority,m) in mvs {
			let mut pv = pv.clone();
			pv.push(m.to_applied_move());

			processed_nodes += 1;
			let nodes = node_count * mvs.len() as u64 - processed_nodes as u64;

			match search.startup_strategy(teban,state,mc,m,
											mhash,shash,
										 	priority,
											oute_kyokumen_map,
											current_kyokumen_map,
											depth,responded_oute) {
				Some(r) => {
					let (depth,obtained,mhash,shash,
						 oute_kyokumen_map,
						 current_kyokumen_map,
						 is_sennichite) = r;

					let prev_state = Some(state.clone());
					let prev_mc = Some(mc.clone());

					let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

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
									best_move = Some(m.to_applied_move());
									if scoreval >= beta {
										return Evaluation::Result(scoreval,best_move);
									}
								}

								if alpha < scoreval {
									alpha = scoreval;
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
									-b,-alpha,Some(m.to_applied_move()),&mc,
									&pv,
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
												best_move = Some(m.to_applied_move());
											}
										}

										return match best_move {
											Some(best_move) => Evaluation::Timeout(Some(scoreval),Some(best_move)),
											None => Evaluation::Timeout(Some(scoreval),Some(m.to_applied_move())),
										};
									},
									Evaluation::Result(s,_) => {
										if let Some(&(_,d)) = env.kyokumen_score_map.get(teban.opposite(),&mhash,&shash) {
											if d < depth {
												env.kyokumen_score_map.insert(teban.opposite(), mhash, shash, (s,depth));
											}
										} else {
											env.kyokumen_score_map.insert(teban.opposite(), mhash, shash, (s,depth));
										}

										if let Some(&(_,d)) = env.kyokumen_score_map.get(teban,&mhash,&shash) {
											 if d < depth {
												env.kyokumen_score_map.insert(teban, mhash, shash, (-s,depth));
											}
										} else {
											env.kyokumen_score_map.insert(teban, mhash, shash, (-s,depth));
										}

										if -s > scoreval {
											search.send_info(env, base_depth,current_depth,&pv);
											search.send_score(&mut env.info_sender,&env.on_error_handler,teban,-s);

											scoreval = -s;
											best_move = Some(m.to_applied_move());
											if scoreval >= beta {
												return Evaluation::Result(scoreval,best_move);
											}
										}
										if alpha < -s {
											alpha = -s;
										} else {
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
							None => Evaluation::Timeout(Some(scoreval),Some(m.to_applied_move())),
						};
					} else if search.timeout_expected(search,env,start_time,current_depth,nodes,processed_nodes) {
						search.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
						return Evaluation::Result(scoreval,best_move);
					}
				},
				None => (),
			}
		}

		Evaluation::Result(scoreval,best_move)
	}

	fn parallel_search<L,S>(search:&Arc<Search<NN>>,
								env:&mut Environment<L,S,NN>,
								event_dispatcher:&mut UserEventDispatcher<Search<NN>,CommonError,L>,
								_:&mut UserEventDispatcher<Solver<CommonError,NN>,CommonError,L>,
								self_nn_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
								opponent_nn_snapshot:&Arc<(<NN as PreTrain<f32>>::OutStack,<NN as PreTrain<f32>>::OutStack)>,
								teban:Teban,state:&Arc<State>,pv:&Vec<AppliedMove>,
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
		-> Evaluation where L: Logger, S: InfoSender,
							Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let mut scoreval = Score::NEGINFINITE;
		let mut best_move:Option<AppliedMove> = None;

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
								best_move = Some(m);
							}
						}

						match search.termination(&receiver, threads, env, scoreval, best_move) {
							Evaluation::Error => {
								return Evaluation::Error;
							},
							Evaluation::Timeout(None,_) => {
								return Evaluation::Timeout(None,best_move);
							},
							Evaluation::Timeout(Some(scoreval),_) |
							Evaluation::Result(scoreval,_) => {
								return Evaluation::Timeout(Some(scoreval),best_move);
							}
						}
					},
					(Evaluation::Result(s,_),m) => {
						if let Some(&(_,d)) = env.kyokumen_score_map.get(teban.opposite(),&mhash,&shash) {
							if d < depth {
								env.kyokumen_score_map.insert(teban.opposite(), mhash, shash, (s,depth));
							}
						} else {
							env.kyokumen_score_map.insert(teban.opposite(), mhash, shash, (s,depth));
						}

						if let Some(&(_,d)) = env.kyokumen_score_map.get(teban,&mhash,&shash) {
							if d < depth {
								env.kyokumen_score_map.insert(teban, mhash, shash, (-s,depth));
							}
						} else {
							env.kyokumen_score_map.insert(teban, mhash, shash, (-s,depth));
						}

						if -s > scoreval {
							let mut pv = pv.clone();
							pv.push(m);

							search.send_info(env, base_depth,current_depth,&pv);
							search.send_score(&mut env.info_sender,&env.on_error_handler,teban,-s);

							scoreval = -s;
							best_move = Some(m);
							if scoreval >= beta {
								return search.termination(&receiver, threads, env, scoreval, best_move);
							}
							if alpha < scoreval {
								alpha = scoreval;
							}
						}

						if search.timeout_expected(search,env,start_time,current_depth,nodes,processed_nodes) {
							search.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
							return search.termination(&receiver, threads, env, scoreval, best_move);
						}
					},
					(Evaluation::Error,_) => {
						let _ = search.termination(&receiver, threads, env, scoreval, best_move);
						return Evaluation::Error;
					}
				}
			} else if let Some(&(priority,m)) = it.next() {
				let mut pv = pv.clone();
				pv.push(m.to_applied_move());

				match search.startup_strategy(teban,state,mc,m,
												mhash,shash,
											 	priority,
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
										best_move = Some(m);
										if alpha < scoreval {
											alpha = scoreval;
										}
										if scoreval >= beta {
											return search.termination(&receiver, threads, env, scoreval, best_move);
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
											-b,-a,Some(m),&mc,
											&pv,
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
							let r = search.termination(&receiver, threads, env, scoreval, best_move);

							return match r {
								Evaluation::Result(scoreval,_) => {
									Evaluation::Timeout(Some(scoreval),best_move)
								},
								Evaluation::Timeout(scoreval,_) => {
									Evaluation::Timeout(scoreval,best_move)
								},
								Evaluation::Error => {
									Evaluation::Error
								}
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
									best_move = Some(m);
								}
							}

							match search.termination(&receiver, threads, env, scoreval, best_move) {
								Evaluation::Timeout(None,_) => {
									return Evaluation::Timeout(None,best_move)
								},
								Evaluation::Timeout(Some(scoreval),_) |
								Evaluation::Result(scoreval,_) => {
									return Evaluation::Timeout(Some(scoreval),best_move);
								},
								Evaluation::Error => {
									return Evaluation::Error;
								}
							}
						},
						(Evaluation::Result(s,_),m) => {
							if -s > scoreval {
								scoreval = -s;
								best_move = Some(m);
								if alpha < scoreval {
									alpha = scoreval;
								}
								if scoreval >= beta {
									return search.termination(&receiver, threads, env, scoreval, best_move);
								}
							}

							let nodes = node_count * mvs_count - processed_nodes as u64;

							if search.timeout_expected(search,env,start_time,current_depth,nodes,processed_nodes) {
								search.send_message(&mut env.info_sender, &env.on_error_handler, "think timeout!");
								return search.termination(&receiver, threads, env, scoreval, best_move);
							}
						},
						(Evaluation::Error,_) => {
							let _ = search.termination(&receiver, threads, env, scoreval, best_move);
							return Evaluation::Error;
						}
					}
				},
				Err(ref e) => {
					let _ = env.on_error_handler.lock().map(|h| h.call(e));
					return Evaluation::Error;
				}
			};
		}

		Evaluation::Result(scoreval,best_move)
	}

	fn termination<L,S>(&self,r:&Receiver<(Evaluation,AppliedMove)>,
				   threads:u32,env:&mut Environment<L,S,NN>,
				   score:Score,best_move:Option<AppliedMove>) -> Evaluation where L: Logger, S: InfoSender {
		env.stop.store(true,atomic::Ordering::Release);

		let mut score = score;
		let mut best_move = best_move;
		let mut has_error = false;

		for _ in threads..self.max_threads {
			match r.recv() {
				Ok((r,m)) => {
					match r {
						Evaluation::Result(s, _) |
						Evaluation::Timeout(Some(s), _) => {
							if -s > score {
								score = -s;
								best_move = Some(m);
							}
						},
						Evaluation::Error => {
							has_error = true;
						}
						_ => ()
					};
				},
				Err(ref e) => {
					let _ = env.on_error_handler.lock().map(|h| h.call(e));
					return Evaluation::Error;
				}
			}
		}

		if has_error {
			Evaluation::Error
		} else if best_move.is_none() {
			Evaluation::Result(Score::NEGINFINITE, best_move)
		} else {
			Evaluation::Result(score, best_move)
		}
	}

	#[inline]
	pub fn calc_main_hash(&self,h:u64,t:Teban,b:&Banmen,mc:&MochigomaCollections,m:AppliedMove,obtained:&Option<MochigomaKind>) -> u64 {
		self.kyokumenhash.calc_main_hash(h,t,b,mc,m,obtained)
	}

	#[inline]
	pub fn calc_sub_hash(&self,h:u64,t:Teban,b:&Banmen,mc:&MochigomaCollections,m:AppliedMove,obtained:&Option<MochigomaKind>) -> u64 {
		self.kyokumenhash.calc_sub_hash(h,t,b,mc,m,obtained)
	}

	#[inline]
	fn calc_initial_hash(&self,b:&Banmen,
		ms:&Mochigoma,mg:&Mochigoma) -> (u64,u64) {
		self.kyokumenhash.calc_initial_hash(b,ms,mg)
	}
}
pub struct NNShogiPlayer<NN>
	where NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
			  PreTrain<f32> + ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static,
	{
	search:Arc<Search<NN>>,
	kyokumen:Option<Kyokumen>,
	mhash:u64,
	shash:u64,
	oute_kyokumen_map:KyokumenMap<u64,()>,
	kyokumen_map:KyokumenMap<u64,u32>,
	remaining_turns:u32,
	evalutor:Option<Arc<Intelligence<NN>>>,
	evalutor_creator: Box<dyn Fn() -> Result<Intelligence<NN>,ApplicationError> + Send + 'static>,
	pub history:Vec<(Banmen,MochigomaCollections,u64,u64)>,
	count_of_move_started:u32,
	moved:bool,
}
impl<NN> fmt::Debug for NNShogiPlayer<NN>
	where NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
			  PreTrain<f32> + ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static,
	{
		fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "NNShogiPlayer")
	}
}
impl<NN> NNShogiPlayer<NN>
	where NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
			  PreTrain<f32> + ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static,
	{

	pub fn new<C: Fn() -> Result<Intelligence<NN>,ApplicationError> + Send + 'static>(evalutor_creator:C)
		-> NNShogiPlayer<NN> {

		NNShogiPlayer {
			search:Arc::new(Search::new()),
			kyokumen:None,
			mhash:0,
			shash:0,
			oute_kyokumen_map:KyokumenMap::new(),
			kyokumen_map:KyokumenMap::new(),
			remaining_turns:TURN_COUNT,
			evalutor:None,
			evalutor_creator:Box::new(evalutor_creator),
			history:Vec::new(),
			count_of_move_started:0,
			moved:false,
		}
	}
}
impl<NN> USIPlayer<CommonError> for NNShogiPlayer<NN>
	where NN: ForwardAll<Input=DiffInput<DiffArr<f32,2517>,f32,2517,256>,Output=Arr<f32,1>> +
			  PreTrain<f32> + ForwardDiff<f32> + AskDiffInput<f32,DiffInput=Arr<f32,256>> + Send + Sync + 'static,
	{

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
	fn take_ready<W,L>(&mut self, _:OnKeepAlive<W,L>)
		-> Result<(),CommonError> where W: USIOutputWriter + Send + 'static,
							  L: Logger + Send + 'static {
		match self.evalutor {
			Some(_) => (),
			None => {
				self.evalutor = Some(Arc::new((self.evalutor_creator)()?));
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
					ms:Mochigoma,mg:Mochigoma,_:u32,m:Vec<Move>)
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
				&Some(m) => {
					let mhash = s.search.calc_main_hash(prev_mhash,t,&banmen,&mc,m,&o);
					let shash = s.search.calc_sub_hash(prev_shash,t,&banmen,&mc,m,&o);

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
	fn think<L,S,P>(&mut self,think_start_time:Instant,
			limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<UserEventQueue>>,
			info_sender:S,periodically_info:P,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<BestMove,CommonError>
		where L: Logger + Send + 'static,
			  S: InfoSender,
			  P: PeriodicallyInfo {
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
																think_start_time.clone(),
													limit,current_limit);

				let mut event_dispatcher = self.search.create_event_dispatcher(&on_error_handler,&env.stop,&env.quited);
				let mut solver_event_dispatcher = self.search.create_event_dispatcher(&on_error_handler,&env.stop,&env.quited);

				let _pinfo_sender = {
					let nodes = env.nodes.clone();
					let think_start_time = think_start_time.clone();
					let on_error_handler = env.on_error_handler.clone();

					periodically_info.start(100,move || {
						let mut commands = vec![];
						commands.push(UsiInfoSubCommand::Nodes(nodes.load(Ordering::Acquire)));

						let sec = (Instant::now() - think_start_time).as_secs();

						if sec > 0 {
							commands.push(UsiInfoSubCommand::Nps(nodes.load(Ordering::Acquire) / sec));
						}

						commands
					}, &on_error_handler)
				};

				let result = match self.search.negascout(
							&mut env,
							&mut event_dispatcher,
							&mut solver_event_dispatcher,
							&Arc::new(self_nn_snapshot),&Arc::new(opponent_nn_snapshot),
							teban,&Arc::new(state.clone()), Score::NEGINFINITE,
							Score::INFINITE, None,&Arc::new(mc.clone()),
							&Vec::new(),
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
						BestMove::Move(m.to_move(),None)
					},
					Evaluation::Timeout(Some(Score::NEGINFINITE),_) => {
						BestMove::Resign
					}
					Evaluation::Timeout(_,Some(m)) => {
						BestMove::Move(m.to_move(),None)
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
							let m = m.to_applied_move();

							let (next,nmc,o) = Rule::apply_move_none_check(&State::new(banmen.clone()),teban,mc,m);
							self.moved = true;
							let mhash = self.search.calc_main_hash(mhash,teban,banmen,mc,m,&o);
							let shash = self.search.calc_sub_hash(shash,teban,banmen,mc,m,&o);
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
	fn think_ponder<L,S,P>(&mut self,_:&UsiGoTimeLimit,_:Arc<Mutex<UserEventQueue>>,
			_:S,_:P,_:Arc<Mutex<OnErrorHandler<L>>>)
			-> Result<BestMove,CommonError> where L: Logger + Send + 'static, S: InfoSender,
												  P: PeriodicallyInfo + Send + 'static {
		unimplemented!();
	}

	fn think_mate<L,S,P>(&mut self,limit:&UsiGoMateTimeLimit,event_queue:Arc<Mutex<UserEventQueue>>,
			info_sender:S,_:P,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<CheckMate,CommonError>
		where L: Logger + Send + 'static,
			  S: InfoSender,
			  P: PeriodicallyInfo {
		let (teban,state,mc) = self.kyokumen.as_ref().map(|k| (k.teban,&k.state,&k.mc)).ok_or(
			UsiProtocolError::InvalidState(
						String::from("Position information is not initialized."))
		)?;

		let (mhash,shash) = (self.mhash.clone(), self.shash.clone());

		let limit = limit.to_instant(Instant::now());

		let search = Search::<NN>::new();

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
	fn gameover<L>(&mut self,_:&GameEndState,
		_:Arc<Mutex<UserEventQueue>>, _:Arc<Mutex<OnErrorHandler<L>>>) -> Result<(),CommonError> where L: Logger, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
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
