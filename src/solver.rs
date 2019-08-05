use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic;
use std::sync::atomic::AtomicBool;

use usiagent::shogi::*;
use usiagent::rule::*;
use usiagent::hash::*;
use usiagent::logger::*;
use usiagent::event::*;
use usiagent::error::PlayerError;
use usiagent::TryFrom;
use player::Search;

#[derive(Debug)]
pub enum SolveError {
	Timeout(String),
}
impl fmt::Display for SolveError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			SolveError::Timeout(ref s) => write!(f, "{}",s),
		}
	}
}
impl error::Error for SolveError {
	fn description(&self) -> &str {
		match *self {
			SolveError::Timeout(_) => "Time limit reached while checking if checkmate",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			SolveError::Timeout(_) => None,
		}
	}
}
pub enum MaybeMate {
	Mate(u32),
	Nomate,
	Timeout,
}
pub struct Solver<E> where E: PlayerError {
	error_type:PhantomData<E>
}
impl<E> Solver<E> where E: PlayerError {
	pub fn new()
	-> Solver<E> {
		Solver {
			error_type:PhantomData::<E>
		}
	}
	pub fn checkmate<L,F,S>(& mut self,
							teban:Teban,state:&State,
							mc:&MochigomaCollections,
							m:LegalMove,
							oute_kyokumen_map:&mut KyokumenMap<u64,()>,
							already_oute_kyokumen_map:&mut KyokumenMap<u64,bool>,
							current_kyokumen_map:&mut KyokumenMap<u64,u32>,
							hasher:&Search,
							mhash:u64,shash:u64,
							current_depth:u32,
							check_timelimit:&mut F,
							stop:&Arc<AtomicBool>,
							on_searchstart:&mut S,
							event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
							event_dispatcher:&mut USIEventDispatcher<UserEventKind,
																UserEvent,Solver<E>,L,E>,)
	-> MaybeMate where E: PlayerError, L: Logger, F: FnMut() -> bool, S: FnMut(u32) {

		self.response_oute(teban,state,
							mc,m,&KyokumenMap::new(),
							oute_kyokumen_map,
							already_oute_kyokumen_map,
							current_kyokumen_map,
							hasher,
							mhash,shash,
							current_depth+1,
							check_timelimit,
							stop,
							on_searchstart,
							event_queue,
							event_dispatcher)
	}

	fn response_oute<L,F,S>(&mut self,
							teban:Teban,state:&State,
							mc:&MochigomaCollections,
							m:LegalMove,
							ignore_kyokumen_map:&KyokumenMap<u64,()>,
							oute_kyokumen_map:&mut KyokumenMap<u64,()>,
							already_oute_kyokumen_map:&mut KyokumenMap<u64,bool>,
							current_kyokumen_map:&mut KyokumenMap<u64,u32>,
							hasher:&Search,
							mhash:u64,shash:u64,
							current_depth:u32,
							check_timelimit:&mut F,
							stop:&Arc<AtomicBool>,
							on_searchstart:&mut S,
							event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
							event_dispatcher:&mut USIEventDispatcher<UserEventKind,
																UserEvent,Solver<E>,L,E>
	) -> MaybeMate where E: PlayerError, L: Logger, F: FnMut() -> bool, S: FnMut(u32) {
		on_searchstart(current_depth);

		let mvs = Rule::legal_moves_all(teban, state, mc);

		let _ = event_dispatcher.dispatch_events(self,&*event_queue);

		if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
			return MaybeMate::Timeout;
		}

		if mvs.len() == 0 {
			MaybeMate::Mate(current_depth)
		} else {
			let mut pmvs = Vec::with_capacity(mvs.len());

			for m in mvs.into_iter() {
				let o = match m {
					LegalMove::To(ref m) => {
						match m.obtained() {
							Some(ObtainKind::Ou) => {
								return MaybeMate::Nomate
							},
							Some(o) => {
								MochigomaKind::try_from(o).ok()
							},
							None => None,
						}
					},
					_ => None,
				};

				let mhash = hasher.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = hasher.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();

				if let Some(()) = ignore_kyokumen_map.get(teban,&mhash,&shash) {
					continue;
				} else {
					ignore_kyokumen_map.insert(teban,mhash,shash,());
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

				let next = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

				match next {
					(ref next,ref mc,_) => {
						pmvs.push((m,Rule::oute_only_moves_all(teban.opposite(), next, mc).len()));
					}
				}

				let _ = event_dispatcher.dispatch_events(self,&*event_queue);

				if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
					return MaybeMate::Timeout;
				}
			}

			pmvs.sort_by(|a,b| b.1.cmp(&a.1));

			for (m,_) in pmvs.into_iter() {
				let o = match m {
					LegalMove::To(ref m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
					_ => None,
				};

				let mhash = hasher.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = hasher.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

				match next {
					(ref next,ref mc,_) => {
						if let Some(true) = already_oute_kyokumen_map.get(teban,&mhash,&shash) {
							continue;
						}

						let mut oute_kyokumen_map = {
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

								oute_kyokumen_map
							} else {
								let mut oute_kyokumen_map = oute_kyokumen_map.clone();
								oute_kyokumen_map.clear(teban);
								oute_kyokumen_map
							}
						};

						match self.oute_only(teban,next,
													mc,m,&ignore_kyokumen_map,
													&mut oute_kyokumen_map,
													already_oute_kyokumen_map,
													current_kyokumen_map,
													hasher,
													mhash,shash,
													current_depth+1,
													check_timelimit,
													stop,
													on_searchstart,
													event_queue,
													event_dispatcher) {
							MaybeMate::Mate(_) => (),
							MaybeMate::Nomate => {
								return MaybeMate::Nomate;
							},
							r @ _ => {
								return r;
							}
						}
					}
				}

				let _ = event_dispatcher.dispatch_events(self,&*event_queue);

				if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
					return MaybeMate::Timeout;
				}
			}

			MaybeMate::Mate(current_depth)
		}
	}

	fn oute_only<L,F,S>(&mut self,
							teban:Teban,state:&State,
							mc:&MochigomaCollections,
							m:LegalMove,
							ignore_kyokumen_map:&KyokumenMap<u64,()>,
							oute_kyokumen_map:&mut KyokumenMap<u64,()>,
							already_oute_kyokumen_map:&mut KyokumenMap<u64,bool>,
							current_kyokumen_map:&mut KyokumenMap<u64,u32>,
							hasher:&Search,
							mhash:u64,shash:u64,
							current_depth:u32,
							check_timelimit:&mut F,
							stop:&Arc<AtomicBool>,
							on_searchstart:&mut S,
							event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
							event_dispatcher:&mut USIEventDispatcher<UserEventKind,
																UserEvent,Solver<E>,L,E>
	) -> MaybeMate where E: PlayerError, L: Logger, F: FnMut() -> bool, S: FnMut(u32) {
		on_searchstart(current_depth);

		let mvs = Rule::oute_only_moves_all(teban, state, mc);

		let _ = event_dispatcher.dispatch_events(self,&*event_queue);

		if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
			return MaybeMate::Timeout;
		}

		if mvs.len() == 0 {
			return MaybeMate::Nomate;
		} else {
			let mut pmvs = Vec::with_capacity(mvs.len());

			for m in mvs.into_iter() {
				match m {
					LegalMove::To(ref m) => {
						if let Some(ObtainKind::Ou) = m.obtained() {
							return MaybeMate::Mate(current_depth);
						}
					},
					_ => ()
				}

				let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

				let o = match m {
					LegalMove::To(ref m) => {
						m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
					},
					_ => None,
				};

				let mhash = hasher.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = hasher.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				let completed = already_oute_kyokumen_map.get(teban,&mhash,&shash);

				if let Some(true) = completed {
					return MaybeMate::Mate(current_depth);
				}

				let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();

				if let Some(()) = ignore_kyokumen_map.get(teban,&mhash,&shash) {
					continue;
				} else {
					ignore_kyokumen_map.insert(teban,mhash,shash,());
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
						pmvs.push((m,Rule::legal_moves_all(teban.opposite(), next, mc).len()));
					}
				}

				let _ = event_dispatcher.dispatch_events(self,&*event_queue);

				if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
					return MaybeMate::Timeout;
				}
			}

			pmvs.sort_by(|a,b| a.1.cmp(&b.1));

			for (m,_) in pmvs.into_iter() {
				let is_put_fu = match m {
					LegalMove::Put(ref m) if m.kind() == MochigomaKind::Fu => true,
					_ => false,
				};

				let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

				let o = match m {
					LegalMove::To(ref m) => {
						m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
					},
					_ => None,
				};

				let mhash = hasher.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = hasher.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				match next {
					(ref next,ref mc,_) => {
						match self.response_oute(teban,next,
													mc,m,&ignore_kyokumen_map,
													oute_kyokumen_map,
													already_oute_kyokumen_map,
													current_kyokumen_map,
													hasher,
													mhash,shash,
													current_depth+1,
													check_timelimit,
													stop,
													on_searchstart,
													event_queue,
													event_dispatcher) {
							MaybeMate::Nomate => (),
							MaybeMate::Mate(d) if !(is_put_fu && d - current_depth == 2)=> {
								already_oute_kyokumen_map.insert(teban,mhash,shash,true);
								return MaybeMate::Mate(d);
							},
							r @ _ => {
								return r;
							},
						}
					}
				}

				let _ = event_dispatcher.dispatch_events(self,&*event_queue);

				if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
					return MaybeMate::Timeout;
				}
			}

			MaybeMate::Nomate
		}
	}
}