use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use error::*;

use usiagent::player::*;
use usiagent::command::*;
use usiagent::event::*;
use usiagent::shogi::*;
use usiagent::OnErrorHandler;
use usiagent::Logger;
use usiagent::error::PlayerError;

#[derive(Debug)]
pub struct NNShogiPlayer {
	stop:bool
}
impl NNShogiPlayer {
	pub fn new() -> NNShogiPlayer {
		NNShogiPlayer {
			stop:false
		}
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
	fn set_position(&mut self,teban:Teban,ban:[KomaKind; 81],
					ms:Vec<MochigomaKind>,mg:Vec<MochigomaKind>,n:u32,m:Vec<Move>)
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
	fn gameover(&mut self,s:&GameEndState) -> Result<(),CommonError> {
		Ok(())
	}
	fn quit(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
}
