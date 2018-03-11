use std::fmt;
use std::error;
use std::io;
use std::convert::From;
use usiagent::event::SystemEventKind;
use usiagent::event::UserEventKind;
use usiagent::error::USIAgentStartupError;
use usiagent::error::EventHandlerError;
use usiagent::error::PlayerError;
use simplenn::error::InvalidStateError;
use simplenn::error::PersistenceError;

#[derive(Debug)]
pub enum CommonError {
	Fail(String),
}
impl PlayerError for CommonError {

}
impl fmt::Display for CommonError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			CommonError::Fail(ref s) => write!(f, "{}",s),
		}
	}
}
impl error::Error for CommonError {
	fn description(&self) -> &str {
		match *self {
			CommonError::Fail(_) => "Player error.",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			CommonError::Fail(_) => None,
		}
	}
}
impl From<CommonError> for EventHandlerError<SystemEventKind,CommonError>
	where SystemEventKind: fmt::Debug {
	fn from(err: CommonError) -> EventHandlerError<SystemEventKind,CommonError> {
		EventHandlerError::PlayerError(err)
	}
}
impl<'a> From<CommonError> for USIAgentStartupError<'a,SystemEventKind,CommonError>
	where SystemEventKind: fmt::Debug {
	fn from(err: CommonError) -> USIAgentStartupError<'a,SystemEventKind,CommonError> {
		USIAgentStartupError::PlayerError(err)
	}
}
impl From<CommonError> for EventHandlerError<UserEventKind,CommonError>
	where UserEventKind: fmt::Debug {
	fn from(err: CommonError) -> EventHandlerError<UserEventKind,CommonError> {
		EventHandlerError::PlayerError(err)
	}
}
impl<'a> From<CommonError> for USIAgentStartupError<'a,UserEventKind,CommonError>
	where UserEventKind: fmt::Debug {
	fn from(err: CommonError) -> USIAgentStartupError<'a,UserEventKind,CommonError> {
		USIAgentStartupError::PlayerError(err)
	}
}
impl From<InvalidStateError> for CommonError {
	fn from(_: InvalidStateError) -> CommonError {
		CommonError::Fail(String::from("invalid state."))
	}
}
impl From<io::Error> for CommonError {
	fn from(_: io::Error) -> CommonError {
		CommonError::Fail(String::from("I/O Error."))
	}
}
impl<E> From<PersistenceError<E>> for CommonError {
	fn from(_: PersistenceError<E>) -> CommonError {
		CommonError::Fail(String::from("An error occurred while saving model of NN."))
	}
}

