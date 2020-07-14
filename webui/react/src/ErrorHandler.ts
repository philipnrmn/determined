import { notification } from 'antd';
import axios from 'axios';

import history from 'routes/history';
import { clone, isAsyncFunction } from 'utils/data';
import Logger, { LoggerInterface } from 'utils/Logger';
import { listToStr } from 'utils/string';

const logger = new Logger('EH');

export const isDaError = (e: unknown): e is DaError => {
  if (!e || typeof e !== 'object') return false;
  if (e === null) return false; // TS: check cannot be included in the previous check.
  const keys = Object.keys(e);
  const requiredKeys = [ 'level', 'type', 'silent' ];
  return !requiredKeys.find(reqKey => !keys.includes(reqKey));
};

export enum ErrorLevel {
  Fatal = 'fatal',
  Error = 'error',
  Warn = 'warning',
}

export enum ErrorType {
  Server = 'server', // internal apis and server errors.
  Auth = 'auth',
  Ui = 'ui',
  Input = 'input',
  ApiBadResponse = 'apiBadResponse',
  Api = 'api', // third-party api
}

export interface DaError {
  error?: Error;
  id?: string; // slug unique to each place in the codebase that we will use this.
  level?: ErrorLevel;
  message: string; // internal message.
  payload?: unknown;
  silent?: boolean;
  isUserTriggered?: boolean; // whether the error was caused by an active interaction.
  logger?: LoggerInterface;
  publicSubject?: string;
  publicMessage?: string;
  type: ErrorType;
}

const openNotification = (e: DaError): void => {
  const config = {
    description: e.publicMessage,
    message: e.publicSubject || listToStr([ e.type, e.level ]),
  };
  switch (e.level) {
    case ErrorLevel.Fatal:
    case ErrorLevel.Error:
      notification.error(config);
      break;
    case ErrorLevel.Warn:
      notification.warn(config);
      break;
  }
};

const defaultErrorParameters = {
  level: ErrorLevel.Error,
  silent: false,
};

const handleError = (e: DaError): DaError => {
  // set the defaults.
  e = { ...defaultErrorParameters, ...e };

  // ignore request cancellation errors
  if (axios.isCancel(e)) return e;

  if (e.type === ErrorType.Auth) {
    if (!window.location.pathname.endsWith('login')) {
      history.replace('/det/logout', { loginRedirect: clone(window.location ) });
    }
    return e;
  }

  // TODO add support and checking for saving and dismissing class of errors as user preference
  // using id.
  const skipNotification = e.silent || (e.level === ErrorLevel.Warn && !e.publicMessage);
  if (!skipNotification) openNotification(e);

  // TODO generate stack trace if error is missing? http://www.stacktracejs.com/

  // log the error if needed.
  if (!e.silent) {
    const msg = listToStr([ `${e.type}:`, e.publicMessage, e.message ]);
    const targetLogger = e.logger || logger;
    switch (e.level) {
      case ErrorLevel.Fatal:
      case ErrorLevel.Error:
        targetLogger.error(msg);
        break;
      case ErrorLevel.Warn:
        targetLogger.warn(msg);
        break;
    }
  }

  // TODO SEP handle transient failures? eg only take action IF.. (requires keeping state)

  // Report to segment.
  if (window.analytics) { // analytics can be turned off
    window.analytics.track(`EH:${e.level}`, e);
  }

  // TODO SEP capture a screenshot or more context (generate a call stack)?
  // https://stackblitz.com/edit/react-screen-capture?file=index.js

  return e;
};

export const handleErrorForFn = async <T>(daError: DaError, func: () => T): Promise<T|DaError> => {
  try {
    return isAsyncFunction(func) ? await func() : func();
  } catch (e) {
    return handleError({ ...daError, error: daError.error || e });
  }
};

export default handleError;
