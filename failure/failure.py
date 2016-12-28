# -*- coding: utf-8 -*-

#    Copyright (C) 2014 Yahoo! Inc. All Rights Reserved.
#    Copyright (C) 2016 GoDaddy Inc. All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

from __future__ import absolute_import

import collections
import itertools
import os
import sys
import traceback

import jsonschema
import six

from failure import _utils as utils


def _py2_str(self):
    return self.__unicode__().encode('utf-8')


def _py3_str(self):
    return self.__unicode__()


if six.PY2:
    _py_str = _py2_str
else:
    _py_str = _py3_str


class InvalidFormat(ValueError):
    """Exception raised when data is not in the right format."""


class WrappedFailure(Exception):
    """Wraps one or several failure objects.

    When exception/s cannot be re-raised (for example, because the value and
    traceback are lost in serialization) or there are several exceptions active
    at the same time (due to more than one thread raising exceptions), we will
    wrap the corresponding failure objects into this exception class and
    *may* reraise this exception type to allow users to handle the contained
    failures/causes as they see fit...

    See the failure class documentation for a more comprehensive set of reasons
    why this object *may* be reraised instead of the original exception.

    :param causes: the :py:class:`~failure.Failure` objects
                   that caused this this exception to be raised.
    """

    def __init__(self, causes):
        super(WrappedFailure, self).__init__()
        self._causes = []
        for cause in causes:
            if cause.check(type(self)) and cause.exception is not None:
                # NOTE(imelnikov): flatten wrapped failures.
                self._causes.extend(cause.exception)
            else:
                self._causes.append(cause)

    def __iter__(self):
        """Iterate over failures that caused the exception."""
        return iter(self._causes)

    def __len__(self):
        """Return number of wrapped failures."""
        return len(self._causes)

    def check(self, *exc_classes):
        """Check if any of exception classes caused the failure/s.

        :param exc_classes: exception types/exception type names to
                            search for.

        If any of the contained failures were caused by an exception of a
        given type, the corresponding argument that matched is returned. If
        not then ``None`` is returned.
        """
        if not exc_classes:
            return None
        for cause in self:
            result = cause.check(*exc_classes)
            if result is not None:
                return result
        return None

    def __bytes__(self):
        buf = six.BytesIO()
        buf.write(b'WrappedFailure: [')
        causes_gen = (six.binary_type(cause) for cause in self._causes)
        buf.write(b", ".join(causes_gen))
        buf.write(b']')
        return buf.getvalue()

    def __unicode__(self):
        buf = six.StringIO()
        buf.write(u'WrappedFailure: [')
        causes_gen = (six.text_type(cause) for cause in self._causes)
        buf.write(u", ".join(causes_gen))
        buf.write(u']')
        return buf.getvalue()


class Failure(object):
    """An immutable object that represents failure.

    Failure objects encapsulate exception information so that they can be
    re-used later to re-raise, inspect, examine, log, print, serialize,
    deserialize...

    For those who are curious, here are a few reasons why the original
    exception itself *may* not be reraised and instead a reraised wrapped
    failure exception object will be instead. These explanations are *only*
    applicable when a failure object is serialized and deserialized (when it is
    retained inside the python process that the exception was created in the
    the original exception can be reraised correctly without issue).

    * Traceback objects are not serializable/recreatable, since they contain
      references to stack frames at the location where the exception was
      raised. When a failure object is serialized and sent across a channel
      and recreated it is *not* possible to restore the original traceback and
      originating stack frames.
    * The original exception *type* can not be guaranteed to be found, workers
      can run code that is not accessible/available when the failure is being
      deserialized. Even if it was possible to use pickle safely it would not
      be possible to find the originating exception or associated code in this
      situation.
    * The original exception *type* can not be guaranteed to be constructed in
      a *correct* manner. At the time of failure object creation the exception
      has already been created and the failure object can not assume it has
      knowledge (or the ability) to recreate the original type of the captured
      exception (this is especially hard if the original exception was created
      via a complex process via some custom exception constructor).
    * The original exception *type* can not be guaranteed to be constructed in
      a *safe* manner. Importing *foreign* exception types dynamically can be
      problematic when not done correctly and in a safe manner; since failure
      objects can capture any exception it would be *unsafe* to try to import
      those exception types namespaces and modules on the receiver side
      dynamically (this would create similar issues as the ``pickle`` module in
      python has where foreign modules can be imported, causing those modules
      to have code ran when this happens, and this can cause issues and
      side-effects that the receiver would not have intended to have caused).

    TODO(harlowja): use parts of http://bugs.python.org/issue17911 and the
    backport at https://pypi.python.org/pypi/traceback2/ to (hopefully)
    simplify the methods and contents of this object...
    """

    BASE_EXCEPTIONS = ('exceptions.BaseException', 'exceptions.Exception')
    """
    Root exceptions of all other python exceptions (as a string).

    See: https://docs.python.org/2/library/exceptions.html
    """

    #: Expected failure schema (in json schema format).
    SCHEMA = {
        "$ref": "#/definitions/cause",
        "definitions": {
            "cause": {
                "type": "object",
                'properties': {
                    'exc_args': {
                        "type": "array",
                        "minItems": 0,
                    },
                    'exc_kwargs': {
                        "type": "object",
                        "additionalProperties": True,
                    },
                    'exception_str': {
                        "type": "string",
                    },
                    'traceback_str': {
                        "type": "string",
                    },
                    'exc_type_names': {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "minItems": 1,
                    },
                    'cause': {
                        "type": "object",
                        "$ref": "#/definitions/cause",
                    },
                },
                "required": [
                    "exception_str",
                    'traceback_str',
                    'exc_type_names',
                ],
                "additionalProperties": True,
            },
        },
    }

    __str__ = _py_str

    def __init__(self, exc_info=None, exc_args=None,
                 exc_kwargs=None, exception_str='',
                 exc_type_names=None, cause=None,
                 traceback_str=''):
        exc_type_names = utils.to_tuple(exc_type_names)
        if not exc_type_names:
            raise ValueError("Invalid exception type (no type names"
                             " provided)")
        self._exc_type_names = exc_type_names
        self._exc_info = utils.to_tuple(exc_info, on_none=None)
        self._exc_args = utils.to_tuple(exc_args)
        if exc_kwargs:
            self._exc_kwargs = dict(exc_kwargs)
        else:
            self._exc_kwargs = {}
        self._exception_str = exception_str
        self._cause = cause
        self._traceback_str = traceback_str

    @classmethod
    def from_exc_info(cls, exc_info=None, retain_exc_info=True):
        """Creates a failure object from a ``sys.exc_info()`` tuple."""
        if exc_info is None:
            exc_info = sys.exc_info()
            if not any(exc_info):
                raise RuntimeError("No exception currently being handled")
        # This should always be the (type, value, traceback) tuple,
        # either from a prior sys.exc_info() call or from some other
        # creation...
        if len(exc_info) != 3:
            raise ValueError("Provided 'exc_info' must contain three"
                             " elements")
        exc_type, exc_val, exc_tb = exc_info
        try:
            if exc_type is None or exc_val is None:
                raise ValueError("Invalid exception tuple (exception"
                                 " type and exception value must"
                                 " be provided)")
            exc_args = tuple(getattr(exc_val, 'args', []))
            exc_kwargs = dict(getattr(exc_val, 'kwargs', {}))
            exc_type_names = utils.extract_roots(exc_type)
            if not exc_type_names:
                # This should only be possible if the exception provided
                # was not really an exception...
                raise TypeError("Invalid exception type '%s' (not an"
                                " exception)" % (exc_type,))
            exception_str = utils.exception_message(exc_val)
            if hasattr(exc_val, '__traceback_str__'):
                traceback_str = exc_val.__traceback_str__
            else:
                if exc_tb is not None:
                    traceback_str = '\n'.join(
                        traceback.format_exception(*exc_info))
                else:
                    traceback_str = ''
            if not retain_exc_info:
                exc_info = None
            return cls(exc_info=exc_info, exc_args=exc_args,
                       exc_kwargs=exc_kwargs, exception_str=exception_str,
                       exc_type_names=exc_type_names,
                       cause=cls._extract_cause(exc_val),
                       traceback_str=traceback_str)
        finally:
            del exc_type, exc_val, exc_tb

    @classmethod
    def from_exception(cls, exception, retain_exc_info=True):
        """Creates a failure object from a exception instance."""
        exc_info = (
            type(exception),
            exception,
            getattr(exception, '__traceback__', None)
        )
        return cls.from_exc_info(exc_info=exc_info,
                                 retain_exc_info=retain_exc_info)

    @classmethod
    def validate(cls, data):
        """Validate input data matches expected failure ``dict`` format."""
        try:
            jsonschema.validate(
                data, cls.SCHEMA,
                # See: https://github.com/Julian/jsonschema/issues/148
                types={'array': (list, tuple)})
        except jsonschema.ValidationError as e:
            raise InvalidFormat("Failure data not of the"
                                " expected format: %s" % (e.message))
        else:
            # Ensure that all 'exc_type_names' originate from one of
            # base exceptions, because those are the root exceptions that
            # python mandates/provides and anything else is invalid...
            causes = collections.deque([data])
            while causes:
                cause = causes.popleft()
                root_exc_type = cause['exc_type_names'][-1]
                if root_exc_type not in cls.BASE_EXCEPTIONS:
                    raise InvalidFormat(
                        "Failure data 'exc_type_names' must"
                        " have an initial exception type that is one"
                        " of %s types: '%s' is not one of those"
                        " types" % (cls.BASE_EXCEPTIONS, root_exc_type))
                sub_cause = cause.get('cause')
                if sub_cause is not None:
                    causes.append(sub_cause)

    def _matches(self, other):
        if self is other:
            return True
        return (self.exception_type_names == other.exception_type_names and
                self.exception_args == other.exception_args and
                self.exception_kwargs == other.exception_kwargs and
                self.exception_str == other.exception_str and
                self.traceback_str == other.traceback_str and
                self.cause == other.cause)

    def matches(self, other):
        """Checks if another object is equivalent to this object.

        :returns: checks if another object is equivalent to this object
        :rtype: boolean
        """
        if not isinstance(other, Failure):
            return False
        if self.exc_info is None or other.exc_info is None:
            return self._matches(other)
        else:
            return self == other

    def __eq__(self, other):
        if not isinstance(other, Failure):
            return NotImplemented
        return (self._matches(other) and
                utils.are_equal_exc_info_tuples(self.exc_info,
                                                other.exc_info))

    def __ne__(self, other):
        return not (self == other)

    # NOTE(imelnikov): obj.__hash__() should return same values for equal
    # objects, so we should redefine __hash__. Failure equality semantics
    # is a bit complicated, so for now we just mark Failure objects as
    # unhashable. See python docs on object.__hash__  for more info:
    # http://docs.python.org/2/reference/datamodel.html#object.__hash__
    __hash__ = None

    @property
    def exception(self):
        """Exception value, or ``None`` if exception value is not present.

        Exception value *may* be lost during serialization.
        """
        if self._exc_info:
            return self._exc_info[1]
        else:
            return None

    @property
    def exception_str(self):
        """String representation of exception."""
        return self._exception_str

    @property
    def exception_args(self):
        """Tuple of arguments given to the exception constructor."""
        return self._exc_args

    @property
    def exception_kwargs(self):
        """Dict of keyword arguments given to the exception constructor."""
        return self._exc_kwargs

    @property
    def exception_type_names(self):
        """Tuple of current exception type **names**."""
        return self._exc_type_names

    @property
    def exc_info(self):
        """Exception info tuple or none.

        See: https://docs.python.org/2/library/sys.html#sys.exc_info for what
             the contents of this tuple are (if none, then no contents can
             be examined).
        """
        return self._exc_info

    @property
    def traceback_str(self):
        """Exception traceback as string."""
        return self._traceback_str

    @staticmethod
    def reraise_if_any(failures, cause_cls_finder=None):
        """Re-raise exceptions if argument is not empty.

        If argument is empty list/tuple/iterator, this method returns
        None. If argument is converted into a list with a
        single ``Failure`` object in it, that failure is reraised. Else, a
        :class:`~.WrappedFailure` exception is raised with the failure
        list as causes.
        """
        if not isinstance(failures, (list, tuple)):
            # Convert generators/other into a list...
            failures = list(failures)
        if len(failures) == 1:
            failures[0].reraise(cause_cls_finder=cause_cls_finder)
        elif len(failures) > 1:
            raise WrappedFailure(failures)

    def reraise(self, cause_cls_finder=None):
        """Re-raise captured exception (possibly trying to recreate)."""
        if self._exc_info:
            six.reraise(*self._exc_info)
        else:
            # Attempt to regenerate the full chain (and then raise
            # from the root); without a traceback, oh well...
            root = None
            parent = None
            for cause in itertools.chain([self], self.iter_causes()):
                if cause_cls_finder is not None:
                    cause_cls = cause_cls_finder(cause)
                else:
                    cause_cls = None
                if cause_cls is None:
                    # Unable to find where this cause came from, give up...
                    raise WrappedFailure([self])
                exc = cause_cls(
                    *cause.exception_args, **cause.exception_kwargs)
                # Saving this will ensure that if this same exception
                # is serialized again that we will extract the traceback
                # from it directly (thus proxying along the original
                # traceback as much as we can).
                exc.__traceback_str__ = cause.traceback_str
                if root is None:
                    root = exc
                if parent is not None:
                    parent.__cause__ = exc
                parent = exc
            six.reraise(type(root), root, tb=None)

    def check(self, *exc_classes):
        """Check if any of ``exc_classes`` caused the failure.

        Arguments of this method can be exception types or type
        names (strings **fully qualified**). If captured exception is
        an instance of exception of given type, the corresponding argument
        is returned, otherwise ``None`` is returned.
        """
        for cls in exc_classes:
            cls_name = utils.cls_to_cls_name(cls)
            if cls_name in self._exc_type_names:
                return cls
        return None

    @property
    def cause(self):
        """Nested failure *cause* of this failure.

        This property is typically only useful on 3.x or newer versions
        of python as older versions do **not** have associated causes.

        Refer to :pep:`3134` and :pep:`409` and :pep:`415` for what
        this is examining to find failure causes.
        """
        return self._cause

    def __unicode__(self):
        return self.pformat()

    def pformat(self, traceback=False):
        """Pretty formats the failure object into a string."""
        buf = six.StringIO()
        if not self._exc_type_names:
            buf.write('Failure: %s' % (self._exception_str))
        else:
            buf.write('Failure: %s: %s' % (self._exc_type_names[0],
                                           self._exception_str))
        if traceback:
            if self._traceback_str is not None:
                traceback_str = self._traceback_str.rstrip()
            else:
                traceback_str = None
            if traceback_str:
                buf.write(os.linesep)
                buf.write(traceback_str)
            else:
                buf.write(os.linesep)
                buf.write('Traceback not available.')
        return buf.getvalue()

    def iter_causes(self):
        """Iterate over all causes."""
        curr = self._cause
        while curr is not None:
            yield curr
            curr = curr._cause

    def __getstate__(self):
        dct = self.to_dict()
        if self._exc_info:
            # Avoids 'TypeError: can't pickle traceback objects'
            dct['exc_info'] = self._exc_info[0:2]
        return dct

    def __setstate__(self, dct):
        self._exception_str = dct['exception_str']
        if 'exc_args' in dct:
            self._exc_args = tuple(dct['exc_args'])
        else:
            # Guess we got an older version somehow, before this
            # was added, so at that point just set to an empty tuple...
            self._exc_args = ()
        if 'exc_kwargs' in dct:
            self._exc_kwargs = dict(dct['exc_kwargs'])
        else:
            self._exc_kwargs = {}
        self._traceback_str = dct['traceback_str']
        self._exc_type_names = dct['exc_type_names']
        if 'exc_info' in dct:
            # Tracebacks can't be serialized/deserialized, but since we
            # provide a traceback string (and more) this should be
            # acceptable...
            #
            # TODO(harlowja): in the future we could do something like
            # what the twisted people have done, see for example
            # twisted-13.0.0/twisted/python/failure.py#L89 for how they
            # created a fake traceback object...
            exc_info = list(dct['exc_info'])
            while len(exc_info) < 3:
                dct['exc_info'].append(None)
            self._exc_info = tuple(exc_info[0:3])
        else:
            self._exc_info = None
        cause = dct.get('cause')
        if cause is not None:
            cause = self.from_dict(cause)
        self._cause = cause

    @classmethod
    def _extract_cause(cls, exc_val):
        """Helper routine to extract nested cause (if any)."""
        # See: https://www.python.org/dev/peps/pep-3134/ for why/what
        # these are...
        #
        # '__cause__' attribute for explicitly chained exceptions
        # '__context__' attribute for implicitly chained exceptions
        # '__traceback__' attribute for the traceback
        #
        # See: https://www.python.org/dev/peps/pep-0415/ for why/what
        # the '__suppress_context__' is/means/implies...
        suppress_context = getattr(exc_val,
                                   '__suppress_context__', False)
        if suppress_context:
            attr_lookups = ['__cause__']
        else:
            attr_lookups = ['__cause__', '__context__']
        nested_exc_val = None
        for attr_name in attr_lookups:
            attr_val = getattr(exc_val, attr_name, None)
            if attr_val is None:
                continue
            nested_exc_val = attr_val
            break
        if nested_exc_val is not None:
            return cls.from_exception(nested_exc_val)
        else:
            return None

    @classmethod
    def from_dict(cls, data):
        """Converts this from a dictionary to a object."""
        data = dict(data)
        cause = data.get('cause')
        if cause is not None:
            data['cause'] = cls.from_dict(cause)
        return cls(**data)

    def to_dict(self, include_args=True, include_kwargs=True):
        """Converts this object to a dictionary.

        :param include_args: boolean indicating whether to include the
                             exception args in the output.
        :param include_kwargs: boolean indicating whether to include the
                               exception kwargs in the output.
        """
        data = {
            'exception_str': self.exception_str,
            'traceback_str': self.traceback_str,
            'exc_type_names': self.exception_type_names,
            'exc_args': self.exception_args if include_args else tuple(),
            'exc_kwargs': self.exception_kwargs if include_kwargs else {},
        }
        if self._cause is not None:
            data['cause'] = self._cause.to_dict(include_args=include_args,
                                                include_kwargs=include_kwargs)
        return data

    def copy(self):
        """Copies this object."""
        cause = self._cause
        if cause is not None:
            cause = cause.copy()
        return Failure(exc_info=utils.copy_exc_info(self.exc_info),
                       exception_str=self.exception_str,
                       traceback_str=self.traceback_str,
                       exc_args=self.exception_args[:],
                       exc_kwargs=self.exception_kwargs.copy(),
                       exc_type_names=self._exc_type_names[:],
                       cause=cause)
