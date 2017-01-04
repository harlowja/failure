# -*- coding: utf-8 -*-

#    Copyright (C) 2013 Yahoo! Inc. All Rights Reserved.
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

from oslo_utils import encodeutils
from oslotest import base
import six
from six.moves import cPickle as pickle
import testtools

from failure import _utils as utils

from failure import failure

# NOTE(harlowja): These change on py2.x and py3.x so that's why
# we figure them out at runtime...
EXCP_BASE = str(Exception.__module__)
RUNTIME_ERROR_CLASSES = utils.extract_roots(RuntimeError)


def _captured_failure(msg):
    try:
        raise RuntimeError(msg)
    except Exception:
        return failure.Failure.from_exc_info()


class GeneralFailureObjTestsMixin(object):

    def test_captures_message(self):
        self.assertEqual('Woot!', self.fail_obj.exception_str)

    def test_str(self):
        self.assertEqual('Failure: %s.RuntimeError: Woot!' % EXCP_BASE,
                         str(self.fail_obj))

    def test_exception_types(self):
        self.assertEqual(RUNTIME_ERROR_CLASSES,
                         self.fail_obj.exception_type_names)

    def test_pformat_no_traceback(self):
        text = self.fail_obj.pformat()
        self.assertNotIn("Traceback", text)

    def test_check_str(self):
        val = '%s.Exception' % EXCP_BASE
        self.assertEqual(val, self.fail_obj.check(val))

    def test_check_str_not_there(self):
        val = '%s.ValueError' % EXCP_BASE
        self.assertIsNone(self.fail_obj.check(val))

    def test_check_type(self):
        self.assertIs(self.fail_obj.check(RuntimeError), RuntimeError)

    def test_check_type_not_there(self):
        self.assertIsNone(self.fail_obj.check(ValueError))


class CaptureFailureTestCase(base.BaseTestCase,
                             GeneralFailureObjTestsMixin):

    def setUp(self):
        super(CaptureFailureTestCase, self).setUp()
        self.fail_obj = _captured_failure('Woot!')

    def test_captures_value(self):
        self.assertIsInstance(self.fail_obj.exception, RuntimeError)

    def test_captures_exc_info(self):
        exc_info = self.fail_obj.exc_info
        self.assertEqual(3, len(exc_info))
        self.assertEqual(RuntimeError, exc_info[0])
        self.assertIs(exc_info[1], self.fail_obj.exception)

    def test_reraises(self):
        self.assertRaisesRegexp(RuntimeError, '^Woot!$', self.fail_obj.reraise)


class ReCreatedFailureTestCase(base.BaseTestCase,
                               GeneralFailureObjTestsMixin):

    def setUp(self):
        super(ReCreatedFailureTestCase, self).setUp()
        fail_obj = _captured_failure('Woot!')
        self.fail_obj = failure.Failure(
            exception_str=fail_obj.exception_str,
            traceback_str=fail_obj.traceback_str,
            exc_type_names=fail_obj.exception_type_names)

    def test_value_lost(self):
        self.assertIsNone(self.fail_obj.exception)

    def test_no_exc_info(self):
        self.assertIsNone(self.fail_obj.exc_info)

    def test_pformat_traceback(self):
        text = self.fail_obj.pformat(traceback=True)
        self.assertIn("Traceback (most recent call last):", text)

    def test_reraises(self):
        exc = self.assertRaises(failure.WrappedFailure,
                                self.fail_obj.reraise)
        self.assertIs(exc.check(RuntimeError), RuntimeError)

    def test_no_type_names(self):
        fail_obj = _captured_failure('Woot!')
        self.assertRaises(ValueError, failure.Failure,
                          exception_str=fail_obj.exception_str,
                          traceback_str=fail_obj.traceback_str,
                          exc_type_names=[])


class FromExceptionTestCase(base.BaseTestCase,
                            GeneralFailureObjTestsMixin):

    def setUp(self):
        super(FromExceptionTestCase, self).setUp()
        self.fail_obj = failure.Failure.from_exception(RuntimeError('Woot!'))

    def test_pformat_no_traceback(self):
        text = self.fail_obj.pformat(traceback=True)
        self.assertIn("Traceback not available", text)


class FailureObjectTestCase(base.BaseTestCase):

    def test_invalids(self):
        f = {
            'exception_str': 'blah',
            'traceback_str': 'blah',
            'exc_type_names': [],
        }
        self.assertRaises(failure.InvalidFormat,
                          failure.Failure.validate, f)
        f = {
            'exception_str': 'blah',
            'exc_type_names': ['Exception'],
        }
        self.assertRaises(failure.InvalidFormat,
                          failure.Failure.validate, f)
        f = {
            'exception_str': 'blah',
            'traceback_str': 'blah',
            'exc_type_names': ['Exception'],
            'version': -1,
        }
        self.assertRaises(failure.InvalidFormat,
                          failure.Failure.validate, f)

    def test_valid_from_dict_to_dict(self):
        f = _captured_failure('Woot!')
        d_f = f.to_dict()
        failure.Failure.validate(d_f)
        f2 = failure.Failure.from_dict(d_f)
        self.assertTrue(f.matches(f2))

    def test_bad_root_exception(self):
        f = _captured_failure('Woot!')
        d_f = f.to_dict()
        d_f['exc_type_names'] = ['Junk']
        self.assertRaises(failure.InvalidFormat,
                          failure.Failure.validate, d_f)

    def test_valid_from_dict_to_dict_2(self):
        f = _captured_failure('Woot!')
        d_f = f.to_dict()
        d_types = ['RuntimeError', 'Exception', 'BaseException']
        for i, v in enumerate(d_types):
            v = EXCP_BASE + "." + v
            d_types[i] = v
        d_f['exc_type_names'] = d_types
        failure.Failure.validate(d_f)

    def test_cause_exception_args(self):
        f = _captured_failure('Woot!')
        d_f = f.to_dict()
        self.assertEqual(1, len(d_f['exc_args']))
        self.assertEqual(("Woot!",), d_f['exc_args'])
        f2 = failure.Failure.from_dict(d_f)
        self.assertEqual(f.exception_args, f2.exception_args)

    def test_empty_does_not_reraise(self):
        self.assertIsNone(failure.Failure.reraise_if_any([]))

    def test_reraises_one(self):
        fls = [_captured_failure('Woot!')]
        self.assertRaisesRegexp(RuntimeError, '^Woot!$',
                                failure.Failure.reraise_if_any, fls)

    def test_reraises_several(self):
        fls = [
            _captured_failure('Woot!'),
            _captured_failure('Oh, not again!')
        ]
        exc = self.assertRaises(failure.WrappedFailure,
                                failure.Failure.reraise_if_any, fls)
        self.assertEqual(fls, list(exc))

    def test_failure_copy(self):
        fail_obj = _captured_failure('Woot!')

        copied = fail_obj.copy()
        self.assertIsNot(fail_obj, copied)
        self.assertEqual(fail_obj, copied)
        self.assertTrue(fail_obj.matches(copied))

    def test_failure_copy_recaptured(self):
        captured = _captured_failure('Woot!')
        fail_obj = failure.Failure(
            exception_str=captured.exception_str,
            traceback_str=captured.traceback_str,
            exc_type_names=captured.exception_type_names)
        copied = fail_obj.copy()
        self.assertIsNot(fail_obj, copied)
        self.assertEqual(fail_obj, copied)
        self.assertFalse(fail_obj != copied)
        self.assertTrue(fail_obj.matches(copied))

    def test_recaptured_not_eq(self):
        captured = _captured_failure('Woot!')
        fail_obj = failure.Failure(
            exception_str=captured.exception_str,
            traceback_str=captured.traceback_str,
            exc_type_names=captured.exception_type_names,
            exc_args=captured.exception_args,
            generated_on=captured.generated_on)
        self.assertFalse(fail_obj == captured)
        self.assertTrue(fail_obj != captured)
        self.assertTrue(fail_obj.matches(captured))

    def test_two_captured_eq(self):
        captured = _captured_failure('Woot!')
        captured2 = _captured_failure('Woot!')
        self.assertEqual(captured, captured2)

    def test_two_recaptured_neq(self):
        captured = _captured_failure('Woot!')
        fail_obj = failure.Failure(
            exception_str=captured.exception_str,
            traceback_str=captured.traceback_str,
            exc_type_names=captured.exception_type_names)
        new_exc_str = captured.exception_str.replace('Woot', 'w00t')
        fail_obj2 = failure.Failure(
            exception_str=new_exc_str,
            traceback_str=captured.traceback_str,
            exc_type_names=captured.exception_type_names)
        self.assertNotEqual(fail_obj, fail_obj2)
        self.assertFalse(fail_obj2.matches(fail_obj))

    def test_compares_to_none(self):
        captured = _captured_failure('Woot!')
        self.assertIsNotNone(captured)
        self.assertFalse(captured.matches(None))

    def test_pformat_traceback(self):
        captured = _captured_failure('Woot!')
        text = captured.pformat(traceback=True)
        self.assertIn("Traceback (most recent call last):", text)

    def test_pformat_traceback_captured_no_exc_info(self):
        captured = _captured_failure('Woot!')
        captured = failure.Failure.from_dict(captured.to_dict())
        text = captured.pformat(traceback=True)
        self.assertIn("Traceback (most recent call last):", text)

    def test_no_capture_exc_args(self):
        captured = _captured_failure(Exception("I am not valid JSON"))
        fail_obj = failure.Failure(
            exception_str=captured.exception_str,
            traceback_str=captured.traceback_str,
            exc_type_names=captured.exception_type_names,
            exc_args=captured.exception_args)
        fail_json = fail_obj.to_dict(include_args=False)
        self.assertNotEqual(fail_obj.exception_args, fail_json['exc_args'])
        self.assertEqual(fail_json['exc_args'], tuple())


class WrappedFailureTestCase(base.BaseTestCase):

    def test_simple_iter(self):
        fail_obj = _captured_failure('Woot!')
        wf = failure.WrappedFailure([fail_obj])
        self.assertEqual(1, len(wf))
        self.assertEqual([fail_obj], list(wf))

    def test_simple_check(self):
        fail_obj = _captured_failure('Woot!')
        wf = failure.WrappedFailure([fail_obj])
        self.assertEqual(RuntimeError, wf.check(RuntimeError))
        self.assertIsNone(wf.check(ValueError))

    def test_two_failures(self):
        fls = [
            _captured_failure('Woot!'),
            _captured_failure('Oh, not again!')
        ]
        wf = failure.WrappedFailure(fls)
        self.assertEqual(2, len(wf))
        self.assertEqual(fls, list(wf))

    def test_flattening(self):
        f1 = _captured_failure('Wrap me')
        f2 = _captured_failure('Wrap me, too')
        f3 = _captured_failure('Woot!')
        try:
            raise failure.WrappedFailure([f1, f2])
        except Exception:
            fail_obj = failure.Failure.from_exc_info()

        wf = failure.WrappedFailure([fail_obj, f3])
        self.assertEqual([f1, f2, f3], list(wf))


class NonAsciiExceptionsTestCase(base.BaseTestCase):

    def test_exception_with_non_ascii_str(self):
        bad_string = chr(200)
        excp = ValueError(bad_string)
        fail = failure.Failure.from_exception(excp)
        self.assertEqual(encodeutils.exception_to_unicode(excp),
                         fail.exception_str)
        # This is slightly different on py2 vs py3... due to how
        # __str__ or __unicode__ is called and what is expected from
        # both...
        if six.PY2:
            msg = encodeutils.exception_to_unicode(excp)
            expected = 'Failure: %s.ValueError: %s' % (EXCP_BASE,
                                                       msg.encode('utf-8'))
        else:
            expected = u'Failure: %s.ValueError: \xc8' % (EXCP_BASE)
        self.assertEqual(expected, str(fail))

    def test_exception_non_ascii_unicode(self):
        hi_ru = u'привет'
        fail = failure.Failure.from_exception(ValueError(hi_ru))
        self.assertEqual(hi_ru, fail.exception_str)
        self.assertIsInstance(fail.exception_str, six.text_type)
        self.assertEqual(u'Failure: %s.ValueError: %s' % (EXCP_BASE, hi_ru),
                         six.text_type(fail))

    def test_wrapped_failure_non_ascii_unicode(self):
        hi_cn = u'嗨'
        fail = ValueError(hi_cn)
        self.assertEqual(hi_cn, encodeutils.exception_to_unicode(fail))
        fail = failure.Failure.from_exception(fail)
        wrapped_fail = failure.WrappedFailure([fail])
        expected_result = (u"WrappedFailure: "
                           "[Failure: %s.ValueError: %s]" % (EXCP_BASE, hi_cn))
        self.assertEqual(expected_result, six.text_type(wrapped_fail))

    def test_failure_equality_with_non_ascii_str(self):
        bad_string = chr(200)
        fail = failure.Failure.from_exception(ValueError(bad_string))
        copied = fail.copy()
        self.assertEqual(fail, copied)

    def test_failure_equality_non_ascii_unicode(self):
        hi_ru = u'привет'
        fail = failure.Failure.from_exception(ValueError(hi_ru))
        copied = fail.copy()
        self.assertEqual(fail, copied)


@testtools.skipIf(not six.PY3, 'this test only works on python 3.x')
class FailureCausesTest(base.BaseTestCase):

    @classmethod
    def _raise_many(cls, messages):
        if not messages:
            return
        msg = messages.pop(0)
        e = RuntimeError(msg)
        try:
            cls._raise_many(messages)
            raise e
        except RuntimeError as e1:
            six.raise_from(e, e1)

    def test_causes(self):
        f = None
        try:
            self._raise_many(["Still still not working",
                              "Still not working", "Not working"])
        except RuntimeError:
            f = failure.Failure.from_exc_info()

        self.assertIsNotNone(f)
        causes = list(f.iter_causes())
        self.assertEqual(2, len(causes))
        self.assertEqual("Still not working", causes[0].exception_str)
        self.assertEqual("Not working", causes[1].exception_str)

        f = causes[0]
        causes = list(f.iter_causes())
        self.assertEqual(1, len(causes))
        self.assertEqual("Not working", causes[0].exception_str)

        f = causes[0]
        causes = list(f.iter_causes())
        self.assertEqual(0, len(causes))

    def test_causes_to_from_dict(self):
        f = None
        try:
            self._raise_many(["Still still not working",
                              "Still not working", "Not working"])
        except RuntimeError:
            f = failure.Failure.from_exc_info()

        self.assertIsNotNone(f)
        d_f = f.to_dict()
        failure.Failure.validate(d_f)
        f = failure.Failure.from_dict(d_f)
        causes = list(f.iter_causes())
        self.assertEqual(2, len(causes))
        self.assertEqual("Still not working", causes[0].exception_str)
        self.assertEqual("Not working", causes[1].exception_str)

        f = causes[0]
        causes = list(f.iter_causes())
        self.assertEqual(1, len(causes))
        self.assertEqual("Not working", causes[0].exception_str)

        f = causes[0]
        causes = list(f.iter_causes())
        self.assertEqual(0, len(causes))

    def test_causes_pickle(self):
        f = None
        try:
            self._raise_many(["Still still not working",
                              "Still not working", "Not working"])
        except RuntimeError:
            f = failure.Failure.from_exc_info()

        self.assertIsNotNone(f)
        p_f = pickle.dumps(f)
        f = pickle.loads(p_f)

        causes = list(f.iter_causes())
        self.assertEqual(2, len(causes))
        self.assertEqual("Still not working", causes[0].exception_str)
        self.assertEqual("Not working", causes[1].exception_str)

        f = causes[0]
        causes = list(f.iter_causes())
        self.assertEqual(1, len(causes))
        self.assertEqual("Not working", causes[0].exception_str)

        f = causes[0]
        causes = list(f.iter_causes())
        self.assertEqual(0, len(causes))

    def test_causes_suppress_context(self):
        f = None
        try:
            try:
                self._raise_many(["Still still not working",
                                  "Still not working", "Not working"])
            except RuntimeError as e:
                six.raise_from(e, None)
        except RuntimeError:
            f = failure.Failure.from_exc_info()

        self.assertIsNotNone(f)
        causes = list(f.iter_causes())
        self.assertEqual([], list(causes))
