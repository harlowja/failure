# -*- coding: utf-8 -*-

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

import pickle

from oslotest import base

import failure
from failure import finders


class MySpecialError(Exception):
    pass


class FindersTest(base.BaseTestCase):
    def test_regeneration_cls_finder(self):
        f = None
        try:
            raise IOError("I broke")
        except IOError:
            f = failure.from_exc_info()
        self.assertIsNotNone(f)
        f = pickle.dumps(f)
        f = pickle.loads(f)
        e = self.assertRaises(
            IOError,
            f.reraise, cause_cls_finder=failure.match_classes([IOError]))
        self.assertEqual(str(e), "I broke")

    def test_regeneration_no_cls_finder(self):
        f = None
        try:
            raise IOError("I broke")
        except IOError:
            f = failure.from_exc_info()
        self.assertIsNotNone(f)
        f = f.to_dict()
        f = failure.Failure.from_dict(f)
        e = self.assertRaises(
            failure.WrappedFailure,
            f.reraise,
            cause_cls_finder=failure.match_classes([MySpecialError]))

    def test_regeneration_mod_finder(self):
        f = None
        try:
            raise IOError("I broke")
        except IOError:
            f = failure.from_exc_info()
        self.assertIsNotNone(f)
        f = pickle.dumps(f)
        f = pickle.loads(f)
        e = self.assertRaises(
            IOError,
            f.reraise,
            cause_cls_finder=failure.match_modules([IOError.__module__]))
        self.assertEqual(str(e), "I broke")
