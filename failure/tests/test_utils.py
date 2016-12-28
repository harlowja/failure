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

import sys

from oslotest import base

from failure import _utils as utils


def _make_exc_info(msg):
    try:
        raise RuntimeError(msg)
    except Exception:
        return sys.exc_info()


class ExcInfoUtilsTest(base.BaseTestCase):
    def test_copy_none(self):
        result = utils.copy_exc_info(None)
        self.assertIsNone(result)

    def test_copy_exc_info(self):
        exc_info = _make_exc_info("Woot!")
        result = utils.copy_exc_info(exc_info)
        self.assertIsNot(result, exc_info)
        self.assertIs(result[0], RuntimeError)
        self.assertIsNot(result[1], exc_info[1])
        self.assertIs(result[2], exc_info[2])

    def test_none_equals(self):
        self.assertTrue(utils.are_equal_exc_info_tuples(None, None))

    def test_none_ne_tuple(self):
        exc_info = _make_exc_info("Woot!")
        self.assertFalse(utils.are_equal_exc_info_tuples(None, exc_info))

    def test_tuple_nen_none(self):
        exc_info = _make_exc_info("Woot!")
        self.assertFalse(utils.are_equal_exc_info_tuples(exc_info, None))

    def test_tuple_equals_itself(self):
        exc_info = _make_exc_info("Woot!")
        self.assertTrue(utils.are_equal_exc_info_tuples(exc_info, exc_info))

    def test_tuple_equals_copy(self):
        exc_info = _make_exc_info("Woot!")
        copied = utils.copy_exc_info(exc_info)
        self.assertTrue(utils.are_equal_exc_info_tuples(exc_info, copied))
