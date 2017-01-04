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

import copy
import traceback
import types

import six

from oslo_utils import encodeutils
from oslo_utils import reflection

exception_message = encodeutils.exception_to_unicode


class StrMixin(object):
    """Mixin that helps deal with the PY2 and PY3 method differences.

    http://lucumr.pocoo.org/2011/1/22/forwards-compatible-python/ explains
    why this is quite useful...
    """

    if six.PY2:
        def __str__(self):
            try:
                return self.__bytes__()
            except AttributeError:
                return self.__unicode__().encode('utf-8')
    else:
        def __str__(self):
            return self.__unicode__()


def mod_to_mod_name(mod):
    if isinstance(mod, types.ModuleType):
        mod_name = mod.__name__
    else:
        mod_name = str(mod)
    return mod_name


def cls_to_cls_name(cls):
    if isinstance(cls, type):
        cls_name = reflection.get_class_name(cls, truncate_builtins=False)
    else:
        cls_name = str(cls)
    return cls_name


def extract_roots(exc_type):
    return to_tuple(
        reflection.get_all_class_names(exc_type, up_to=BaseException,
                                       truncate_builtins=False))


def array_prefix_matches(src, cmp_to, on_src_empty=False):
    src_len = len(src)
    if src_len == 0:
        return on_src_empty
    return cmp_to[0:src_len] == src


def to_tuple(vals, on_none=()):
    if isinstance(vals, tuple):
        return vals
    else:
        if vals is None:
            return on_none
        return tuple(vals)


def copy_exc_info(exc_info, deep=False):
    if exc_info is None:
        return None
    exc_type, exc_value, exc_tb = exc_info
    # NOTE(imelnikov): there is no need to copy the exception type, and
    # a shallow copy of the value is fine and we can't copy the traceback since
    # it contains reference to the internal stack frames...
    if deep:
        return (exc_type, copy.deepcopy(exc_value), exc_tb)
    else:
        return (exc_type, copy.copy(exc_value), exc_tb)


def are_equal_exc_info_tuples(ei1, ei2):
    if ei1 == ei2:
        return True
    if ei1 is None or ei2 is None:
        return False  # if both are None, we returned True above
    # NOTE(imelnikov): we can't compare exceptions with '=='
    # because we want exc_info be equal to it's copy made with
    # copy_exc_info above.
    if ei1[0] is not ei2[0]:
        return False
    if not all((type(ei1[1]) == type(ei2[1]),  # noqa
                exception_message(ei1[1]) == exception_message(ei2[1]),
                repr(ei1[1]) == repr(ei2[1]))):
        return False
    if ei1[2] == ei2[2]:
        return True
    tb1 = traceback.format_tb(ei1[2])
    tb2 = traceback.format_tb(ei2[2])
    return tb1 == tb2
