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

from __future__ import absolute_import

import itertools

from oslo_utils import importutils
from oslo_utils import reflection

from failure import _utils as utils


class InvalidTypeError(TypeError):
    pass


def ensure_base_exception(cause_type_name, cls):
    # Ensure source class is correct (ie that it has the right
    # root that **all** python exceptions must have); if not right then
    # it will be discarded.
    if not issubclass(cls, BaseException):
        raise InvalidTypeError(
            "Cause with type '%s' was regenerated as a non-exception"
            " base class '%s'" % (cause_type_name,
                                  reflection.get_class_name(cls)))
    else:
        return cls


def match_modules(allowed_modules):
    """Creates a matcher that matches a list/set/tuple of allowed modules."""
    cleaned_allowed_modules = [
        utils.mod_to_mod_name(tmp_mod)
        for tmp_mod in allowed_modules
    ]
    cleaned_split_allowed_modules = [
        tmp_mod.split(".")
        for tmp_mod in cleaned_allowed_modules
    ]
    cleaned_allowed_modules = []
    del cleaned_allowed_modules

    def matcher(cause):
        cause_cls = None
        cause_type_name = cause.exception_type_names[0]
        # Rip off the class name (usually at the end).
        cause_type_name_pieces = cause_type_name.split(".")
        cause_type_name_mod_pieces = cause_type_name_pieces[0:-1]
        # Do any modules provided match the provided causes module?
        mod_match = any(
            utils.array_prefix_matches(mod_pieces,
                                       cause_type_name_mod_pieces)
            for mod_pieces in cleaned_split_allowed_modules)
        if mod_match:
            cause_cls = importutils.import_class(cause_type_name)
            cause_cls = ensure_base_exception(cause_type_name, cause_cls)
        return cause_cls

    return matcher


def match_classes(allowed_classes):
    """Creates a matcher that matches a list/tuple of allowed classes."""
    cleaned_allowed_classes = [
        utils.cls_to_cls_name(tmp_cls)
        for tmp_cls in allowed_classes
    ]

    def matcher(cause):
        cause_cls = None
        cause_type_name = cause.exception_type_names[0]
        try:
            cause_cls_idx = cleaned_allowed_classes.index(cause_type_name)
        except ValueError:
            pass
        else:
            cause_cls = allowed_classes[cause_cls_idx]
            if not isinstance(cause_cls, type):
                cause_cls = importutils.import_class(cause_cls)
            cause_cls = ensure_base_exception(cause_type_name, cause_cls)
        return cause_cls

    return matcher


def combine_or(matcher, *more_matchers):
    """Combines more than one matcher together (first that matches wins)."""

    def matcher(cause):
        for sub_matcher in itertools.chain([matcher], more_matchers):
            cause_cls = sub_matcher(cause)
            if cause_cls is not None:
                return cause_cls
        return None

    return matcher
